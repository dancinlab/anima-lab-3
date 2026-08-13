from types import SimpleNamespace
from pathlib import Path

import torch

import anima_alive
import anima_unified
from anima_unified import AnimaUnified


def test_claude_dialogue_uses_stdin_and_disables_tools(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="맥락을 기억한 답변\n", stderr="")

    monkeypatch.delenv("ANIMA_CLAUDE_BIN", raising=False)
    monkeypatch.setattr(anima_alive.shutil, "which", lambda name: "/bin/echo")
    monkeypatch.setattr(anima_alive.subprocess, "run", fake_run)

    response = anima_alive.ask_claude(
        "비밀 프로젝트를 기억해",
        "관련 기억 없음",
        [{"role": "user", "content": "앞선 맥락"}],
    )

    assert response == "맥락을 기억한 답변"
    assert "비밀 프로젝트" in captured["input"]
    assert "비밀 프로젝트" not in " ".join(captured["command"])
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
    assert "--safe-mode" in captured["command"]
    assert "--no-session-persistence" in captured["command"]


def test_claude_dialogue_failure_returns_none(monkeypatch):
    monkeypatch.setenv("ANIMA_CLAUDE_BIN", "/bin/echo")
    monkeypatch.setattr(
        anima_alive.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="error"),
    )
    assert anima_alive.ask_claude("질문", "상태", []) is None


def test_runtime_claude_path_uses_history_once_and_falls_back(monkeypatch):
    runtime = AnimaUnified.__new__(AnimaUnified)
    runtime.args = SimpleNamespace(dialogue_backend="claude")
    runtime.history = [
        {"role": "user", "content": "앞선 말"},
        {"role": "assistant", "content": "앞선 답"},
        {"role": "user", "content": "현재 질문"},
    ]
    seen = {}

    def fake_ask(text, state, history):
        seen.update(text=text, state=state, history=history)
        return None

    monkeypatch.setattr(anima_unified, "ask_claude", fake_ask)
    result = runtime._generate_dialogue_answer("현재 질문", "현재 상태", "순수 발화")

    assert result == "순수 발화"
    assert seen["history"] == runtime.history[:-1]
    assert seen["text"] == "현재 질문"


def test_runtime_defaults_to_existing_pure_path():
    runtime = AnimaUnified.__new__(AnimaUnified)
    runtime.args = SimpleNamespace()
    runtime.history = []
    runtime.model = None
    assert runtime._generate_dialogue_answer("질문", "상태", "순수 발화") == "순수 발화"


def test_dialogue_report_header_uses_self_model_estimates_for_all_affect():
    metacognition = {
        'reported_tension': 0.2,
        'reported_curiosity': 0.1,
    }
    direction = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    header = AnimaUnified._build_dialogue_report_header(
        metacognition, direction, 0.0, 2, 0, 'meta',
    )

    assert 'reported_tension=0.200' in header
    assert 'reported_curiosity=0.100' in header
    assert f"mood={anima_alive.compute_mood(0.2, 0.1)}" in header
    assert 'control states are withheld from language' in header
    assert '1.900' not in header


def test_native_dialogue_backend_receives_state_and_prior_history():
    calls = []

    class Native:
        model_type = "native-dialogue"

        def generate_dialogue(self, text, state, history):
            calls.append((text, state, history))
            return "자체 답변"

    runtime = AnimaUnified.__new__(AnimaUnified)
    runtime.args = SimpleNamespace(dialogue_backend="model")
    runtime.model = Native()
    runtime.history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답"},
        {"role": "user", "content": "현재 질문"},
    ]
    assert runtime._generate_dialogue_answer("현재 질문", "현재 상태", "순수 발화") == "자체 답변"
    assert calls == [("현재 질문", "현재 상태", runtime.history[:-1])]


def test_field_runtime_disables_all_execution_paths():
    payload = __import__("scripts.deploy_gate_runtime3", fromlist=["_runtime_payload"])._runtime_payload()
    arguments = payload["ProgramArguments"]
    assert "--dialogue-backend" in arguments
    assert "--no-actions" in arguments
    assert "--no-agent-tools" in arguments
    assert "--no-web-sense" in arguments
    assert "--no-autonomous-learning" in arguments
    assert "--no-dream" in arguments
    assert payload["EnvironmentVariables"]["ANIMA_CLAUDE_BIN"]


def test_explicit_data_root_never_imports_global_legacy_memory(tmp_path, monkeypatch):
    legacy_root = tmp_path / "repository"
    legacy_root.mkdir()
    (legacy_root / "memory_alive.json").write_text('{"turns": [{"text": "legacy"}]}')
    runtime_root = tmp_path / "isolated"
    runtime_root.mkdir()

    runtime = AnimaUnified.__new__(AnimaUnified)
    runtime.model_name = "conscious-lm"
    runtime._uses_default_data_root = False
    runtime.paths = {
        "memory": runtime_root / "memory.json",
        "state": runtime_root / "state.pt",
        "growth": runtime_root / "growth.json",
        "web_memories": runtime_root / "web_memories.json",
    }
    monkeypatch.setattr(anima_unified, "ANIMA_DIR", legacy_root)

    runtime._migrate_legacy_files()

    assert not runtime.paths["memory"].exists()
