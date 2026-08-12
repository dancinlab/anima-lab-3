from pathlib import Path

import pytest

from native_dialogue_lm import NativeDialogueTokenizer
from measurement.native_dialogue_registry import checkpoint_spec_sha256


def test_native_tokenizer_roundtrip_and_role_stop(tmp_path: Path):
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "user: 안녕\nassistant: 반가워요.\n\n"
        "user: hello\nassistant: nice to meet you.\n",
        encoding="utf-8",
    )
    tokenizer = NativeDialogueTokenizer.train([corpus], vocab_size=320)
    encoded = tokenizer.encode("<user>안녕\n<assistant>반가워요.<eos>")
    assert "안녕" in tokenizer.decode(encoded)
    response = tokenizer.encode("반가워요.<eos><user>누출")
    assert tokenizer.trim_response(response) == "반가워요."


def test_native_prompt_keeps_current_turn_and_drops_old_prefix(tmp_path: Path):
    corpus = tmp_path / "train.txt"
    corpus.write_text("사용자 현재 질문 assistant answer history state", encoding="utf-8")
    tokenizer = NativeDialogueTokenizer.train([corpus], vocab_size=320)
    prompt = tokenizer.format_prompt(
        "현재 질문",
        "관련 기억",
        [{"role": "user", "content": "오래된 말 " * 100}],
        max_tokens=40,
    )
    decoded = tokenizer.decode(prompt)
    assert "현재 질문" in decoded
    assert "<assistant>" not in decoded


def test_native_tokenizer_rejects_empty_training_set():
    with pytest.raises(ValueError):
        NativeDialogueTokenizer.train([], vocab_size=320)


def test_checkpoint_compatibility_hash_ignores_research_thresholds(monkeypatch):
    from measurement import native_dialogue_registry as registry

    before = checkpoint_spec_sha256()
    monkeypatch.setitem(registry.NATIVE_DIALOGUE_SPEC["thresholds"], "maximum_empty", 99)
    assert checkpoint_spec_sha256() == before
