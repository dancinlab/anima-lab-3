#!/usr/bin/env python3
"""Install or inspect the canonical local GATE-RUNTIME-3 services on macOS."""
from __future__ import annotations

import json
import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


RUNTIME_LABEL = "com.dancinlab.anima-lab3-field"
COLLECTOR_LABEL = "com.dancinlab.anima-lab3-field-collector"
HOST = "127.0.0.1"
ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".local" / "gate-runtime-venv" / "bin" / "python"
ENTRYPOINT = ROOT / "anima_unified.py"
COLLECTOR = ROOT / "gate_runtime3.py"
DATA_ROOT = ROOT / ".local" / "gate-runtime3" / "data"
STATE_ROOT = ROOT / ".local" / "gate-runtime3"
LOG_ROOT = STATE_ROOT / "logs"
PORT = 8765
DOMAIN = f"gui/{os.getuid()}"
PLIST_ROOT = Path.home() / "Library" / "LaunchAgents"


def _plist(label: str) -> Path:
    return PLIST_ROOT / f"{label}.plist"


def _check_files(dialogue_backend: str = "claude") -> None:
    missing = [str(path) for path in (PYTHON, ENTRYPOINT, COLLECTOR) if not path.is_file()]
    if missing:
        raise SystemExit("missing runtime file(s): " + ", ".join(missing))
    subprocess.run(
        [str(PYTHON), "-c", "import faiss, torch, transformers, websockets"],
        cwd=ROOT,
        check=True,
    )
    if dialogue_backend == "claude":
        claude = shutil.which("claude")
        if not claude or not os.access(claude, os.X_OK):
            raise SystemExit("claude CLI is required for the contextual dialogue runtime")
    elif dialogue_backend == "native":
        missing_model = [
            str(path) for path in (
                ROOT / "models" / "anima-native" / "final.pt",
                ROOT / "models" / "anima-native" / "tokenizer.json",
            ) if not path.is_file()
        ]
        if missing_model:
            raise SystemExit("missing native model file(s): " + ", ".join(missing_model))
    else:
        raise SystemExit(f"unsupported dialogue backend: {dialogue_backend}")


def _runtime_payload(
    dialogue_backend: str = "claude",
    data_root: Path = DATA_ROOT,
    log_root: Path = LOG_ROOT,
) -> dict[str, object]:
    arguments = [
        str(PYTHON), "-u", str(ENTRYPOINT), "--web", "--port", str(PORT),
        "--data-root", str(data_root), "--max-cells", "64", "--no-camera",
        "--no-vision", "--no-telepathy", "--no-cloud",
        "--no-actions", "--no-agent-tools", "--no-web-sense",
        "--no-autonomous-learning", "--no-dream", "--memory-gate-shadow",
    ]
    environment = {}
    if dialogue_backend == "claude":
        claude = shutil.which("claude")
        if not claude:
            raise SystemExit("claude CLI is required for the contextual dialogue runtime")
        arguments.extend(("--no-conscious-lm", "--dialogue-backend", "claude"))
        environment["ANIMA_CLAUDE_BIN"] = claude
    elif dialogue_backend == "native":
        arguments.extend(("--model", "anima-native", "--dialogue-backend", "model"))
    else:
        raise SystemExit(f"unsupported dialogue backend: {dialogue_backend}")
    return {
        "Label": RUNTIME_LABEL,
        "ProgramArguments": arguments,
        "EnvironmentVariables": environment,
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(log_root / "runtime.out.log"),
        "StandardErrorPath": str(log_root / "runtime.err.log"),
    }


def _collector_payload(
    data_root: Path = DATA_ROOT,
    state_root: Path = STATE_ROOT,
    log_root: Path = LOG_ROOT,
) -> dict[str, object]:
    return {
        "Label": COLLECTOR_LABEL,
        "ProgramArguments": [
            str(PYTHON), str(COLLECTOR),
            "--source", str(data_root / "conscious-lm" / "memory.db"),
            "--output", str(state_root / "status.json"),
            "--verdict", str(state_root / "verdict.json"),
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "StartInterval": 900,
        "ProcessType": "Background",
        "StandardOutPath": str(log_root / "collector.out.log"),
        "StandardErrorPath": str(log_root / "collector.err.log"),
    }


def _write_plist(label: str, payload: dict[str, object]) -> None:
    path = _plist(label)
    temporary = path.with_suffix(".plist.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)


def _restore_plist(label: str, previous: bytes | None) -> None:
    path = _plist(label)
    if previous is None:
        subprocess.run(
            ["launchctl", "bootout", DOMAIN, str(path)], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        path.unlink(missing_ok=True)
        return
    temporary = path.with_suffix(".plist.rollback")
    temporary.write_bytes(previous)
    os.replace(temporary, path)
    _bootstrap(label)


def _loaded(label: str) -> bool:
    return subprocess.run(
        ["launchctl", "print", f"{DOMAIN}/{label}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _healthy(timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def _bootstrap(label: str) -> None:
    path = _plist(label)
    subprocess.run(
        ["launchctl", "bootout", DOMAIN, str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["launchctl", "bootstrap", DOMAIN, str(path)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"{DOMAIN}/{label}"], check=True)


def install(dialogue_backend: str = "claude", data_root: Path = DATA_ROOT) -> int:
    data_root = data_root.expanduser().resolve()
    state_root = data_root.parent
    log_root = state_root / "logs"
    _check_files(dialogue_backend)
    for directory in (data_root, state_root, log_root, PLIST_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    labels = (RUNTIME_LABEL, COLLECTOR_LABEL)
    previous = {
        label: _plist(label).read_bytes() if _plist(label).is_file() else None
        for label in labels
    }
    try:
        _write_plist(
            RUNTIME_LABEL,
            _runtime_payload(dialogue_backend, data_root, log_root),
        )
        _write_plist(
            COLLECTOR_LABEL,
            _collector_payload(data_root, state_root, log_root),
        )
        _bootstrap(RUNTIME_LABEL)
        if not _healthy():
            raise RuntimeError(f"runtime failed health check; inspect {log_root}")
        source = data_root / "conscious-lm" / "memory.db"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not source.is_file():
            time.sleep(0.25)
        if not source.is_file():
            raise RuntimeError("runtime did not initialize the isolated memory database")
        _bootstrap(COLLECTOR_LABEL)
        deadline = time.monotonic() + 10
        status_path = state_root / "status.json"
        while time.monotonic() < deadline and not status_path.is_file():
            time.sleep(0.25)
        if not status_path.is_file():
            raise RuntimeError(f"collector did not produce status; inspect {log_root}")
    except Exception as exc:
        for label in labels:
            _restore_plist(label, previous[label])
        raise SystemExit(f"deployment rolled back: {exc}") from exc
    return status(data_root)


def status(data_root: Path = DATA_ROOT) -> int:
    state_root = data_root.expanduser().resolve().parent
    runtime_loaded = _loaded(RUNTIME_LABEL)
    collector_loaded = _loaded(COLLECTOR_LABEL)
    healthy = _healthy(timeout=1.0)
    status_path = state_root / "status.json"
    counts = {}
    verdict = None
    if status_path.is_file():
        counts = json.loads(status_path.read_text()).get("audit", {}).get("counts", {})
    verdict_path = state_root / "verdict.json"
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text()).get("verdict")
    print(json.dumps({
        "runtime_loaded": runtime_loaded,
        "collector_loaded": collector_loaded,
        "healthy": healthy,
        "counts": counts,
        "verdict": verdict,
        "raw_text_in_status": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if (
        runtime_loaded and collector_loaded and healthy
        and status_path.is_file() and verdict_path.is_file()
    ) else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("install", "status"))
    parser.add_argument("--dialogue-backend", choices=("claude", "native"), default="claude")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)
    return (
        install(args.dialogue_backend, args.data_root)
        if args.command == "install" else status(args.data_root)
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
