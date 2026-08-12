#!/usr/bin/env python3
"""Install or inspect the canonical local GATE-RUNTIME-3 services on macOS."""
from __future__ import annotations

import json
import os
import plistlib
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


def _check_files() -> None:
    missing = [str(path) for path in (PYTHON, ENTRYPOINT, COLLECTOR) if not path.is_file()]
    if missing:
        raise SystemExit("missing runtime file(s): " + ", ".join(missing))
    subprocess.run(
        [str(PYTHON), "-c", "import faiss, torch, transformers, websockets"],
        cwd=ROOT,
        check=True,
    )


def _runtime_payload() -> dict[str, object]:
    return {
        "Label": RUNTIME_LABEL,
        "ProgramArguments": [
            str(PYTHON), "-u", str(ENTRYPOINT), "--web", "--port", str(PORT),
            "--data-root", str(DATA_ROOT), "--max-cells", "64", "--no-camera",
            "--no-vision", "--no-telepathy", "--no-cloud", "--no-conscious-lm",
            "--memory-gate-shadow",
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_ROOT / "runtime.out.log"),
        "StandardErrorPath": str(LOG_ROOT / "runtime.err.log"),
    }


def _collector_payload() -> dict[str, object]:
    return {
        "Label": COLLECTOR_LABEL,
        "ProgramArguments": [
            str(PYTHON), str(COLLECTOR),
            "--source", str(DATA_ROOT / "conscious-lm" / "memory.db"),
            "--output", str(STATE_ROOT / "status.json"),
            "--verdict", str(STATE_ROOT / "verdict.json"),
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "StartInterval": 900,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_ROOT / "collector.out.log"),
        "StandardErrorPath": str(LOG_ROOT / "collector.err.log"),
    }


def _write_plist(label: str, payload: dict[str, object]) -> None:
    path = _plist(label)
    temporary = path.with_suffix(".plist.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)


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


def install() -> int:
    _check_files()
    for directory in (DATA_ROOT, STATE_ROOT, LOG_ROOT, PLIST_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    _write_plist(RUNTIME_LABEL, _runtime_payload())
    _write_plist(COLLECTOR_LABEL, _collector_payload())
    _bootstrap(RUNTIME_LABEL)
    if not _healthy():
        raise SystemExit(f"runtime failed health check; inspect {LOG_ROOT}")
    source = DATA_ROOT / "conscious-lm" / "memory.db"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not source.is_file():
        time.sleep(0.25)
    if not source.is_file():
        raise SystemExit("runtime did not initialize the isolated memory database")
    _bootstrap(COLLECTOR_LABEL)
    deadline = time.monotonic() + 10
    status_path = STATE_ROOT / "status.json"
    while time.monotonic() < deadline and not status_path.is_file():
        time.sleep(0.25)
    if not status_path.is_file():
        raise SystemExit(f"collector did not produce status; inspect {LOG_ROOT}")
    return status()


def status() -> int:
    runtime_loaded = _loaded(RUNTIME_LABEL)
    collector_loaded = _loaded(COLLECTOR_LABEL)
    healthy = _healthy(timeout=1.0)
    status_path = STATE_ROOT / "status.json"
    counts = {}
    verdict = None
    if status_path.is_file():
        counts = json.loads(status_path.read_text()).get("audit", {}).get("counts", {})
    verdict_path = STATE_ROOT / "verdict.json"
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
    if len(argv) != 1 or argv[0] not in {"install", "status"}:
        print("usage: deploy_gate_runtime3.py install|status", file=sys.stderr)
        return 2
    return install() if argv[0] == "install" else status()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
