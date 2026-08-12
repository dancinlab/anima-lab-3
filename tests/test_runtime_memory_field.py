import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from gate_runtime2 import scan_source
from measurement.runtime_memory_field_gate import adjudicate
from measurement.runtime_memory_field_registry import (
    RUNTIME_MEMORY_FIELD_SPEC,
    spec_sha256,
)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, role TEXT NOT NULL, "
            "text TEXT NOT NULL, timestamp TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO memories(id, role, text, timestamp) VALUES (?, ?, ?, ?)",
            [
                (1, "user", "첫 기록", "2026-08-01T00:00:00+00:00"),
                (2, "assistant", "응답", "2026-08-01T00:00:01+00:00"),
                (3, "user", "둘째 기록", "2026-08-01T00:31:00+00:00"),
            ],
        )


def test_source_scan_is_read_only_and_raw_text_free(tmp_path):
    path = tmp_path / "memory.db"
    _database(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    audit, rows = scan_source(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert audit["eligible_user_turns"] == 2
    assert audit["unique_user_turns"] == 2
    assert audit["sessions"] == 2
    assert audit["raw_text_leaks"] == 0
    encoded = json.dumps(audit, ensure_ascii=False)
    assert "첫 기록" not in encoded
    assert "둘째 기록" not in encoded
    assert [row["role"] for row in rows] == ["user", "user"]


def test_field_registration_is_pinned():
    spec = RUNTIME_MEMORY_FIELD_SPEC
    assert spec["preregistration_commit"] == "edb9f36e2"
    assert spec["audit"]["default_enabled"] is False
    assert spec["audit"]["raw_text_allowed"] is False
    assert spec["audit"]["source_read_only"] is True
    assert len(spec_sha256()) == 64


def test_field_gate_fails_closed_on_registration_change():
    changed = copy.deepcopy(RUNTIME_MEMORY_FIELD_SPEC)
    changed["thresholds"]["minimum_user_turns"] = 1
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(changed),
    }
    assert adjudicate(payload)["verdict"] == "GR20_INVALID"


def test_recorded_field_result_replays():
    path = Path("measurement/runtime_memory_field_results.json")
    if not path.exists():
        return
    assert adjudicate(json.loads(path.read_text()))["verdict"] in {
        "GR21_INSUFFICIENT_FIELD_DATA",
        "GR22_REVIEW_INCOMPLETE",
        "GR23_FIELD_SELECTION_UNSAFE",
        "GR24_FIELD_SHADOW_SAFE",
    }
