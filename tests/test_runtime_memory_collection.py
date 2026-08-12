import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from gate_runtime3 import _manifest_sha256, run, scan_collection_source
from measurement.runtime_memory_collection_gate import adjudicate
from measurement.runtime_memory_collection_registry import (
    RUNTIME_MEMORY_COLLECTION_SPEC,
    spec_sha256,
)


def _database(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, role TEXT NOT NULL, "
            "text TEXT NOT NULL, timestamp TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO memories(id, role, text, timestamp) VALUES (?, ?, ?, ?)",
            rows,
        )


def _spec(path: Path, baseline_rows: list[tuple]) -> dict:
    spec = copy.deepcopy(RUNTIME_MEMORY_COLLECTION_SPEC)
    spec["source_database"] = str(path)
    spec["preregistration_commit"] = "0" * 40
    spec["baseline"] = {
        "total_rows": len(baseline_rows),
        "eligible_user_turns": sum(row[1] == "user" for row in baseline_rows),
        "source_manifest_sha256": _manifest_sha256(baseline_rows),
    }
    spec["thresholds"] = {
        "minimum_user_turns": 3,
        "minimum_unique_user_turns": 3,
        "minimum_active_days": 2,
        "minimum_sessions": 2,
        "maximum_raw_text_leaks": 0,
    }
    return spec


def test_collection_scan_accepts_append_only_growth_without_raw_text(tmp_path):
    path = tmp_path / "memory.db"
    baseline = [
        (1, "user", "기준 대화", "2026-08-01T00:00:00+00:00"),
        (2, "assistant", "기준 응답", "2026-08-01T00:00:01+00:00"),
    ]
    _database(path, baseline)
    spec = _spec(path, baseline)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "INSERT INTO memories(id, role, text, timestamp) VALUES (?, ?, ?, ?)",
            [
                (3, "user", "새 대화 하나", "2026-08-02T00:00:00+00:00"),
                (4, "user", "새 대화 둘", "2026-08-02T00:31:00+00:00"),
            ],
        )
    audit = scan_collection_source(path, spec)
    assert audit["source_append_only"] is True
    assert audit["new_user_turns"] == 2
    assert audit["collection_status"] == "ready_for_review"
    encoded = json.dumps(audit, ensure_ascii=False)
    assert "기준 대화" not in encoded
    assert "새 대화" not in encoded


def test_collection_scan_detects_baseline_mutation(tmp_path):
    path = tmp_path / "memory.db"
    baseline = [(1, "user", "원본", "2026-08-01T00:00:00+00:00")]
    _database(path, baseline)
    spec = _spec(path, baseline)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE memories SET text = '변조' WHERE id = 1")
    assert scan_collection_source(path, spec)["source_append_only"] is False


def test_collection_gate_closes_on_mutation_and_opens_on_threshold(tmp_path):
    path = tmp_path / "memory.db"
    rows = [
        (1, "user", "하나", "2026-08-01T00:00:00+00:00"),
        (2, "user", "둘", "2026-08-02T00:00:00+00:00"),
        (3, "user", "셋", "2026-08-02T00:31:00+00:00"),
    ]
    _database(path, rows)
    spec = _spec(path, [])
    payload = run(path, spec)
    assert adjudicate(payload, spec)["verdict"] == "GR32_READY_FOR_REVIEW"
    wrong_source = copy.deepcopy(payload)
    wrong_source["source"] = str(tmp_path / "other.db")
    assert adjudicate(wrong_source, spec)["verdict"] == "GR30_INVALID"
    payload["audit"]["source_append_only"] = False
    assert adjudicate(payload, spec)["verdict"] == "GR30_INVALID"


def test_collection_registration_is_preregistered_after_pin():
    spec = RUNTIME_MEMORY_COLLECTION_SPEC
    assert spec["preregistration_commit"] == "55fc9dfd50155435c060ed542820e0a6a386092d"
    assert spec["audit"]["raw_text_allowed"] is False
    assert spec["audit"]["source_append_only"] is True
    assert spec["runtime"]["data_root"].startswith(".local/")
    assert len(spec_sha256()) == 64


def test_collection_rejects_empty_dialogue(tmp_path):
    path = tmp_path / "memory.db"
    _database(path, [(1, "user", "", "2026-08-01T00:00:00+00:00")])
    with pytest.raises(ValueError, match="empty dialogue"):
        scan_collection_source(path, _spec(path, []))
