import copy
import json
from pathlib import Path

import pytest

from memory_shadow import (
    AUDIT_FORMAT,
    RuntimeMemoryShadow,
    create_memory_shadow,
    observe_shadow_search,
    observe_shadow_writes,
)
from measurement.runtime_memory_shadow_gate import adjudicate
from measurement.runtime_memory_shadow_registry import (
    RUNTIME_MEMORY_SHADOW_SPEC,
    spec_sha256,
)


class FakeScorer:
    threshold = 0.5

    def score_rows(self, rows):
        return [1.0 if "기억" in row["text"] else 0.0 for row in rows]


def test_shadow_audit_is_append_only_and_raw_text_free(tmp_path):
    path = tmp_path / "memory_shadow.jsonl"
    shadow = RuntimeMemoryShadow(FakeScorer(), path, clock=lambda: "fixed")
    rows = [
        {"role": "user", "text": "이 약속을 기억해 줘", "memory_id": 1},
        {"role": "assistant", "text": "오늘은 맑아요", "memory_id": 2},
    ]
    events = shadow.record_writes(rows)
    shadow.record_search("약속이 뭐였지?", [
        {"id": 1, "role": "user", "text": rows[0]["text"], "similarity": 0.9},
        {"id": 2, "role": "assistant", "text": rows[1]["text"], "similarity": 0.8},
    ])
    raw = path.read_text()
    assert rows[0]["text"] not in raw
    assert rows[1]["text"] not in raw
    assert events[0]["selected"] is True
    parsed = [json.loads(line) for line in raw.splitlines()]
    assert [row["event"] for row in parsed] == ["write", "write", "search"]
    assert all(row["format"] == AUDIT_FORMAT for row in parsed)
    assert parsed[-1]["candidates"][0]["selected"] is True


def test_shadow_failures_do_not_escape():
    errors = []

    class Broken:
        def record_writes(self, rows):
            raise OSError("write")

        def record_search(self, query, candidates):
            raise OSError("search")

    assert create_memory_shadow(
        lambda: (_ for _ in ()).throw(RuntimeError("init")),
        on_error=lambda error: errors.append(str(error)),
    ) is None
    observe_shadow_writes(Broken(), [{"role": "user", "text": "x"}],
                          on_error=lambda error: errors.append(str(error)))
    observe_shadow_search(Broken(), "q", [],
                          on_error=lambda error: errors.append(str(error)))
    assert errors == ["init", "write", "search"]


def test_shadow_rejects_raw_text_at_any_nesting_level():
    with pytest.raises(ValueError, match="raw dialogue text"):
        RuntimeMemoryShadow._validate_event({
            "event": "search",
            "candidates": [{"text": "must not be copied"}],
        })


def test_runtime_shadow_registration_is_pinned():
    spec = RUNTIME_MEMORY_SHADOW_SPEC
    assert spec["preregistration_commit"] == "94dba1566"
    assert spec["audit"]["default_enabled"] is False
    assert spec["audit"]["filters_primary_memory"] is False
    assert len(spec_sha256()) == 64


def test_runtime_shadow_gate_fails_closed_on_registration_change():
    changed = copy.deepcopy(RUNTIME_MEMORY_SHADOW_SPEC)
    changed["thresholds"]["minimum_important_selection_rate"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(changed),
    }
    assert adjudicate(payload)["verdict"] == "GR0_INVALID"


def test_recorded_runtime_shadow_result_replays():
    path = Path("measurement/runtime_memory_shadow_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    assert adjudicate(json.loads(path.read_text()))["verdict"] in {
        "GR1_SHADOW_RUNTIME_SAFE",
        "GR2_SELECTION_NOT_GENERAL",
        "GR3_RUNTIME_INTERFERENCE",
    }
