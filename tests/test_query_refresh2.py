from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from measurement.query_refresh2_gate import _classify, adjudicate
from measurement.query_refresh2_registry import (
    QUERY_REFRESH2_SPEC, canonical_spec, spec_sha256,
)
from query_refresh2 import _runtime_spec


def test_registry_matches_preregistration():
    assert QUERY_REFRESH2_SPEC["preregistration_commit"] == "cde0a15a4"
    assert QUERY_REFRESH2_SPEC["runtime_conditions"] == {
        "baseline_6": 6, "refreshed_8": 8,
        "disabled_6": 6, "recovered_8": 8,
    }
    assert QUERY_REFRESH2_SPEC["missing_fraction"] == 0.25
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_runtime_spec_changes_only_registered_query_steps():
    baseline = _runtime_spec(6)
    refreshed = _runtime_spec(8)
    assert baseline["settled_context_steps"] == refreshed["settled_context_steps"] == 6
    assert baseline["key_sense_steps"] == refreshed["key_sense_steps"] == 3
    assert baseline["query_context_sense_steps"] == 6
    assert refreshed["query_context_sense_steps"] == 8
    try:
        _runtime_spec(12)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered query step count must fail")


def test_closed_verdict_order():
    assert _classify(True, True, True, True, True, True)[0] == "QRI1_FULL_PATH_RECOVERED"
    assert _classify(True, True, False, False, True, True)[0] == "QRI2_CONTEXT_PATH_RECOVERED"
    assert _classify(True, False, False, False, True, False)[0] == "QRI3_BEHAVIOR_IMPROVED_NOT_RECOVERED"
    assert _classify(True, False, False, False, False, False)[0] == "QRI4_REFRESH_NOT_CAUSAL"
    assert _classify(False, True, True, True, True, True)[0] == "QRI5_FULL_BEHAVIOR_REGRESSION"


def test_committed_result_replays_and_fails_closed():
    results_path = Path("measurement/query_refresh2_results.json")
    verdict_path = Path("measurement/query_refresh2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["conditions"]["refreshed_8"]["state_audit"][
        "context_step_calls"
    ] += 1
    assert adjudicate(changed)["verdict"] == "QRI0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["conditions"]["recovered_8"]["record_digests"][
        "full_cue"
    ] = "0" * 64
    assert adjudicate(changed)["verdict"] == "QRI0_INVALID"
