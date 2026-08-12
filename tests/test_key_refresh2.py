from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from key_refresh2 import _runtime_spec
from measurement.key_refresh2_gate import _classify, adjudicate
from measurement.key_refresh2_registry import (
    KEY_REFRESH2_SPEC, canonical_spec, spec_sha256,
)


def test_registry_matches_preregistration():
    assert KEY_REFRESH2_SPEC["preregistration_commit"] == "4b8e8c45f"
    assert KEY_REFRESH2_SPEC["query_context_sense_steps"] == 8
    assert KEY_REFRESH2_SPEC["key_sense_steps"] == 3
    assert KEY_REFRESH2_SPEC["query_key_sense_steps"] == 4
    assert KEY_REFRESH2_SPEC["runtime_conditions"] == {
        "baseline_3": 3, "integrated_4": 4,
        "disabled_3": 3, "recovered_4": 4,
    }
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_runtime_applies_only_registered_query_key_default():
    baseline = _runtime_spec(3)
    integrated = _runtime_spec(KEY_REFRESH2_SPEC["query_key_sense_steps"])
    assert baseline["settled_context_steps"] == integrated["settled_context_steps"] == 6
    assert baseline["query_context_sense_steps"] == integrated["query_context_sense_steps"] == 8
    assert baseline["key_sense_steps"] == integrated["key_sense_steps"] == 3
    assert baseline["query_key_sense_steps"] == 3
    assert integrated["query_key_sense_steps"] == 4
    try:
        _runtime_spec(6)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered query-key step count must fail")


def test_closed_verdict_order():
    assert _classify(True, True, True, True, True)[0] == "KR2I_FULL_PATH_RECOVERED"
    assert _classify(True, True, False, False, True)[0] == "KR2I_PARTIAL_PATH_RECOVERED"
    assert _classify(False, True, True, True, True)[0] == "KR2I_FULL_BEHAVIOR_REGRESSION"
    assert _classify(True, True, True, True, False)[0] == "KR2I_NOT_CAUSAL"


def test_committed_result_replays_and_fails_closed():
    results_path = Path("measurement/key_refresh2_results.json")
    verdict_path = Path("measurement/key_refresh2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["conditions"]["integrated_4"]["state_audit"][
        "key_step_calls"
    ] += 1
    assert adjudicate(changed)["verdict"] == "KR2I_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["conditions"]["recovered_4"]["record_digests"][
        "full_cue"
    ] = "0" * 64
    assert adjudicate(changed)["verdict"] == "KR2I_INVALID"
