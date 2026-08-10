from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from measurement.conjunction2_gate import adjudicate
from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256


def test_conjunction2_registry_matches_preregistration():
    assert CONJUNCTION2_SPEC["preregistration_commit"] == "22891223d"
    assert CONJUNCTION2_SPEC["events_per_episode"] == 16
    assert CONJUNCTION2_SPEC["value_address_dim"] == 32
    assert CONJUNCTION2_SPEC["value_transform_calls_per_episode"] == 16
    assert len(CONJUNCTION2_SPEC["evaluation_combinations"]) == 4
    assert len(spec_sha256()) == 64


def test_committed_conjunction2_result_replays_and_rejects_call_changes():
    results_path = Path("measurement/conjunction2_results.json")
    verdict_path = Path("measurement/conjunction2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["evaluations"][0]["path_audit"]["integrated_stable_conjunction_normal"]["value_calls"] += 1
    assert adjudicate(changed)["verdict"] == "CJ2_0_INVALID"
