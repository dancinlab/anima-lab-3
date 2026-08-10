from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from measurement.projector_gate import _classify, adjudicate
from measurement.projector_registry import (
    PROJECTOR_SPEC,
    canonical_spec,
    projector_name,
    spec_sha256,
)


def test_projector_registry_matches_preregistration():
    assert PROJECTOR_SPEC["preregistration_commit"] == "e0aa03beb"
    assert len(PROJECTOR_SPEC["training_combinations"]) == 4
    assert len(PROJECTOR_SPEC["evaluation_combinations"]) == 4
    assert PROJECTOR_SPEC["event_count"] == 4
    assert PROJECTOR_SPEC["settling_updates"] == 8
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_projector_grid_verdicts():
    low, high = PROJECTOR_SPEC["factor_seeds"]
    names = {
        (c, t): projector_name({"calibration_seed": c, "training_seed": t})
        for c in (low, high) for t in (low, high)
    }
    def grid(a, b, c, d):
        return {names[(low, low)]: a, names[(low, high)]: b,
                names[(high, low)]: c, names[(high, high)]: d}
    assert _classify(grid(False, False, True, True), low, high)[0] == "PD1_CALIBRATION_STREAM_CAUSAL"
    assert _classify(grid(False, True, False, True), low, high)[0] == "PD2_TRAINING_RANDOMNESS_CAUSAL"
    assert _classify(grid(False, True, True, True), low, high)[0] == "PD3_EITHER_FACTOR_SUFFICIENT"
    assert _classify(grid(False, False, False, True), low, high)[0] == "PD4_BOTH_FACTORS_REQUIRED"
    assert _classify(grid(True, False, True, False), low, high)[0] == "PD5_FACTOR_INTERACTION_OR_MIXED"


def test_committed_projector_result_replays_and_fails_closed():
    results_path = Path("measurement/projector_results.json")
    verdict_path = Path("measurement/projector_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["projectors"][0]["training_seed"] = 99
    assert adjudicate(tampered)["verdict"] == "PD0_INVALID"
