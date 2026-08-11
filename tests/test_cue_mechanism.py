from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from cue_mechanism import _component_diagnostic
from measurement.cue_mechanism_gate import adjudicate
from measurement.cue_mechanism_registry import (
    CUE_MECHANISM_SPEC, canonical_spec, spec_sha256,
)
from key_stability import StableKeyProjector


def test_cue_mechanism_registry_matches_preregistration():
    assert CUE_MECHANISM_SPEC["preregistration_commit"] == "29eea3a6a"
    assert CUE_MECHANISM_SPEC["missing_fraction"] == 0.25
    assert CUE_MECHANISM_SPEC["conditions"]["context_quarter_missing"] == [0.25, 0.0]
    assert CUE_MECHANISM_SPEC["conditions"]["key_quarter_missing"] == [0.0, 0.25]
    assert CUE_MECHANISM_SPEC["thresholds"]["component_category_accuracy"] == 0.90
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_component_diagnostic_is_observational_and_finite():
    projector = StableKeyProjector(4, 4, 4, 0.1, True).eval()
    state = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    before = state.clone()
    reference = projector.address(state.mean(0).unsqueeze(0))[0].detach()
    row = _component_diagnostic(projector, state, 1, reference)
    assert torch.equal(state, before)
    assert set(row) == {
        "prediction", "correct_similarity", "closest_wrong_similarity",
        "center_margin", "full_address_similarity",
    }
    assert -1.000001 <= row["full_address_similarity"] <= 1.000001


def test_committed_cue_mechanism_result_replays_and_fails_closed():
    results_path = Path("measurement/cue_mechanism_results.json")
    verdict_path = Path("measurement/cue_mechanism_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["restoration_audit"]["context_restored_to_full"] = False
    assert adjudicate(changed)["verdict"] == "CM0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["memory_path_audit"]["full_cue"]["stores"] -= 1
    assert adjudicate(changed)["verdict"] == "CM0_INVALID"
