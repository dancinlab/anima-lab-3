from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from completion import _partial_state
from measurement.completion_registry import cue_mask_indices as completion1_mask
from measurement.completion2_gate import _classify, adjudicate
from measurement.completion2_registry import (
    COMPLETION2_SPEC, canonical_spec, cue_mask_indices, mask_plan_audit, spec_sha256,
)


def test_registry_matches_preregistration():
    assert COMPLETION2_SPEC["preregistration_commit"] == "fb6dcdfe0"
    assert COMPLETION2_SPEC["query_context_sense_steps"] == 8
    assert COMPLETION2_SPEC["query_key_sense_steps"] == 4
    assert COMPLETION2_SPEC["missing_fractions"] == [0.25, 0.50, 0.75, 1.0]
    assert len(COMPLETION2_SPEC["conditions"]) == 13
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_registered_masks_preserve_quarter_source_and_exact_counts():
    for component in COMPLETION2_SPEC["mask_components"]:
        assert cue_mask_indices(0, component, 0.25) == completion1_mask(
            0, component, 0.25
        )
        for fraction, count in ((0.25, 24), (0.50, 48), (0.75, 72), (1.0, 96)):
            assert len(cue_mask_indices(17, component, fraction)) == count
    audit = mask_plan_audit()
    assert audit["context:1.00"]["unique_masks"] == 1
    assert audit["key:1.00"]["removed_per_episode"] == 96


def test_full_category_removal_zeroes_without_mutating_input():
    state = torch.arange(192, dtype=torch.float32).reshape(2, 96)
    original = state.clone()
    removed = _partial_state(state, 0, "context", 1.0, COMPLETION2_SPEC)
    assert torch.equal(state, original)
    assert torch.count_nonzero(removed) == 0


def test_closed_verdict_order():
    assert _classify(75)[0] == "C2_BOUNDARY_75"
    assert _classify(50)[0] == "C2_BOUNDARY_50"
    assert _classify(25)[0] == "C2_BOUNDARY_25"


def test_committed_result_replays_and_fails_closed():
    results_path = Path("measurement/completion2_results.json")
    verdict_path = Path("measurement/completion2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["state_audit"]["key_step_calls"] += 1
    assert adjudicate(changed)["verdict"] == "C20_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["reference_audit"]["both_quarter_missing"][
        "record_match"
    ] = False
    assert adjudicate(changed)["verdict"] == "C20_INVALID"
