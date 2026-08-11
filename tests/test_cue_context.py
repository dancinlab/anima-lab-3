from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from cue_context import _mask_rows
from measurement.cue_context_gate import adjudicate
from measurement.cue_context_registry import (
    CUE_CONTEXT_SPEC,
    calibration_pairs,
    canonical_spec,
    spec_sha256,
    training_mask_indices,
)


def test_cue_context_registry_matches_preregistration():
    assert CUE_CONTEXT_SPEC["preregistration_commit"] == "90808831e"
    assert CUE_CONTEXT_SPEC["combined_schedule"] == "even_storage_odd_query"
    assert CUE_CONTEXT_SPEC["missing_fraction"] == 0.25
    assert CUE_CONTEXT_SPEC["fit_method"] == "canonical_ridge"
    assert CUE_CONTEXT_SPEC["thresholds"]["category_accuracy"] == 0.90
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_training_masks_are_deterministic_and_have_registered_width():
    first = training_mask_indices(0)
    assert len(first) == 24
    assert first == training_mask_indices(0)
    assert first != training_mask_indices(1)
    assert calibration_pairs() == 1024


def test_mask_rows_is_non_mutating_and_fail_closed():
    states = torch.ones(2, CUE_CONTEXT_SPEC["state_dim"])
    before = states.clone()
    masks = [training_mask_indices(0), training_mask_indices(1)]
    masked = _mask_rows(states, masks)
    assert torch.equal(states, before)
    assert torch.all((masked == 0).sum(1) == 24)
    try:
        _mask_rows(states[:1], masks)
    except ValueError:
        pass
    else:
        raise AssertionError("state and mask roster mismatch must fail")


def test_committed_cue_context_result_replays_and_fails_closed():
    results_path = Path("measurement/cue_context_results.json")
    verdict_path = Path("measurement/cue_context_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["calibration_evaluation_overlap"] = 1
    assert adjudicate(changed)["verdict"] == "CC0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["source_reference_audit"]["query_full_match"] = False
    assert adjudicate(changed)["verdict"] == "CC0_INVALID"
