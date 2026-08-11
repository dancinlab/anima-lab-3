from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from cue_robust import _masked_rows
from measurement.cue_robust_gate import adjudicate
from measurement.cue_robust_registry import (
    CUE_ROBUST_SPEC,
    canonical_spec,
    spec_sha256,
    training_examples_per_component,
    training_mask_indices,
)


def test_cue_robust_registry_matches_preregistration():
    assert CUE_ROBUST_SPEC["preregistration_commit"] == "eee24d2f2"
    assert CUE_ROBUST_SPEC["fit_method"] == "canonical_ridge"
    assert CUE_ROBUST_SPEC["training_missing_fraction"] == 0.25
    assert CUE_ROBUST_SPEC["component_address_dim"] == 32
    assert CUE_ROBUST_SPEC["composite_address_dim"] == 64
    assert CUE_ROBUST_SPEC["thresholds"]["partial_category_accuracy"] == 0.90
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_training_masks_are_deterministic_and_component_specific():
    first = training_mask_indices(0, "context")
    assert len(first) == 24
    assert first == training_mask_indices(0, "context")
    assert first != training_mask_indices(1, "context")
    assert first != training_mask_indices(0, "key")


def test_masked_rows_validate_shape_and_do_not_mutate_input():
    total = training_examples_per_component()
    states = torch.ones(total, CUE_ROBUST_SPEC["state_dim"])
    before = states.clone()
    masked = _masked_rows(states, "context")
    assert torch.equal(states, before)
    assert masked.shape == states.shape
    assert torch.all((masked == 0).sum(1) == 24)

    try:
        _masked_rows(states[:1], "context")
    except ValueError:
        pass
    else:
        raise AssertionError("wrong training state shape must fail")


def test_committed_cue_robust_result_replays_and_fails_closed():
    results_path = Path("measurement/cue_robust_results.json")
    verdict_path = Path("measurement/cue_robust_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["context_fit"]["full_refit_matches_source"] = False
    assert adjudicate(changed)["verdict"] == "CR0_INVALID"

    changed = deepcopy(payload)
    changed["mask_overlap_audit"]["context"]["exact_overlap"] = 1
    assert adjudicate(changed)["verdict"] == "CR0_INVALID"
