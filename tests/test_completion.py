from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from completion import _partial_state
from measurement.completion_gate import adjudicate
from measurement.completion_registry import (
    COMPLETION_SPEC, canonical_spec, cue_mask_indices, mask_plan_audit, spec_sha256,
)


def test_completion_registry_matches_preregistration():
    assert COMPLETION_SPEC["preregistration_commit"] == "eafd591b4"
    assert COMPLETION_SPEC["settled_context_steps"] == 6
    assert COMPLETION_SPEC["missing_fractions"] == [0.25, 0.50, 0.75]
    assert COMPLETION_SPEC["thresholds"]["both_half_selection_accuracy"] == 0.75
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_completion_masks_are_deterministic_balanced_and_label_free():
    first = cue_mask_indices(0, "context", 0.50)
    assert first == cue_mask_indices(0, "context", 0.50)
    assert len(first) == 48
    assert first != cue_mask_indices(1, "context", 0.50)
    assert first != cue_mask_indices(0, "key", 0.50)
    audit = mask_plan_audit()
    assert audit["context:0.25"]["removed_per_episode"] == 24
    assert audit["key:0.75"]["removed_per_episode"] == 72
    assert all(row["episodes"] == COMPLETION_SPEC["eval_episodes"] for row in audit.values())
    assert all(len(row["sha256"]) == 64 for row in audit.values())


def test_partial_state_preserves_input_and_masks_registered_coordinates():
    state = torch.ones(3, COMPLETION_SPEC["state_dim"])
    partial = _partial_state(state, 7, "key", 0.25)
    assert torch.equal(state, torch.ones_like(state))
    assert partial.shape == state.shape
    assert int((partial[0] == 0).sum()) == 24
    assert torch.equal(partial[0], partial[1])


def test_committed_completion_result_replays_and_fails_closed():
    results_path = Path("measurement/completion_results.json")
    verdict_path = Path("measurement/completion_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["mask_plan_audit"]["context:0.50"]["removed_per_episode"] -= 1
    assert adjudicate(changed)["verdict"] == "CP0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["memory_path_audit"]["both_half_missing"]["stores"] -= 1
    assert adjudicate(changed)["verdict"] == "CP0_INVALID"
