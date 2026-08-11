from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from cue_align import _fit_deterministic, _transform, _wrong_targets
from measurement.cue_align_gate import adjudicate
from measurement.cue_align_registry import (
    CUE_ALIGN_SPEC, calibration_pairs, canonical_spec, spec_sha256,
)


def test_cue_align_registry_matches_preregistration():
    assert CUE_ALIGN_SPEC["preregistration_commit"] == "407384ca3"
    assert CUE_ALIGN_SPEC["global_alignment_uses_labels"] is False
    assert CUE_ALIGN_SPEC["category_oracle_uses_true_label"] is True
    assert CUE_ALIGN_SPEC["fit_method"] == "canonical_affine_ridge"
    assert CUE_ALIGN_SPEC["thresholds"]["minimum_damaged_gain"] == 0.02
    assert calibration_pairs() == 1024
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_affine_fit_is_deterministic_and_recovers_known_map():
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(64, 4, generator=generator)
    weight = torch.tensor([[1.0, 0.2, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0],
                           [0.0, 0.0, 1.1, 0.1], [0.1, 0.0, 0.0, 1.0]])
    targets = inputs @ weight.T + 0.25
    spec = {**CUE_ALIGN_SPEC, "state_dim": 4}
    state, audit = _fit_deterministic(inputs, targets, spec)
    assert audit["deterministic"] is True
    assert torch.mean((_transform(state, inputs) - targets) ** 2) < 1e-8


def test_wrong_pair_targets_never_keep_the_same_label():
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    targets = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    wrong = _wrong_targets(targets, labels, 4)
    for index, label in enumerate(labels.tolist()):
        source_index = torch.where(torch.all(targets == wrong[index], dim=1))[0].item()
        assert labels[source_index].item() == (label + 1) % 4


def test_committed_cue_align_result_replays_and_fails_closed():
    results_path = Path("measurement/cue_align_results.json")
    verdict_path = Path("measurement/cue_align_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["label_use_audit"]["global_affine"] = True
    assert adjudicate(changed)["verdict"] == "CA0_INVALID"
    changed = deepcopy(payload)
    changed["evaluations"][0]["source_reference_audit"]["query_full"] = False
    assert adjudicate(changed)["verdict"] == "CA0_INVALID"

