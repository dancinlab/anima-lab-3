from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from address_margin import _center, _classification, _join, _margin
from conjunction import build_episodes
from key_stability import StableKeyProjector
from measurement.address_margin_gate import adjudicate
from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC, canonical_spec, spec_sha256
from measurement.conjunction2_registry import CONJUNCTION2_SPEC


def test_address_margin_registry_matches_preregistration():
    assert ADDRESS_MARGIN_SPEC["preregistration_commit"] == "799e48bad"
    assert ADDRESS_MARGIN_SPEC["settled_context_steps"] == 6
    assert ADDRESS_MARGIN_SPEC["thresholds"]["minimum_selection_gain"] == 0.08
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_registered_source_dataset_can_be_built_without_runtime_only_fields():
    episodes = build_episodes({**CONJUNCTION2_SPEC, "eval_episodes": 8})
    assert len(episodes) == 8


def test_center_uses_frozen_registered_prototype():
    model = StableKeyProjector(4, 4, 2, 0.1)
    with torch.no_grad():
        model.prototypes.copy_(torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0]]))
    assert torch.equal(_center(model, 0), torch.tensor([1.0, 0, 0, 0]))
    assert torch.equal(_center(model, 1), torch.tensor([0.0, 1.0, 0, 0]))


def test_join_and_margin_preserve_component_width_and_correct_gap():
    spec = {"component_weight": 1.0, "composite_address_dim": 4}
    addresses = [
        _join(torch.tensor([1.0, 0]), torch.tensor([1.0, 0]), spec),
        _join(torch.tensor([0.0, 1]), torch.tensor([0.0, 1]), spec),
    ]
    correct, wrong, margin = _margin(addresses, addresses[1], 1)
    assert correct == pytest.approx(1.0)
    assert wrong == 0.0
    assert margin == pytest.approx(1.0)


def test_classification_reports_balanced_confusion():
    metric = _classification([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metric["accuracy"] == 0.75
    assert metric["per_class_recall"] == [0.5, 1.0]
    assert metric["confusion_matrix"] == [[1, 1], [0, 2]]


def test_committed_address_margin_result_replays_and_fails_closed():
    results_path = Path("measurement/address_margin_results.json")
    verdict_path = Path("measurement/address_margin_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["evaluations"][0]["path_audit"]["retrievals"] += 1
    assert adjudicate(changed)["verdict"] == "AM0_INVALID"
