from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from context2 import CompositeStateTransform
from key_stability import StableKeyProjector
from measurement.context2_gate import adjudicate
from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256
from measurement.context_registry import CONTEXT_SPEC
from trinity import VectorMemory


def _projector(classes: int = 4) -> StableKeyProjector:
    model = StableKeyProjector(4, 4, classes, 0.1, True)
    model.eval()
    model.requires_grad_(False)
    return model


def test_context2_registry_matches_preregistration():
    assert CONTEXT2_SPEC["preregistration_commit"] == "eddc01a4d"
    assert CONTEXT2_SPEC["components_per_key"] == 2
    assert CONTEXT2_SPEC["stores_per_episode"] == 4
    assert CONTEXT2_SPEC["retrievals_per_episode"] == 1
    assert CONTEXT2_SPEC["transform_calls_per_episode"] == 5
    assert CONTEXT2_SPEC["composite_address_dim"] == 64
    assert CONTEXT2_SPEC["temperature"] == CONTEXT_SPEC["temperature"]
    assert CONTEXT2_SPEC["bias"] == CONTEXT_SPEC["bias"]
    assert len(CONTEXT2_SPEC["evaluation_combinations"]) == 4
    assert len(spec_sha256()) == 64


def test_vector_memory_routes_composite_keys_only_through_optional_transform():
    calls = []

    def transform(components):
        calls.append(tuple(component.clone() for component in components))
        return torch.cat(tuple(component.mean(0) for component in components))

    memory = VectorMemory(capacity=2, dim=2, key_transform=transform)
    first = (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    second = [torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])]
    memory.store(first, torch.tensor([3.0, 4.0]))
    memory.store(second, torch.tensor([7.0, 8.0]))
    assert torch.equal(memory.retrieve(first, top_k=1)[0], torch.tensor([3.0, 4.0]))
    assert len(calls) == 3
    assert all(len(row) == 2 for row in calls)

    with pytest.raises(TypeError, match="require a key_transform"):
        VectorMemory().store(first, torch.ones(2))
    with pytest.raises(ValueError, match="must not be empty"):
        memory.store((), torch.ones(2))
    with pytest.raises(TypeError, match="components must be"):
        memory.store((torch.ones(1, 2), "not-a-tensor"), torch.ones(2))
    with pytest.raises(ValueError, match="non-finite"):
        memory.store((torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2)), torch.ones(2))


def test_composite_state_transform_validates_registered_state_shape():
    spec = deepcopy(CONTEXT2_SPEC)
    spec.update({
        "state_dim": 4, "component_address_dim": 4, "composite_address_dim": 8,
        "minimum_cells": 1, "maximum_cells": 3,
    })
    transform = CompositeStateTransform(_projector(), _projector(), spec)
    address = transform((torch.randn(2, 4), torch.randn(3, 4)))
    assert address.shape == (8,)
    assert transform.calls == 1
    assert transform.component_counts == [2]
    with pytest.raises(ValueError, match="component count"):
        transform((torch.randn(2, 4),))
    with pytest.raises(ValueError, match="changed shape"):
        transform((torch.randn(2, 5), torch.randn(2, 4)))


def test_composite_state_transform_optionally_uses_predicted_context_center():
    spec = deepcopy(CONTEXT2_SPEC)
    spec.update({
        "state_dim": 4, "component_address_dim": 4, "composite_address_dim": 8,
        "minimum_cells": 1, "maximum_cells": 3, "component_weight": 1.0,
    })
    context = _projector(4)
    key = _projector(4)
    with torch.no_grad():
        context.projection.weight.copy_(torch.eye(4))
        context.projection.bias.zero_()
        context.prototypes.copy_(torch.eye(4))
        key.projection.weight.copy_(torch.eye(4))
        key.projection.bias.zero_()
    states = (torch.tensor([[3.0, 0, 0, 0]]), torch.tensor([[0, 2.0, 0, 0]]))
    legacy = CompositeStateTransform(context, key, spec)(states)
    centered = CompositeStateTransform(context, key, spec, center_context=True)(states)
    assert torch.equal(legacy, centered)

    shifted = (torch.tensor([[3.0, 0.2, 0, 0]]), states[1])
    legacy_shifted = CompositeStateTransform(context, key, spec)(shifted)
    centered_shifted = CompositeStateTransform(
        context, key, spec, center_context=True
    )(shifted)
    assert not torch.equal(legacy_shifted, centered_shifted)
    assert torch.equal(centered_shifted[:4], torch.tensor([1.0, 0, 0, 0]))
    assert torch.equal(
        CompositeStateTransform(
            context, key, spec, center_context=True, mask_context=True
        )(shifted)[:4],
        torch.zeros(4),
    )


def test_committed_context2_result_replays_and_fails_closed():
    results_path = Path("measurement/context2_results.json")
    verdict_path = Path("measurement/context2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed_calls = deepcopy(payload)
    changed_calls["evaluations"][0]["memory_path_audit"][
        "integrated_composite_normal"
    ]["maximum_calls"] += 1
    assert adjudicate(changed_calls)["verdict"] == "CX2I0_INVALID"

    path_loss = deepcopy(payload)
    for row in path_loss["evaluations"]:
        normal = row["arms"]["integrated_composite_normal"]
        normal["selection_accuracy"] = 0.5
        normal["accuracy"] = 0.5
    assert adjudicate(path_loss)["verdict"] == "CX2I_MEMORY_PATH_LOSS"

    value_loss = deepcopy(payload)
    for row in value_loss["evaluations"]:
        normal = row["arms"]["integrated_composite_normal"]
        normal["selection_accuracy"] = 1.0
        normal["accuracy"] = 0.5
        normal["per_value_recall"] = [0.5] * CONTEXT2_SPEC["values"]
    assert adjudicate(value_loss)["verdict"] == "CX2I_VALUE_READOUT_LOSS"
