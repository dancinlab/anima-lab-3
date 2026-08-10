from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from conjunction import (
    _exact_addresses, _latin_valid, _memory_outcome, build_episodes, dataset_audit,
)
from context2 import CompositeStateTransform
from key_stability import StableKeyProjector
from measurement.conjunction_gate import adjudicate
from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256
from trinity import VectorMemory


def _projector() -> StableKeyProjector:
    model = StableKeyProjector(4, 4, 4, 0.1, True)
    model.eval()
    model.requires_grad_(False)
    return model


def test_conjunction_registry_matches_preregistration():
    spec = CONJUNCTION_SPEC
    assert spec["preregistration_commit"] == "5129d96d8"
    assert spec["events_per_episode"] == 16
    assert spec["stores_per_episode"] == 16
    assert spec["transform_calls_per_episode"] == 17
    assert spec["components_per_key"] == 2
    assert len(spec["evaluation_combinations"]) == 4
    assert len(spec_sha256()) == 64


def test_balanced_conjunction_dataset_requires_both_components():
    spec = deepcopy(CONJUNCTION_SPEC)
    spec["eval_episodes"] = spec["contexts"] * spec["keys"] * spec["values"]
    episodes = build_episodes(spec)
    audit = dataset_audit(episodes, spec)
    assert audit["episodes"] == audit["unique_fingerprints"] == spec["eval_episodes"]
    assert audit["latin_valid_episodes"] == spec["eval_episodes"]
    assert set(audit["query_triple_counts"].values()) == {1}
    assert all(_latin_valid(episode) for episode in episodes)
    for episode in episodes[:32]:
        exact, query = _exact_addresses(episode, spec=spec)
        context_only, context_query = _exact_addresses(episode, mask_key=True, spec=spec)
        key_only, key_query = _exact_addresses(episode, mask_context=True, spec=spec)
        assert len(exact) == len(context_only) == len(key_only) == 16
        assert query.numel() == context_query.numel() == key_query.numel() == 16
        assert len({(episode.contexts[i], episode.keys[i]) for i in range(16)}) == 16


def test_composite_transform_can_mask_either_registered_component():
    spec = deepcopy(CONJUNCTION_SPEC)
    spec.update({
        "state_dim": 4, "component_address_dim": 4, "composite_address_dim": 8,
        "minimum_cells": 1, "maximum_cells": 3,
    })
    components = (torch.randn(2, 4), torch.randn(3, 4))
    context_masked = CompositeStateTransform(
        _projector(), _projector(), spec, mask_context=True
    )(components)
    key_masked = CompositeStateTransform(
        _projector(), _projector(), spec, mask_key=True
    )(components)
    assert torch.equal(context_masked[:4], torch.zeros(4))
    assert torch.equal(key_masked[4:], torch.zeros(4))


def test_diagnostic_selection_uses_the_common_memory_tie_rule():
    memory = VectorMemory(capacity=4, dim=2)
    keys = [torch.tensor([1.0, 0.0])] * 4
    values = [torch.tensor([[float(index), 0.0]]) for index in range(4)]
    for key, value in zip(keys, values):
        memory.store(key, value)
    prototypes = torch.stack([value.mean(0) for value in values])
    _, selected, api_match, _ = _memory_outcome(
        memory, torch.tensor([1.0, 0.0]), values, prototypes
    )
    expected_selected = int(torch.ones(4).topk(1).indices[0])
    assert selected == expected_selected
    assert api_match is True


def test_committed_conjunction_result_replays_and_fails_closed():
    results_path = Path("measurement/conjunction_results.json")
    verdict_path = Path("measurement/conjunction_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    if expected["verdict"] == "CJ0_INVALID":
        return

    changed_balance = deepcopy(payload)
    first = next(iter(changed_balance["dataset_audit"]["query_triple_counts"]))
    changed_balance["dataset_audit"]["query_triple_counts"][first] += 1
    assert adjudicate(changed_balance)["verdict"] == "CJ0_INVALID"

    changed_calls = deepcopy(payload)
    changed_calls["evaluations"][0]["memory_path_audit"][
        "integrated_conjunction_normal"
    ]["maximum_calls"] += 1
    assert adjudicate(changed_calls)["verdict"] == "CJ0_INVALID"

    not_conjunctive = deepcopy(payload)
    for row in not_conjunctive["evaluations"]:
        row["arms"]["integrated_context_masked"]["accuracy"] = 0.5
    assert adjudicate(not_conjunctive)["verdict"] == "CJ4_NOT_CONJUNCTIVE"

    collision = deepcopy(payload)
    for row in collision["evaluations"]:
        normal = row["arms"]["integrated_conjunction_normal"]
        normal["selection_accuracy"] = 0.5
        normal["accuracy"] = 0.5
    assert adjudicate(collision)["verdict"] == "CJ2_COMPONENT_COLLISION"

    readout = deepcopy(payload)
    for row in readout["evaluations"]:
        normal = row["arms"]["integrated_conjunction_normal"]
        normal["selection_accuracy"] = 1.0
        normal["accuracy"] = 0.5
        normal["per_value_recall"] = [0.5] * CONJUNCTION_SPEC["values"]
    assert adjudicate(readout)["verdict"] == "CJ3_VALUE_READOUT_LOSS"
