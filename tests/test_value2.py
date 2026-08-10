from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from measurement.value2_gate import adjudicate
from measurement.value2_registry import VALUE2_SPEC, spec_sha256
from trinity import VectorMemory


def test_value2_registry_matches_preregistration():
    assert VALUE2_SPEC["preregistration_commit"] == "1f1efbf52"
    assert VALUE2_SPEC["fit_method"] == "canonical_ridge"
    assert VALUE2_SPEC["address_dim"] == 32
    assert VALUE2_SPEC["query_positions"] == [0, 4, 8, 12, 15]
    assert len(spec_sha256()) == 64


def test_vector_memory_default_value_path_is_backward_compatible():
    memory = VectorMemory(capacity=2, dim=2)
    key = torch.tensor([1.0, 0.0])
    value = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    memory.store(key, value)
    assert torch.equal(memory.values[0], value.mean(0))
    assert torch.equal(memory.retrieve(key, top_k=1)[0], value.mean(0))


def test_vector_memory_applies_and_validates_one_value_transform_per_store():
    calls = []

    def transform(value):
        calls.append(value.clone())
        return value[:2] * 2

    memory = VectorMemory(capacity=2, dim=2, value_transform=transform)
    key = torch.tensor([1.0, 0.0])
    value = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    memory.store(key, value)
    assert len(calls) == 1
    assert torch.equal(calls[0], value.mean(0))
    assert torch.equal(memory.retrieve(key, top_k=1)[0], value.mean(0)[:2] * 2)

    bad = VectorMemory(value_transform=lambda _: torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="non-empty 1D"):
        bad.store(key, value)
    changing = VectorMemory(value_transform=lambda row: row[:1] if not calls else row[:2])
    changing.value_transform = lambda row: row[:1]
    changing.store(key, value)
    changing.value_transform = lambda row: row[:2]
    with pytest.raises(ValueError, match="changed the value width"):
        changing.store(key, value)


def test_committed_value2_result_replays_and_rejects_call_changes():
    results_path = Path("measurement/value2_results.json")
    verdict_path = Path("measurement/value2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["evaluations"][0]["positions"][0]["path_audit"]["maximum_calls"] += 1
    assert adjudicate(changed)["verdict"] == "VT0_INVALID"
