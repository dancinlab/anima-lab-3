import copy
import hashlib

import pytest
import torch

from gate_control1 import attention_mask_mean
from memory_gate import fit_canonical_ridge
from measurement.semantic_memory_write_gate import _checkpoint_valid, adjudicate
from measurement.semantic_memory_write_registry import SEMANTIC_MEMORY_WRITE_SPEC, spec_sha256


def test_semantic_control_is_preregistered_and_pinned():
    assert SEMANTIC_MEMORY_WRITE_SPEC["preregistration_commit"] == "6a7f03c2e"
    assert SEMANTIC_MEMORY_WRITE_SPEC["encoder"]["revision"] == (
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    )
    assert SEMANTIC_MEMORY_WRITE_SPEC["encoder"]["feature_dim"] == 386
    assert len(spec_sha256()) == 64


def test_attention_mask_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(attention_mask_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))
    with pytest.raises(ValueError):
        attention_mask_mean(hidden, torch.ones(1, 2))


def test_shared_canonical_ridge_is_deterministic_and_validates_inputs():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    first = fit_canonical_ridge(features, labels)
    second = fit_canonical_ridge(features, labels)
    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert first[3]["feature_dim"] == 2
    with pytest.raises(ValueError):
        fit_canonical_ridge(torch.ones(2, 2), torch.ones(2))
    with pytest.raises(ValueError):
        fit_canonical_ridge(torch.tensor([[float("nan")], [0.0]]), torch.tensor([0.0, 1.0]))


def test_semantic_adjudicator_fails_closed_on_changed_model_revision():
    changed = copy.deepcopy(SEMANTIC_MEMORY_WRITE_SPEC)
    changed["encoder"]["revision"] = "0" * 40
    payload = {
        "experiment": SEMANTIC_MEMORY_WRITE_SPEC["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GC0_INVALID"


def test_semantic_checkpoint_requires_complete_finite_payload(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text("{}")
    receipt = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not _checkpoint_valid(receipt, SEMANTIC_MEMORY_WRITE_SPEC["encoder"])
