import json

import pytest
import torch

from gate1 import build_calibration, build_evaluation, dataset_audit
from memory_gate import DialogueMemoryGate, fit_canonical_memory_gate, memory_gate_features
from measurement.memory_write_gate import adjudicate
from measurement.memory_write_gate_registry import MEMORY_WRITE_GATE_SPEC, spec_sha256


def test_registered_spec_is_preregistered():
    assert MEMORY_WRITE_GATE_SPEC["preregistration_commit"] == "092b818e3"
    assert len(spec_sha256()) == 64


def test_dataset_is_balanced_unique_and_disjoint():
    calibration = build_calibration(1337)
    evaluation = build_evaluation(1337)
    audit = dataset_audit(calibration, evaluation, MEMORY_WRITE_GATE_SPEC)
    assert audit["calibration_positive"] == audit["calibration_negative"] == 2048
    assert audit["calibration_unique"] == 4096
    assert audit["evaluation_unique"] == 8192
    assert audit["overlap"] == 0
    assert set(audit["fact_counts"].values()) == {256}


def test_canonical_fit_and_checkpoint_round_trip(tmp_path):
    gate, audit = fit_canonical_memory_gate(build_calibration(1337))
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate.to_payload()))
    restored = DialogueMemoryGate.load(path)
    row = build_evaluation(1337)[0]["candidates"][0]
    assert restored.score(row["role"], row["text"]) == pytest.approx(
        gate.score(row["role"], row["text"]), abs=1e-12
    )
    assert audit["feature_dim"] == 130


def test_feature_validation():
    assert memory_gate_features("user", "기억할 문장").shape == (130,)
    with pytest.raises(ValueError):
        memory_gate_features("system", "기억할 문장")
    with pytest.raises(ValueError):
        memory_gate_features("user", " ")


def test_adjudicator_fails_closed_on_wrong_spec():
    payload = {
        "experiment": MEMORY_WRITE_GATE_SPEC["experiment"],
        "spec": {**MEMORY_WRITE_GATE_SPEC, "top_k": 4},
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "G0_INVALID"


def test_gate_rejects_non_finite_checkpoint():
    with pytest.raises(ValueError):
        DialogueMemoryGate(torch.full((130,), float("nan")), 0.0, 0.5)
