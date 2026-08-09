from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from episode_control import (
    DynamicRelationGRU,
    _input_dim,
    _metrics,
    _no_memory_predictions,
    _vector_memory_predictions,
    build_split,
    build_splits,
    dataset_audit,
    encode_episodes,
    labels,
)
from measurement.episode_control_gate import adjudicate
from measurement.episode_control_registry import CONTROL_SPEC, spec_sha256


def test_control_dataset_is_balanced_disjoint_and_encodable():
    splits = build_splits()
    audit = dataset_audit(splits)
    assert set(audit["overlap"].values()) == {0}
    for name, count in CONTROL_SPEC["splits"].items():
        row = audit["splits"][name]
        assert row["episodes"] == row["unique_fingerprints"] == count
        assert len(set(row["target_counts"].values())) == 1
        assert len(set(row["query_key_counts"].values())) == 1
        assert len(set(row["query_position_counts"].values())) == 1
    encoded = encode_episodes(splits["validation"][:3])
    assert encoded.shape == (3, 5, _input_dim(CONTROL_SPEC))
    assert torch.all(encoded[:, :, :3].sum(-1) == 1)


def test_existing_vector_memory_is_exact_and_no_memory_is_at_chance():
    splits = build_splits()
    train = splits["train"]
    evaluate = splits["eval"]
    expected = labels(evaluate)
    vector = _vector_memory_predictions(evaluate, CONTROL_SPEC)
    no_memory = _no_memory_predictions(train, evaluate, CONTROL_SPEC)
    assert torch.equal(vector, expected)
    assert float((no_memory == expected).float().mean()) == 1 / CONTROL_SPEC["values"]


def test_control_gru_uses_registered_shapes():
    model = DynamicRelationGRU(_input_dim(CONTROL_SPEC), CONTROL_SPEC["state_dim"],
                               CONTROL_SPEC["values"])
    logits = model(torch.zeros(2, 5, _input_dim(CONTROL_SPEC)))
    assert logits.shape == (2, CONTROL_SPEC["values"])


def _passing_payload(tmp_path: Path) -> dict:
    splits = build_splits()
    expected = labels(splits["eval"])
    perfect = _metrics(expected, expected, CONTROL_SPEC["values"])
    wrong = _metrics(expected, (expected + 1) % CONTROL_SPEC["values"], CONTROL_SPEC["values"])
    chance = _metrics(expected, torch.zeros_like(expected), CONTROL_SPEC["values"])
    seeds = []
    for seed in CONTROL_SPEC["seeds"]:
        checkpoint = tmp_path / f"seed_{seed}.pt"
        torch.save({
            "experiment": CONTROL_SPEC["experiment"],
            "spec_sha256": spec_sha256(CONTROL_SPEC),
            "seed": seed,
            "selected_step": 100,
            "model": {},
        }, checkpoint)
        seeds.append({
            "seed": seed,
            "selected_step": 100,
            "validation_accuracy": 1.0,
            "arms": {
                "gru": deepcopy(perfect),
                "vector_memory": deepcopy(perfect),
                "no_memory": deepcopy(chance),
                "shuffled_labels": deepcopy(wrong),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
        })
    return {
        "experiment": CONTROL_SPEC["experiment"],
        "spec": deepcopy(CONTROL_SPEC),
        "spec_sha256": spec_sha256(CONTROL_SPEC),
        "dataset_audit": dataset_audit(splits),
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": seeds,
    }


def test_control_gate_passes_and_fails_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "P1_POSITIVE_CONTROL_VALID"

    failed = deepcopy(value)
    failed["seeds"][0]["arms"]["gru"]["accuracy"] = 0.50
    assert adjudicate(failed)["verdict"] == "P2_TRAINING_PATH_INVALID"

    invalid = deepcopy(value)
    invalid["dataset_audit"]["overlap"]["train:eval"] = 1
    assert adjudicate(invalid)["verdict"] == "P0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["checkpoint"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "P0_INVALID"
