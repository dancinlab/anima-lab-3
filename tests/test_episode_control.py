from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from episode_control import (
    DynamicRelationGRU,
    KeyedRelationAttention,
    OnlineEpisodeStream,
    _input_dim,
    _metrics,
    _no_memory_predictions,
    _vector_memory_predictions,
    build_split,
    build_reference_splits,
    build_splits,
    dataset_audit,
    encode_episodes,
    labels,
    relation_tensors,
)
from measurement.episode_control_gate import adjudicate
from measurement.episode_control_registry import (
    ATTENTION_CONTROL_SPEC, CONTROL_SPEC, ONLINE_CONTROL_SPEC, spec_sha256,
)


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


def test_keyed_attention_uses_shared_keys_and_registered_shapes():
    model = KeyedRelationAttention(
        ATTENTION_CONTROL_SPEC["keys"], ATTENTION_CONTROL_SPEC["values"],
        ATTENTION_CONTROL_SPEC["state_dim"], ATTENTION_CONTROL_SPEC["attention_heads"],
        ATTENTION_CONTROL_SPEC["attention_dropout"],
    )
    episodes = build_reference_splits(ATTENTION_CONTROL_SPEC)["validation"][:3]
    stores, values, queries = relation_tensors(episodes)
    logits, weights = model(stores, values, queries, need_weights=True)
    assert logits.shape == (3, ATTENTION_CONTROL_SPEC["values"])
    assert weights.shape == (3, ATTENTION_CONTROL_SPEC["relations_per_episode"])
    assert model.attention.num_heads == ATTENTION_CONTROL_SPEC["attention_heads"]
    assert model.attention.dropout == ATTENTION_CONTROL_SPEC["attention_dropout"]


def test_online_stream_is_fresh_balanced_disjoint_and_deterministic():
    fixed = build_reference_splits()
    control1 = build_splits()
    for name in ONLINE_CONTROL_SPEC["splits"]:
        assert [row.fingerprint() for row in fixed[name]] == [
            row.fingerprint() for row in control1[name]
        ]
    excluded = {row.fingerprint() for rows in fixed.values() for row in rows}
    left = OnlineEpisodeStream(ONLINE_CONTROL_SPEC, excluded)
    right = OnlineEpisodeStream(ONLINE_CONTROL_SPEC, excluded)
    for _ in range(3):
        left_batch = left.next_batch()
        right_batch = right.next_batch()
        assert [row.fingerprint() for row in left_batch] == [
            row.fingerprint() for row in right_batch
        ]
    audit = left.audit()
    assert audit["examples"] == audit["unique_fingerprints"] == 384
    assert audit["fixed_split_overlap"] == 0
    assert audit["balanced_batches"] == 3
    assert len(set(audit["target_counts"].values())) == 1
    assert len(set(audit["query_key_counts"].values())) == 1
    assert len(set(audit["query_position_counts"].values())) == 1
    assert audit == right.audit()


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


def _online_passing_payload(tmp_path: Path) -> dict:
    fixed = build_reference_splits()
    expected = labels(fixed["eval"])
    perfect = _metrics(expected, expected, ONLINE_CONTROL_SPEC["values"])
    wrong = _metrics(
        expected, (expected + 1) % ONLINE_CONTROL_SPEC["values"],
        ONLINE_CONTROL_SPEC["values"],
    )
    chance = _metrics(expected, torch.zeros_like(expected), ONLINE_CONTROL_SPEC["values"])
    examples = ONLINE_CONTROL_SPEC["online_train_examples"]
    training_audit = {
        "examples": examples,
        "unique_fingerprints": examples,
        "fixed_split_overlap": 0,
        "balanced_batches": ONLINE_CONTROL_SPEC["train_steps"],
        "target_counts": {str(index): examples // 8 for index in range(8)},
        "query_key_counts": {str(index): examples // 8 for index in range(8)},
        "query_position_counts": {str(index): examples // 2 for index in range(2)},
        "ordered_fingerprint_sha256": "a" * 64,
        "key_value_counts": [[examples // 64] * 8 for _ in range(8)],
    }
    seeds = []
    for seed in ONLINE_CONTROL_SPEC["seeds"]:
        checkpoint = tmp_path / f"online_seed_{seed}.pt"
        torch.save({
            "experiment": ONLINE_CONTROL_SPEC["experiment"],
            "spec_sha256": spec_sha256(ONLINE_CONTROL_SPEC),
            "seed": seed,
            "selected_step": 100,
            "training_stream_sha256": training_audit["ordered_fingerprint_sha256"],
            "model": {},
        }, checkpoint)
        seeds.append({
            "seed": seed,
            "selected_step": 100,
            "validation_accuracy": 1.0,
            "training_audit": deepcopy(training_audit),
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
        "experiment": ONLINE_CONTROL_SPEC["experiment"],
        "spec": deepcopy(ONLINE_CONTROL_SPEC),
        "spec_sha256": spec_sha256(ONLINE_CONTROL_SPEC),
        "dataset_audit": dataset_audit(fixed, ONLINE_CONTROL_SPEC),
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": seeds,
    }


def test_online_control_gate_passes_and_fails_closed(tmp_path):
    value = _online_passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "O1_ONLINE_CONTROL_VALID"

    failed = deepcopy(value)
    failed["seeds"][0]["arms"]["gru"]["accuracy"] = 0.50
    assert adjudicate(failed)["verdict"] == "O2_ONLINE_TRAINING_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["training_audit"]["unique_fingerprints"] -= 1
    assert adjudicate(invalid)["verdict"] == "O0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][1]["training_audit"]["ordered_fingerprint_sha256"] = "b" * 64
    assert adjudicate(invalid)["verdict"] == "O0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["training_audit"]["target_counts"] = {
        str(index): 0 for index in range(8)
    }
    assert adjudicate(invalid)["verdict"] == "O0_INVALID"


def _attention_passing_payload(tmp_path: Path) -> dict:
    fixed = build_reference_splits(ATTENTION_CONTROL_SPEC)
    expected = labels(fixed["eval"])
    perfect = _metrics(expected, expected, ATTENTION_CONTROL_SPEC["values"])
    wrong = _metrics(
        expected, (expected + 1) % ATTENTION_CONTROL_SPEC["values"],
        ATTENTION_CONTROL_SPEC["values"],
    )
    chance = _metrics(expected, torch.zeros_like(expected), ATTENTION_CONTROL_SPEC["values"])
    examples = ATTENTION_CONTROL_SPEC["online_train_examples"]
    training_audit = {
        "examples": examples,
        "unique_fingerprints": examples,
        "fixed_split_overlap": 0,
        "balanced_batches": ATTENTION_CONTROL_SPEC["train_steps"],
        "target_counts": {str(index): examples // 8 for index in range(8)},
        "query_key_counts": {str(index): examples // 8 for index in range(8)},
        "query_position_counts": {str(index): examples // 2 for index in range(2)},
        "ordered_fingerprint_sha256": "a" * 64,
        "key_value_counts": [[examples // 64] * 8 for _ in range(8)],
    }
    seeds = []
    for seed in ATTENTION_CONTROL_SPEC["seeds"]:
        checkpoint = tmp_path / f"attention_seed_{seed}.pt"
        torch.save({
            "experiment": ATTENTION_CONTROL_SPEC["experiment"],
            "spec_sha256": spec_sha256(ATTENTION_CONTROL_SPEC),
            "seed": seed,
            "selected_step": 100,
            "training_stream_sha256": training_audit["ordered_fingerprint_sha256"],
            "model_class": ATTENTION_CONTROL_SPEC["model_class"],
            "model": {},
        }, checkpoint)
        seeds.append({
            "seed": seed,
            "selected_step": 100,
            "validation_accuracy": 1.0,
            "training_audit": deepcopy(training_audit),
            "arms": {
                "attention": deepcopy(perfect),
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
        "experiment": ATTENTION_CONTROL_SPEC["experiment"],
        "spec": deepcopy(ATTENTION_CONTROL_SPEC),
        "spec_sha256": spec_sha256(ATTENTION_CONTROL_SPEC),
        "dataset_audit": dataset_audit(fixed, ATTENTION_CONTROL_SPEC),
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": seeds,
    }


def test_attention_control_gate_passes_and_fails_closed(tmp_path):
    value = _attention_passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "A1_KEYED_ATTENTION_VALID"

    failed = deepcopy(value)
    failed["seeds"][0]["arms"]["attention"]["accuracy"] = 0.50
    assert adjudicate(failed)["verdict"] == "A2_ATTENTION_PATH_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["checkpoint"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "A0_INVALID"

    invalid = deepcopy(value)
    checkpoint = Path(invalid["seeds"][0]["checkpoint"]["path"])
    stored = torch.load(checkpoint, weights_only=True)
    stored["model_class"] = "custom.Attention"
    torch.save(stored, checkpoint)
    invalid["seeds"][0]["checkpoint"]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert adjudicate(invalid)["verdict"] == "A0_INVALID"
