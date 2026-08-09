"""Single source of truth for CONTROL-1 dynamic relation memory."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy


CONTROL_SPEC = {
    "experiment": "control1_dynamic_relation_positive_control",
    "seeds": [1337, 7331],
    "data_seed": 20260810,
    "keys": 8,
    "values": 8,
    "distractors": 16,
    "relations_per_episode": 2,
    "distractor_steps": 2,
    "splits": {
        "train": 8192,
        "validation": 1024,
        "eval": 2048,
    },
    "split_seed_offsets": {
        "train": 0,
        "validation": 1_000_000,
        "eval": 2_000_000,
    },
    "state_dim": 96,
    "batch_size": 128,
    "train_steps": 2000,
    "validate_every": 50,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "device": "cpu",
    "arms": ["gru", "vector_memory", "no_memory", "shuffled_labels"],
    "thresholds": {
        "gru_accuracy": 0.90,
        "gru_min_value_recall": 0.80,
        "vector_memory_accuracy": 1.0,
        "negative_control_max_accuracy": 0.25,
        "maximum_balance_delta": 1,
    },
}


ONLINE_CONTROL_SPEC = {
    "experiment": "control2_online_dynamic_relation_positive_control",
    "reference_experiment": CONTROL_SPEC["experiment"],
    "seeds": [1337, 7331],
    "online_data_seed": 20260811,
    "keys": 8,
    "values": 8,
    "distractors": 16,
    "relations_per_episode": 2,
    "distractor_steps": 2,
    "splits": {
        "validation": 1024,
        "eval": 2048,
    },
    "state_dim": 96,
    "batch_size": 128,
    "train_steps": 2000,
    "online_train_examples": 256000,
    "validate_every": 50,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "device": "cpu",
    "arms": ["gru", "vector_memory", "no_memory", "shuffled_labels"],
    "thresholds": {
        "gru_accuracy": 0.90,
        "gru_min_value_recall": 0.80,
        "vector_memory_accuracy": 1.0,
        "negative_control_max_accuracy": 0.25,
        "maximum_balance_delta": 0,
    },
}


ATTENTION_CONTROL_SPEC = {
    "experiment": "control3_keyed_attention_positive_control",
    "reference_experiment": CONTROL_SPEC["experiment"],
    "seeds": [1337, 7331],
    "online_data_seed": 20260811,
    "keys": 8,
    "values": 8,
    "distractors": 16,
    "relations_per_episode": 2,
    "distractor_steps": 2,
    "splits": {
        "validation": 1024,
        "eval": 2048,
    },
    "state_dim": 96,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "batch_size": 128,
    "train_steps": 2000,
    "online_train_examples": 256000,
    "validate_every": 50,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "device": "cpu",
    "model_class": "torch.nn.MultiheadAttention",
    "arms": ["attention", "vector_memory", "no_memory", "shuffled_labels"],
    "thresholds": {
        "attention_accuracy": 0.90,
        "attention_min_value_recall": 0.80,
        "vector_memory_accuracy": 1.0,
        "negative_control_max_accuracy": 0.25,
        "maximum_balance_delta": 0,
    },
}


REGISTERED_EXPERIMENTS = {
    CONTROL_SPEC["experiment"]: CONTROL_SPEC,
    ONLINE_CONTROL_SPEC["experiment"]: ONLINE_CONTROL_SPEC,
    ATTENTION_CONTROL_SPEC["experiment"]: ATTENTION_CONTROL_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown CONTROL-1 experiment: {name}") from exc


def canonical_spec(spec: dict = CONTROL_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTROL_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
