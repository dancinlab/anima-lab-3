"""Single source of truth for KEY-1 temporal key stabilization."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.episode_registry import EPISODE_SPEC
except ModuleNotFoundError:
    from episode_registry import EPISODE_SPEC


KEY_SPEC = {
    "experiment": "key1_temporal_key_stabilization",
    "preregistration_commit": "8512f1c04",
    "source_experiment": EPISODE_SPEC["experiment"],
    "source_verdict": "E2_KEY_RETRIEVAL_LOSS",
    "source_results": "measurement/episode_results.json",
    "source_verdict_path": "measurement/episode_verdict.json",
    "source_checkpoint_dir": "checkpoints/episode1",
    "seeds": deepcopy(EPISODE_SPEC["seeds"]),
    "calibration_split": "validation",
    "eval_split": EPISODE_SPEC["eval_split"],
    "calibration_episodes": 1024,
    "eval_episodes": EPISODE_SPEC["splits"][EPISODE_SPEC["eval_split"]],
    "keys": EPISODE_SPEC["keys"],
    "values": EPISODE_SPEC["values"],
    "relations_per_episode": EPISODE_SPEC["relations_per_episode"],
    "input_dim": EPISODE_SPEC["state_dim"],
    "address_dim": 32,
    "model_class": "key_stability.StableKeyProjector",
    "bias": True,
    "train_steps": 1000,
    "batch_size": 256,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "temperature": 0.1,
    "calibration_seed_base": 5_000_000_000,
    "eval_seed_base": EPISODE_SPEC["episode_seed_base"],
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "shuffle_seed_base": 6_000_000_000,
    "device": "cpu",
    "arms": [
        "stabilized_memory_normal",
        "stabilized_memory_partner_swap",
        "stabilized_memory_recovered",
        "raw_quantum_memory",
        "sensory_memory",
        "keyed_attention",
        "no_memory",
        "shuffled_label_projector",
    ],
    "thresholds": {
        "key_classification_accuracy": 0.90,
        "selection_accuracy": 0.90,
        "final_accuracy": 0.75,
        "minimum_value_recall": 0.60,
        "content_readout_accuracy": 0.90,
        "positive_control_accuracy": 0.90,
        "negative_control_max_accuracy": 0.25,
        "recovery_prediction_match": 1.0,
        "maximum_balance_delta": 0,
    },
}


def canonical_spec(spec: dict = KEY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = KEY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
