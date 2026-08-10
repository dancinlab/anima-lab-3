"""Single source of truth for EPISODE-2 stable-key memory integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.episode_registry import EPISODE_SPEC
    from measurement.key_registry import KEY_SPEC
except ModuleNotFoundError:
    from episode_registry import EPISODE_SPEC
    from key_registry import KEY_SPEC


EPISODE2_SPEC = {
    "experiment": "episode2_integrated_stable_memory_path",
    "preregistration_commit": "e197a9ac6",
    "source_experiment": KEY_SPEC["experiment"],
    "source_verdict": "K1_STABLE_KEY_VALID_NOT_UNIQUE",
    "source_results": "measurement/key_results.json",
    "source_verdict_path": "measurement/key_verdict.json",
    "source_checkpoint_dir": "checkpoints/key1",
    "seeds": deepcopy(KEY_SPEC["seeds"]),
    "eval_split": KEY_SPEC["eval_split"],
    "eval_episodes": KEY_SPEC["eval_episodes"],
    "keys": KEY_SPEC["keys"],
    "values": KEY_SPEC["values"],
    "relations_per_episode": KEY_SPEC["relations_per_episode"],
    "state_dim": EPISODE_SPEC["state_dim"],
    "address_dim": KEY_SPEC["address_dim"],
    "model_class": KEY_SPEC["model_class"],
    "memory_class": "trinity.VectorMemory",
    "memory_transform_argument": "key_transform",
    "expected_transform_calls_per_episode": 3,
    "device": "cpu",
    "arms": [
        "integrated_stable_normal",
        "integrated_stable_partner_swap",
        "integrated_stable_recovered",
        "manual_stable_reference",
        "transform_disabled",
        "sensory_memory",
        "keyed_attention",
        "no_memory",
    ],
    "thresholds": {
        "selection_accuracy": 0.90,
        "final_accuracy": 0.75,
        "minimum_value_recall": 0.60,
        "content_readout_accuracy": 0.90,
        "positive_control_accuracy": 0.90,
        "negative_control_max_accuracy": 0.25,
        "recovery_prediction_match": 1.0,
        "reference_prediction_match": 1.0,
    },
}


def canonical_spec(spec: dict = EPISODE2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = EPISODE2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
