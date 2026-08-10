"""Single source of truth for SEPARATION-1 similar-episode collision mapping."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.episode2_registry import EPISODE2_SPEC
    from measurement.episode_registry import EPISODE_SPEC
except ModuleNotFoundError:
    from episode2_registry import EPISODE2_SPEC
    from episode_registry import EPISODE_SPEC


SEPARATION_SPEC = {
    "experiment": "separation1_similar_episode_collision",
    "preregistration_commit": "2dc5dc0fd",
    "protocol_repair_commit": "f1350f8b1",
    "source_experiment": EPISODE2_SPEC["experiment"],
    "source_verdict": "E2I_PATH_RECOVERED_NOT_UNIQUE",
    "source_results": "measurement/episode2_results.json",
    "source_verdict_path": "measurement/episode2_verdict.json",
    "seeds": deepcopy(EPISODE2_SPEC["seeds"]),
    "eval_episodes": 2048,
    "events_per_episode": 4,
    "keys": EPISODE2_SPEC["keys"],
    "values": EPISODE2_SPEC["values"],
    "contexts": len(EPISODE_SPEC["distractor_words"]),
    "distractor_steps": 2,
    "state_dim": EPISODE2_SPEC["state_dim"],
    "minimum_cells": 2,
    "maximum_cells": EPISODE_SPEC["cells"],
    "state_pooling": "cell_mean",
    "address_dim": EPISODE2_SPEC["address_dim"],
    "model_class": EPISODE2_SPEC["model_class"],
    "memory_class": EPISODE2_SPEC["memory_class"],
    "data_seed": 7_100_000_000,
    "episode_seed_base": 7_200_000_000,
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "expected_stable_transform_calls_per_episode": 5,
    "device": "cpu",
    "arms": [
        "stable_similar_normal",
        "raw_similar_normal",
        "stable_distinct_key_control",
        "exact_context_key_control",
        "exact_context_key_partner_swap",
        "exact_context_key_recovered",
        "context_removed_control",
    ],
    "thresholds": {
        "similar_selection_accuracy": 0.75,
        "similar_final_accuracy": 0.75,
        "similar_minimum_value_recall": 0.60,
        "content_readout_accuracy": 0.90,
        "distinct_selection_accuracy": 0.90,
        "distinct_final_accuracy": 0.90,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "context_removed_max_accuracy": 0.35,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
        "maximum_balance_delta": 0,
    },
}


def canonical_spec(spec: dict = SEPARATION_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SEPARATION_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
