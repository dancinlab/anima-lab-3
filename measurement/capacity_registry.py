"""Single source of truth for CAPACITY-1 stable-address event boundary."""
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


CAPACITY_SPEC = {
    "experiment": "capacity1_stable_address_event_boundary",
    "preregistration_commit": "24fe3e1a9",
    "source_experiment": EPISODE2_SPEC["experiment"],
    "source_verdict": "E2I_PATH_RECOVERED_NOT_UNIQUE",
    "source_results": "measurement/episode2_results.json",
    "source_verdict_path": "measurement/episode2_verdict.json",
    "seeds": deepcopy(EPISODE2_SPEC["seeds"]),
    "event_counts": [2, 3, 4],
    "eval_episodes_per_count": 1536,
    "balance_mode": "exact_marginals",
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
    "data_seed_base": 7_300_000_000,
    "episode_seed_base": 7_400_000_000,
    "event_count_seed_stride": 10_000_000,
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "device": "cpu",
    "arms": [
        "stable_distinct_normal",
        "raw_distinct_control",
        "exact_key_control",
        "exact_key_partner_swap",
        "exact_key_recovered",
    ],
    "thresholds": {
        "stable_selection_accuracy": 0.90,
        "stable_final_accuracy": 0.90,
        "stable_minimum_value_recall": 0.75,
        "content_readout_accuracy": 0.90,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
        "maximum_balance_delta": 0,
    },
}


def canonical_spec(spec: dict = CAPACITY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CAPACITY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
