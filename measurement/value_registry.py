"""Single source of truth for VALUE-1 event-count readout boundary."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.conjunction_registry import CONJUNCTION_SPEC


VALUE_SPEC = {
    "experiment": "value1_event_count_readout_boundary",
    "preregistration_commit": "ec4dca3fc",
    "source_experiment": CONJUNCTION_SPEC["experiment"],
    "source_verdict": "CJ0_INVALID",
    "source_results": "measurement/conjunction_results.json",
    "source_verdict_path": "measurement/conjunction_verdict.json",
    "evaluation_combinations": deepcopy(CONJUNCTION_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(CONJUNCTION_SPEC["evaluation_names"]),
    "eval_episodes": 512,
    "event_counts": [4, 8, 12, 16],
    "events_per_episode": CONJUNCTION_SPEC["events_per_episode"],
    "active_contexts_per_episode": CONJUNCTION_SPEC["active_contexts_per_episode"],
    "active_keys_per_episode": CONJUNCTION_SPEC["active_keys_per_episode"],
    "active_values_per_episode": CONJUNCTION_SPEC["active_values_per_episode"],
    "keys": CONJUNCTION_SPEC["keys"],
    "values": CONJUNCTION_SPEC["values"],
    "contexts": CONJUNCTION_SPEC["contexts"],
    "distractor_steps": CONJUNCTION_SPEC["distractor_steps"],
    "settling_updates": CONJUNCTION_SPEC["settling_updates"],
    "pre_query_updates": CONJUNCTION_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(CONJUNCTION_SPEC["pre_query_dynamics_ablation"]),
    "state_dim": CONJUNCTION_SPEC["state_dim"],
    "minimum_cells": CONJUNCTION_SPEC["minimum_cells"],
    "maximum_cells": CONJUNCTION_SPEC["maximum_cells"],
    "state_pooling": CONJUNCTION_SPEC["state_pooling"],
    "memory_class": CONJUNCTION_SPEC["memory_class"],
    "data_seed": 8_800_000_000,
    "episode_seed_base": 8_900_000_000,
    "seed_stride": CONJUNCTION_SPEC["seed_stride"],
    "device": "cpu",
    "address_mode": "exact_context_key_one_hot",
    "order_mode": "query_first_value_balanced_prefix",
    "stores_per_count": {str(count): count for count in (4, 8, 12, 16)},
    "retrievals_per_episode": 1,
    "arms": [
        "exact_value_normal",
        "exact_value_partner_swap",
        "exact_value_recovered",
    ],
    "thresholds": {
        "selection_accuracy": 0.99,
        "final_accuracy": 0.90,
        "minimum_value_recall": 0.75,
        "content_readout_accuracy": 0.90,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
    },
}


def canonical_spec(spec: dict = VALUE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = VALUE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
