"""Single source of truth for DECAY-1 memory competition/time decomposition."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.capacity_registry import CAPACITY_SPEC
    from measurement.episode_registry import EPISODE_SPEC
except ModuleNotFoundError:
    from capacity_registry import CAPACITY_SPEC
    from episode_registry import EPISODE_SPEC


DECAY_SPEC = {
    "experiment": "decay1_memory_competition_time_decomposition",
    "preregistration_commit": "1ea8ba092",
    "source_experiment": CAPACITY_SPEC["experiment"],
    "source_verdict": "C3_CAPACITY_BOUNDARY_2",
    "source_results": "measurement/capacity_results.json",
    "source_verdict_path": "measurement/capacity_verdict.json",
    "seeds": deepcopy(CAPACITY_SPEC["seeds"]),
    "prepared_events": 3,
    "queryable_events": 2,
    "distractor_steps": [0, 2, 4, 8],
    "baseline_distractor_steps": 2,
    "eval_episodes_per_delay": 768,
    "balance_mode": "exact_marginals",
    "keys": CAPACITY_SPEC["keys"],
    "values": CAPACITY_SPEC["values"],
    "contexts": CAPACITY_SPEC["contexts"],
    "state_dim": CAPACITY_SPEC["state_dim"],
    "minimum_cells": CAPACITY_SPEC["minimum_cells"],
    "maximum_cells": CAPACITY_SPEC["maximum_cells"],
    "state_pooling": CAPACITY_SPEC["state_pooling"],
    "address_dim": CAPACITY_SPEC["address_dim"],
    "model_class": CAPACITY_SPEC["model_class"],
    "memory_class": CAPACITY_SPEC["memory_class"],
    "data_seed": 7_500_000_000,
    "episode_seed_base": 7_600_000_000,
    "delay_seed_stride": 10_000_000,
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "device": "cpu",
    "stable_arms": [
        "two_stream_two_candidates",
        "three_stream_two_candidates",
        "three_stream_three_candidates",
    ],
    "arms": [
        "two_stream_two_candidates",
        "three_stream_two_candidates",
        "three_stream_three_candidates",
        "raw_three_stream_three_candidates",
        "exact_three_candidates",
        "exact_three_partner_swap",
        "exact_three_recovered",
    ],
    "expected_transform_calls": {
        "two_stream_two_candidates": 3,
        "three_stream_two_candidates": 3,
        "three_stream_three_candidates": 4,
    },
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


def canonical_spec(spec: dict = DECAY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = DECAY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
