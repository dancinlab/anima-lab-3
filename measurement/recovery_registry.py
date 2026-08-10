"""Single source of truth for RECOVERY-1 dense memory recovery curve."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.decay_registry import DECAY_SPEC
except ModuleNotFoundError:
    from decay_registry import DECAY_SPEC


RECOVERY_SPEC = {
    "experiment": "recovery1_dense_memory_recovery_curve",
    "preregistration_commit": "5634a0ab1",
    "source_experiment": DECAY_SPEC["experiment"],
    "source_verdict": "D5_NON_MONOTONIC_OR_MIXED",
    "source_results": "measurement/decay_results.json",
    "source_verdict_path": "measurement/decay_verdict.json",
    "seeds": deepcopy(DECAY_SPEC["seeds"]),
    "replicates": [0, 1, 2],
    "episodes_per_replicate": 512,
    "prepared_events": 3,
    "queryable_events": 2,
    "distractor_steps": list(range(9)),
    "keys": DECAY_SPEC["keys"],
    "values": DECAY_SPEC["values"],
    "contexts": DECAY_SPEC["contexts"],
    "state_dim": DECAY_SPEC["state_dim"],
    "minimum_cells": DECAY_SPEC["minimum_cells"],
    "maximum_cells": DECAY_SPEC["maximum_cells"],
    "state_pooling": DECAY_SPEC["state_pooling"],
    "address_dim": DECAY_SPEC["address_dim"],
    "model_class": DECAY_SPEC["model_class"],
    "memory_class": DECAY_SPEC["memory_class"],
    "data_seed_base": 7_700_000_000,
    "episode_seed_base": 7_800_000_000,
    "replicate_seed_stride": 10_000_000,
    "seed_stride": DECAY_SPEC["seed_stride"],
    "device": "cpu",
    "stable_arms": ["stable_three_candidates", "stable_two_candidates"],
    "arms": [
        "stable_three_candidates",
        "stable_two_candidates",
        "exact_three_candidates",
        "exact_three_partner_swap",
        "exact_three_recovered",
    ],
    "expected_transform_calls": {
        "stable_three_candidates": 4,
        "stable_two_candidates": 3,
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
        "minimum_recovery_delta": 0.05,
        "maximum_balance_delta": 0,
    },
}


def canonical_spec(spec: dict = RECOVERY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RECOVERY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
