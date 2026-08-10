"""Single source of truth for VALUE-2 stable value integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC


VALUE2_SPEC = {
    "experiment": "value2_stable_value_memory_path",
    "preregistration_commit": "1f1efbf52",
    "source_experiment": VALUE_MECHANISM_SPEC["experiment"],
    "source_verdict": "VP2_LATE_POSITION_LOSS",
    "source_results": "measurement/value_mechanism_results.json",
    "source_verdict_path": "measurement/value_mechanism_verdict.json",
    "checkpoint_path": "checkpoints/value2/canonical_value_projector.pt",
    "evaluation_combinations": deepcopy(VALUE_MECHANISM_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(VALUE_MECHANISM_SPEC["evaluation_names"]),
    "calibration_engine_seeds": [1337, 7331],
    "calibration_episodes": 512,
    "eval_episodes": VALUE_MECHANISM_SPEC["eval_episodes"],
    "query_positions": deepcopy(VALUE_MECHANISM_SPEC["query_positions"]),
    "events_per_episode": VALUE_MECHANISM_SPEC["events_per_episode"],
    "keys": VALUE_MECHANISM_SPEC["keys"],
    "values": VALUE_MECHANISM_SPEC["values"],
    "contexts": VALUE_MECHANISM_SPEC["contexts"],
    "distractor_steps": VALUE_MECHANISM_SPEC["distractor_steps"],
    "settling_updates": VALUE_MECHANISM_SPEC["settling_updates"],
    "pre_query_updates": VALUE_MECHANISM_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(VALUE_MECHANISM_SPEC["pre_query_dynamics_ablation"]),
    "state_dim": VALUE_MECHANISM_SPEC["state_dim"],
    "input_dim": VALUE_MECHANISM_SPEC["state_dim"],
    "address_dim": 32,
    "minimum_cells": VALUE_MECHANISM_SPEC["minimum_cells"],
    "maximum_cells": VALUE_MECHANISM_SPEC["maximum_cells"],
    "model_class": "key_stability.StableKeyProjector",
    "memory_class": VALUE_MECHANISM_SPEC["memory_class"],
    "fit_method": "canonical_ridge",
    "bias": True,
    "weight_decay": 0.0001,
    "temperature": 0.1,
    "calibration_data_seed": 9_100_000_000,
    "calibration_seed_base": 9_200_000_000,
    "eval_seed_base": VALUE_MECHANISM_SPEC["episode_seed_base"],
    "seed_stride": VALUE_MECHANISM_SPEC["seed_stride"],
    "device": "cpu",
    "stores_per_episode": VALUE_MECHANISM_SPEC["events_per_episode"],
    "retrievals_per_episode": 1,
    "value_transform_calls_per_episode": VALUE_MECHANISM_SPEC["events_per_episode"],
    "arms": [
        "integrated_stable_value_normal",
        "external_stable_value_reference",
        "raw_value_control",
        "integrated_stable_value_partner_swap",
        "integrated_stable_value_recovered",
    ],
    "thresholds": {
        "value_classification_accuracy": 0.90,
        "value_classification_minimum_recall": 0.75,
        "selection_accuracy": 0.99,
        "final_accuracy": 0.90,
        "minimum_value_recall": 0.75,
        "reference_prediction_match": 1.0,
        "raw_late_max_accuracy": 0.85,
        "minimum_causal_drop": 0.15,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
        "deterministic_tolerance": 0.0,
    },
}


def canonical_spec(spec: dict = VALUE2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = VALUE2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
