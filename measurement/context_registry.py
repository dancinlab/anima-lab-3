"""Single source of truth for CONTEXT-1 composite memory addresses."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.canonical2_registry import CANONICAL2_SPEC
from measurement.episode_registry import EPISODE_SPEC
from measurement.projector_registry import evaluation_name
from measurement.separation2_registry import SEPARATION2_SPEC


CONTEXT_SPEC = {
    "experiment": "context1_composite_memory_address",
    "preregistration_commit": "db66f1163",
    "source_experiment": SEPARATION2_SPEC["experiment"],
    "source_verdict": "SP3_CONTEXT_NOT_IN_KEY_STATE",
    "source_results": "measurement/separation2_results.json",
    "source_verdict_path": "measurement/separation2_verdict.json",
    "fit_method": CANONICAL2_SPEC["fit_method"],
    "calibration_engine_seeds": deepcopy(CANONICAL2_SPEC["calibration_seeds"]),
    "calibration_episodes": 1024,
    "calibration_data_seed": 8_300_000_000,
    "calibration_episode_seed_base": 8_350_000_000,
    "calibration_exact_marginal_balance": True,
    "evaluation_combinations": deepcopy(SEPARATION2_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(SEPARATION2_SPEC["evaluation_names"]),
    "eval_episodes": 2048,
    "events_per_episode": SEPARATION2_SPEC["events_per_episode"],
    "keys": SEPARATION2_SPEC["keys"],
    "values": SEPARATION2_SPEC["values"],
    "contexts": SEPARATION2_SPEC["contexts"],
    "distractor_steps": SEPARATION2_SPEC["distractor_steps"],
    "settling_updates": SEPARATION2_SPEC["settling_updates"],
    "pre_query_updates": SEPARATION2_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(SEPARATION2_SPEC["pre_query_dynamics_ablation"]),
    "state_dim": SEPARATION2_SPEC["state_dim"],
    "minimum_cells": SEPARATION2_SPEC["minimum_cells"],
    "maximum_cells": SEPARATION2_SPEC["maximum_cells"],
    "state_pooling": SEPARATION2_SPEC["state_pooling"],
    "component_address_dim": SEPARATION2_SPEC["address_dim"],
    "composite_address_dim": SEPARATION2_SPEC["address_dim"] * 2,
    "component_weight": 1.0,
    "model_class": SEPARATION2_SPEC["model_class"],
    "memory_class": SEPARATION2_SPEC["memory_class"],
    "bias": CANONICAL2_SPEC["bias"],
    "weight_decay": CANONICAL2_SPEC["weight_decay"],
    "temperature": CANONICAL2_SPEC["temperature"],
    "data_seed": 8_400_000_000,
    "episode_seed_base": 8_500_000_000,
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "states_per_episode": SEPARATION2_SPEC["events_per_episode"] + 1,
    "device": "cpu",
    "arms": [
        "composite_context_key_normal",
        "context_masked_control",
        "key_masked_control",
        "composite_distinct_key_control",
        "exact_context_key_control",
        "exact_context_key_partner_swap",
        "composite_context_key_recovered",
    ],
    "thresholds": {
        "context_classification_accuracy": 0.99,
        "context_minimum_recall": 0.95,
        "normal_selection_accuracy": 0.90,
        "normal_final_accuracy": 0.90,
        "normal_minimum_value_recall": 0.75,
        "content_readout_accuracy": 0.90,
        "distinct_selection_accuracy": 0.90,
        "distinct_final_accuracy": 0.90,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "context_masked_max_accuracy": 0.35,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
    },
}


def canonical_spec(spec: dict = CONTEXT_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTEXT_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
