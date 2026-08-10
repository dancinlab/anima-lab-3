"""Single source of truth for CONTEXT-2 common composite-memory integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.context_registry import CONTEXT_SPEC


CONTEXT2_SPEC = {
    "experiment": "context2_integrated_composite_memory_path",
    "preregistration_commit": "eddc01a4d",
    "source_experiment": CONTEXT_SPEC["experiment"],
    "source_verdict": "CX1_CONTEXT_KEY_VALID_NOT_UNIQUE",
    "source_results": "measurement/context_results.json",
    "source_verdict_path": "measurement/context_verdict.json",
    "evaluation_combinations": deepcopy(CONTEXT_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(CONTEXT_SPEC["evaluation_names"]),
    "eval_episodes": CONTEXT_SPEC["eval_episodes"],
    "events_per_episode": CONTEXT_SPEC["events_per_episode"],
    "keys": CONTEXT_SPEC["keys"],
    "values": CONTEXT_SPEC["values"],
    "contexts": CONTEXT_SPEC["contexts"],
    "distractor_steps": CONTEXT_SPEC["distractor_steps"],
    "settling_updates": CONTEXT_SPEC["settling_updates"],
    "pre_query_updates": CONTEXT_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(CONTEXT_SPEC["pre_query_dynamics_ablation"]),
    "state_dim": CONTEXT_SPEC["state_dim"],
    "minimum_cells": CONTEXT_SPEC["minimum_cells"],
    "maximum_cells": CONTEXT_SPEC["maximum_cells"],
    "state_pooling": CONTEXT_SPEC["state_pooling"],
    "component_address_dim": CONTEXT_SPEC["component_address_dim"],
    "composite_address_dim": CONTEXT_SPEC["composite_address_dim"],
    "component_weight": CONTEXT_SPEC["component_weight"],
    "model_class": CONTEXT_SPEC["model_class"],
    "memory_class": CONTEXT_SPEC["memory_class"],
    "fit_method": CONTEXT_SPEC["fit_method"],
    "data_seed": CONTEXT_SPEC["data_seed"],
    "episode_seed_base": CONTEXT_SPEC["episode_seed_base"],
    "seed_stride": CONTEXT_SPEC["seed_stride"],
    "device": "cpu",
    "components_per_key": 2,
    "stores_per_episode": CONTEXT_SPEC["events_per_episode"],
    "retrievals_per_episode": 1,
    "transform_calls_per_episode": CONTEXT_SPEC["events_per_episode"] + 1,
    "arms": [
        "integrated_composite_normal",
        "external_composite_reference",
        "integrated_context_masked",
        "exact_context_key_control",
        "exact_context_key_partner_swap",
        "integrated_composite_recovered",
    ],
    "thresholds": {
        "normal_selection_accuracy": 0.90,
        "normal_final_accuracy": 0.90,
        "normal_minimum_value_recall": 0.75,
        "content_readout_accuracy": 0.90,
        "reference_prediction_match": 1.0,
        "reference_selection_match": 1.0,
        "context_masked_max_accuracy": 0.35,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
    },
}


def canonical_spec(spec: dict = CONTEXT2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTEXT2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
