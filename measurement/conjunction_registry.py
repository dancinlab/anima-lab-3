"""Single source of truth for CONJUNCTION-1 context+key pair retrieval."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.context2_registry import CONTEXT2_SPEC
from measurement.projector_registry import evaluation_name


CONJUNCTION_SPEC = {
    "experiment": "conjunction1_context_key_conjunction",
    "preregistration_commit": "5129d96d8",
    "source_experiment": CONTEXT2_SPEC["experiment"],
    "source_verdict": "CX2I_PATH_RECOVERED_NOT_UNIQUE",
    "source_results": "measurement/context2_results.json",
    "source_verdict_path": "measurement/context2_verdict.json",
    "evaluation_combinations": deepcopy(CONTEXT2_SPEC["evaluation_combinations"]),
    "evaluation_names": [
        evaluation_name(row) for row in CONTEXT2_SPEC["evaluation_combinations"]
    ],
    "eval_episodes": 1024,
    "active_contexts_per_episode": 4,
    "active_keys_per_episode": 4,
    "active_values_per_episode": 4,
    "events_per_episode": 16,
    "keys": CONTEXT2_SPEC["keys"],
    "values": CONTEXT2_SPEC["values"],
    "contexts": CONTEXT2_SPEC["contexts"],
    "distractor_steps": CONTEXT2_SPEC["distractor_steps"],
    "settling_updates": CONTEXT2_SPEC["settling_updates"],
    "pre_query_updates": CONTEXT2_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(
        CONTEXT2_SPEC["pre_query_dynamics_ablation"]
    ),
    "state_dim": CONTEXT2_SPEC["state_dim"],
    "minimum_cells": CONTEXT2_SPEC["minimum_cells"],
    "maximum_cells": CONTEXT2_SPEC["maximum_cells"],
    "state_pooling": CONTEXT2_SPEC["state_pooling"],
    "component_address_dim": CONTEXT2_SPEC["component_address_dim"],
    "composite_address_dim": CONTEXT2_SPEC["composite_address_dim"],
    "component_weight": CONTEXT2_SPEC["component_weight"],
    "model_class": CONTEXT2_SPEC["model_class"],
    "memory_class": CONTEXT2_SPEC["memory_class"],
    "fit_method": CONTEXT2_SPEC["fit_method"],
    "temperature": CONTEXT2_SPEC["temperature"],
    "bias": CONTEXT2_SPEC["bias"],
    "data_seed": 8_600_000_000,
    "episode_seed_base": 8_700_000_000,
    "seed_stride": CONTEXT2_SPEC["seed_stride"],
    "device": "cpu",
    "components_per_key": 2,
    "stores_per_episode": 16,
    "retrievals_per_episode": 1,
    "transform_calls_per_episode": 17,
    "arms": [
        "integrated_conjunction_normal",
        "external_conjunction_reference",
        "integrated_context_masked",
        "integrated_key_masked",
        "exact_context_key_control",
        "exact_context_only_control",
        "exact_key_only_control",
        "exact_context_key_partner_swap",
        "integrated_conjunction_recovered",
    ],
    "thresholds": {
        "normal_selection_accuracy": 0.90,
        "normal_final_accuracy": 0.90,
        "normal_minimum_value_recall": 0.75,
        "content_readout_accuracy": 0.90,
        "reference_prediction_match": 1.0,
        "reference_selection_match": 1.0,
        "component_masked_max_accuracy": 0.35,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "exact_component_only_max_accuracy": 0.35,
        "partner_swap_max_accuracy": 0.05,
        "recovery_prediction_match": 1.0,
        "retrieval_api_match": 1.0,
    },
}


def canonical_spec(spec: dict = CONJUNCTION_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONJUNCTION_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
