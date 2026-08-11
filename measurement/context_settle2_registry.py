"""Single source of truth for CONTEXT-SETTLE-2 integrated transition path."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC
from measurement.episode_registry import EPISODE_SPEC


CONTEXT_SETTLE2_SPEC = {
    "experiment": "context_settle2_integrated_transition_path",
    "preregistration_commit": "e5023e810",
    "source_settle_experiment": CONTEXT_SETTLE_SPEC["experiment"],
    "source_settle_verdict": "CT1_MINIMUM_SETTLING_FOUND",
    "source_settle_results": "measurement/context_settle_results.json",
    "source_settle_verdict_path": "measurement/context_settle_verdict.json",
    "source_component_experiment": "component2_stable_composite_path",
    "source_component_verdict": "CS2_COMPONENT_FIT_INVALID",
    "source_component_results": "measurement/component2_results.json",
    "source_component_verdict_path": "measurement/component2_verdict.json",
    "evaluation_combinations": deepcopy(CONJUNCTION2_SPEC["evaluation_combinations"]),
    "eval_episodes": CONJUNCTION2_SPEC["eval_episodes"],
    "events_per_episode": CONJUNCTION2_SPEC["events_per_episode"],
    "distractor_steps": CONJUNCTION2_SPEC["distractor_steps"],
    "contexts": CONJUNCTION2_SPEC["contexts"],
    "keys": CONJUNCTION2_SPEC["keys"],
    "values": CONJUNCTION2_SPEC["values"],
    "data_seed": CONJUNCTION2_SPEC["data_seed"],
    "episode_seed_base": CONJUNCTION2_SPEC["episode_seed_base"],
    "seed_stride": CONJUNCTION2_SPEC["seed_stride"],
    "baseline_context_steps": EPISODE_SPEC["sense_steps"],
    "settled_context_steps": 6,
    "key_sense_steps": EPISODE_SPEC["sense_steps"],
    "value_sense_steps": EPISODE_SPEC["sense_steps"],
    "distractor_sense_steps": EPISODE_SPEC["distractor_sense_steps"],
    "conditions": ["baseline_3", "settled_6"],
    "components_per_key": CONJUNCTION2_SPEC["components_per_key"],
    "stores_per_episode": CONJUNCTION2_SPEC["stores_per_episode"],
    "retrievals_per_episode": CONJUNCTION2_SPEC["retrievals_per_episode"],
    "transform_calls_per_episode": CONJUNCTION2_SPEC["transform_calls_per_episode"],
    "value_transform_calls_per_episode": CONJUNCTION2_SPEC["value_transform_calls_per_episode"],
    "state_dim": CONJUNCTION2_SPEC["state_dim"],
    "component_address_dim": CONJUNCTION2_SPEC["component_address_dim"],
    "composite_address_dim": CONJUNCTION2_SPEC["composite_address_dim"],
    "value_address_dim": CONJUNCTION2_SPEC["value_address_dim"],
    "minimum_cells": CONJUNCTION2_SPEC["minimum_cells"],
    "maximum_cells": CONJUNCTION2_SPEC["maximum_cells"],
    "component_weight": CONJUNCTION2_SPEC["component_weight"],
    "temperature": CONJUNCTION2_SPEC["temperature"],
    "bias": CONJUNCTION2_SPEC["bias"],
    "arms": deepcopy(CONJUNCTION2_SPEC["arms"]),
    "device": "cpu",
    "thresholds": {
        "baseline_selection_max_accuracy": 0.85,
        "selection_accuracy": 0.90,
        "final_accuracy": 0.90,
        "minimum_value_recall": 0.75,
        "minimum_causal_gain": 0.15,
        "component_masked_max_accuracy": 0.35,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = CONTEXT_SETTLE2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTEXT_SETTLE2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
