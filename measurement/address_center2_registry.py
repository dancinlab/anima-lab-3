"""Single source of truth for ADDRESS-CENTER-2."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC


ADDRESS_CENTER2_SPEC = {
    "experiment": "address_center2_integrated_context_center",
    "preregistration_commit": "09c3ea80e",
    "source_experiment": ADDRESS_MARGIN_SPEC["experiment"],
    "source_verdict": "AM1_WITHIN_CLASS_MARGIN_LOSS",
    "source_results": "measurement/address_margin_results.json",
    "source_verdict_path": "measurement/address_margin_verdict.json",
    "evaluation_combinations": deepcopy(ADDRESS_MARGIN_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(ADDRESS_MARGIN_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
        )
    },
    "components_per_key": 2,
    "stores_per_episode": ADDRESS_MARGIN_SPEC["events_per_episode"],
    "retrievals_per_episode": 1,
    "transform_calls_per_episode": ADDRESS_MARGIN_SPEC["events_per_episode"] + 1,
    "arms": [
        "integrated_context_center", "integrated_center_disabled",
        "integrated_context_masked", "integrated_context_center_recovered",
        "exact_context_key_control", "exact_context_key_partner_swap",
    ],
    "thresholds": {
        "center_selection_accuracy": 0.90,
        "center_final_accuracy": 0.90,
        "minimum_value_recall": 0.75,
        "disabled_selection_max_accuracy": 0.85,
        "minimum_selection_gain": 0.08,
        "context_masked_max_accuracy": 0.35,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = ADDRESS_CENTER2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = ADDRESS_CENTER2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
