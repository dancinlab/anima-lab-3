"""Single source of truth for CUE-MECHANISM-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.completion_registry import COMPLETION_SPEC


CUE_MECHANISM_SPEC = {
    "experiment": "cue_mechanism1_partial_cue_decomposition",
    "preregistration_commit": "29eea3a6a",
    "source_experiment": COMPLETION_SPEC["experiment"],
    "source_verdict": "CP2_FRAGILE_CUE_PATH",
    "source_results": "measurement/completion_results.json",
    "source_verdict_path": "measurement/completion_verdict.json",
    "evaluation_combinations": deepcopy(COMPLETION_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(COMPLETION_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
            "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode", "mask_salt", "mask_components",
        )
    },
    "missing_fraction": 0.25,
    "conditions": {
        "full_cue": [0.0, 0.0],
        "context_quarter_missing": [0.25, 0.0],
        "key_quarter_missing": [0.0, 0.25],
        "both_quarter_missing": [0.25, 0.25],
    },
    "arms": [
        "full_cue", "context_quarter_missing", "key_quarter_missing",
        "both_quarter_missing", "exact_context_key_control",
        "exact_context_key_partner_swap",
    ],
    "thresholds": {
        "full_selection_accuracy": 0.90,
        "full_final_accuracy": 0.90,
        "full_minimum_value_recall": 0.75,
        "component_category_accuracy": 0.90,
        "component_minimum_category_recall": 0.75,
        "single_quarter_selection_accuracy": 0.90,
        "single_quarter_final_accuracy": 0.90,
        "both_quarter_selection_accuracy": 0.90,
        "both_quarter_final_accuracy": 0.90,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = CUE_MECHANISM_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CUE_MECHANISM_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
