"""Single source of truth for KEY-REFRESH-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.query_refresh2_registry import QUERY_REFRESH2_SPEC


KEY_REFRESH_SPEC = {
    "experiment": "key_refresh1_query_key_refresh",
    "preregistration_commit": "804c5f4ac",
    "source_experiment": QUERY_REFRESH2_SPEC["experiment"],
    "source_verdict": "QRI2_CONTEXT_PATH_RECOVERED",
    "source_results": "measurement/query_refresh2_results.json",
    "source_verdict_path": "measurement/query_refresh2_verdict.json",
    "evaluation_combinations": deepcopy(QUERY_REFRESH2_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(QUERY_REFRESH2_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
            "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode", "conditions", "arms",
        )
    },
    "missing_fraction": 0.25,
    "query_context_sense_steps": 8,
    "query_key_steps": [3, 4, 6, 8, 12],
    "baseline_query_key_steps": 3,
    "thresholds": {
        "full_selection_accuracy": 0.90,
        "full_final_accuracy": 0.90,
        "full_minimum_value_recall": 0.75,
        "damaged_selection_accuracy": 0.90,
        "damaged_final_accuracy": 0.90,
        "minimum_key_final_gain": 0.01,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = KEY_REFRESH_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = KEY_REFRESH_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
