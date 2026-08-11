"""Single source of truth for QUERY-REFRESH-2."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.cue_robust_registry import CUE_ROBUST_SPEC


QUERY_REFRESH2_SPEC = {
    "experiment": "query_refresh2_integrated_query_refresh",
    "preregistration_commit": "cde0a15a4",
    "source_refresh_experiment": "query_refresh1_query_state_refresh",
    "source_refresh_verdict": "QR1_REFRESH_RECOVERS_AND_CONVERGES",
    "source_refresh_results": "measurement/query_refresh_results.json",
    "source_refresh_verdict_path": "measurement/query_refresh_verdict.json",
    "source_robust_experiment": CUE_ROBUST_SPEC["experiment"],
    "source_robust_verdict": "CR2_CONTEXT_CATEGORY_NOT_RECOVERED",
    "source_robust_results": "measurement/cue_robust_results.json",
    "source_robust_verdict_path": "measurement/cue_robust_verdict.json",
    "evaluation_combinations": deepcopy(CUE_ROBUST_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_ROBUST_SPEC[name])
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
    "runtime_conditions": {
        "baseline_6": 6,
        "refreshed_8": 8,
        "disabled_6": 6,
        "recovered_8": 8,
    },
    "baseline_condition": "baseline_6",
    "refreshed_condition": "refreshed_8",
    "disabled_condition": "disabled_6",
    "recovered_condition": "recovered_8",
    "thresholds": {
        "full_selection_accuracy": 0.90,
        "full_final_accuracy": 0.90,
        "full_minimum_value_recall": 0.75,
        "damaged_selection_accuracy": 0.90,
        "damaged_final_accuracy": 0.90,
        "minimum_context_final_gain": 0.05,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = QUERY_REFRESH2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = QUERY_REFRESH2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
