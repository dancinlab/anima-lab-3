"""Single source of truth for CONJUNCTION-2 stable-value pair retrieval."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.conjunction_registry import CONJUNCTION_SPEC
from measurement.value2_registry import VALUE2_SPEC


CONJUNCTION2_SPEC = {
    "experiment": "conjunction2_stable_value_conjunction",
    "preregistration_commit": "22891223d",
    "source_value_experiment": VALUE2_SPEC["experiment"],
    "source_value_verdict": "VT1_STABLE_VALUE_PATH_VALID_NOT_UNIQUE",
    "source_value_results": "measurement/value2_results.json",
    "source_value_verdict_path": "measurement/value2_verdict.json",
    "source_conjunction_experiment": CONJUNCTION_SPEC["experiment"],
    "source_conjunction_verdict": "CJ0_INVALID",
    "source_conjunction_results": "measurement/conjunction_results.json",
    "source_conjunction_verdict_path": "measurement/conjunction_verdict.json",
    "evaluation_combinations": deepcopy(CONJUNCTION_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(CONJUNCTION_SPEC["evaluation_names"]),
    **{
        name: deepcopy(CONJUNCTION_SPEC[name])
        for name in (
            "eval_episodes", "active_contexts_per_episode", "active_keys_per_episode",
            "active_values_per_episode", "events_per_episode", "keys", "values",
            "contexts", "distractor_steps", "settling_updates", "pre_query_updates",
            "pre_query_dynamics_ablation", "state_dim", "minimum_cells", "maximum_cells",
            "state_pooling", "component_address_dim", "composite_address_dim",
            "component_weight", "model_class", "memory_class", "fit_method",
            "temperature", "bias", "data_seed", "episode_seed_base", "seed_stride",
            "device", "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode",
        )
    },
    "value_address_dim": VALUE2_SPEC["address_dim"],
    "value_transform_calls_per_episode": CONJUNCTION_SPEC["events_per_episode"],
    "arms": [
        "integrated_stable_conjunction_normal",
        "external_stable_conjunction_reference",
        "integrated_stable_context_masked",
        "integrated_stable_key_masked",
        "exact_stable_context_key_control",
        "exact_stable_context_only_control",
        "exact_stable_key_only_control",
        "exact_stable_partner_swap",
        "integrated_stable_conjunction_recovered",
        "integrated_raw_value_control",
    ],
    "thresholds": deepcopy(CONJUNCTION_SPEC["thresholds"]),
}


def canonical_spec(spec: dict = CONJUNCTION2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONJUNCTION2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
