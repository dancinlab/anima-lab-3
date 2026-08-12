"""Single source of truth for KEY-REFRESH-2."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.key_refresh_registry import KEY_REFRESH_SPEC


KEY_REFRESH2_SPEC = {
    "experiment": "key_refresh2_integrated_query_key_refresh",
    "preregistration_commit": "4b8e8c45f",
    "source_experiment": KEY_REFRESH_SPEC["experiment"],
    "source_verdict": "KRF1_KEY_PATH_RECOVERED_AND_SUSTAINED",
    "source_results": "measurement/key_refresh_results.json",
    "source_verdict_path": "measurement/key_refresh_verdict.json",
    "evaluation_combinations": deepcopy(KEY_REFRESH_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(KEY_REFRESH_SPEC[name])
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
    "missing_fraction": KEY_REFRESH_SPEC["missing_fraction"],
    "query_context_sense_steps": KEY_REFRESH_SPEC["query_context_sense_steps"],
    "baseline_query_key_sense_steps": KEY_REFRESH_SPEC["baseline_query_key_steps"],
    "query_key_sense_steps": 4,
    "runtime_conditions": {
        "baseline_3": 3,
        "integrated_4": 4,
        "disabled_3": 3,
        "recovered_4": 4,
    },
    "baseline_condition": "baseline_3",
    "integrated_condition": "integrated_4",
    "disabled_condition": "disabled_3",
    "recovered_condition": "recovered_4",
    "thresholds": deepcopy(KEY_REFRESH_SPEC["thresholds"]),
}


def canonical_spec(spec: dict = KEY_REFRESH2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = KEY_REFRESH2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
