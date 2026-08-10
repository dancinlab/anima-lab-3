"""Single source of truth for COMPONENT-1 address serial stability."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.conjunction2_registry import CONJUNCTION2_SPEC


COMPONENT_SPEC = {
    "experiment": "component1_address_serial_stability",
    "preregistration_commit": "23078054b",
    "source_experiment": CONJUNCTION2_SPEC["experiment"],
    "source_verdict": "CJ2_2_COMPONENT_COLLISION",
    "source_results": "measurement/conjunction2_results.json",
    "source_verdict_path": "measurement/conjunction2_verdict.json",
    "engine_seeds": [1337, 7331],
    "eval_episodes": 512,
    "positions": list(range(16)),
    **{
        name: deepcopy(CONJUNCTION2_SPEC[name])
        for name in (
            "active_contexts_per_episode", "active_keys_per_episode",
            "active_values_per_episode", "events_per_episode", "keys", "values",
            "contexts", "distractor_steps", "settling_updates", "pre_query_updates",
            "pre_query_dynamics_ablation", "state_dim", "minimum_cells", "maximum_cells",
            "component_address_dim", "model_class", "fit_method", "temperature", "bias",
            "seed_stride", "device",
        )
    },
    "data_seed": 9_300_000_000,
    "episode_seed_base": 9_400_000_000,
    "thresholds": {
        "classification_accuracy": 0.90,
        "minimum_class_recall": 0.75,
    },
}


def canonical_spec(spec: dict = COMPONENT_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = COMPONENT_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
