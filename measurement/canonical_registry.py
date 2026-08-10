"""Single source of truth for CANONICAL-1 deterministic stable addresses."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.training_registry import TRAINING_SPEC


CANONICAL_SPEC = {
    "experiment": "canonical1_deterministic_address",
    "preregistration_commit": "ad8009478",
    "source_experiment": TRAINING_SPEC["experiment"],
    "source_verdict": "TR1_INITIALIZATION_CAUSAL",
    "source_results": "measurement/training_results.json",
    "source_verdict_path": "measurement/training_verdict.json",
    "calibration_arms": [
        {"name": "calibration_1337", "calibration_seeds": [1337]},
        {"name": "calibration_7331", "calibration_seeds": [7331]},
        {"name": "pooled", "calibration_seeds": [1337, 7331]},
    ],
    "evaluation_combinations": deepcopy(TRAINING_SPEC["evaluation_combinations"]),
    "method": "ridge_fixed_orthogonal_targets",
    "order_tolerance": 1e-6,
    "repeat_tolerance": 0.0,
    **{
        name: deepcopy(TRAINING_SPEC[name])
        for name in (
            "factor_seeds", "calibration_episodes", "calibration_split", "input_dim",
            "address_dim", "keys", "values", "contexts", "model_class", "bias",
            "weight_decay", "temperature", "seed_stride", "event_count", "eval_episodes",
            "settling_updates", "minimum_cells", "maximum_cells", "episode_seed_base",
            "event_count_seed_stride", "device", "arms", "thresholds",
        )
    },
}


def canonical_spec(spec: dict = CANONICAL_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CANONICAL_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
