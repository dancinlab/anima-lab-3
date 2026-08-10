"""Single source of truth for CANONICAL-2 default address integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from key_stability import DEFAULT_STABLE_KEY_FIT_METHOD
from measurement.canonical_registry import CANONICAL_SPEC


CANONICAL2_SPEC = {
    "experiment": "canonical2_integrated_default_address",
    "preregistration_commit": "3f02c9441",
    "source_experiment": CANONICAL_SPEC["experiment"],
    "source_verdict": "CN1_CANONICAL_ADDRESS_VALID",
    "source_results": "measurement/canonical_results.json",
    "source_verdict_path": "measurement/canonical_verdict.json",
    "fit_method": DEFAULT_STABLE_KEY_FIT_METHOD,
    "calibration_seeds": deepcopy(CANONICAL_SPEC["factor_seeds"]),
    "event_counts": [2, 3, 4],
    "evaluation_combinations": deepcopy(CANONICAL_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CANONICAL_SPEC[name])
        for name in (
            "calibration_episodes", "calibration_split", "input_dim", "address_dim",
            "keys", "values", "contexts", "model_class", "bias", "weight_decay",
            "temperature", "seed_stride", "eval_episodes", "settling_updates",
            "minimum_cells", "maximum_cells", "episode_seed_base", "event_count_seed_stride",
            "device", "arms", "thresholds",
        )
    },
    "minimum_event4_block_delta": 0.50,
}


def canonical_spec(spec: dict = CANONICAL2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CANONICAL2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
