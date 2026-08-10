"""Single source of truth for TRAINING-1 address randomness crossing."""
from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy

from measurement.projector_registry import PROJECTOR_SPEC


_SEEDS = deepcopy(PROJECTOR_SPEC["factor_seeds"])


TRAINING_SPEC = {
    "experiment": "training1_address_randomness_factorial",
    "preregistration_commit": "aff5bee25",
    "source_experiment": PROJECTOR_SPEC["experiment"],
    "source_verdict": "PD2_TRAINING_RANDOMNESS_CAUSAL",
    "source_results": "measurement/projector_results.json",
    "source_verdict_path": "measurement/projector_verdict.json",
    "factor_seeds": _SEEDS,
    "calibration_seed": _SEEDS[0],
    "training_combinations": [
        {"initialization_seed": initialization, "batch_seed": batch}
        for initialization, batch in itertools.product(_SEEDS, repeat=2)
    ],
    "evaluation_combinations": deepcopy(PROJECTOR_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(PROJECTOR_SPEC[name])
        for name in (
            "calibration_episodes", "calibration_split", "input_dim", "address_dim",
            "keys", "values", "contexts", "model_class", "bias", "train_steps",
            "batch_size", "learning_rate", "weight_decay", "gradient_clip", "temperature",
            "calibration_seed_base", "seed_stride", "event_count", "eval_episodes",
            "settling_updates", "minimum_cells", "maximum_cells", "episode_seed_base",
            "event_count_seed_stride", "device", "arms", "thresholds",
        )
    },
}


def training_name(row: dict) -> str:
    return f"i{row['initialization_seed']}_b{row['batch_seed']}"


def canonical_spec(spec: dict = TRAINING_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = TRAINING_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
