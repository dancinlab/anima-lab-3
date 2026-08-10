"""Single source of truth for SEEDMAP-1 capacity seed-factor crossing."""
from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy

try:
    from measurement.capacity2_registry import CAPACITY2_SPEC
except ModuleNotFoundError:
    from capacity2_registry import CAPACITY2_SPEC


_SEEDS = deepcopy(CAPACITY2_SPEC["seeds"])


SEEDMAP_SPEC = {
    "experiment": "seedmap1_capacity_factorial",
    "preregistration_commit": "3e828054a",
    "source_experiment": CAPACITY2_SPEC["experiment"],
    "source_verdict": "CP4_SEED_CONDITIONAL_CAPACITY",
    "source_results": "measurement/capacity2_results.json",
    "source_verdict_path": "measurement/capacity2_verdict.json",
    "factor_seeds": _SEEDS,
    "event_count": 4,
    "eval_episodes": CAPACITY2_SPEC["eval_episodes_per_count"],
    "settling_updates": CAPACITY2_SPEC["settling_updates"],
    "keys": CAPACITY2_SPEC["keys"],
    "values": CAPACITY2_SPEC["values"],
    "contexts": CAPACITY2_SPEC["contexts"],
    "state_dim": CAPACITY2_SPEC["state_dim"],
    "minimum_cells": CAPACITY2_SPEC["minimum_cells"],
    "maximum_cells": CAPACITY2_SPEC["maximum_cells"],
    "address_dim": CAPACITY2_SPEC["address_dim"],
    "model_class": CAPACITY2_SPEC["model_class"],
    "memory_class": CAPACITY2_SPEC["memory_class"],
    "episode_seed_base": CAPACITY2_SPEC["episode_seed_base"],
    "event_count_seed_stride": CAPACITY2_SPEC["event_count_seed_stride"],
    "seed_stride": CAPACITY2_SPEC["seed_stride"],
    "device": "cpu",
    "factors": ["projector_seed", "prototype_seed", "engine_seed"],
    "combinations": [
        {
            "projector_seed": projector,
            "prototype_seed": prototype,
            "engine_seed": engine,
        }
        for projector, prototype, engine in itertools.product(_SEEDS, repeat=3)
    ],
    "arms": deepcopy(CAPACITY2_SPEC["arms"]),
    "thresholds": {
        **deepcopy(CAPACITY2_SPEC["thresholds"]),
        "minimum_factor_delta": 0.02,
        "paired_exact_p_maximum": 0.01,
        "source_metric_tolerance": 1e-7,
    },
}


def combination_name(row: dict) -> str:
    return f"p{row['projector_seed']}_v{row['prototype_seed']}_e{row['engine_seed']}"


def canonical_spec(spec: dict = SEEDMAP_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SEEDMAP_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
