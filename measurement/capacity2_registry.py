"""Single source of truth for CAPACITY-2 post-settling capacity boundary."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from quantum_engine_fast import CANONICAL_DYNAMICS_COMPONENTS

try:
    from measurement.capacity_registry import CAPACITY_SPEC
except ModuleNotFoundError:
    from capacity_registry import CAPACITY_SPEC


CAPACITY2_SPEC = {
    "experiment": "capacity2_settled_address_boundary",
    "preregistration_commit": "0f05e9066",
    "source_experiment": CAPACITY_SPEC["experiment"],
    "source_verdict": "C3_CAPACITY_BOUNDARY_2",
    "source_results": "measurement/capacity_results.json",
    "source_verdict_path": "measurement/capacity_verdict.json",
    "seeds": deepcopy(CAPACITY_SPEC["seeds"]),
    "event_counts": deepcopy(CAPACITY_SPEC["event_counts"]),
    "eval_episodes_per_count": CAPACITY_SPEC["eval_episodes_per_count"],
    "balance_mode": CAPACITY_SPEC["balance_mode"],
    "keys": CAPACITY_SPEC["keys"],
    "values": CAPACITY_SPEC["values"],
    "contexts": CAPACITY_SPEC["contexts"],
    "distractor_steps": CAPACITY_SPEC["distractor_steps"],
    "state_dim": CAPACITY_SPEC["state_dim"],
    "minimum_cells": CAPACITY_SPEC["minimum_cells"],
    "maximum_cells": CAPACITY_SPEC["maximum_cells"],
    "state_pooling": CAPACITY_SPEC["state_pooling"],
    "address_dim": CAPACITY_SPEC["address_dim"],
    "model_class": CAPACITY_SPEC["model_class"],
    "memory_class": CAPACITY_SPEC["memory_class"],
    "data_seed_base": CAPACITY_SPEC["data_seed_base"],
    "episode_seed_base": CAPACITY_SPEC["episode_seed_base"],
    "event_count_seed_stride": CAPACITY_SPEC["event_count_seed_stride"],
    "seed_stride": CAPACITY_SPEC["seed_stride"],
    "device": "cpu",
    "settling_updates": 8,
    "mechanism_component": "frustration_regulation",
    "canonical_dynamics_components": list(CANONICAL_DYNAMICS_COMPONENTS),
    "conditions": [
        {"name": "baseline", "updates": 0, "disabled": []},
        {"name": "settled", "updates": 8, "disabled": []},
        {
            "name": "without_frustration_regulation",
            "updates": 8,
            "disabled": ["frustration_regulation"],
        },
    ],
    "arms": deepcopy(CAPACITY_SPEC["arms"]),
    "thresholds": {
        **deepcopy(CAPACITY_SPEC["thresholds"]),
        "minimum_mechanism_accuracy_drop": 0.05,
        "paired_exact_p_maximum": 0.01,
        "source_metric_tolerance": 1e-7,
    },
}


def canonical_spec(spec: dict = CAPACITY2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CAPACITY2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
