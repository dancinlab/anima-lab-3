"""Single source of truth for MECHANISM-1 autonomous settling ablations."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from quantum_engine_fast import CANONICAL_DYNAMICS_COMPONENTS

try:
    from measurement.settle_registry import SETTLE_SPEC
except ModuleNotFoundError:
    from settle_registry import SETTLE_SPEC


_INTERVENTIONS = [
    {"name": "intact", "mode": "autonomous", "disabled": []},
    {"name": "frozen", "mode": "frozen", "disabled": []},
    *[
        {
            "name": f"without_{component}",
            "mode": "autonomous",
            "disabled": [component],
        }
        for component in CANONICAL_DYNAMICS_COMPONENTS
    ],
]


MECHANISM_SPEC = {
    "experiment": "mechanism1_autonomous_settling_components",
    "preregistration_commit": "76a7141d1",
    "source_experiment": SETTLE_SPEC["experiment"],
    "source_verdict": "ST1_AUTONOMOUS_SETTLING_CAUSAL",
    "source_results": "measurement/settle_results.json",
    "source_verdict_path": "measurement/settle_verdict.json",
    "seeds": deepcopy(SETTLE_SPEC["seeds"]),
    "replicates": deepcopy(SETTLE_SPEC["replicates"]),
    "episodes_per_replicate": SETTLE_SPEC["episodes_per_replicate"],
    "prepared_events": SETTLE_SPEC["prepared_events"],
    "queryable_events": SETTLE_SPEC["queryable_events"],
    "update_steps": [8],
    "update_modes": ["autonomous", "frozen"],
    "preserve_query_rng": True,
    "components": list(CANONICAL_DYNAMICS_COMPONENTS),
    "interventions": _INTERVENTIONS,
    "keys": SETTLE_SPEC["keys"],
    "values": SETTLE_SPEC["values"],
    "contexts": SETTLE_SPEC["contexts"],
    "state_dim": SETTLE_SPEC["state_dim"],
    "minimum_cells": SETTLE_SPEC["minimum_cells"],
    "maximum_cells": SETTLE_SPEC["maximum_cells"],
    "state_pooling": SETTLE_SPEC["state_pooling"],
    "address_dim": SETTLE_SPEC["address_dim"],
    "model_class": SETTLE_SPEC["model_class"],
    "memory_class": SETTLE_SPEC["memory_class"],
    "data_seed_base": 8_400_000_000,
    "distractor_seed_base": 8_500_000_000,
    "episode_seed_base": 8_600_000_000,
    "replicate_seed_stride": SETTLE_SPEC["replicate_seed_stride"],
    "seed_stride": SETTLE_SPEC["seed_stride"],
    "distractor_steps": list(range(9)),
    "device": "cpu",
    "stable_arms": deepcopy(SETTLE_SPEC["stable_arms"]),
    "arms": deepcopy(SETTLE_SPEC["arms"]),
    "expected_transform_calls": deepcopy(SETTLE_SPEC["expected_transform_calls"]),
    "thresholds": {
        **deepcopy(SETTLE_SPEC["thresholds"]),
        "minimum_accuracy_drop": 0.05,
        "paired_exact_p_maximum": 0.01,
        "minimum_worsening_replicates": 5,
    },
}


def canonical_spec(spec: dict = MECHANISM_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = MECHANISM_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
