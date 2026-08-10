"""Single source of truth for SETTLE-1 autonomous memory settling."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.reset_registry import RESET_SPEC
except ModuleNotFoundError:
    from reset_registry import RESET_SPEC


SETTLE_SPEC = {
    "experiment": "settle1_autonomous_memory_settling",
    "preregistration_commit": "0a1bffc6a",
    "source_experiment": RESET_SPEC["experiment"],
    "source_verdict": "RS5_MIXED_MECHANISM",
    "source_results": "measurement/reset_results.json",
    "source_verdict_path": "measurement/reset_verdict.json",
    "seeds": deepcopy(RESET_SPEC["seeds"]),
    "replicates": [0, 1, 2, 3, 4, 5],
    "episodes_per_replicate": 512,
    "prepared_events": RESET_SPEC["prepared_events"],
    "queryable_events": RESET_SPEC["queryable_events"],
    "update_steps": [0, 2, 4, 8],
    "update_modes": ["autonomous", "frozen"],
    "preserve_query_rng": True,
    "keys": RESET_SPEC["keys"],
    "values": RESET_SPEC["values"],
    "contexts": RESET_SPEC["contexts"],
    "state_dim": RESET_SPEC["state_dim"],
    "minimum_cells": RESET_SPEC["minimum_cells"],
    "maximum_cells": RESET_SPEC["maximum_cells"],
    "state_pooling": RESET_SPEC["state_pooling"],
    "address_dim": RESET_SPEC["address_dim"],
    "model_class": RESET_SPEC["model_class"],
    "memory_class": RESET_SPEC["memory_class"],
    "data_seed_base": 8_100_000_000,
    "distractor_seed_base": 8_200_000_000,
    "episode_seed_base": 8_300_000_000,
    "replicate_seed_stride": RESET_SPEC["replicate_seed_stride"],
    "seed_stride": RESET_SPEC["seed_stride"],
    "distractor_steps": list(range(9)),
    "device": "cpu",
    "stable_arms": deepcopy(RESET_SPEC["stable_arms"]),
    "arms": deepcopy(RESET_SPEC["arms"]),
    "expected_transform_calls": deepcopy(RESET_SPEC["expected_transform_calls"]),
    "thresholds": {
        **deepcopy(RESET_SPEC["thresholds"]),
        "paired_exact_p_maximum": 0.01,
        "minimum_improving_replicates": 5,
    },
}


def canonical_spec(spec: dict = SETTLE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SETTLE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
