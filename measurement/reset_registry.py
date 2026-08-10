"""Single source of truth for RESET-1 recovery mechanism separation."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.episode_registry import EPISODE_SPEC
    from measurement.recovery_registry import RECOVERY_SPEC
except ModuleNotFoundError:
    from episode_registry import EPISODE_SPEC
    from recovery_registry import RECOVERY_SPEC


RESET_SPEC = {
    "experiment": "reset1_recovery_mechanism_separation",
    "preregistration_commit": "d6be8b58e",
    "source_experiment": RECOVERY_SPEC["experiment"],
    "source_verdict": "RC1_ORDERED_RECOVERY_REPRODUCED",
    "source_results": "measurement/recovery_results.json",
    "source_verdict_path": "measurement/recovery_verdict.json",
    "seeds": deepcopy(RECOVERY_SPEC["seeds"]),
    "replicates": deepcopy(RECOVERY_SPEC["replicates"]),
    "episodes_per_replicate": RECOVERY_SPEC["episodes_per_replicate"],
    "prepared_events": RECOVERY_SPEC["prepared_events"],
    "queryable_events": RECOVERY_SPEC["queryable_events"],
    "update_steps": [0, 2, 4, 8],
    "update_modes": ["varied_sensory", "repeated_sensory", "autonomous"],
    "repeated_neutral_context": 6,
    "repeated_neutral_word": EPISODE_SPEC["distractor_words"][6],
    "require_distinct_varied_inputs": True,
    "keys": RECOVERY_SPEC["keys"],
    "values": RECOVERY_SPEC["values"],
    "contexts": RECOVERY_SPEC["contexts"],
    "state_dim": RECOVERY_SPEC["state_dim"],
    "minimum_cells": RECOVERY_SPEC["minimum_cells"],
    "maximum_cells": RECOVERY_SPEC["maximum_cells"],
    "state_pooling": RECOVERY_SPEC["state_pooling"],
    "address_dim": RECOVERY_SPEC["address_dim"],
    "model_class": RECOVERY_SPEC["model_class"],
    "memory_class": RECOVERY_SPEC["memory_class"],
    "data_seed_base": RECOVERY_SPEC["data_seed_base"],
    "distractor_seed_base": 7_900_000_000,
    "episode_seed_base": 8_000_000_000,
    "replicate_seed_stride": RECOVERY_SPEC["replicate_seed_stride"],
    "seed_stride": RECOVERY_SPEC["seed_stride"],
    "device": "cpu",
    "stable_arms": deepcopy(RECOVERY_SPEC["stable_arms"]),
    "arms": deepcopy(RECOVERY_SPEC["arms"]),
    "expected_transform_calls": deepcopy(RECOVERY_SPEC["expected_transform_calls"]),
    "thresholds": deepcopy(RECOVERY_SPEC["thresholds"]),
}


def canonical_spec(spec: dict = RESET_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RESET_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
