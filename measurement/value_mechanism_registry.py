"""Single source of truth for VALUE-MECHANISM-1 serial-position test."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.value_registry import VALUE_SPEC


VALUE_MECHANISM_SPEC = {
    "experiment": "value_mechanism1_serial_position",
    "preregistration_commit": "61cf10232",
    "source_experiment": VALUE_SPEC["experiment"],
    "source_verdict": "VB1_READOUT_VALID_THROUGH_16",
    "source_results": "measurement/value_results.json",
    "source_verdict_path": "measurement/value_verdict.json",
    "evaluation_combinations": deepcopy(VALUE_SPEC["evaluation_combinations"]),
    "evaluation_names": deepcopy(VALUE_SPEC["evaluation_names"]),
    "eval_episodes": VALUE_SPEC["eval_episodes"],
    "query_positions": [0, 4, 8, 12, 15],
    "query_position_labels": [1, 5, 9, 13, 16],
    "events_per_episode": VALUE_SPEC["events_per_episode"],
    "active_contexts_per_episode": VALUE_SPEC["active_contexts_per_episode"],
    "active_keys_per_episode": VALUE_SPEC["active_keys_per_episode"],
    "active_values_per_episode": VALUE_SPEC["active_values_per_episode"],
    "keys": VALUE_SPEC["keys"],
    "values": VALUE_SPEC["values"],
    "contexts": VALUE_SPEC["contexts"],
    "distractor_steps": VALUE_SPEC["distractor_steps"],
    "settling_updates": VALUE_SPEC["settling_updates"],
    "pre_query_updates": VALUE_SPEC["pre_query_updates"],
    "pre_query_dynamics_ablation": deepcopy(VALUE_SPEC["pre_query_dynamics_ablation"]),
    "state_dim": VALUE_SPEC["state_dim"],
    "minimum_cells": VALUE_SPEC["minimum_cells"],
    "maximum_cells": VALUE_SPEC["maximum_cells"],
    "memory_class": VALUE_SPEC["memory_class"],
    "data_seed": VALUE_SPEC["data_seed"],
    "episode_seed_base": 9_000_000_000,
    "seed_stride": VALUE_SPEC["seed_stride"],
    "device": "cpu",
    "address_mode": VALUE_SPEC["address_mode"],
    "position_move": "swap_first_with_target",
    "stores_per_episode": VALUE_SPEC["events_per_episode"],
    "retrievals_per_episode": 1,
    "arms": deepcopy(VALUE_SPEC["arms"]),
    "thresholds": {
        **deepcopy(VALUE_SPEC["thresholds"]),
        "minimum_position_effect": 0.05,
    },
}


def canonical_spec(spec: dict = VALUE_MECHANISM_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = VALUE_MECHANISM_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
