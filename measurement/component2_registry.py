"""Single source of truth for COMPONENT-2 stable composite integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.component_registry import COMPONENT_SPEC
from measurement.conjunction2_registry import CONJUNCTION2_SPEC


COMPONENT2_SPEC = {
    "experiment": "component2_stable_composite_path",
    "preregistration_commit": "3a3ea619c",
    "source_experiment": COMPONENT_SPEC["experiment"],
    "source_verdict": "AC3_BOTH_COMPONENTS_LOSS",
    "source_results": "measurement/component_results.json",
    "source_verdict_path": "measurement/component_verdict.json",
    "checkpoint_path": "checkpoints/component2/stable_components.pt",
    "calibration_episodes": 512,
    "calibration_engine_seeds": [1337, 7331],
    "calibration_data_seed": 9_500_000_000,
    "calibration_seed_base": 9_600_000_000,
    "engine_seeds": deepcopy(COMPONENT_SPEC["engine_seeds"]),
    "diagnostic_episodes": COMPONENT_SPEC["eval_episodes"],
    "evaluation_combinations": deepcopy(CONJUNCTION2_SPEC["evaluation_combinations"]),
    "eval_episodes": CONJUNCTION2_SPEC["eval_episodes"],
    "positions": deepcopy(COMPONENT_SPEC["positions"]),
    "events_per_episode": CONJUNCTION2_SPEC["events_per_episode"],
    "active_contexts_per_episode": CONJUNCTION2_SPEC["active_contexts_per_episode"],
    "active_keys_per_episode": CONJUNCTION2_SPEC["active_keys_per_episode"],
    "active_values_per_episode": CONJUNCTION2_SPEC["active_values_per_episode"],
    "contexts": CONJUNCTION2_SPEC["contexts"], "keys": CONJUNCTION2_SPEC["keys"],
    "values": CONJUNCTION2_SPEC["values"], "state_dim": CONJUNCTION2_SPEC["state_dim"],
    "input_dim": CONJUNCTION2_SPEC["state_dim"], "address_dim": 32,
    "minimum_cells": CONJUNCTION2_SPEC["minimum_cells"],
    "maximum_cells": CONJUNCTION2_SPEC["maximum_cells"],
    "fit_method": "canonical_ridge", "model_class": "key_stability.StableKeyProjector",
    "bias": True, "weight_decay": 0.0001, "temperature": 0.1,
    "seed_stride": CONJUNCTION2_SPEC["seed_stride"], "device": "cpu",
    "thresholds": {
        "classification_accuracy": 0.90, "minimum_class_recall": 0.75,
        "selection_accuracy": 0.90, "final_accuracy": 0.90,
        "minimum_value_recall": 0.75, "old_selection_max_accuracy": 0.85,
        "minimum_causal_gain": 0.15,
    },
}


def canonical_spec(spec: dict = COMPONENT2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = COMPONENT2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
