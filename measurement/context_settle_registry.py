"""Single source of truth for CONTEXT-SETTLE-1 transition settling."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.component2_registry import COMPONENT2_SPEC
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.episode_registry import EPISODE_SPEC


CONTEXT_SETTLE_SPEC = {
    "experiment": "context_settle1_transition_settling",
    "preregistration_commit": "aa3330d6e",
    "source_experiment": COMPONENT2_SPEC["experiment"],
    "source_verdict": "CS2_COMPONENT_FIT_INVALID",
    "source_results": "measurement/component2_results.json",
    "source_verdict_path": "measurement/component2_verdict.json",
    "source_checkpoint_path": COMPONENT2_SPEC["checkpoint_path"],
    "engine_seeds": [1337, 7331],
    "eval_episodes": 512,
    "data_seed": 9_700_000_000,
    "episode_seed_base": 9_800_000_000,
    "seed_stride": CONJUNCTION2_SPEC["seed_stride"],
    "context_steps": [3, 4, 6, 9],
    "baseline_steps": EPISODE_SPEC["sense_steps"],
    "transition_positions": [8, 12],
    "positions": list(range(CONJUNCTION2_SPEC["events_per_episode"])),
    "events_per_episode": CONJUNCTION2_SPEC["events_per_episode"],
    "active_contexts_per_episode": CONJUNCTION2_SPEC["active_contexts_per_episode"],
    "active_keys_per_episode": CONJUNCTION2_SPEC["active_keys_per_episode"],
    "active_values_per_episode": CONJUNCTION2_SPEC["active_values_per_episode"],
    "contexts": CONJUNCTION2_SPEC["contexts"],
    "keys": CONJUNCTION2_SPEC["keys"],
    "values": CONJUNCTION2_SPEC["values"],
    "key_steps": EPISODE_SPEC["sense_steps"],
    "value_steps": EPISODE_SPEC["sense_steps"],
    "state_dim": CONJUNCTION2_SPEC["state_dim"],
    "address_dim": COMPONENT2_SPEC["address_dim"],
    "minimum_cells": CONJUNCTION2_SPEC["minimum_cells"],
    "maximum_cells": CONJUNCTION2_SPEC["maximum_cells"],
    "fit_method": COMPONENT2_SPEC["fit_method"],
    "model_class": COMPONENT2_SPEC["model_class"],
    "temperature": COMPONENT2_SPEC["temperature"],
    "bias": COMPONENT2_SPEC["bias"],
    "device": "cpu",
    "source_overlap_sets": [
        "component2_calibration", "component1_evaluation", "conjunction2_evaluation",
    ],
    "thresholds": {
        "classification_accuracy": 0.90,
        "minimum_class_recall": 0.75,
    },
}


def canonical_spec(spec: dict = CONTEXT_SETTLE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTEXT_SETTLE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
