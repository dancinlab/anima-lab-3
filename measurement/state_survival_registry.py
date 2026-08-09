"""Single source of truth for STATE-1 information-survival mapping."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.graft_behavior_registry import BEHAVIOR_SPEC


STATE_SURVIVAL_SPEC = {
    "experiment": "state_information_survival",
    "seeds": deepcopy(BEHAVIOR_SPEC["seeds"]),
    "situations": deepcopy(BEHAVIOR_SPEC["situations"]),
    "nuisance_words": deepcopy(BEHAVIOR_SPEC["nuisance_words"]),
    "cells": BEHAVIOR_SPEC["cells"],
    "engine_dim": BEHAVIOR_SPEC.get("engine_dim", BEHAVIOR_SPEC["state_dim"]),
    "warm_steps": BEHAVIOR_SPEC["warm_steps"],
    "sense_steps": BEHAVIOR_SPEC["sense_steps"],
    "delay_steps": [0, 1, 2, 4, 8, 16, 32],
    "train_examples_per_situation": 24,
    "eval_examples_per_situation": 16,
    "probe_ridge": 1.0,
    "bridge": {"hub_dim": 8, "output_dim": 96, "readout": "phase"},
    "channels": [
        "sense_input",
        "phase",
        "amplitude",
        "phase_velocity",
        "tension_frustration",
        "full_state",
        "temporal_phase",
        "bridge_cells",
        "bridge_pooled",
        "bridge_gate",
    ],
    "thresholds": {
        "signal_accuracy": 0.75,
        "positive_control_accuracy": 0.90,
        "shuffled_label_max_accuracy": 0.40,
    },
    "downstream_behavior": {
        "verdict_path": "measurement/graft_behavior_phase_state_repair_verdict.json",
        "required_experiment": "graft_behavior_causality_phase_state_memory_control_repair",
    },
}


def canonical_spec(spec: dict | None = None) -> str:
    return json.dumps(spec or STATE_SURVIVAL_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict | None = None) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
