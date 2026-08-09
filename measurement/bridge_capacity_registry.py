"""Single source of truth for the STATE-2 ThalamicBridge width sweep."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.state_survival_registry import STATE_SURVIVAL_SPEC
except ModuleNotFoundError:
    from state_survival_registry import STATE_SURVIVAL_SPEC


BRIDGE_CAPACITY_SPEC = {
    "experiment": "thalamic_bridge_capacity",
    "source_experiment": STATE_SURVIVAL_SPEC["experiment"],
    "source_verdict": "S4_BRIDGE_TRANSFORM_LOSS",
    "seeds": deepcopy(STATE_SURVIVAL_SPEC["seeds"]),
    "situations": deepcopy(STATE_SURVIVAL_SPEC["situations"]),
    "nuisance_words": deepcopy(STATE_SURVIVAL_SPEC["nuisance_words"]),
    "cells": STATE_SURVIVAL_SPEC["cells"],
    "engine_dim": STATE_SURVIVAL_SPEC["engine_dim"],
    "warm_steps": STATE_SURVIVAL_SPEC["warm_steps"],
    "sense_steps": STATE_SURVIVAL_SPEC["sense_steps"],
    "delay_steps": deepcopy(STATE_SURVIVAL_SPEC["delay_steps"]),
    "train_examples_per_situation": STATE_SURVIVAL_SPEC["train_examples_per_situation"],
    "eval_examples_per_situation": STATE_SURVIVAL_SPEC["eval_examples_per_situation"],
    "probe_ridge": STATE_SURVIVAL_SPEC["probe_ridge"],
    "label_control": deepcopy(STATE_SURVIVAL_SPEC["label_control"]),
    "bridge": {
        "baseline_hub_dim": STATE_SURVIVAL_SPEC["bridge"]["hub_dim"],
        "hub_dims": [8, 16, 32, 48, 96],
        "output_dim": STATE_SURVIVAL_SPEC["bridge"]["output_dim"],
        "readout": STATE_SURVIVAL_SPEC["bridge"]["readout"],
        "pooling": "mean",
    },
    "channels": [
        "sense_input",
        "phase",
        "full_state",
        "bridge_cells",
        "bridge_pooled",
        "bridge_gate",
    ],
    "thresholds": deepcopy(STATE_SURVIVAL_SPEC["thresholds"]),
    "selection": {
        "positive_controls": ["sense_input", "phase", "full_state"],
        "ordered_stages": ["bridge_cells", "bridge_pooled", "bridge_gate"],
        "rule": "minimum_hub_dim_passing_both_seeds_at_every_delay",
        "deploy_only_if_full_path_passes": True,
    },
}


def canonical_spec(spec: dict | None = None) -> str:
    return json.dumps(
        spec or BRIDGE_CAPACITY_SPEC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def spec_sha256(spec: dict | None = None) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
