"""Single source of truth for PROJECTOR-1 address-training factor crossing."""
from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy

from measurement.key_registry import KEY_SPEC
from measurement.seedmap_registry import SEEDMAP_SPEC


_SEEDS = deepcopy(SEEDMAP_SPEC["factor_seeds"])


PROJECTOR_SPEC = {
    "experiment": "projector1_address_training_factorial",
    "preregistration_commit": "e0aa03beb",
    "source_experiment": SEEDMAP_SPEC["experiment"],
    "source_verdict": "SM1_SINGLE_FACTOR_CAUSAL",
    "source_results": "measurement/seedmap_results.json",
    "source_verdict_path": "measurement/seedmap_verdict.json",
    "source_key_experiment": KEY_SPEC["experiment"],
    "source_key_verdict": "K1_STABLE_KEY_VALID_NOT_UNIQUE",
    "source_key_results": "measurement/key_results.json",
    "source_key_verdict_path": "measurement/key_verdict.json",
    "factor_seeds": _SEEDS,
    "training_combinations": [
        {"calibration_seed": calibration, "training_seed": training}
        for calibration, training in itertools.product(_SEEDS, repeat=2)
    ],
    "evaluation_combinations": [
        {"prototype_seed": prototype, "engine_seed": engine}
        for prototype, engine in itertools.product(_SEEDS, repeat=2)
    ],
    "calibration_episodes": KEY_SPEC["calibration_episodes"],
    "calibration_split": KEY_SPEC["calibration_split"],
    "input_dim": KEY_SPEC["input_dim"],
    "address_dim": KEY_SPEC["address_dim"],
    "keys": KEY_SPEC["keys"],
    "values": SEEDMAP_SPEC["values"],
    "contexts": SEEDMAP_SPEC["contexts"],
    "model_class": KEY_SPEC["model_class"],
    "bias": KEY_SPEC["bias"],
    "train_steps": KEY_SPEC["train_steps"],
    "batch_size": KEY_SPEC["batch_size"],
    "learning_rate": KEY_SPEC["learning_rate"],
    "weight_decay": KEY_SPEC["weight_decay"],
    "gradient_clip": KEY_SPEC["gradient_clip"],
    "temperature": KEY_SPEC["temperature"],
    "calibration_seed_base": KEY_SPEC["calibration_seed_base"],
    "seed_stride": KEY_SPEC["seed_stride"],
    "event_count": SEEDMAP_SPEC["event_count"],
    "eval_episodes": SEEDMAP_SPEC["eval_episodes"],
    "settling_updates": SEEDMAP_SPEC["settling_updates"],
    "minimum_cells": SEEDMAP_SPEC["minimum_cells"],
    "maximum_cells": SEEDMAP_SPEC["maximum_cells"],
    "episode_seed_base": SEEDMAP_SPEC["episode_seed_base"],
    "event_count_seed_stride": SEEDMAP_SPEC["event_count_seed_stride"],
    "device": "cpu",
    "arms": deepcopy(SEEDMAP_SPEC["arms"]),
    "thresholds": deepcopy(SEEDMAP_SPEC["thresholds"]),
}


def projector_name(row: dict) -> str:
    return f"c{row['calibration_seed']}_t{row['training_seed']}"


def evaluation_name(row: dict) -> str:
    return f"v{row['prototype_seed']}_e{row['engine_seed']}"


def canonical_spec(spec: dict = PROJECTOR_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = PROJECTOR_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
