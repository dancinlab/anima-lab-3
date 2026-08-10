"""Single source of truth for SEPARATION-2 canonical similar-episode retrieval."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.canonical2_registry import CANONICAL2_SPEC
from measurement.episode_registry import EPISODE_SPEC
from measurement.projector_registry import evaluation_name
from measurement.separation_registry import SEPARATION_SPEC


SEPARATION2_SPEC = {
    "experiment": "separation2_canonical_similar_episode",
    "preregistration_commit": "805623f16",
    "source_experiment": CANONICAL2_SPEC["experiment"],
    "source_verdict": "CI1_CANONICAL_DEFAULT_INTEGRATED",
    "source_results": "measurement/canonical2_results.json",
    "source_verdict_path": "measurement/canonical2_verdict.json",
    "fit_method": CANONICAL2_SPEC["fit_method"],
    "evaluation_combinations": deepcopy(CANONICAL2_SPEC["evaluation_combinations"]),
    "evaluation_names": [
        evaluation_name(row) for row in CANONICAL2_SPEC["evaluation_combinations"]
    ],
    "eval_episodes": 2048,
    "events_per_episode": 4,
    "keys": CANONICAL2_SPEC["keys"],
    "values": CANONICAL2_SPEC["values"],
    "contexts": CANONICAL2_SPEC["contexts"],
    "distractor_steps": SEPARATION_SPEC["distractor_steps"],
    "settling_updates": CANONICAL2_SPEC["settling_updates"],
    "pre_query_updates": CANONICAL2_SPEC["settling_updates"],
    "pre_query_dynamics_ablation": [],
    "state_dim": CANONICAL2_SPEC["input_dim"],
    "minimum_cells": CANONICAL2_SPEC["minimum_cells"],
    "maximum_cells": CANONICAL2_SPEC["maximum_cells"],
    "state_pooling": SEPARATION_SPEC["state_pooling"],
    "address_dim": CANONICAL2_SPEC["address_dim"],
    "model_class": CANONICAL2_SPEC["model_class"],
    "memory_class": SEPARATION_SPEC["memory_class"],
    "data_seed": 8_100_000_000,
    "episode_seed_base": 8_200_000_000,
    "seed_stride": EPISODE_SPEC["seed_stride"],
    "expected_stable_transform_calls_per_episode": 5,
    "device": "cpu",
    "arms": deepcopy(SEPARATION_SPEC["arms"]),
    "thresholds": deepcopy(SEPARATION_SPEC["thresholds"]),
}


def canonical_spec(spec: dict = SEPARATION2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SEPARATION2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
