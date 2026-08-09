"""Single source of truth for EPISODE-1 one-shot relation memory."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
except ModuleNotFoundError:
    from episode_control_registry import ATTENTION_CONTROL_SPEC


EPISODE_SPEC = {
    "experiment": "episode1_one_shot_relation_memory",
    "source_experiment": ATTENTION_CONTROL_SPEC["experiment"],
    "source_verdict": "A1_KEYED_ATTENTION_VALID",
    "source_results": "measurement/episode_control3_results.json",
    "source_verdict_path": "measurement/episode_control3_verdict.json",
    "source_checkpoint_dir": "checkpoints/episode_control3",
    "seeds": deepcopy(ATTENTION_CONTROL_SPEC["seeds"]),
    "splits": deepcopy(ATTENTION_CONTROL_SPEC["splits"]),
    "eval_split": "eval",
    "keys": ATTENTION_CONTROL_SPEC["keys"],
    "values": ATTENTION_CONTROL_SPEC["values"],
    "relations_per_episode": ATTENTION_CONTROL_SPEC["relations_per_episode"],
    "distractor_steps": ATTENTION_CONTROL_SPEC["distractor_steps"],
    "key_words": ["붉음", "푸름", "초록", "금빛", "은빛", "원형", "사각", "삼각"],
    "value_words": ["폭풍", "갈증", "피로", "평온", "위험", "메마름", "졸림", "안전"],
    "distractor_words": [
        "아침", "저녁", "실내", "실외", "가까이", "멀리", "조용히", "빠르게",
    ],
    "cells": 48,
    "engine_dim": 48,
    "state_dim": 96,
    "warm_steps": 5,
    "sense_steps": 3,
    "distractor_sense_steps": 1,
    "prototype_repeats_per_value": 16,
    "prototype_seed_base": 3_000_000_000,
    "episode_seed_base": 4_000_000_000,
    "seed_stride": 10_000_000,
    "device": "cpu",
    "memory_class": "trinity.VectorMemory",
    "readout": "phase_cell_mean_cosine",
    "arms": [
        "quantum_memory_normal",
        "quantum_memory_partner_swap",
        "quantum_memory_recovered",
        "sensory_memory_normal",
        "sensory_memory_partner_swap",
        "keyed_attention",
        "no_memory",
    ],
    "thresholds": {
        "quantum_accuracy": 0.75,
        "quantum_selection_accuracy": 0.75,
        "quantum_min_value_recall": 0.60,
        "content_readout_accuracy": 0.90,
        "positive_control_accuracy": 0.90,
        "negative_control_max_accuracy": 0.25,
        "maximum_balance_delta": 0,
        "recovery_prediction_match": 1.0,
    },
}


def canonical_spec(spec: dict = EPISODE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = EPISODE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
