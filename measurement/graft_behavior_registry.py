"""Single source of truth for the GRAFT hidden-situation behavior experiment."""
from __future__ import annotations

import hashlib
import json


BEHAVIOR_SPEC = {
    "experiment": "graft_behavior_causality",
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "seeds": [1337, 7331],
    "situations": [
        {"id": "danger", "words": ["폭풍", "위험"], "action": "A"},
        {"id": "thirst", "words": ["갈증", "메마름"], "action": "B"},
        {"id": "fatigue", "words": ["피로", "졸림"], "action": "C"},
        {"id": "safe", "words": ["평온", "안전"], "action": "D"},
    ],
    "nuisance_words": ["아침", "저녁", "실내", "실외", "가까이", "멀리", "조용히", "빠르게"],
    "question": (
        "A hidden situation was sensed. Choose exactly one action: "
        "A seek shelter; B drink water; C rest; D continue safely. Answer:"
    ),
    "neutral_prompts": [
        "The capital of France is", "Two plus two equals", "Water freezes at",
        "A short greeting is",
    ],
    "interventions": ["normal", "off", "shuffle", "noise", "recovered"],
    "cells": 48,
    "state_dim": 48,
    "warm_steps": 5,
    "sense_steps": 3,
    "delay_steps": [1, 4],
    "train_examples_per_situation": 24,
    "eval_examples_per_situation": 16,
    "train_steps": 400,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "gate_strength": 0.1,
    "gate_rms_max": 4.0,
    "gate_rho": 1.0,
    "thresholds": {
        "normal_accuracy": 0.75,
        "positive_control_accuracy": 0.80,
        "minimum_off_drop": 0.25,
        "content_control_max_accuracy": 0.40,
        "recovery_accuracy_tolerance": 0.0,
        "neutral_kl_nats": 0.50,
        "memory_equivalence_margin": 0.05,
    },
}


def canonical_spec() -> str:
    return json.dumps(BEHAVIOR_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256() -> str:
    return hashlib.sha256(canonical_spec().encode()).hexdigest()
