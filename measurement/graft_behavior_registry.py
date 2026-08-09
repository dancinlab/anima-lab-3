"""Single source of truth for the GRAFT hidden-situation behavior experiment."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy


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

LANGUAGE_PRESERVED_SPEC = deepcopy(BEHAVIOR_SPEC)
LANGUAGE_PRESERVED_SPEC.update({
    "experiment": "graft_behavior_causality_language_preserved",
    "language_kl_weight": 1.0,
})

PHASE_STATE_SPEC = deepcopy(LANGUAGE_PRESERVED_SPEC)
PHASE_STATE_SPEC.update({
    "experiment": "graft_behavior_causality_phase_state",
    "readout": "phase",
    "state_dim": 2 * BEHAVIOR_SPEC["state_dim"],
})

REGISTERED_EXPERIMENTS = {
    BEHAVIOR_SPEC["experiment"]: BEHAVIOR_SPEC,
    LANGUAGE_PRESERVED_SPEC["experiment"]: LANGUAGE_PRESERVED_SPEC,
    PHASE_STATE_SPEC["experiment"]: PHASE_STATE_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown GRAFT behavior experiment: {name}") from exc


def canonical_spec(spec: dict | None = None) -> str:
    return json.dumps(spec or BEHAVIOR_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict | None = None) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
