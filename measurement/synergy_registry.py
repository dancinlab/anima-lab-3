"""Single source of truth for SYNERGY-1 split-cue integration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.bridge_config import THALAMIC_BRIDGE_HUB_DIM
except ModuleNotFoundError:
    from bridge_config import THALAMIC_BRIDGE_HUB_DIM


SYNERGY_SPEC = {
    "experiment": "synergy1_split_cue_modular_sum",
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "seeds": [1337, 7331],
    "arms": ["quantum_pair", "direct_memory", "gru"],
    "module_a_cues": ["붉음", "푸름", "초록", "금빛"],
    "module_b_cues": ["원형", "사각", "삼각", "별빛"],
    "nuisance_words": ["아침", "저녁", "실내", "실외", "가까이", "멀리", "조용히", "빠르게"],
    "target_rule": "(module_a_index + module_b_index) modulo 4",
    "actions": ["A", "B", "C", "D"],
    "question": (
        "Two hidden modules each sensed one clue. Combine both clues and choose exactly one "
        "registered action A, B, C, or D. Answer:"
    ),
    "neutral_prompts": [
        "The capital of France is", "Two plus two equals", "Water freezes at",
        "A short greeting is",
    ],
    "cells_per_module": 48,
    "engine_dim": 48,
    "state_dim": 96,
    "bridge_hub_dim": THALAMIC_BRIDGE_HUB_DIM,
    "warm_steps": 5,
    "sense_steps": 3,
    "delay_steps": [1, 4],
    "train_repeats_per_pair": 6,
    "eval_repeats_per_pair": 4,
    "train_steps": 800,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "gate_strength": 0.1,
    "gate_rms_max": 4.0,
    "gate_rho": 1.0,
    "language_kl_weight": 1.0,
    "gru_hidden_dim": 96,
    "interventions": [
        "normal", "module_a_only", "module_b_only", "partner_shuffle", "recovered"
    ],
    "pair_shuffle_constraint": "different_module_b_and_different_target",
    "thresholds": {
        "joint_accuracy": 0.75,
        "positive_control_accuracy": 0.80,
        "single_module_max_accuracy": 0.40,
        "partner_shuffle_max_accuracy": 0.40,
        "minimum_joint_gain": 0.35,
        "recovery_accuracy_tolerance": 0.0,
        "neutral_kl_nats": 0.50,
        "quantum_advantage_margin": 0.05,
    },
}


SYNERGY_CONTROL_REPAIR_SPEC = deepcopy(SYNERGY_SPEC)
SYNERGY_CONTROL_REPAIR_SPEC.update({
    "experiment": "synergy1_split_cue_modular_sum_control_role_repair",
    # A direct store preserves clues but is not guaranteed to compute their nonlinear
    # modular relation. The trained recurrent arm is the registered task control.
    "validation_arms": ["gru"],
    "comparison_arms": ["direct_memory", "gru"],
})


REGISTERED_EXPERIMENTS = {
    SYNERGY_SPEC["experiment"]: SYNERGY_SPEC,
    SYNERGY_CONTROL_REPAIR_SPEC["experiment"]: SYNERGY_CONTROL_REPAIR_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown SYNERGY-1 experiment: {name}") from exc


def canonical_spec(spec: dict | None = None) -> str:
    return json.dumps(
        spec or SYNERGY_SPEC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def spec_sha256(spec: dict | None = None) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
