"""Single source of truth for RELATION-1 role-content binding."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.workspace_registry import WORKSPACE_CONTROL_SEED_REPAIR_SPEC
except ModuleNotFoundError:
    from workspace_registry import WORKSPACE_CONTROL_SEED_REPAIR_SPEC


SOURCE = WORKSPACE_CONTROL_SEED_REPAIR_SPEC

RELATION_SPEC = {
    "experiment": "relation1_hippocampal_role_content_binding",
    "source_experiment": SOURCE["experiment"],
    "source_verdict": "W3_NOT_INTEGRATED",
    "source_results": "measurement/workspace_results.json",
    "source_verdict_path": "measurement/workspace_verdict.json",
    "model": SOURCE["model"],
    "seeds": deepcopy(SOURCE["seeds"]),
    "module_a_cues": ["붉음", "푸름", "초록", "금빛", "은빛"],
    "module_b_cues": ["원형", "사각", "삼각", "별빛", "물결"],
    "nuisance_words": deepcopy(SOURCE["nuisance_words"]),
    "target_rule": "(module_a_index + 2 * module_b_index) modulo 5",
    "target_table": [[(a + 2 * b) % 5 for b in range(5)] for a in range(5)],
    "actions": ["A", "B", "C", "D", "E"],
    "question": (
        "Two hidden modules sensed clues with different roles. Bind both role-clue pairs "
        "and choose exactly one registered action A, B, C, D, or E. Answer:"
    ),
    "neutral_prompts": deepcopy(SOURCE["neutral_prompts"]),
    "cells_per_module": SOURCE["cells_per_module"],
    "engine_dim": SOURCE["engine_dim"],
    "state_dim": SOURCE["state_dim"],
    "bridge_hub_dim": SOURCE["bridge_hub_dim"],
    "workspace_rounds": [2],
    "relation_rounds": 1,
    "relation_rule": (
        "shared bridge cell transform, separate bias-free role projections, "
        "elementwise role-content product, one width-32 relation projection"
    ),
    "warm_steps": SOURCE["warm_steps"],
    "sense_steps": SOURCE["sense_steps"],
    "delay_steps": deepcopy(SOURCE["delay_steps"]),
    "train_repeats_per_pair": SOURCE["train_repeats_per_pair"],
    "eval_repeats_per_pair": SOURCE["eval_repeats_per_pair"],
    "train_steps": SOURCE["train_steps"],
    "batch_size": SOURCE["batch_size"],
    "learning_rate": SOURCE["learning_rate"],
    "weight_decay": SOURCE["weight_decay"],
    "gate_strength": SOURCE["gate_strength"],
    "gate_rms_max": SOURCE["gate_rms_max"],
    "gate_rho": SOURCE["gate_rho"],
    "language_kl_weight": SOURCE["language_kl_weight"],
    "gru_hidden_dim": SOURCE["gru_hidden_dim"],
    "arms": ["quantum_workspace_2", "quantum_relation", "memory_relation", "gru"],
    "relation_arm": "quantum_relation",
    "baseline_arm": "quantum_workspace_2",
    "validation_arms": ["gru"],
    "comparison_arms": ["memory_relation", "gru"],
    "arm_seed_offsets": {
        "quantum_workspace_2": 0,
        "quantum_relation": 100_000,
        "memory_relation": 200_000,
        "gru": 300_000,
    },
    "interventions": [
        "normal", "module_a_only", "module_b_only", "partner_shuffle",
        "role_swap", "recovered",
    ],
    "pair_shuffle_constraint": "different_module_b_and_different_target",
    "thresholds": {
        "joint_accuracy": 0.75,
        "positive_control_accuracy": 0.80,
        "single_module_max_accuracy": 0.40,
        "partner_shuffle_max_accuracy": 0.40,
        "role_swap_max_accuracy": 0.40,
        "minimum_joint_gain": 0.35,
        "recovery_accuracy_tolerance": 0.0,
        "neutral_kl_nats": 0.50,
        "quantum_advantage_margin": 0.05,
    },
}


REGISTERED_EXPERIMENTS = {RELATION_SPEC["experiment"]: RELATION_SPEC}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown RELATION-1 experiment: {name}") from exc


def canonical_spec(spec: dict = RELATION_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RELATION_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
