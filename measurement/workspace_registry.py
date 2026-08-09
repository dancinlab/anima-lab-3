"""Single source of truth for WORKSPACE-1 split-cue survival and recurrence."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.synergy_registry import SYNERGY_CONTROL_REPAIR_SPEC
except ModuleNotFoundError:
    from synergy_registry import SYNERGY_CONTROL_REPAIR_SPEC


SOURCE = SYNERGY_CONTROL_REPAIR_SPEC

WORKSPACE_INFORMATION_SPEC = {
    "experiment": "workspace1_split_cue_information_map",
    "source_experiment": SOURCE["experiment"],
    "source_verdict": "Y3_NOT_INTEGRATED",
    "source_results": "measurement/synergy_results.json",
    "source_verdict_path": "measurement/synergy_verdict.json",
    "checkpoint_dir": "checkpoints/synergy1_control_role_repair",
    "seeds": deepcopy(SOURCE["seeds"]),
    "module_a_cues": deepcopy(SOURCE["module_a_cues"]),
    "module_b_cues": deepcopy(SOURCE["module_b_cues"]),
    "nuisance_words": deepcopy(SOURCE["nuisance_words"]),
    "cells_per_module": SOURCE["cells_per_module"],
    "engine_dim": SOURCE["engine_dim"],
    "state_dim": SOURCE["state_dim"],
    "bridge_hub_dim": SOURCE["bridge_hub_dim"],
    "warm_steps": SOURCE["warm_steps"],
    "sense_steps": SOURCE["sense_steps"],
    "delay_steps": deepcopy(SOURCE["delay_steps"]),
    "train_repeats_per_pair": SOURCE["train_repeats_per_pair"],
    "eval_repeats_per_pair": SOURCE["eval_repeats_per_pair"],
    "probe_ridge": 1.0,
    "label_control": {"method": "mean_random_permutation", "permutations": 32},
    "channels": [
        "raw_pair",
        "bridge_cells",
        "bridge_pooled",
        "bridge_gate",
        "normalized_code",
    ],
    "labels": ["module_a", "module_b"],
    "thresholds": {
        "signal_accuracy": 0.75,
        "positive_control_accuracy": 0.90,
        "shuffled_label_max_accuracy": 0.40,
    },
}


WORKSPACE_SPEC = {
    "experiment": "workspace1_recurrent_split_cue_integration",
    "source_map_experiment": WORKSPACE_INFORMATION_SPEC["experiment"],
    "source_map_verdicts": [
        "I1_LOCAL_TRANSFORM_LOSS",
        "I2_POOLING_LOSS",
        "I3_GATE_LOSS",
        "I4_RELATION_COMPUTATION_LOSS",
    ],
    "source_map_results": "measurement/workspace_information_results.json",
    "source_map_verdict": "measurement/workspace_information_verdict.json",
    "model": SOURCE["model"],
    "seeds": deepcopy(SOURCE["seeds"]),
    "module_a_cues": deepcopy(SOURCE["module_a_cues"]),
    "module_b_cues": deepcopy(SOURCE["module_b_cues"]),
    "nuisance_words": deepcopy(SOURCE["nuisance_words"]),
    "target_rule": SOURCE["target_rule"],
    "actions": deepcopy(SOURCE["actions"]),
    "question": SOURCE["question"],
    "neutral_prompts": deepcopy(SOURCE["neutral_prompts"]),
    "cells_per_module": SOURCE["cells_per_module"],
    "engine_dim": SOURCE["engine_dim"],
    "state_dim": SOURCE["state_dim"],
    "bridge_hub_dim": SOURCE["bridge_hub_dim"],
    "workspace_rounds": [1, 2, 4],
    "workspace_rule": (
        "shared ThalamicBridge local transform, GRUCell workspace update, "
        "broadcast to both module summaries after every round"
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
    "arms": [
        "quantum_single_pass",
        "quantum_workspace_1",
        "quantum_workspace_2",
        "quantum_workspace_4",
        "memory_workspace_1",
        "memory_workspace_2",
        "memory_workspace_4",
        "gru",
    ],
    "validation_arms": ["gru"],
    "comparison_arm_prefix": "memory_workspace_",
    "interventions": deepcopy(SOURCE["interventions"]),
    "pair_shuffle_constraint": SOURCE["pair_shuffle_constraint"],
    "thresholds": deepcopy(SOURCE["thresholds"]),
    "selection": {
        "rule": "minimum_quantum_workspace_rounds_passing_both_seeds",
        "require_single_pass_failure": True,
    },
}


REGISTERED_EXPERIMENTS = {
    WORKSPACE_INFORMATION_SPEC["experiment"]: WORKSPACE_INFORMATION_SPEC,
    WORKSPACE_SPEC["experiment"]: WORKSPACE_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown WORKSPACE-1 experiment: {name}") from exc


def canonical_spec(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
