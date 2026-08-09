"""Single source of truth for META-1 self-monitoring calibration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy


METACOGNITION_SPEC = {
    "experiment": "meta1_bridge32_self_monitoring",
    "action_experiment": "graft_behavior_causality_phase_state_bridge32_memory_control_repair",
    "action_spec_sha256": "a4217c7ddd9f7af46a30da012bf7acf198e4471073c21cd703c3caf39f39807c",
    "archive": {
        "repo": "dancinlab/anima-lab-research-archive",
        "revision": "graft-bridge32-behavior",
        "checkpoint_sha256": {
            "1337": {
                "consciousness": "994dcd5e62a8b3d65ad097ab476e875ea5e7dbd00c590eadb491e22f4bfd9ff1",
                "memory": "8c2446bf2ae6961473692a00e10842d927a5e698297006881a1610fba5298bf6",
            },
            "7331": {
                "consciousness": "bf8ef98ec71ea79bac9d1d1dcb311b1ed0167fe9790edb11aedc40db2aa1799b",
                "memory": "047b9ca6361c885cd0ca9d38852d36511d61adfbc0fef0e76520386b05349a0a",
            },
        },
    },
    "seeds": [1337, 7331],
    "arms": ["consciousness", "memory"],
    # Circular radians for QuantumC phase; relative RMS for direct memory.
    "readout_noise_levels": [0.0, 0.25, 0.5, 1.0, 2.0],
    "reader": {
        "hidden_dim": 32,
        "train_steps": 500,
        "batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "ece_bins": 10,
        "selective_fraction": 0.5,
    },
    "interventions": ["normal", "off", "shuffle", "noise", "recovered"],
    "shuffle_constraint": "different_predicted_action_when_available",
    "thresholds": {
        "clean_action_accuracy": 0.90,
        "hard_action_accuracy_max": 0.50,
        "minimum_correct_examples": 32,
        "minimum_incorrect_examples": 32,
        "auroc": 0.75,
        "brier": 0.20,
        "ece": 0.15,
        "selective_accuracy_gap": 0.20,
        "shuffle_auroc_drop": 0.10,
        "shuffle_brier_increase": 0.05,
        "output_auroc_advantage": 0.05,
        "output_brier_advantage": 0.02,
    },
}

METACOGNITION_MEMORY_NOISE_REPAIR_SPEC = deepcopy(METACOGNITION_SPEC)
METACOGNITION_MEMORY_NOISE_REPAIR_SPEC.update({
    "experiment": "meta1_bridge32_self_monitoring_memory_noise_repair",
    # Direct memory stores one global vector replicated across cells. Perturb that
    # vector once and replicate the perturbation; independent per-cell noise is
    # cancelled by the bridge's registered mean pooling.
    "memory_noise_topology": "global_broadcast",
})

REGISTERED_EXPERIMENTS = {
    METACOGNITION_SPEC["experiment"]: METACOGNITION_SPEC,
    METACOGNITION_MEMORY_NOISE_REPAIR_SPEC["experiment"]:
        METACOGNITION_MEMORY_NOISE_REPAIR_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown META-1 experiment: {name}") from exc


def canonical_spec(spec: dict | None = None) -> str:
    return json.dumps(spec or METACOGNITION_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict | None = None) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
