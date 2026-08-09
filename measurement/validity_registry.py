"""Single source of truth for VALIDITY-1 action-path decomposition."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

try:
    from measurement.relation_registry import RELATION_ROLE_REPAIR_SPEC
except ModuleNotFoundError:
    from relation_registry import RELATION_ROLE_REPAIR_SPEC


SOURCE = RELATION_ROLE_REPAIR_SPEC

VALIDITY_BASE_SPEC = {
    "experiment": "validity1_action_path_decomposition",
    "source_experiment": SOURCE["experiment"],
    "source_verdict": "R0_INVALID",
    "source_results": "measurement/relation_results.json",
    "source_results_sha256": "ac4988bb8253b03d77637228d3db6a223f4eeec80fbe81ea3b6ee17b55f77d17",
    "source_verdict_path": "measurement/relation_verdict.json",
    "source_verdict_sha256": "85fcdfeca7e1d3b81b44dbcb68cb95bd62a76ed0e860b24f5e312b9ced586e4c",
    "checkpoint_dir": "checkpoints/validity1_source/repair/checkpoints",
    "archive": {
        "repo_id": "dancinlab/anima-lab-research-archive",
        "revision": "relation1-role-binding",
        "prefix": "repair/checkpoints",
    },
    "checkpoint_sha256": {
        "1337": {
            "quantum_workspace_2": "b579ef5047a0564afc53307ee6696e322a793fb65fc53577b68c179cf9d429b8",
            "quantum_relation": "1d9f0faf7ef9ad61163f31fe1cbcd80e0df40f754e34c3b45b119046e734c2a0",
            "memory_relation": "4aa7ce30ebc975bc2faec3829fa097d7a21fcf4e5efcad3a81ee6b9dd95f6e21",
            "gru": "6a1237d3eba631cbac42b5a25ff4da330b2196faf32d5badde418b7ddca06303",
        },
        "7331": {
            "quantum_workspace_2": "b9ac017aa24751e2074e2fb1df15c6d1690a91d46ad32bf31ee4bc1bd048e92c",
            "quantum_relation": "2f592b5af633fdf49115a558efac7d53fb728e6fc27726e2d5505bbafd1fab95",
            "memory_relation": "70bc32d7346137e40470b13b7adb200f8e582d6b0fca9e5d9fa8a3015ac6ee87",
            "gru": "3046307f30f7e24c24bbd159a87c3737e3bfacdceb9c136da618dd6cf6dae2bb",
        },
    },
    "model": SOURCE["model"],
    "seeds": deepcopy(SOURCE["seeds"]),
    "arms": deepcopy(SOURCE["arms"]),
    "validation_arm": "gru",
    "reference_arm": "memory_relation",
    "actions": deepcopy(SOURCE["actions"]),
    "probe_ridge": 1.0,
    "label_control": {"method": "mean_random_permutation", "permutations": 32},
    "normalization_modes": ["train_style", "runtime_style"],
    "stages": ["sensory", "relation", "direct_action", "language"],
    "thresholds": {
        "sensory_accuracy": 0.90,
        "relation_accuracy": 0.80,
        "direct_action_accuracy": 0.80,
        "language_accuracy": 0.80,
        "shuffled_label_max_accuracy": 0.40,
    },
}


VALIDITY_REPLAY_REPAIR_SPEC = deepcopy(VALIDITY_BASE_SPEC)
VALIDITY_REPLAY_REPAIR_SPEC.update({
    "experiment": "validity1_action_path_decomposition_numerical_replay_repair",
    "invalid_results": "measurement/validity_invalid_results.json",
    "invalid_results_sha256": "e9a208ff48ce94da47f8de6d0afe5d3d9b3c239563ea307ff8e035202025adeb",
    "invalid_verdict": "measurement/validity_invalid_verdict.json",
    "invalid_verdict_sha256": "d50e384d58e0ae60d05c879be362db1ac6f801016f7571e8cd88abf1502b8bd4",
    "model_revision": "63a8b081895390a26e140280378bc85ec8bce07a",
    "language_replay": {
        "maximum_accuracy_delta": 0.01,
        "minimum_exact_arms": 6,
        "require_same_threshold_side": True,
    },
})

VALIDITY_SPEC = VALIDITY_REPLAY_REPAIR_SPEC

REGISTERED_EXPERIMENTS = {
    VALIDITY_BASE_SPEC["experiment"]: VALIDITY_BASE_SPEC,
    VALIDITY_REPLAY_REPAIR_SPEC["experiment"]: VALIDITY_REPLAY_REPAIR_SPEC,
}


def experiment(name: str) -> dict:
    try:
        return deepcopy(REGISTERED_EXPERIMENTS[name])
    except KeyError as exc:
        raise ValueError(f"unknown VALIDITY-1 experiment: {name}") from exc


def canonical_spec(spec: dict = VALIDITY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = VALIDITY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
