from copy import deepcopy
import json
from pathlib import Path

from measurement.workspace_gate import adjudicate
from measurement.workspace_registry import (
    WORKSPACE_CONTROL_SEED_REPAIR_SPEC,
    WORKSPACE_SPEC,
    spec_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def arm(normal=0.90, a_only=0.25, b_only=0.25, shuffled=0.25, kl=0.1):
    return {
        "conditions": {
            "normal": {"accuracy": normal},
            "module_a_only": {"accuracy": a_only},
            "module_b_only": {"accuracy": b_only},
            "partner_shuffle": {"accuracy": shuffled},
            "recovered": {"accuracy": normal, "logits_identical": True},
        },
        "neutral_kl_nats": kl,
    }


def payload(q_rounds=None, memory=0.90, gru=0.90, spec=WORKSPACE_SPEC):
    q_rounds = q_rounds or {1: 0.90, 2: 0.90, 4: 0.90}
    map_results = json.loads((ROOT / spec["source_map_results"]).read_text())
    map_verdict = json.loads((ROOT / spec["source_map_verdict"]).read_text())
    arms = {
        "quantum_single_pass": arm(normal=0.30),
        "gru": arm(normal=gru),
    }
    for rounds in spec["workspace_rounds"]:
        arms[f"quantum_workspace_{rounds}"] = arm(normal=q_rounds[rounds])
        arms[f"memory_workspace_{rounds}"] = arm(normal=memory)
    return {
        "experiment": spec["experiment"],
        "spec": deepcopy(spec),
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": {
            split: {
                "examples": 16 * spec[f"{split}_repeats_per_pair"],
                "pair_count": 16,
                "examples_per_pair": spec[f"{split}_repeats_per_pair"],
                "target_counts": {str(i): 4 * spec[f"{split}_repeats_per_pair"] for i in range(4)},
                "module_a_target_counts": {
                    str(cue): {str(i): spec[f"{split}_repeats_per_pair"] for i in range(4)}
                    for cue in range(4)
                },
                "module_b_target_counts": {
                    str(cue): {str(i): spec[f"{split}_repeats_per_pair"] for i in range(4)}
                    for cue in range(4)
                },
            }
            for split in ("train", "eval")
        },
        "source_map": {"results": map_results, "verdict": map_verdict},
        "seeds": [
            {
                "seed": seed,
                "arms": deepcopy(arms),
                "checkpoints": {
                    name: {"path": f"seed_{seed}_{name}.pt", "sha256": f"{index + 1:x}" * 64}
                    for index, name in enumerate(spec["arms"])
                },
            }
            for seed in spec["seeds"]
        ],
    }


def test_workspace_gate_selects_minimum_rounds_and_general_equivalence():
    value = payload(q_rounds={1: 0.60, 2: 0.90, 4: 0.95})
    verdict = adjudicate(value)
    assert verdict["verdict"] == "W1_INTEGRATED_NOT_UNIQUE"
    assert verdict["selected_workspace_rounds"] == 2


def test_workspace_gate_distinguishes_failure_confound_and_invalid_control():
    failed = payload(q_rounds={1: 0.60, 2: 0.65, 4: 0.70})
    assert adjudicate(failed)["verdict"] == "W3_NOT_INTEGRATED"
    confounded = payload()
    confounded["seeds"][0]["arms"]["quantum_workspace_1"]["neutral_kl_nats"] = 0.8
    assert adjudicate(confounded)["verdict"] == "W4_CONFOUNDED"
    weak_control = payload(gru=0.50)
    assert adjudicate(weak_control)["verdict"] == "W0_INVALID"


def test_workspace_gate_fails_closed_on_map_checkpoint_and_spec_drift():
    changed_map = payload()
    changed_map["source_map"]["verdict"]["verdict"] = "I4_RELATION_COMPUTATION_LOSS"
    assert adjudicate(changed_map)["verdict"] == "W0_INVALID"
    bad_hash = payload()
    bad_hash["seeds"][0]["checkpoints"]["gru"]["sha256"] = "bad"
    assert adjudicate(bad_hash)["verdict"] == "W0_INVALID"
    drift = payload()
    drift["spec"]["workspace_rounds"] = [1, 3, 4]
    assert adjudicate(drift)["verdict"] == "W0_INVALID"


def test_workspace_gate_accepts_registered_control_seed_repair():
    value = payload(spec=WORKSPACE_CONTROL_SEED_REPAIR_SPEC)
    assert adjudicate(value)["verdict"] == "W1_INTEGRATED_NOT_UNIQUE"
