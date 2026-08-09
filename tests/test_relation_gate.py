from copy import deepcopy
import json
from pathlib import Path

from measurement.relation_gate import adjudicate
from measurement.relation_registry import RELATION_SPEC, spec_sha256
from measurement.synergy_gate import _expected_audit


ROOT = Path(__file__).resolve().parents[1]


def arm(normal=0.90, single=0.20, shuffled=0.20, swapped=0.20, kl=0.1):
    return {
        "conditions": {
            "normal": {"accuracy": normal},
            "module_a_only": {"accuracy": single},
            "module_b_only": {"accuracy": single},
            "partner_shuffle": {"accuracy": shuffled},
            "role_swap": {"accuracy": swapped},
            "recovered": {"accuracy": normal, "logits_identical": True},
        },
        "neutral_kl_nats": kl,
    }


def payload(relation=0.90, baseline=0.30, memory=0.90, gru=0.90):
    spec = RELATION_SPEC
    arms = {
        "quantum_workspace_2": arm(normal=baseline),
        "quantum_relation": arm(normal=relation),
        "memory_relation": arm(normal=memory),
        "gru": arm(normal=gru),
    }
    return {
        "experiment": spec["experiment"],
        "spec": deepcopy(spec),
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": {
            split: _expected_audit(spec, split) for split in ("train", "eval")
        },
        "source": {
            "results": json.loads((ROOT / spec["source_results"]).read_text()),
            "verdict": json.loads((ROOT / spec["source_verdict_path"]).read_text()),
        },
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


def test_relation_gate_accepts_bound_not_unique_and_quantum_advantage():
    assert adjudicate(payload())["verdict"] == "R1_BOUND_NOT_UNIQUE"
    assert adjudicate(payload(relation=0.96, memory=0.85, gru=0.85))["verdict"] == "R2_QUANTUM_ADVANTAGE"


def test_relation_gate_distinguishes_failure_confound_and_existing_sufficiency():
    assert adjudicate(payload(relation=0.60))["verdict"] == "R3_NOT_BOUND"
    confounded = payload()
    confounded["seeds"][0]["arms"]["quantum_relation"]["neutral_kl_nats"] = 0.8
    assert adjudicate(confounded)["verdict"] == "R4_CONFOUNDED"
    assert adjudicate(payload(baseline=0.90))["verdict"] == "R5_EXISTING_WORKSPACE_SUFFICIENT"


def test_relation_gate_fails_closed_on_role_control_source_and_receipt_drift():
    weak_control = payload()
    weak_control["seeds"][0]["arms"]["gru"]["conditions"]["role_swap"]["accuracy"] = 0.80
    assert adjudicate(weak_control)["verdict"] == "R0_INVALID"
    changed_source = payload()
    changed_source["source"]["verdict"]["verdict"] = "W1_INTEGRATED_NOT_UNIQUE"
    assert adjudicate(changed_source)["verdict"] == "R0_INVALID"
    bad_hash = payload()
    bad_hash["seeds"][0]["checkpoints"]["gru"]["sha256"] = "bad"
    assert adjudicate(bad_hash)["verdict"] == "R0_INVALID"
