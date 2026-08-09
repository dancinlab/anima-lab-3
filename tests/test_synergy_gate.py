from copy import deepcopy
import json
from pathlib import Path

import torch

from measurement.synergy_gate import _expected_audit, adjudicate
from measurement.synergy_registry import (
    SYNERGY_CONTROL_REPAIR_SPEC,
    SYNERGY_SPEC,
    spec_sha256,
)
from synergy import SplitCueExample, _partner_permutation, audit_examples


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


def payload(quantum=None, memory=None, gru=None, spec=SYNERGY_SPEC):
    quantum = quantum or arm()
    memory = memory or arm(normal=0.90)
    gru = gru or arm(normal=0.90)
    return {
        "experiment": spec["experiment"],
        "spec": deepcopy(spec),
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": {
            split: _expected_audit(spec, split) for split in ("train", "eval")
        },
        "seeds": [
            {
                "seed": seed,
                "arms": {
                    "quantum_pair": deepcopy(quantum),
                    "direct_memory": deepcopy(memory),
                    "gru": deepcopy(gru),
                },
                "checkpoints": {
                    name: {"path": f"seed_{seed}_{name}.pt", "sha256": str(index + 1) * 64}
                    for index, name in enumerate(spec["arms"])
                },
            }
            for seed in spec["seeds"]
        ],
    }


def test_synergy_gate_distinguishes_advantage_equivalence_and_failure():
    assert adjudicate(payload())[
        "verdict"
    ] == "Y1_INTEGRATED_NOT_UNIQUE"
    advantage = payload(memory=arm(normal=0.80), gru=arm(normal=0.80))
    assert adjudicate(advantage)["verdict"] == "Y2_QUANTUM_ADVANTAGE"
    failed = payload(quantum=arm(normal=0.60))
    assert adjudicate(failed)["verdict"] == "Y3_NOT_INTEGRATED"


def test_synergy_gate_fails_closed_on_balance_controls_hash_and_recovery():
    unbalanced = payload()
    unbalanced["dataset_audit"]["eval"]["target_counts"]["0"] += 1
    assert adjudicate(unbalanced)["verdict"] == "Y0_INVALID"
    weak_control = payload(gru=arm(normal=0.50))
    assert adjudicate(weak_control)["verdict"] == "Y0_INVALID"
    bad_hash = payload()
    bad_hash["seeds"][0]["checkpoints"]["gru"]["sha256"] = "wrong"
    assert adjudicate(bad_hash)["verdict"] == "Y0_INVALID"
    bad_recovery = payload()
    bad_recovery["seeds"][0]["arms"]["quantum_pair"]["conditions"]["recovered"][
        "logits_identical"
    ] = False
    assert adjudicate(bad_recovery)["verdict"] == "Y3_NOT_INTEGRATED"


def test_control_role_repair_keeps_direct_memory_as_comparison_not_validator():
    repaired = payload(
        memory=arm(normal=0.35, kl=2.0),
        spec=SYNERGY_CONTROL_REPAIR_SPEC,
    )
    assert adjudicate(repaired)["verdict"] == "Y1_INTEGRATED_NOT_UNIQUE"
    weak_gru = payload(
        memory=arm(normal=0.35, kl=2.0),
        gru=arm(normal=0.50),
        spec=SYNERGY_CONTROL_REPAIR_SPEC,
    )
    assert adjudicate(weak_gru)["verdict"] == "Y0_INVALID"


def test_synergy_gate_rejects_non_finite_and_spec_drift():
    non_finite = payload()
    non_finite["seeds"][0]["arms"]["quantum_pair"]["neutral_kl_nats"] = float("nan")
    assert adjudicate(non_finite)["verdict"] == "Y0_INVALID"
    drift = payload()
    drift["spec"]["train_steps"] += 1
    assert adjudicate(drift)["verdict"] == "Y0_INVALID"


def test_split_cue_audit_is_balanced_and_partner_shuffle_breaks_the_target():
    examples = []
    state = torch.zeros(2, 4)
    for module_a in range(4):
        for module_b in range(4):
            for _ in range(2):
                examples.append(SplitCueExample(
                    quantum=(state, state), memory=(state, state),
                    module_a=module_a, module_b=module_b,
                    target=(module_a + module_b) % 4,
                ))
    audit = audit_examples(examples)
    assert audit["pair_count"] == 16
    assert set(audit["target_counts"].values()) == {8}
    permutation = _partner_permutation(examples, len(SYNERGY_SPEC["actions"]))
    assert all(examples[index].module_b != examples[source].module_b
               for index, source in enumerate(permutation))
    assert all(
        (examples[index].module_a + examples[source].module_b) % 4 != examples[index].target
        for index, source in enumerate(permutation)
    )


def test_committed_synergy_results_reproduce_registered_verdicts():
    root = Path(__file__).resolve().parents[1]
    for stem, expected in (("synergy_invalid", "Y0_INVALID"), ("synergy", "Y3_NOT_INTEGRATED")):
        result = json.loads((root / f"measurement/{stem}_results.json").read_text())
        verdict = json.loads((root / f"measurement/{stem}_verdict.json").read_text())
        assert adjudicate(result) == verdict
        assert verdict["verdict"] == expected
