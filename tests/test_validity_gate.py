from copy import deepcopy
import json
from pathlib import Path

from measurement.validity_gate import adjudicate
from measurement.validity_registry import VALIDITY_SPEC, spec_sha256


def probe(accuracy=0.95, shuffled=0.20):
    matrix = [[0] * 5 for _ in range(5)]
    for index in range(5):
        matrix[index][index] = 20
    return {
        "accuracy": accuracy,
        "shuffled_label_accuracy": shuffled,
        "shuffled_label_accuracy_std": 0.01,
        "shuffled_label_permutations": 32,
        "confusion_matrix": matrix,
        "per_class_recall": [1.0] * 5,
    }


def payload():
    spec = VALIDITY_SPEC
    audit = {"train": {"examples": 150}, "eval": {"examples": 100}}
    source = {
        "experiment": spec["source_experiment"],
        "verdict": spec["source_verdict"],
        "results_sha256": spec["source_results_sha256"],
        "verdict_sha256": spec["source_verdict_sha256"],
        "dataset_audit": audit,
        "reproduced": True,
    }
    root = Path(__file__).resolve().parents[1]
    value = {
        "experiment": spec["experiment"],
        "spec": deepcopy(spec),
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "model_revision": spec["model_revision"],
        "dataset_audit": audit,
        "action_tokens": {
            "token_ids": {action: index for index, action in enumerate(spec["actions"])},
            "unique_single_tokens": True,
        },
        "seeds": [
            {
                "seed": seed,
                "sensory": {
                    source_name: {label: probe() for label in ("module_a", "module_b")}
                    for source_name in ("quantum", "memory")
                },
                "arms": {
                    arm: {
                        "relation": probe(),
                        "direct_action": {
                            mode: probe() for mode in spec["normalization_modes"]
                        },
                        "normalization": {
                            "eval_rms_difference": 0.1,
                            "population_mean_ready": True,
                        },
                        "language": {
                            "accuracy": 0.95,
                            "source_accuracy": 0.95,
                            "source_accuracy_exact": True,
                            "selection_counts": {action: 20 for action in spec["actions"]},
                            "confusion_matrix": [[20 if i == j else 0 for j in range(5)]
                                                 for i in range(5)],
                            "per_class_recall": [1.0] * 5,
                        },
                    }
                    for arm in spec["arms"]
                },
                "checkpoints": {
                    arm: {"path": f"seed_{seed}_{arm}.pt",
                          "sha256": spec["checkpoint_sha256"][str(seed)][arm]}
                    for arm in spec["arms"]
                },
            }
            for seed in spec["seeds"]
        ],
    }
    value["invalid_run"] = {
        "results": json.loads((root / spec["invalid_results"]).read_text()),
        "verdict": json.loads((root / spec["invalid_verdict"]).read_text()),
        "results_sha256": spec["invalid_results_sha256"],
        "verdict_sha256": spec["invalid_verdict_sha256"],
    }
    return value


def row(value, seed=1337, arm="gru"):
    return next(item for item in value["seeds"] if item["seed"] == seed)["arms"][arm]


def test_validity_gate_locates_registered_failure_stages():
    assert adjudicate(payload())["verdict"] == "V5_PATH_VALID"

    value = payload()
    next(item for item in value["seeds"] if item["seed"] == 1337)["sensory"]["memory"][
        "module_a"
    ] = probe(0.50)
    assert adjudicate(value)["verdict"] == "V1_SENSE_LOSS"

    value = payload()
    row(value)["relation"] = probe(0.50)
    assert adjudicate(value)["verdict"] == "V2_RELATION_LOSS"

    value = payload()
    row(value)["direct_action"]["runtime_style"] = probe(0.50)
    assert adjudicate(value)["verdict"] == "V4_PROTOCOL_LOSS"

    value = payload()
    row(value)["language"]["accuracy"] = 0.50
    row(value)["language"]["source_accuracy"] = 0.50
    assert adjudicate(value)["verdict"] == "V3_LANGUAGE_LOSS"

    value = payload()
    row(value)["language"]["accuracy"] = 0.94
    row(value)["language"]["source_accuracy_exact"] = False
    assert adjudicate(value)["verdict"] == "V5_PATH_VALID"


def test_validity_gate_fails_closed_on_source_checkpoint_fake_and_tokens():
    value = payload()
    value["source"]["results_sha256"] = "bad"
    assert adjudicate(value)["verdict"] == "V0_INVALID"

    value = payload()
    value["seeds"][0]["checkpoints"]["gru"]["sha256"] = "0" * 64
    assert adjudicate(value)["verdict"] == "V0_INVALID"

    value = payload()
    row(value)["relation"] = probe(0.95, 0.50)
    assert adjudicate(value)["verdict"] == "V0_INVALID"

    value = payload()
    value["action_tokens"]["unique_single_tokens"] = False
    assert adjudicate(value)["verdict"] == "V0_INVALID"

    value = payload()
    value["model_revision"] = "main"
    assert adjudicate(value)["verdict"] == "V0_INVALID"


def test_validity_gate_rejects_missing_action_and_spec_drift():
    value = payload()
    row(value)["direct_action"]["runtime_style"]["confusion_matrix"][0] = [20, 0]
    assert adjudicate(value)["verdict"] == "V0_INVALID"

    value = payload()
    value["spec"]["probe_ridge"] = 2.0
    assert adjudicate(value)["verdict"] == "V0_INVALID"
