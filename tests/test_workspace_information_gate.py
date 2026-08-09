from copy import deepcopy

from measurement.workspace_information_gate import adjudicate
from measurement.workspace_registry import WORKSPACE_INFORMATION_SPEC, spec_sha256


def metric(accuracy=0.95, shuffled=0.25):
    return {
        "accuracy": accuracy,
        "shuffled_label_accuracy": shuffled,
        "shuffled_label_accuracy_std": 0.01,
        "shuffled_label_permutations": 32,
    }


def payload():
    spec = WORKSPACE_INFORMATION_SPEC
    return {
        "experiment": spec["experiment"],
        "spec": deepcopy(spec),
        "spec_sha256": spec_sha256(spec),
        "source": {"experiment": spec["source_experiment"], "verdict": spec["source_verdict"]},
        "seeds": [
            {
                "seed": seed,
                "checkpoint": {"path": f"seed_{seed}_quantum_pair.pt", "sha256": "a" * 64},
                "channels": {
                    channel: {label: metric() for label in spec["labels"]}
                    for channel in spec["channels"]
                },
            }
            for seed in spec["seeds"]
        ],
    }


def test_information_gate_locates_each_registered_loss_stage():
    assert adjudicate(payload())["verdict"] == "I4_RELATION_COMPUTATION_LOSS"
    for channel, expected in (
        ("bridge_cells", "I1_LOCAL_TRANSFORM_LOSS"),
        ("bridge_pooled", "I2_POOLING_LOSS"),
        ("bridge_gate", "I3_GATE_LOSS"),
        ("normalized_code", "I3_GATE_LOSS"),
    ):
        value = payload()
        value["seeds"][0]["channels"][channel]["module_a"] = metric(0.50)
        assert adjudicate(value)["verdict"] == expected


def test_information_gate_fails_closed_on_source_seed_hash_and_spec():
    value = payload()
    value["source"]["verdict"] = "Y1_INTEGRATED_NOT_UNIQUE"
    assert adjudicate(value)["verdict"] == "I0_INVALID"
    value = payload()
    value["seeds"].pop()
    assert adjudicate(value)["verdict"] == "I0_INVALID"
    value = payload()
    value["seeds"][0]["checkpoint"]["sha256"] = "bad"
    assert adjudicate(value)["verdict"] == "I0_INVALID"
    value = payload()
    value["spec"]["probe_ridge"] = 2.0
    assert adjudicate(value)["verdict"] == "I0_INVALID"
