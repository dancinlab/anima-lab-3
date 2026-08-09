from copy import deepcopy
import json
from pathlib import Path

from measurement.bridge_capacity_gate import adjudicate
from measurement.bridge_capacity_registry import BRIDGE_CAPACITY_SPEC, spec_sha256


def _metrics(passed=True, positive=False):
    threshold = 0.95 if positive else 0.85
    return {"accuracy": threshold if passed else 0.5, "shuffled_label_accuracy": 0.25}


def _payload():
    widths = []
    for width in BRIDGE_CAPACITY_SPEC["bridge"]["hub_dims"]:
        seeds = []
        for seed in BRIDGE_CAPACITY_SPEC["seeds"]:
            delays = {}
            for delay in BRIDGE_CAPACITY_SPEC["delay_steps"]:
                delays[str(delay)] = {
                    channel: _metrics(positive=channel == "sense_input")
                    for channel in BRIDGE_CAPACITY_SPEC["channels"]
                }
            seeds.append({"seed": seed, "delays": delays})
        widths.append({"hub_dim": width, "pooling": "mean", "seeds": seeds})
    return {
        "experiment": BRIDGE_CAPACITY_SPEC["experiment"],
        "spec_sha256": spec_sha256(),
        "widths": widths,
    }


def test_capacity_gate_selects_the_minimum_full_path_width():
    payload = _payload()
    for row in payload["widths"]:
        if row["hub_dim"] < 32:
            for seed in row["seeds"]:
                seed["delays"]["32"]["bridge_cells"] = _metrics(False)
    verdict = adjudicate(payload)
    assert verdict["verdict"] == "C4_FULL_PATH_RECOVERY"
    assert verdict["selected_hub_dim"] == 32


def test_capacity_gate_localizes_mean_pooling_and_fails_closed():
    payload = _payload()
    for row in payload["widths"]:
        for seed in row["seeds"]:
            seed["delays"]["32"]["bridge_pooled"] = _metrics(False)
    assert adjudicate(payload)["verdict"] == "C2_TRANSFORM_RECOVERED_POOLING_LOSS"
    invalid = deepcopy(payload)
    invalid["spec_sha256"] = "wrong"
    assert adjudicate(invalid)["verdict"] == "C0_INVALID"


def test_committed_capacity_result_reproduces_the_registered_verdict():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "measurement/bridge_capacity_results.json").read_text())
    verdict = json.loads((root / "measurement/bridge_capacity_verdict.json").read_text())

    assert adjudicate(payload) == verdict
    assert verdict["verdict"] == "C4_FULL_PATH_RECOVERY"
    assert verdict["selected_hub_dim"] == 32
