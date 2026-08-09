from copy import deepcopy
import json
from pathlib import Path

import torch

from measurement.state_survival_gate import adjudicate
from measurement.state_survival_registry import STATE_SURVIVAL_SPEC, spec_sha256
from state_survival import probe_channel
from trinity import QuantumC, ThalamicBridge
from measurement.bridge_config import THALAMIC_BRIDGE_HUB_DIM


def test_quantum_state_channels_are_read_only_and_complete():
    torch.manual_seed(7)
    engine = QuantumC(nc=4, dim=6, max_cells=4)
    before = engine.engine.snapshot()
    channels = engine.get_state_channels()
    after = engine.engine.snapshot()

    assert set(channels) == {"amplitude", "phase", "phase_velocity", "tension_frustration", "full_state"}
    assert channels["phase"].shape == (4, 12)
    assert channels["full_state"].shape == (4, 26)
    for key in ("amplitudes", "phases", "phase_velocities", "frustrations"):
        assert torch.equal(before[key], after[key])


def test_bridge_trace_is_the_exact_forward_path_and_read_only():
    torch.manual_seed(11)
    bridge = ThalamicBridge(c_dim=12, d_model=16, hub_dim=4).eval()
    states = torch.randn(5, 12)
    before = states.clone()
    trace = bridge.trace(states, seq_len=3)

    assert trace["cells"].shape == (1, 5, 4)
    assert trace["pooled"].shape == (1, 1, 4)
    assert torch.equal(trace["gate"], bridge(states, seq_len=3))
    assert torch.equal(states, before)


def test_bridge_default_is_upgraded_and_legacy_checkpoint_width_is_inferred():
    current = ThalamicBridge(c_dim=12, d_model=16)
    legacy = ThalamicBridge(c_dim=12, d_model=16, hub_dim=8)

    assert current.compress.out_features == THALAMIC_BRIDGE_HUB_DIM
    assert ThalamicBridge.hub_dim_from_state_dict(legacy.state_dict()) == 8


def _metrics(passed=True):
    return {"accuracy": 0.9 if passed else 0.5, "shuffled_label_accuracy": 0.25}


def _payload():
    delays = {}
    for delay in STATE_SURVIVAL_SPEC["delay_steps"]:
        delays[str(delay)] = {channel: _metrics() for channel in STATE_SURVIVAL_SPEC["channels"]}
    return {
        "experiment": STATE_SURVIVAL_SPEC["experiment"],
        "spec_sha256": spec_sha256(),
        "seeds": [{"seed": seed, "delays": deepcopy(delays)} for seed in STATE_SURVIVAL_SPEC["seeds"]],
        "downstream_behavior": {
            "experiment": STATE_SURVIVAL_SPEC["downstream_behavior"]["required_experiment"],
            "verdict": "B3_NOT_CAUSAL",
        },
    }


def test_state_gate_fails_closed_and_localizes_pooling_loss():
    payload = _payload()
    assert adjudicate(payload)["verdict"] == "S7_BEHAVIOR_GROUNDING_LOSS"
    payload["seeds"][0]["delays"]["8"]["bridge_pooled"] = _metrics(False)
    assert adjudicate(payload)["verdict"] == "S5_CELL_POOLING_LOSS"
    payload["spec_sha256"] = "wrong"
    assert adjudicate(payload)["verdict"] == "S0_INVALID"


def test_state_gate_rejects_non_finite_metrics():
    payload = _payload()
    payload["seeds"][0]["delays"]["0"]["phase"]["accuracy"] = float("nan")
    assert adjudicate(payload)["verdict"] == "S0_INVALID"


def test_probe_averages_the_registered_number_of_label_permutations():
    train_y = torch.arange(4).repeat_interleave(8)
    train_x = torch.nn.functional.one_hot(train_y, num_classes=4).float()
    metrics = probe_channel(train_x, train_y, train_x, train_y, seed=19)

    assert metrics["accuracy"] == 1.0
    assert metrics["shuffled_label_permutations"] == STATE_SURVIVAL_SPEC["label_control"]["permutations"]
    assert metrics["shuffled_label_accuracy"] <= STATE_SURVIVAL_SPEC["thresholds"]["shuffled_label_max_accuracy"]


def test_committed_state_result_reproduces_the_registered_verdict():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "measurement/state_survival_results.json").read_text())
    verdict = json.loads((root / "measurement/state_survival_verdict.json").read_text())

    assert adjudicate(payload) == verdict
    assert verdict["verdict"] == "S4_BRIDGE_TRANSFORM_LOSS"
