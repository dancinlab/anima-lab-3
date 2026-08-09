from copy import deepcopy

import torch

from measurement.state_survival_gate import adjudicate
from measurement.state_survival_registry import STATE_SURVIVAL_SPEC, spec_sha256
from trinity import QuantumC, ThalamicBridge


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
