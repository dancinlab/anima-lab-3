import pytest
import torch

from measurement.workspace_registry import WORKSPACE_CONTROL_SEED_REPAIR_SPEC
from synergy import _arm_seed
from trinity import RecurrentWorkspaceBridge, ThalamicBridge


def test_thalamic_refactor_preserves_forward_and_trace_identity():
    torch.manual_seed(7)
    bridge = ThalamicBridge(c_dim=6, d_model=10, hub_dim=4, alpha=0.5).eval()
    states = torch.randn(5, 6)
    trace = bridge.trace(states, seq_len=3)
    assert torch.equal(bridge(states, seq_len=3), trace["gate"])
    assert torch.equal(bridge.transform_cells(states), trace["cells"])
    final = bridge.gate_from_pooled(trace["pooled"], seq_len=3)
    assert torch.equal(final["gate"], trace["gate"])


def test_recurrent_workspace_reuses_bridge_shapes_and_backpropagates():
    torch.manual_seed(11)
    bridge = RecurrentWorkspaceBridge(
        c_dim=6, d_model=10, hub_dim=4, alpha=0.5, rounds=2
    )
    modules = (torch.randn(5, 6), torch.randn(5, 6))
    trace = bridge.trace_modules(modules, seq_len=3)
    assert trace["module_cells"].shape == (2, 5, 4)
    assert trace["module_summaries"].shape == (2, 4)
    assert trace["workspace_timeline"].shape == (1, 2, 4)
    assert trace["gate"].shape == (1, 3, 10)
    trace["gate"].sum().backward()
    assert bridge.workspace_cell.weight_hh.grad is not None


def test_recurrent_workspace_rejects_missing_module_or_round():
    with pytest.raises(ValueError):
        RecurrentWorkspaceBridge(c_dim=6, d_model=10, hub_dim=4, rounds=0)
    bridge = RecurrentWorkspaceBridge(c_dim=6, d_model=10, hub_dim=4, rounds=1)
    with pytest.raises(ValueError):
        bridge((torch.randn(5, 6),))


def test_repaired_arm_seeds_are_independent_of_roster_order():
    spec = dict(WORKSPACE_CONTROL_SEED_REPAIR_SPEC)
    expected = {
        arm: _arm_seed(1337, arm, spec)
        for arm in spec["arms"]
    }
    spec["arms"] = list(reversed(spec["arms"]))
    assert {
        arm: _arm_seed(1337, arm, spec)
        for arm in spec["arms"]
    } == expected
    assert expected["gru"] == 201_337
    assert expected["quantum_workspace_2"] == 201_337
    assert expected["memory_workspace_4"] == 601_337
