import torch

import graft_behavior
from measurement.graft_behavior_registry import (
    BEHAVIOR_SPEC,
    PHASE_STATE_BRIDGE32_MEMORY_CONTROL_REPAIR_SPEC,
    PHASE_STATE_MEMORY_CONTROL_REPAIR_SPEC,
    PHASE_STATE_SPEC,
    experiment,
)
from trinity import QuantumC


def test_quantum_phase_readout_is_wrap_safe_and_read_only():
    torch.manual_seed(7)
    engine = QuantumC(nc=4, dim=6, max_cells=4)
    before = engine.engine._phases.clone()
    states = engine.get_phase_states()
    engine.engine._phases.add_(2 * torch.pi)
    wrapped = engine.get_phase_states()
    engine.engine._phases.copy_(before)

    assert states.shape == (4, 12)
    assert torch.allclose(states, wrapped, atol=1e-5)
    assert torch.equal(engine.engine._phases, before)


def test_phase_examples_keep_engine_and_readout_dimensions_separate(monkeypatch):
    spec = experiment(PHASE_STATE_SPEC["experiment"])
    spec["situations"] = spec["situations"][:1]
    spec["train_examples_per_situation"] = 1
    spec["warm_steps"] = 0
    spec["sense_steps"] = 1
    spec["delay_steps"] = [0, 0]
    monkeypatch.setattr(graft_behavior, "BEHAVIOR_SPEC", spec)

    row = graft_behavior.build_examples(7, "train")[0]

    assert row.state.shape == (spec["cells"], spec["state_dim"])
    assert row.memory.shape == (spec["cells"], spec["state_dim"])


def test_phase_repair_keeps_the_validated_memory_control_shape(monkeypatch):
    spec = experiment(PHASE_STATE_MEMORY_CONTROL_REPAIR_SPEC["experiment"])
    spec["situations"] = spec["situations"][:1]
    spec["train_examples_per_situation"] = 1
    spec["warm_steps"] = 0
    spec["sense_steps"] = 1
    spec["delay_steps"] = [0, 0]
    monkeypatch.setattr(graft_behavior, "BEHAVIOR_SPEC", spec)

    row = graft_behavior.build_examples(7, "train")[0]

    assert row.state.shape == (spec["cells"], 2 * BEHAVIOR_SPEC["state_dim"])
    assert row.memory.shape == (spec["cells"], BEHAVIOR_SPEC["state_dim"])


def test_bridge32_repair_changes_only_the_quantumc_arm_width():
    spec = experiment(PHASE_STATE_BRIDGE32_MEMORY_CONTROL_REPAIR_SPEC["experiment"])

    assert graft_behavior.bridge_hub_dim_for_arm(spec, "consciousness") == 32
    assert graft_behavior.bridge_hub_dim_for_arm(spec, "memory") == 8
