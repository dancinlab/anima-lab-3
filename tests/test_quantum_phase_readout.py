import torch

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
