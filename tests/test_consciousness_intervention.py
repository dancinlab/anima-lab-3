import torch

from conscious_lm import ConsciousLM


def tiny_model():
    torch.manual_seed(7)
    return ConsciousLM(
        vocab_size=256, d_model=24, n_head=4, n_layer=2,
        block_size=8, dropout=0.0,
    ).eval()


def test_normal_path_is_stable_and_off_changes_only_the_forward_condition():
    model = tiny_model()
    x = torch.arange(8).unsqueeze(0)
    normal_a = model.set_consciousness_intervention("normal")(x)[0]
    normal_b = model.set_consciousness_intervention("normal")(x)[0]
    off_a = model.set_consciousness_intervention("off")(x)[0]
    off_b = model.set_consciousness_intervention("off")(x)[0]
    assert torch.equal(normal_a, normal_b)
    assert torch.equal(off_a, off_b)
    assert not torch.equal(normal_a, off_a)


def test_shuffle_and_norm_matched_noise_are_reproducible():
    model = tiny_model()
    signal = torch.randn(2, 8, 24)
    shuffled_a = model.set_consciousness_intervention("shuffle", 11)._intervene_consciousness(signal)
    shuffled_b = model.set_consciousness_intervention("shuffle", 11)._intervene_consciousness(signal)
    noise_a = model.set_consciousness_intervention("noise", 11)._intervene_consciousness(signal)
    noise_b = model.set_consciousness_intervention("noise", 11)._intervene_consciousness(signal)
    assert torch.equal(shuffled_a, shuffled_b)
    assert torch.equal(noise_a, noise_b)
    assert torch.allclose(noise_a.norm(dim=-1), signal.norm(dim=-1), atol=1e-6)


def test_unknown_intervention_fails_closed():
    model = tiny_model()
    try:
        model.set_consciousness_intervention("wishful")
    except ValueError as exc:
        assert "unknown consciousness intervention" in str(exc)
    else:
        raise AssertionError("unknown intervention was accepted")
