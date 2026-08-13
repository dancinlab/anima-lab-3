import pytest
import torch

import conscious_lm
import train_conscious_lm
from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC, preset


def test_native_target_keeps_registered_303m_shape():
    target = preset("target")
    assert (target["dim"], target["heads"], target["layers"]) == (896, 14, 11)
    assert target["dim"] // target["heads"] == 64
    training = NATIVE_DIALOGUE_SPEC["native_dialogue5"]
    assert training["parameters"] == 303_628_504
    assert training["global_batch"] % training["micro_batch"] == 0
    assert training["gradient_accumulation"] == (
        training["global_batch"] // training["micro_batch"]
    )
    assert training["micro_batch_profiles"] == {
        "rtx_5090_32gb": 16,
        "h100_80gb": 32,
    }
    assert all(
        training["global_batch"] % micro_batch == 0
        for micro_batch in training["micro_batch_profiles"].values()
    )
    assert training["preprocessing_workers"] == 4


def test_checkpoint_config_builds_registered_ffn_variant():
    config = {
        "dim": 24,
        "heads": 4,
        "layers": 2,
        "block_size": 16,
        "dropout": 0.37,
        "ffn_type": "standard",
    }
    model = conscious_lm.build_model_from_config(config, dropout=0.0)
    assert model.ffn_type == "standard"
    assert isinstance(model.blocks[0].ffn, conscious_lm.StandardFFN)
    logits, _, activity = model(torch.zeros((1, 8), dtype=torch.long))
    assert logits.shape == (1, 8, 256)
    assert len(activity) == 2


def test_legacy_checkpoint_config_defaults_to_pure_field():
    config = {"dim": 24, "heads": 4, "layers": 1, "block_size": 16}
    model = conscious_lm.build_model_from_config(config, dropout=0.0)
    assert model.ffn_type == "pure_field"
    assert isinstance(model.blocks[0].ffn, conscious_lm.PureFieldFFN)


def test_checkpoint_config_supports_native_vocabulary_and_signal_settings():
    config = {
        "vocab_size": 512,
        "dim": 24,
        "heads": 4,
        "layers": 1,
        "block_size": 16,
        "ffn_type": "standard",
        "gate_strength": 0.02,
        "n_ca_rules": 3,
    }
    model = conscious_lm.build_model_from_config(config, dropout=0.0)
    assert model.vocab_size == 512
    assert model.blocks[0].gate_strength == 0.02
    assert model.blocks[0].n_ca_rules == 3
    logits, _, _ = model(torch.zeros((1, 4), dtype=torch.long))
    assert logits.shape == (1, 4, 512)


def test_internal_signal_cannot_leak_suffix_tokens_into_prefix_logits():
    torch.manual_seed(7)
    model = conscious_lm.ConsciousLM(
        vocab_size=32, d_model=24, n_head=4, n_layer=3,
        block_size=16, dropout=0.0, gate_strength=0.1,
        n_ca_rules=3, ffn_type="standard", signal_normalization="causal_prefix",
    ).eval()
    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    first = torch.cat((prefix, torch.tensor([[5, 6]], dtype=torch.long)), dim=1)
    second = torch.cat((prefix, torch.tensor([[7, 8]], dtype=torch.long)), dim=1)

    first_logits, _, _ = model(first)
    second_logits, _, _ = model(second)

    torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4])


def test_causal_zscore_uses_only_available_prefix():
    prefix = torch.tensor([[2.0, 4.0, 8.0]])
    extended = torch.tensor([[2.0, 4.0, 8.0, 1000.0]])
    torch.testing.assert_close(
        conscious_lm.causal_zscore(prefix),
        conscious_lm.causal_zscore(extended)[:, :3],
    )


def test_causal_zscore_has_finite_gradient_at_zero_variance_prefix():
    values = torch.tensor([[2.0, 4.0, 8.0]], requires_grad=True)
    normalized = conscious_lm.causal_zscore(values)
    normalized.square().sum().backward()
    assert torch.isfinite(normalized).all()
    assert torch.isfinite(values.grad).all()


def test_cached_forward_matches_full_causal_path():
    torch.manual_seed(17)
    model = conscious_lm.ConsciousLM(
        vocab_size=32, d_model=24, n_head=4, n_layer=3,
        block_size=16, dropout=0.0, gate_strength=0.1,
        n_ca_rules=3, ffn_type="standard", signal_normalization="causal_prefix",
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    appended = torch.tensor([[5]], dtype=torch.long)

    full_prompt = model(prompt)[0]
    cached_prompt, _, _, cache = model.forward_cached(prompt)
    full_appended = model(torch.cat((prompt, appended), dim=1))[0][:, -1:]
    cached_appended, _, _, cache = model.forward_cached(appended, cache)

    torch.testing.assert_close(cached_prompt, full_prompt, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cached_appended, full_appended, atol=1e-6, rtol=1e-5)
    assert cache["length"] == 5


def test_cached_generation_preserves_uncached_tokens():
    torch.manual_seed(23)
    model = conscious_lm.ConsciousLM(
        vocab_size=32, d_model=24, n_head=4, n_layer=2,
        block_size=32, dropout=0.0, gate_strength=0.1,
        n_ca_rules=3, ffn_type="standard", signal_normalization="causal_prefix",
    ).eval()
    outputs = []
    for use_cache in (False, True):
        torch.manual_seed(29)
        generated, _ = conscious_lm.generate(
            model, [1, 2, 3], max_new=8, temperature=0.7,
            device="cpu", top_p=0.9, repetition_penalty=1.1,
            use_cache=use_cache,
        )
        outputs.append(generated)
    assert outputs[0] == outputs[1]


def test_trainer_exports_canonical_checkpoint_builder_for_scorers():
    assert train_conscious_lm.build_model_from_config is conscious_lm.build_model_from_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new": -1},
        {"temperature": 0},
        {"top_p": 0},
        {"top_p": 1.1},
        {"repetition_penalty": 0.9},
        {"eos_token_id": 999},
    ],
)
def test_generation_rejects_invalid_sampling_settings(kwargs):
    model = conscious_lm.ConsciousLM(
        vocab_size=32, d_model=24, n_head=4, n_layer=1,
        block_size=16, ffn_type="standard",
    )
    with pytest.raises(ValueError):
        conscious_lm.generate(model, [1], device="cpu", **kwargs)
