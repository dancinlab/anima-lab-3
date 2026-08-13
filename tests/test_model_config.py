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
