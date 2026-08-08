import torch

import conscious_lm
import train_conscious_lm


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


def test_trainer_exports_canonical_checkpoint_builder_for_scorers():
    assert train_conscious_lm.build_model_from_config is conscious_lm.build_model_from_config
