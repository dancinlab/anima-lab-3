from pathlib import Path

import numpy as np
import torch

from native_dialogue_lm import NativeDialogueTokenizer
from train_native_dialogue_lm import (
    BatchSource,
    dialogue_events,
    load_dialogue_examples,
    validation_loss,
)


def test_dialogue_parser_preserves_multiline_and_turn_order():
    events = dialogue_events(
        "user: 첫 질문\nassistant: 첫 답\n이어지는 답\nuser: 둘째 질문\nassistant: 둘째 답"
    )
    assert events == [
        ("user", "첫 질문"),
        ("assistant", "첫 답\n이어지는 답"),
        ("user", "둘째 질문"),
        ("assistant", "둘째 답"),
    ]


def test_response_only_batch_masks_user_tokens(tmp_path: Path):
    path = tmp_path / "dialogue.txt"
    path.write_text("user: 질문\nassistant: 정답\n", encoding="utf-8")
    tokenizer = NativeDialogueTokenizer.train([path], vocab_size=320)
    examples = load_dialogue_examples(path, tokenizer)
    source = BatchSource([], examples, block_size=32, seed=7, dialogue_fraction=1.0)
    _, labels = source.batch(1, response_only=True, device=torch.device("cpu"))
    kept = labels[0][labels[0] >= 0].tolist()
    assert tokenizer.ids["<assistant>"] not in kept
    assert tokenizer.ids["<eos>"] in kept
    assert len(kept) > 1


def test_general_batch_has_causal_targets():
    stream = np.arange(40, dtype=np.int32)
    source = BatchSource([stream], [], block_size=8, seed=3, dialogue_fraction=0.0)
    x, y = source.batch(2, response_only=False, device=torch.device("cpu"))
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_validation_replays_the_same_batches():
    from conscious_lm import build_model_from_config
    from measurement.native_dialogue_registry import preset

    config = preset("micro")
    model = build_model_from_config(config, dropout=0.0)
    stream = np.arange(1024, dtype=np.int32) % config["vocab_size"]
    source = BatchSource([stream], [], config["block_size"], 7, 0.0)
    first = validation_loss(model, source, 2, 1, torch.device("cpu"))
    second = validation_loss(model, source, 2, 1, torch.device("cpu"))
    assert first == second


def test_dialogue_files_are_sampled_before_examples():
    mask = np.array([False, True, True])
    short = [(np.array([1, 2, 3]), mask)]
    long = [(np.array([4, 5, 6]), mask)] * 100
    source = BatchSource([], [short, long], block_size=4, seed=11, dialogue_fraction=1.0)
    counts = {1: 0, 4: 0}
    for _ in range(1000):
        sequence, _ = source._dialogue()
        counts[int(sequence[0])] += 1
    assert 400 <= counts[1] <= 600
    assert 400 <= counts[4] <= 600
