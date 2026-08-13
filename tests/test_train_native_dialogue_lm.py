from pathlib import Path

import numpy as np
import pytest
import torch

from native_dialogue_lm import NativeDialogueTokenizer
import train_native_dialogue_lm
from train_native_dialogue_lm import (
    assert_registered_model_is_causal,
    BatchSource,
    dialogue_events,
    load_dialogue_examples,
    prepare_tokenizer,
    sha256_file,
    validation_loss,
)


def test_registered_model_startup_guard_accepts_causal_engine():
    assert_registered_model_is_causal()


def test_registered_model_startup_guard_rejects_future_signal(monkeypatch):
    class NonCausalModel(torch.nn.Module):
        def forward(self, tokens):
            shared_future = tokens.float().mean(dim=1, keepdim=True)
            logits = shared_future.unsqueeze(-1).expand(-1, tokens.shape[1], 2)
            return logits, None, []

    monkeypatch.setattr(
        train_native_dialogue_lm,
        "build_model_from_config",
        lambda *_args, **_kwargs: NonCausalModel(),
    )
    with pytest.raises(RuntimeError, match="future tokens changed prefix logits"):
        assert_registered_model_is_causal()


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


def test_jsonl_dialogue_supports_system_state_and_response_mask(tmp_path: Path):
    path = tmp_path / "dialogue.jsonl"
    path.write_text(
        '{"messages":[{"role":"system","content":"be concise"},'
        '{"role":"user","content":"질문"},{"role":"assistant","content":"정답"}]}\n',
        encoding="utf-8",
    )
    tokenizer = NativeDialogueTokenizer.train([path], vocab_size=320)
    examples = load_dialogue_examples(path, tokenizer)
    assert len(examples) == 1
    ids, mask = examples[0]
    assert tokenizer.ids["<state>"] in ids
    assert mask.sum() > 0
    source = BatchSource([], [examples], block_size=32, seed=7, dialogue_fraction=0.0)
    _, labels = source.batch(
        1, response_only=True, device=torch.device("cpu"), source_mode="dialogue"
    )
    kept = labels[0][labels[0] >= 0].tolist()
    assert tokenizer.ids["<eos>"] in kept


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


def test_continuation_copies_checkpoint_tokenizer_instead_of_retraining(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    corpus = tmp_path / "corpus.txt"
    source.mkdir()
    corpus.write_text("기존 vocabulary text", encoding="utf-8")
    original = NativeDialogueTokenizer.train([corpus], vocab_size=320)
    original.save(source / "tokenizer.json")
    checkpoint = source / "final.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")

    replacement = tmp_path / "replacement.txt"
    replacement.write_text("completely different training text", encoding="utf-8")
    _, copied = prepare_tokenizer(output, original.vocab_size, [replacement], checkpoint)

    assert sha256_file(copied) == sha256_file(source / "tokenizer.json")


def test_continuation_rejects_different_existing_tokenizer(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("가나다라 existing tokens", encoding="utf-8")
    right.write_text("different replacement vocabulary words", encoding="utf-8")
    NativeDialogueTokenizer.train([left], vocab_size=320).save(source / "tokenizer.json")
    NativeDialogueTokenizer.train([right], vocab_size=320).save(output / "tokenizer.json")
    checkpoint = source / "final.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")

    with pytest.raises(ValueError, match="output tokenizer differs"):
        prepare_tokenizer(output, 320, [right], checkpoint)


def test_continuation_rejects_manifest_tokenizer_drift(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("manifest tokenizer identity " * 50, encoding="utf-8")
    tokenizer = NativeDialogueTokenizer.train([corpus], vocab_size=320)
    tokenizer.save(source / "tokenizer.json")
    checkpoint = source / "final.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")

    with pytest.raises(ValueError, match="differs from the data manifest"):
        prepare_tokenizer(
            output,
            tokenizer.vocab_size,
            [corpus],
            checkpoint,
            expected_sha256="0" * 64,
        )
