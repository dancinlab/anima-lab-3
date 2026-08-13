from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from native_dialogue_lm import NativeDialogueTokenizer
from prepare_native_dialogue_data import panel_fingerprints
from prepare_native_instruction_data import (
    dialogue_token_count,
    memory_messages,
    select_instructions,
)


def test_dynamic_memory_is_bilingual_bounded_and_avoids_frozen_values(tmp_path: Path):
    seed = tmp_path / "seed.txt"
    seed.write_text("\n".join(message["content"] for lang in ("en", "ko")
                              for i in range(20) for message in memory_messages(lang, i)))
    tokenizer = NativeDialogueTokenizer.train([seed], vocab_size=512)
    for lang in ("en", "ko"):
        for index in range(20):
            messages = memory_messages(lang, index)
            assert [message["role"] for message in messages] == [
                "user", "assistant", "user", "assistant"
            ]
            assert dialogue_token_count(tokenizer, messages) <= 513
            folded = " ".join(message["content"] for message in messages).casefold()
            assert "blue box" not in folded and "파란 상자" not in folded


def test_instruction_selection_is_deterministic_filtered_and_bounded(tmp_path: Path):
    rows = []
    for index in range(40):
        rows.append({
            "id": index,
            "inputs": f"Explain separate topic number {index}.",
            "targets": f"This is grounded answer number {index}.",
            "dataset_name": "tiny",
        })
    path = tmp_path / "instructions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(row["inputs"] + row["targets"] for row in rows))
    tokenizer = NativeDialogueTokenizer.train([corpus], vocab_size=512)
    panel = panel_fingerprints(Path("measurement/native_dialogue_panel.json"))
    first, stats = select_instructions(path, "en", tokenizer, panel, 10, 2, 128)
    second, _ = select_instructions(path, "en", tokenizer, panel, 10, 2, 128)
    assert first == second
    assert stats["selected"] == 10
    assert all(dialogue_token_count(tokenizer, messages) <= 128 for _, messages in first)
