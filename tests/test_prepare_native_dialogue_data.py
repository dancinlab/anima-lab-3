from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_native_dialogue_data import (
    SampleWriter,
    is_panel_near_duplicate,
    panel_fingerprints,
    prepare_dialogue,
    prepare_general,
)


def test_panel_decontamination_finds_exact_and_keeps_unrelated():
    rows = panel_fingerprints(Path("measurement/native_dialogue_panel.json"))
    assert is_panel_near_duplicate(
        "What happens when ice is left in sunlight, and why?", rows
    )
    assert not is_panel_near_duplicate("A fox crossed the quiet road at dawn.", rows)


def test_prepare_parquet_writes_balanced_jsonl_and_manifest_inputs(tmp_path: Path):
    dialogue_path = tmp_path / "dialogue.parquet"
    pq.write_table(pa.table({
        "custom_id": ["train-row", "validation-row"],
        "messages": [[
            {"role": "user", "content": "오늘 어때?", "content_en": "How are you?"},
            {"role": "assistant", "content": "좋아요.", "content_en": "I am well."},
        ], [
            {"role": "user", "content": "다른 질문", "content_en": "Another question"},
            {"role": "assistant", "content": "다른 답", "content_en": "Another answer"},
        ]],
    }), dialogue_path)
    general_path = tmp_path / "general.parquet"
    pq.write_table(pa.table({"text": ["A separate educational document."]}), general_path)
    output = tmp_path / "out"
    output.mkdir()
    samples = SampleWriter(output, 1000)
    rows = panel_fingerprints(Path("measurement/native_dialogue_panel.json"))
    counts, made = prepare_dialogue(dialogue_path, 0, output, 50, rows, samples)
    general_counts, general_made = prepare_general(
        general_path, "en", 0, output, 50, rows, samples
    )
    samples.close()
    assert sum(counts["en"][key] for key in ("train", "validation")) == 2
    assert sum(counts["ko"][key] for key in ("train", "validation")) == 2
    assert made["en"][0].endswith(".jsonl")
    assert sum(general_counts[key] for key in ("train", "validation")) == 1
    assert (output / general_made[0]).is_file()
