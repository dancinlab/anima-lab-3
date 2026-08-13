import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from native_dialogue_lm import NativeDialogueTokenizer
from prepare_native_conversation_data import ROLE_PREFIX, select_conversations
from prepare_native_dialogue_data import panel_fingerprints


def _tokenizer(tmp_path: Path) -> NativeDialogueTokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("natural answer 질문 대답 memory correction hello world", encoding="utf-8")
    return NativeDialogueTokenizer.train([corpus], vocab_size=512)


def test_small_model_sources_are_deterministic_clean_and_bounded(tmp_path: Path):
    english = tmp_path / "en.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"messages": [{"role": "user", "content": f"Natural question {index}?"},
                      {"role": "assistant", "content": f"Natural answer {index}."}]}
        for index in range(30)
    ]), english)
    korean = tmp_path / "ko.jsonl"
    korean.write_text("".join(json.dumps({
        "prompt": f"Human: 자연스러운 질문 {index}?",
        "response": f"GPT: 자연스러운 대답 {index}.",
    }, ensure_ascii=False) + "\n" for index in range(30)), encoding="utf-8")
    tokenizer = _tokenizer(tmp_path)
    panel = panel_fingerprints(Path("measurement/native_dialogue_panel.json"))
    for lang, paths in (("en", [english]), ("ko", [korean])):
        first, stats = select_conversations(paths, lang, tokenizer, panel, 10, 2, 128)
        second, _ = select_conversations(paths, lang, tokenizer, panel, 10, 2, 128)
        assert first == second
        assert stats["selected"] == 10
        assert all(sum(len(tokenizer.encode(row["content"])) for row in messages) < 128
                   for _, messages in first)
    assert ROLE_PREFIX.sub("", "Human: 실제 질문") == "실제 질문"
    assert ROLE_PREFIX.sub("", "GPT: 실제 답변") == "실제 답변"
