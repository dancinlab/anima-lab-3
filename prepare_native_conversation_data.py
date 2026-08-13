#!/usr/bin/env python3
"""Prepare small-model bilingual conversation and dynamic-memory data."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from pathlib import Path

from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC, spec_sha256
from native_dialogue_lm import NativeDialogueTokenizer, checkpoint_sha256
from prepare_native_dialogue_data import (
    PANEL_PATH,
    is_panel_near_duplicate,
    panel_fingerprints,
    parquet_batches,
    sha256_file,
    validation_row,
)
from prepare_native_instruction_data import dialogue_token_count, memory_messages


ROLE_PREFIX = re.compile(r"^(?:human|gpt)\s*:\s*", re.IGNORECASE)


def download_sources(source_dir: Path) -> dict[str, list[Path]]:
    from huggingface_hub import hf_hub_download

    sources = NATIVE_DIALOGUE_SPEC["native_dialogue4"]["sources"]
    paths: dict[str, list[Path]] = {}
    for lang, source in sources.items():
        paths[lang] = [Path(hf_hub_download(
            repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
            filename=filename, local_dir=source_dir / source["repo_id"].replace("/", "--"),
        )) for filename in source["files"]]
    return paths


def _clean_messages(messages) -> list[dict]:
    cleaned = []
    for message in messages if isinstance(messages, list) else ():
        if not isinstance(message, dict):
            return []
        role, content = message.get("role"), message.get("content")
        if role == "system":
            role = "state"
        if role not in {"state", "user", "assistant"} or not isinstance(content, str):
            return []
        content = content.strip()
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned if any(row["role"] == "user" for row in cleaned) and any(
        row["role"] == "assistant" for row in cleaned
    ) else []


def _smol_rows(paths: list[Path]):
    for path in paths:
        row_index = 0
        for batch in parquet_batches(path, ["messages"]):
            for row in batch.to_pylist():
                yield f"{path.name}:{row_index}", _clean_messages(row.get("messages"))
                row_index += 1


def _korean_rows(paths: list[Path]):
    if len(paths) != 1:
        raise ValueError("the registered Korean source must contain exactly one JSONL file")
    with paths[0].open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Korean source row {row_index + 1}") from exc
            prompt, response = row.get("prompt"), row.get("response")
            if isinstance(prompt, str) and isinstance(response, str):
                messages = [
                    {"role": "user", "content": ROLE_PREFIX.sub("", prompt.strip())},
                    {"role": "assistant", "content": ROLE_PREFIX.sub("", response.strip())},
                ]
            else:
                messages = []
            yield f"{paths[0].name}:{row_index}", _clean_messages(messages)


def select_conversations(paths: list[Path], lang: str, tokenizer: NativeDialogueTokenizer,
                         panel_rows, count: int, multiplier: int, maximum_tokens: int):
    candidates = []
    candidate_count = count * multiplier
    rows = _smol_rows(paths) if lang == "en" else _korean_rows(paths)
    scanned = 0
    removed_panel = removed_length = 0
    for identity, messages in rows:
        scanned += 1
        if not messages:
            continue
        # Validity filters must run before hash ranking. Ranking first made long
        # SmolTalk rows occupy most of the bounded candidate heap, so a valid
        # 100k sample could not be produced even though enough short rows exist.
        if any(is_panel_near_duplicate(row["content"], panel_rows) for row in messages):
            removed_panel += 1
            continue
        if dialogue_token_count(tokenizer, messages) > maximum_tokens:
            removed_length += 1
            continue
        score = int.from_bytes(hashlib.sha256(f"{lang}:{identity}".encode()).digest()[:8], "big")
        item = (-score, identity, messages)
        if len(candidates) < candidate_count:
            heapq.heappush(candidates, item)
        elif score < -candidates[0][0]:
            heapq.heapreplace(candidates, item)
    selected = [
        (identity, messages)
        for _, identity, messages in sorted(candidates, key=lambda item: -item[0])[:count]
    ]
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} valid {lang} conversations after scanning {scanned}")
    return selected, {
        "scanned": scanned, "selected": len(selected),
        "panel_removed": removed_panel, "length_removed": removed_length,
    }


def _atomic_handles(output_dir: Path, names: list[str]):
    paths = {name: output_dir / name for name in names}
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    handles = {name: temporary[name].open("w", encoding="utf-8") for name in names}
    return paths, temporary, handles


def write_dataset(output_dir: Path, tokenizer: NativeDialogueTokenizer, selected, panel_rows):
    spec = NATIVE_DIALOGUE_SPEC["native_dialogue4"]
    shards = spec["conversation_shards_per_language"]
    names = []
    for lang in ("en", "ko"):
        for index in range(shards):
            names.extend((f"conversation.{lang}.{index:03d}.train.jsonl",
                          f"conversation.{lang}.{index:03d}.validation.jsonl"))
        names.extend((f"memory.{lang}.train.jsonl", f"memory.{lang}.validation.jsonl"))
    paths, temporary, handles = _atomic_handles(output_dir, names)
    splits = {"train_general": [], "validation_general": [],
              "train_dialogue": [], "validation_dialogue": []}
    statistics = {"conversation": {}, "memory": {}}
    for lang in ("en", "ko"):
        counts = {"train": 0, "validation": 0}
        for item_index, (identity, messages) in enumerate(selected[lang]):
            shard = item_index % shards
            split = "validation" if validation_row(identity, spec["validation_percent"]) else "train"
            name = f"conversation.{lang}.{shard:03d}.{split}.jsonl"
            handles[name].write(json.dumps({"messages": messages}, ensure_ascii=False,
                                           separators=(",", ":")) + "\n")
            counts[split] += 1
        statistics["conversation"][lang] = counts
        memory_counts = {"train": 0, "validation": 0, "panel_removed": 0}
        for index in range(spec["memory_examples_per_language"]):
            messages = memory_messages(lang, index)
            if any(is_panel_near_duplicate(row["content"], panel_rows) for row in messages):
                memory_counts["panel_removed"] += 1
                continue
            if dialogue_token_count(tokenizer, messages) > spec["maximum_screen_tokens"]:
                raise RuntimeError("registered dynamic memory example exceeds the screen context")
            identity = f"native4-memory:{lang}:{index}"
            split = "validation" if validation_row(identity, spec["validation_percent"]) else "train"
            name = f"memory.{lang}.{split}.jsonl"
            handles[name].write(json.dumps({"messages": messages}, ensure_ascii=False,
                                           separators=(",", ":")) + "\n")
            memory_counts[split] += 1
        statistics["memory"][lang] = memory_counts
    for handle in handles.values():
        handle.close()
    for name in names:
        os.replace(temporary[name], paths[name])
    for lang in ("en", "ko"):
        for index in range(shards):
            splits["train_dialogue"].append(f"conversation.{lang}.{index:03d}.train.jsonl")
            splits["validation_dialogue"].append(f"conversation.{lang}.{index:03d}.validation.jsonl")
        splits["train_dialogue"].append(f"memory.{lang}.train.jsonl")
        splits["validation_dialogue"].append(f"memory.{lang}.validation.jsonl")
    return paths, splits, statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    args = parser.parse_args()
    if not args.tokenizer.is_file():
        parser.error("screen tokenizer must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or args.output_dir / ".source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_paths = download_sources(source_dir)
    tokenizer = NativeDialogueTokenizer.load(args.tokenizer)
    panel_rows = panel_fingerprints(args.panel)
    config = NATIVE_DIALOGUE_SPEC["native_dialogue4"]
    selected, selection = {}, {}
    for lang in ("en", "ko"):
        selected[lang], selection[lang] = select_conversations(
            source_paths[lang], lang, tokenizer, panel_rows,
            config["examples_per_language"], config["candidate_multiplier"],
            config["maximum_screen_tokens"],
        )
    paths, splits, statistics = write_dataset(args.output_dir, tokenizer, selected, panel_rows)
    sources = config["sources"]
    manifest = {
        "format": "anima_native_dialogue_data_v2",
        "profile": "small-model-conversation-screen",
        "spec_sha256": spec_sha256(),
        "panel_sha256": sha256_file(args.panel),
        "base_tokenizer_sha256": checkpoint_sha256(args.tokenizer),
        "splits": splits,
        "tokenizer_files": [],
        "source_files": [
            {"repo_id": sources[lang]["repo_id"], "revision": sources[lang]["revision"],
             "file": filename, "license": sources[lang]["license"],
             "sha256": sha256_file(path)}
            for lang in ("en", "ko")
            for filename, path in zip(sources[lang]["files"], source_paths[lang])
        ],
        "statistics": {**statistics, "selection": selection},
        "outputs": {name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
                    for name, path in paths.items()},
    }
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "manifest.json")
    print(json.dumps({
        "selected": {lang: selection[lang]["selected"] for lang in ("en", "ko")},
        "memory": {lang: sum(statistics["memory"][lang][key] for key in ("train", "validation"))
                   for lang in ("en", "ko")},
        "outputs": len(paths),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
