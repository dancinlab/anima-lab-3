#!/usr/bin/env python3
"""Prepare pinned bilingual general and dialogue corpora for native training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC, spec_sha256


PANEL_PATH = Path(__file__).parent / "measurement" / "native_dialogue_panel.json"
WORD = re.compile(r"[0-9a-z가-힣]+", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def words(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.casefold()))


def shingles(tokens: tuple[str, ...], width: int = 3) -> set[tuple[str, ...]]:
    if len(tokens) < width:
        return {tokens} if tokens else set()
    return set(zip(*(tokens[offset:] for offset in range(width))))


def panel_fingerprints(panel_path: Path) -> list[tuple[tuple[str, ...], set[tuple[str, ...]]]]:
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    result = []
    for item in panel["items"]:
        for turn in item["turns"]:
            token_words = words(turn["user"])
            result.append((token_words, shingles(token_words)))
    return result


def is_panel_near_duplicate(text: str, panel_rows) -> bool:
    token_words = words(text)
    if not token_words:
        return False
    joined = " ".join(token_words)
    text_shingles = shingles(token_words)
    for panel_words, panel_shingles in panel_rows:
        panel_joined = " ".join(panel_words)
        if panel_joined and panel_joined in joined:
            return True
        if panel_shingles and len(text_shingles & panel_shingles) / len(panel_shingles) >= 0.6:
            return True
    return False


def validation_row(identifier: str, percentage: int) -> bool:
    value = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")
    return value % 100 < percentage


class SampleWriter:
    def __init__(self, output_dir: Path, limit: int):
        self.limit = limit
        self.counts = {(lang, kind): 0 for lang in ("en", "ko") for kind in ("general", "dialogue")}
        self.paths = {
            key: output_dir / f"tokenizer.{key[0]}.{key[1]}.txt" for key in self.counts
        }
        self.handles = {key: path.open("w", encoding="utf-8") for key, path in self.paths.items()}

    def add(self, lang: str, kind: str, text: str) -> None:
        key = (lang, kind)
        remaining = self.limit - self.counts[key]
        if remaining <= 0:
            return
        piece = text[:remaining]
        self.handles[key].write(piece)
        self.handles[key].write("\n")
        self.counts[key] += len(piece)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def download(repo_id: str, revision: str, filename: str, source_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=source_dir / repo_id.replace("/", "--"),
    ))


def parquet_batches(path: Path, columns: list[str]):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to prepare native dialogue data") from exc
    yield from parquet.ParquetFile(path).iter_batches(batch_size=1024, columns=columns)


def atomic_handles(train_path: Path, validation_path: Path):
    train_tmp = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_tmp = validation_path.with_suffix(validation_path.suffix + ".tmp")
    return train_tmp, validation_tmp, train_tmp.open("w", encoding="utf-8"), validation_tmp.open("w", encoding="utf-8")


def finish_handles(train_tmp: Path, validation_tmp: Path, train_handle, validation_handle,
                   train_path: Path, validation_path: Path) -> None:
    train_handle.close()
    validation_handle.close()
    os.replace(train_tmp, train_path)
    os.replace(validation_tmp, validation_path)


def prepare_general(source_path: Path, lang: str, index: int, output_dir: Path,
                    validation_percent: int, panel_rows, samples: SampleWriter) -> tuple[dict, list[str]]:
    train_path = output_dir / f"general.{lang}.{index:03d}.train.txt"
    validation_path = output_dir / f"general.{lang}.{index:03d}.validation.txt"
    train_tmp, validation_tmp, train_handle, validation_handle = atomic_handles(train_path, validation_path)
    counts = {"train": 0, "validation": 0, "panel_removed": 0, "characters": 0}
    row_index = 0
    for batch in parquet_batches(source_path, ["text"]):
        for row in batch.to_pylist():
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                row_index += 1
                continue
            text = text.strip()
            if is_panel_near_duplicate(text, panel_rows):
                counts["panel_removed"] += 1
                row_index += 1
                continue
            split = "validation" if validation_row(f"{source_path.name}:{row_index}", validation_percent) else "train"
            handle = validation_handle if split == "validation" else train_handle
            handle.write(text)
            handle.write("\n\n")
            counts[split] += 1
            counts["characters"] += len(text)
            if split == "train":
                samples.add(lang, "general", text)
            row_index += 1
    finish_handles(train_tmp, validation_tmp, train_handle, validation_handle, train_path, validation_path)
    return counts, [train_path.name, validation_path.name]


def clean_messages(messages, lang: str):
    field = "content_en" if lang == "en" else "content"
    cleaned = []
    for message in messages or []:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get(field)
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        if content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    roles = [message["role"] for message in cleaned]
    if "user" not in roles or "assistant" not in roles or roles[-1] != "assistant":
        return None
    return cleaned


def prepare_dialogue(source_path: Path, index: int, output_dir: Path,
                     validation_percent: int, panel_rows, samples: SampleWriter) -> tuple[dict, dict[str, list[str]]]:
    results = {}
    paths = {}
    for lang in ("en", "ko"):
        train_path = output_dir / f"dialogue.{lang}.{index:03d}.train.jsonl"
        validation_path = output_dir / f"dialogue.{lang}.{index:03d}.validation.jsonl"
        train_tmp, validation_tmp, train_handle, validation_handle = atomic_handles(train_path, validation_path)
        results[lang] = {"train": 0, "validation": 0, "panel_removed": 0, "invalid": 0, "characters": 0}
        row_index = 0
        for batch in parquet_batches(source_path, ["messages", "custom_id"]):
            for row in batch.to_pylist():
                messages = clean_messages(row.get("messages"), lang)
                if messages is None:
                    results[lang]["invalid"] += 1
                    row_index += 1
                    continue
                if any(is_panel_near_duplicate(message["content"], panel_rows) for message in messages):
                    results[lang]["panel_removed"] += 1
                    row_index += 1
                    continue
                identifier = str(row.get("custom_id") or f"{source_path.name}:{row_index}")
                split = "validation" if validation_row(identifier, validation_percent) else "train"
                handle = validation_handle if split == "validation" else train_handle
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                results[lang][split] += 1
                characters = sum(len(message["content"]) for message in messages)
                results[lang]["characters"] += characters
                if split == "train":
                    for message in messages:
                        samples.add(lang, "dialogue", message["content"])
                row_index += 1
        finish_handles(train_tmp, validation_tmp, train_handle, validation_handle, train_path, validation_path)
        paths[lang] = [train_path.name, validation_path.name]
    return results, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("screen", "target"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    args = parser.parse_args()

    config = NATIVE_DIALOGUE_SPEC["native_dialogue2"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or args.output_dir / ".source"
    source_dir.mkdir(parents=True, exist_ok=True)
    panel_rows = panel_fingerprints(args.panel)
    samples = SampleWriter(
        args.output_dir,
        config["tokenizer_sample_characters_per_language_and_kind"],
    )
    splits = {name: [] for name in (
        "train_general", "validation_general", "train_dialogue", "validation_dialogue"
    )}
    source_manifest = []
    statistics = {"general": {}, "dialogue": {}}

    dialogue = config["sources"]["dialogue"]
    dialogue_files = dialogue["files"][:1] if args.profile == "screen" else dialogue["files"]
    for index, filename in enumerate(dialogue_files):
        source_path = download(dialogue["repo_id"], dialogue["revision"], filename, source_dir)
        source_manifest.append({"repo_id": dialogue["repo_id"], "revision": dialogue["revision"],
                                "file": filename, "sha256": sha256_file(source_path),
                                "license": dialogue["license"]})
        counts, made = prepare_dialogue(
            source_path, index, args.output_dir, config["validation_percent"], panel_rows, samples
        )
        statistics["dialogue"][filename] = counts
        for lang in ("en", "ko"):
            splits["train_dialogue"].append(made[lang][0])
            splits["validation_dialogue"].append(made[lang][1])

    if args.profile == "target":
        general = config["sources"]["general"]
        for lang, filenames in (("en", general["en_files"]), ("ko", general["ko_files"])):
            for index, filename in enumerate(filenames):
                source_path = download(general["repo_id"], general["revision"], filename, source_dir)
                source_manifest.append({"repo_id": general["repo_id"], "revision": general["revision"],
                                        "file": filename, "sha256": sha256_file(source_path),
                                        "license": general["license"]})
                counts, made = prepare_general(
                    source_path, lang, index, args.output_dir, config["validation_percent"],
                    panel_rows, samples,
                )
                statistics["general"][filename] = counts
                splits["train_general"].append(made[0])
                splits["validation_general"].append(made[1])
    samples.close()

    tokenizer_files = [path.name for path in samples.paths.values() if path.stat().st_size > 0]
    output_manifest = {
        "format": "anima_native_dialogue_data_v2",
        "profile": args.profile,
        "spec_sha256": spec_sha256(),
        "panel_sha256": sha256_file(args.panel),
        "splits": splits,
        "tokenizer_files": tokenizer_files,
        "source_files": source_manifest,
        "statistics": statistics,
        "outputs": {},
    }
    for paths in splits.values():
        for relative in paths:
            path = args.output_dir / relative
            output_manifest["outputs"][relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    for relative in tokenizer_files:
        path = args.output_dir / relative
        output_manifest["outputs"][relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "manifest.json")
    print(json.dumps({
        "profile": args.profile,
        "sources": len(source_manifest),
        "outputs": len(output_manifest["outputs"]),
        "panel_removed": sum(
            cell.get("panel_removed", 0)
            for group in statistics.values() for row in group.values()
            for cell in (row.values() if group is statistics["dialogue"] else (row,))
            if isinstance(cell, dict)
        ),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
