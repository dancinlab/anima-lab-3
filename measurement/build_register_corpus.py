#!/usr/bin/env python3
"""Build disjoint, line-deduplicated train and fresh pools from JSONL prose.

The source record, split rule, and byte targets are inputs so another natural
register does not require another one-off cleaner. Records are assigned by a
stable hash of their source id; all lines from one document stay on one side.
Exact normalized-line deduplication is global across both outputs, preventing a
fresh receipt from reusing a line that entered the training corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


SPACE = re.compile(r"[ \t\u00a0]+")
SENTENCE = re.compile(r"(?<=[.!?。！？]|다|요)\s+")
HANGUL = re.compile(r"[가-힣]")
MIN_BYTES = 30
MAX_BYTES = 400


def record_side(record_id: str, train_fraction: float = 0.5) -> str:
    """Assign whole documents by hash without depending on source order."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if draw < train_fraction else "fresh"


def normalized_lines(text: str):
    """Yield conservative prose lines suitable for the existing λ instruments."""
    text = unicodedata.normalize("NFC", text.replace("\r", "\n"))
    for paragraph in text.split("\n"):
        paragraph = SPACE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        pieces = SENTENCE.split(paragraph) if len(paragraph.encode()) > MAX_BYTES else [paragraph]
        for piece in pieces:
            piece = SPACE.sub(" ", piece).strip()
            raw = piece.encode("utf-8")
            if not MIN_BYTES <= len(raw) <= MAX_BYTES:
                continue
            if len(HANGUL.findall(piece)) * 3 < len(raw) * 0.20:
                continue
            yield raw


def build(source: Path, train_path: Path, fresh_path: Path, train_target_bytes: int,
          fresh_target_bytes: int | None = None, train_fraction: float = 0.5):
    fresh_target_bytes = fresh_target_bytes or train_target_bytes
    targets = {"train": train_target_bytes, "fresh": fresh_target_bytes}
    outputs = {"train": train_path, "fresh": fresh_path}
    handles = {}
    sizes = {"train": 0, "fresh": 0}
    records = {"train": 0, "fresh": 0}
    lines = {"train": 0, "fresh": 0}
    duplicate_lines = 0
    invalid_records = 0
    seen = set()
    try:
        for side, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[side] = path.open("wb")
        with source.open(encoding="utf-8") as stream:
            for raw_record in stream:
                if all(sizes[side] >= targets[side] for side in sizes):
                    break
                try:
                    row = json.loads(raw_record)
                    record_id = str(row["id"])
                    text = row["text"]
                    if not isinstance(text, str):
                        raise TypeError("text is not a string")
                except (KeyError, TypeError, json.JSONDecodeError):
                    invalid_records += 1
                    continue
                side = record_side(record_id, train_fraction)
                if sizes[side] >= targets[side]:
                    continue
                wrote_record = False
                for line in normalized_lines(text):
                    fingerprint = hashlib.sha256(line).digest()
                    if fingerprint in seen:
                        duplicate_lines += 1
                        continue
                    seen.add(fingerprint)
                    if sizes[side] + len(line) + 1 > targets[side]:
                        continue
                    handles[side].write(line + b"\n")
                    sizes[side] += len(line) + 1
                    lines[side] += 1
                    wrote_record = True
                if wrote_record:
                    records[side] += 1
    finally:
        for handle in handles.values():
            handle.close()
    # A line is indivisible, so the closest valid size can be up to one maximum
    # line below the target. That bounded slack is not source exhaustion.
    if any(sizes[side] < targets[side] - (MAX_BYTES + 1) for side in sizes):
        raise RuntimeError(f"source exhausted before both targets: {sizes}")
    return {
        "source": str(source),
        "target_bytes": targets,
        "split": f"sha256(record id) uniform draw <{train_fraction} => train, otherwise fresh",
        "normalization": "NFC, whitespace fold, 30..400 UTF-8 bytes, >=20% Hangul bytes",
        "global_line_dedup": True,
        "sizes": sizes,
        "records": records,
        "lines": lines,
        "duplicate_lines_dropped": duplicate_lines,
        "invalid_records": invalid_records,
        "sha256": {
            side: hashlib.sha256(path.read_bytes()).hexdigest()
            for side, path in outputs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--train-target-bytes", type=int, required=True)
    parser.add_argument("--fresh-target-bytes", type=int, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-license", required=True)
    args = parser.parse_args()
    receipt = build(args.source, args.train, args.fresh, args.train_target_bytes,
                    args.fresh_target_bytes, args.train_fraction)
    receipt["source_url"] = args.source_url
    receipt["source_revision"] = args.source_revision
    receipt["source_license"] = args.source_license
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
