#!/usr/bin/env python3
"""Quantify how much of aiden's nf9 validation span is memorisable from train.

Reproduces train_conscious_lm_nf8.py's split EXACTLY (1MB chunks, every 10th
held out, no line-level dedup) on the byte-identical corpus, then measures:

  1. what fraction of held-out LINES occur verbatim in the train split
     (and what fraction of held-out BYTES those lines account for)
  2. the order-0 (unigram) and order-1 (bigram) BPC floors of THIS train split
     -- baselines must be measured on the split in use, never assumed
  3. writes the honest eval span: the first 262,144 bytes of held-out material
     built ONLY from lines absent from train, so the model's CE can be
     re-measured on material it cannot have memorised.

Output is a JSON blob plus a human-readable summary.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

CORPUS = Path("data/corpus_v2.txt")
OUT_SPAN = Path(sys.argv[1] if len(sys.argv) > 1 else "unseen_val_span.bin")
OUT_JSON = OUT_SPAN.with_suffix(".json")
SPAN_BYTES = 262_144
CHUNK = 1 << 20

data = CORPUS.read_bytes()
n = len(data)

train_parts, val_parts = [], []
for i, start in enumerate(range(0, n, CHUNK)):
    (val_parts if i % 10 == 9 else train_parts).append(data[start:start + CHUNK])
train = b"".join(train_parts)
val = b"".join(val_parts)
print(f"[split] total={n:,} train={len(train):,} val={len(val):,} bytes")
assert len(train) + len(val) == n

# --- 1. line-level leakage ---------------------------------------------------
train_lines = set(train.split(b"\n"))
val_lines = val.split(b"\n")
print(f"[lines] train distinct={len(train_lines):,} val total={len(val_lines):,}")

seen_n = seen_bytes = 0
unseen_lines = []
for ln in val_lines:
    if ln in train_lines:
        seen_n += 1
        seen_bytes += len(ln) + 1
    else:
        unseen_lines.append(ln)

line_leak = seen_n / max(1, len(val_lines))
byte_leak = seen_bytes / max(1, len(val))
print(f"[leak] lines occurring verbatim in train: {seen_n:,}/{len(val_lines):,} "
      f"= {line_leak * 100:.1f}%")
print(f"[leak] bytes covered by those lines: {seen_bytes:,}/{len(val):,} "
      f"= {byte_leak * 100:.1f}%")

# --- 2. measured entropy floors of THIS train split --------------------------
uni = Counter(train)
tot = sum(uni.values())
unigram_bpc = -sum(c / tot * math.log2(c / tot) for c in uni.values())

bi = Counter(zip(train, train[1:]))
ctx = Counter(train[:-1])
bipairs = sum(bi.values())
bigram_bpc = -sum(
    c / bipairs * math.log2(c / ctx[a]) for (a, _b), c in bi.items()
)
print(f"[floor] distinct byte values={len(uni)} "
      f"unigram={unigram_bpc:.4f} BPC  bigram={bigram_bpc:.4f} BPC")

# --- 3. honest eval span (unseen lines only) --------------------------------
span = bytearray()
used = 0
for ln in unseen_lines:
    span += ln + b"\n"
    used += 1
    if len(span) >= SPAN_BYTES:
        break
span = bytes(span[:SPAN_BYTES])
OUT_SPAN.write_bytes(span)
print(f"[span] wrote {len(span):,} bytes from {used:,} unseen lines -> {OUT_SPAN}")

result = {
    "corpus": str(CORPUS),
    "total_bytes": n,
    "train_bytes": len(train),
    "val_bytes": len(val),
    "val_lines_total": len(val_lines),
    "val_lines_in_train": seen_n,
    "line_leak_frac": line_leak,
    "val_bytes_from_leaked_lines": seen_bytes,
    "byte_leak_frac": byte_leak,
    "train_distinct_lines": len(train_lines),
    "distinct_byte_values": len(uni),
    "train_unigram_bpc": unigram_bpc,
    "train_bigram_bpc": bigram_bpc,
    "unseen_span_bytes": len(span),
    "unseen_span_lines": used,
    "unseen_span_path": str(OUT_SPAN),
}
OUT_JSON.write_text(json.dumps(result, indent=2))
print(f"[json] {OUT_JSON}")
