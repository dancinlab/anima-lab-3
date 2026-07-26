#!/usr/bin/env python3
"""Does the 3-arm runs' `unseen lines only` span actually contain novel material?

DATA-3 measured that on raw corpus_v2 the line-level filter moved 32-byte
familiarity the WRONG way (79.5% -> 91.8%): filtering by line identity selects
long lines, and long lines are the most near-duplicated material. The arms'
verdicts are read off BPC measured on exactly that filter
(train_conscious_lm.py:899-905), so the same check has to run on the corpora the
arms actually use before their numbers can be trusted.

Reproduces the arms' split and filter verbatim, then reports for each span what
fraction of w-byte windows occur verbatim inside train:

  unseen -- the span the arm is scored on
  raw    -- the held-out split before the filter
  train  -- sampled from train itself (sanity ceiling: must be ~100%)
"""
import json
import math
import random
import sys
from collections import Counter

CHUNK = 1 << 20
WIDTHS = (16, 32, 64, 128)
N_SAMPLE = 400
SEED = 1337


def split_like_arm(path):
    """Reproduce train_conscious_lm.py's split + `unseen lines only` filter."""
    with open(path, "rb") as f:
        data = f.read()
    train_parts, val_parts = [], []
    for i, start in enumerate(range(0, len(data), CHUNK)):
        (val_parts if i % 10 == 9 else train_parts).append(data[start:start + CHUNK])
    train = b"".join(train_parts)
    val_raw = b"".join(val_parts)
    train_lines = set(train.split(b"\n"))
    unseen = b"\n".join(
        ln for ln in val_raw.split(b"\n")
        if len(ln) > 8 and ln not in train_lines
    )
    return train, val_raw, unseen, train_lines


def floors(train):
    uni = Counter(train)
    tot = sum(uni.values())
    unigram = -sum(c / tot * math.log2(c / tot) for c in uni.values())
    bi = Counter(zip(train, train[1:]))
    ctx = Counter(train[:-1])
    pairs = sum(bi.values())
    bigram = -sum(c / pairs * math.log2(c / ctx[a]) for (a, _b), c in bi.items())
    return len(uni), unigram, bigram


def main():
    path, out = sys.argv[1], sys.argv[2]
    train, val_raw, unseen, train_lines = split_like_arm(path)
    print(f"[{path}]", flush=True)
    print(f"  train={len(train):,}  val_raw={len(val_raw):,}  "
          f"unseen={len(unseen):,} ({len(unseen)/len(val_raw)*100:.1f}% of held-out)",
          flush=True)

    val_lines = val_raw.split(b"\n")
    in_train = sum(1 for ln in val_lines if ln in train_lines)
    unseen_lines = unseen.split(b"\n")
    avg_len = len(unseen) / max(1, len(unseen_lines))
    print(f"  held-out lines in train: {in_train:,}/{len(val_lines):,} "
          f"= {in_train/max(1,len(val_lines))*100:.1f}%  ·  "
          f"unseen span avg line = {avg_len:.0f} B", flush=True)

    nb, uni, big = floors(train)
    print(f"  train floors: {nb} byte values · unigram {uni:.4f} · bigram {big:.4f} BPC",
          flush=True)

    rng = random.Random(SEED)
    spans = {"unseen": unseen, "raw": val_raw, "train(ceiling)": train}
    table = {}
    for name, span in spans.items():
        table[name] = {}
        for w in WIDTHS:
            hi = len(span) - w
            if hi <= 0:
                continue
            hits = sum(1 for _ in range(N_SAMPLE)
                       if train.find(span[(i := rng.randrange(hi)):i + w]) != -1)
            table[name][w] = hits / N_SAMPLE
            print(f"  [{name:15s}] w={w:4d}  in-train {hits:3d}/{N_SAMPLE} = "
                  f"{hits/N_SAMPLE*100:5.1f}%", flush=True)

    with open(out, "w") as f:
        json.dump({"corpus": path, "train_bytes": len(train),
                   "val_raw_bytes": len(val_raw), "unseen_bytes": len(unseen),
                   "unseen_frac_of_heldout": len(unseen) / len(val_raw),
                   "heldout_lines_in_train_frac": in_train / max(1, len(val_lines)),
                   "unseen_avg_line_bytes": avg_len,
                   "distinct_byte_values": nb, "unigram_bpc": uni, "bigram_bpc": big,
                   "in_train_frac": table, "widths": list(WIDTHS),
                   "n_sample": N_SAMPLE, "seed": SEED}, f, indent=2)
    print(f"  [json] {out}", flush=True)


if __name__ == "__main__":
    main()
