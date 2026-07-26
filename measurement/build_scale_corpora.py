#!/usr/bin/env python3
"""Build 25% and 50% subsets of the merged corpus, and measure each one's floors.

DATA-4 concluded that unique data is the binding constraint at 27.7M: 2.28x the
data roughly halved the ratio to the bigram floor. That is one point of evidence
from one comparison. A scaling curve tests it directly -- if data is still
binding, the ratio keeps falling from 25% to 50% to 100%; if it flattens between
50% and 100%, the constraint has moved elsewhere at this model size.

Subsets are taken by LINE INDEX MODULO, not by prefix: the merged corpus is a
concatenation of sources, so the first 25% of its bytes is mostly one source and
would confound "less data" with "different distribution".

Floors are measured on each subset's own TRAIN split (the same interleaved 1MB
every-10th split the trainer uses). A ratio computed against another corpus's
floor is meaningless -- the merged corpus reads 3.6010 where the single one reads
3.5044.
"""
import json
import math
import sys
from collections import Counter

SRC = "data/corpus_merged_dedup.txt"
CHUNK = 1 << 20


def train_split(data):
    parts = [data[s:s + CHUNK] for s in range(0, len(data), CHUNK)]
    return b"".join(p for i, p in enumerate(parts) if i % 10 != 9)


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
    out_json = sys.argv[1]
    with open(SRC, "rb") as f:
        lines = f.read().split(b"\n")
    print(f"[src] {SRC}: {len(lines):,} lines", flush=True)

    result = {}
    for frac, modulo, path in ((0.25, 4, "data/corpus_merged_25.txt"),
                               (0.50, 2, "data/corpus_merged_50.txt"),
                               (1.00, 1, SRC)):
        if modulo > 1:
            kept = b"\n".join(ln for i, ln in enumerate(lines) if i % modulo == 0)
            with open(path, "wb") as f:
                f.write(kept)
            data = kept
        else:
            data = b"\n".join(lines)
        train = train_split(data)
        nb, uni, big = floors(train)
        result[path] = {"target_frac": frac, "total_bytes": len(data),
                        "train_bytes": len(train), "distinct_byte_values": nb,
                        "unigram_bpc": uni, "bigram_bpc": big}
        print(f"[{frac:>5.0%}] {path}: total={len(data):,} train={len(train):,} "
              f"· {nb} byte values · unigram {uni:.4f} · bigram {big:.4f} BPC", flush=True)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] {out_json}", flush=True)


if __name__ == "__main__":
    main()
