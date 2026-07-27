#!/usr/bin/env python3
"""Measure the natural corpus's own train-split floors.

A ratio against another corpus's floor is meaningless and the gate is a ratio,
so lambda1 cannot adjudicate arm_nat until this exists. Same split the trainer
uses -- interleaved 1MB chunks, every 10th held out.
"""
import json
import math
import sys
from collections import Counter

CHUNK = 1 << 20


def main():
    path, out = sys.argv[1], sys.argv[2]
    data = open(path, "rb").read()
    parts = [data[s:s + CHUNK] for s in range(0, len(data), CHUNK)]
    train = b"".join(p for i, p in enumerate(parts) if i % 10 != 9)
    uni = Counter(train)
    tot = sum(uni.values())
    unigram = -sum(c / tot * math.log2(c / tot) for c in uni.values())
    bi = Counter(zip(train, train[1:]))
    ctx = Counter(train[:-1])
    pairs = sum(bi.values())
    bigram = -sum(c / pairs * math.log2(c / ctx[a]) for (a, _b), c in bi.items())
    print(f"[{path.split('/')[-1]}] bytes={len(data):,} train={len(train):,} · "
          f"{len(uni)} byte values · unigram {unigram:.4f} · bigram {bigram:.4f} BPC",
          flush=True)
    json.dump({"path": path, "bytes": len(data), "train_bytes": len(train),
               "byte_values": len(uni), "unigram_bpc": unigram,
               "bigram_bpc": bigram}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
