#!/usr/bin/env python3
"""Build the complement of the failing 50% subset, and measure its own floors.

DATA-6 has crossed off volume, composition, structure, the objective phase,
checkpoint selection, run-to-run noise, exposure per byte, and batch order. One
variable was never varied: WHICH lines the failing subset contains. Every 50% run
used the same modulo phase (i%2==0). Its complement (i%2==1) is the same size,
built by the same rule, and shares no line with it.

That makes it the one control that holds the failing SIZE fixed and swaps the
CONTENT:

  complement PASSES -> the failure belongs to that particular half, not to "half
                       of this corpus". A strict superset of a passing set failing
                       then has to be explained by the specific lines, and §4.6's
                       whole-corpus comparisons were looking at the wrong grain.
  complement FAILS  -> both halves fail at this size while 25% and 100% pass, so
                       the effect is non-monotonic in size and not about content
                       at all -- which points at capacity/size interaction and
                       makes the "added lines broke it" framing wrong.

Either outcome removes a live explanation, which is why it is worth 25 minutes.

Floors are measured on this subset's OWN train split (the same interleaved 1MB
every-10th split the trainer uses) -- a ratio against another corpus's floor is
meaningless, and the gate is a ratio.
"""
import json
import math
import sys
from collections import Counter

SRC = "data/corpus_merged_dedup.txt"
OUT = "data/corpus_merged_50c.txt"
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
    out_json = sys.argv[1] if len(sys.argv) > 1 else "logs/complement_half.json"
    lines = open(SRC, "rb").read().split(b"\n")
    print(f"[src] {SRC}: {len(lines):,} lines", flush=True)

    orig = b"\n".join(ln for i, ln in enumerate(lines) if i % 2 == 0)
    comp = b"\n".join(ln for i, ln in enumerate(lines) if i % 2 == 1)
    open(OUT, "wb").write(comp)

    # Disjointness is the point of the control, so it gets asserted, not assumed.
    # Empty lines are shared by construction and are not content -- excluded.
    a = {ln for i, ln in enumerate(lines) if i % 2 == 0 and ln}
    b = {ln for i, ln in enumerate(lines) if i % 2 == 1 and ln}
    shared = a & b
    print(f"[disjoint] original {len(a):,} distinct · complement {len(b):,} distinct · "
          f"shared {len(shared):,} = {len(shared)/max(1,len(b))*100:.2f}% of the complement",
          flush=True)

    result = {}
    for name, data, path in (("50", orig, "data/corpus_merged_50.txt"),
                             ("50c", comp, OUT)):
        tr = train_split(data)
        nb, u, g = floors(tr)
        size_ratio = len(data) / len(orig)
        print(f"[{name}] {path}: {len(data):,} bytes ({size_ratio*100:.1f}% of the original half) · "
              f"train {len(tr):,} · {nb} byte values · unigram {u:.4f} · bigram {g:.4f} BPC",
              flush=True)
        result[name] = {"path": path, "bytes": len(data), "train_bytes": len(tr),
                        "size_ratio_to_50": size_ratio, "byte_values": nb,
                        "unigram_bpc": u, "bigram_bpc": g}
    result["shared_distinct_lines"] = len(shared)
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"[json] {out_json}", flush=True)


if __name__ == "__main__":
    main()
