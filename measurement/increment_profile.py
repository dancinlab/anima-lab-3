#!/usr/bin/env python3
"""What is IN the lines that adding broke the model?

DATA-6 §4.5: 25% is a strict subset of 50%, so the 50% corpus is the 25% corpus plus
one increment of lines — and the model trained on the superset fails the language gate
while the subset-trained one passes it. Every check so far compared whole corpora and
found them indistinguishable; the increment itself was never looked at separately.

Modulo sampling takes lines by index: base = i%4==0, increment = i%4==2 (their union is
i%2==0 = the 50% corpus). If the file has any period-4 structure, those two phases can
draw different material even though the whole-corpus histograms match.

Profiles base vs increment side by side: size, line length, script mix (ASCII vs Hangul
lead bytes vs other), entropy floors, and how much of the increment is a near-duplicate
of the base at substring level.
"""
import math
import random
import sys
from collections import Counter
from pathlib import Path

SRC = "data/corpus_merged_dedup.txt"
PROBE, N_SAMPLE, SEED = 64, 400, 1337


def floors(b):
    uni = Counter(b)
    tot = sum(uni.values())
    u = -sum(c / tot * math.log2(c / tot) for c in uni.values())
    bi = Counter(zip(b, b[1:]))
    ctx = Counter(b[:-1])
    pairs = sum(bi.values())
    g = -sum(c / pairs * math.log2(c / ctx[a]) for (a, _x), c in bi.items())
    return len(uni), u, g


def script_mix(b):
    ascii_n = sum(1 for x in b if x < 0x80)
    # UTF-8 lead bytes 0xEA-0xED cover the Hangul syllable block
    hangul_lead = sum(1 for x in b if 0xEA <= x <= 0xED)
    return ascii_n / len(b), hangul_lead / len(b)


def profile(name, lines):
    blob = b"\n".join(lines)
    nonempty = [x for x in lines if x]
    nb, u, g = floors(blob)
    a, h = script_mix(blob)
    print(f"[{name}]")
    print(f"  lines={len(lines):,} distinct={len(set(lines)):,} bytes={len(blob):,}")
    print(f"  mean len={len(blob)/max(1,len(lines)):.1f}B · non-empty={len(nonempty):,} "
          f"({len(nonempty)/max(1,len(lines))*100:.1f}%)")
    print(f"  script: ASCII {a*100:.1f}% · Hangul-lead {h*100:.1f}%")
    print(f"  floors: {nb} byte values · unigram {u:.4f} · bigram {g:.4f} BPC")
    return blob


def main():
    lines = Path(SRC).read_bytes().split(b"\n")
    base = [ln for i, ln in enumerate(lines) if i % 4 == 0]
    incr = [ln for i, ln in enumerate(lines) if i % 4 == 2]
    print(f"[src] {SRC}: {len(lines):,} lines → base(i%4==0) {len(base):,} · "
          f"increment(i%4==2) {len(incr):,}\n", flush=True)
    base_b = profile("base = the 25% corpus", base)
    print()
    incr_b = profile("increment = what 50% adds", incr)

    rng = random.Random(SEED)
    hits = 0
    for _ in range(N_SAMPLE):
        i = rng.randrange(len(incr_b) - PROBE)
        if base_b.find(incr_b[i:i + PROBE]) != -1:
            hits += 1
    print(f"\n[overlap] {PROBE}B windows of the increment already inside the base: "
          f"{hits}/{N_SAMPLE} = {hits/N_SAMPLE*100:.1f}%")
    print("  (high = the increment is mostly re-runs of what the base already taught;"
          " low = it is genuinely new material)")


if __name__ == "__main__":
    main()
