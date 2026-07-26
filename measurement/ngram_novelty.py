#!/usr/bin/env python3
"""Test whether "held-out lines absent from train" are actually novel material.

The model scored BETTER on the unseen-line span (0.299 BPC) than on the raw
held-out split (0.654), which refutes line-level leakage as the explanation for
the raw number and raises the opposite question: is line-level verbatim absence
too weak a novelty criterion? Unseen lines skew long (404 B/line), and a long
line that differs from a training line by a few characters is absent as a LINE
while being almost entirely present as SUBSTRINGS.

Measures, for sampled fixed-width windows, what fraction occur verbatim inside
the train split -- for three spans, so the comparison has a floor and a ceiling:

  unseen  -- the honest span (held-out lines absent from train)
  raw     -- the run's own held-out split
  train   -- sampled from train itself (sanity ceiling: must be ~100%)
"""
import json
import random
import sys

CORPUS = "data/corpus_v2.txt"
UNSEEN = sys.argv[1]
OUT = sys.argv[2]
WIDTHS = (16, 32, 64, 128)
N_SAMPLE = 400
SEED = 1337

with open(CORPUS, "rb") as f:
    data = f.read()
chunk = 1 << 20
train_parts, val_parts = [], []
for i, start in enumerate(range(0, len(data), chunk)):
    (val_parts if i % 10 == 9 else train_parts).append(data[start:start + chunk])
train = b"".join(train_parts)
val_raw = b"".join(val_parts)
del data, train_parts, val_parts
with open(UNSEEN, "rb") as f:
    unseen = f.read()

spans = {"unseen": unseen, "raw": val_raw, "train(ceiling)": train}
rng = random.Random(SEED)
results = {}
for name, span in spans.items():
    results[name] = {}
    for w in WIDTHS:
        hi = len(span) - w
        hits = 0
        for _ in range(N_SAMPLE):
            i = rng.randrange(hi)
            if train.find(span[i:i + w]) != -1:
                hits += 1
        frac = hits / N_SAMPLE
        results[name][w] = frac
        print(f"[{name:15s}] w={w:4d}  in-train {hits:3d}/{N_SAMPLE} = {frac*100:5.1f}%",
              flush=True)

with open(OUT, "w") as f:
    json.dump({"widths": list(WIDTHS), "n_sample": N_SAMPLE, "seed": SEED,
               "in_train_frac": results}, f, indent=2)
print(f"[json] {OUT}")
