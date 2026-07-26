#!/usr/bin/env python3
"""Does the 50% arm use context at all, or is it a byte histogram with a transformer around it?

DATA-6 placed the 50% arm at 88% of its unigram floor and called it the
unigram-floor failure mode. That name is a claim about MECHANISM -- that the model
predicts from byte frequency and ignores what came before -- and BPC alone cannot
show it. A model can sit near the unigram number while still using context, and a
model can beat it while using almost none.

So this measures context use directly. For each arm, on the same novelty-controlled
windows, CE is computed three ways:

  true      the window as it is
  shuffled  the context bytes permuted inside the window, targets untouched -- the
            same byte statistics, no usable order
  uniform   context replaced by uniformly random bytes -- neither statistics nor order

A model that reads context loses a lot when the order is destroyed. A byte histogram
loses nothing, because it was never reading the context. The gap is the measurement;
the absolute CE is not.
"""
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CHUNK = 1 << 20
BLOCK = 256
PROBE = 64
TARGET_WINDOWS = 256
STRIDE = 1021
BATCH = 8
SEED = 1337
LN2 = math.log(2)

CORPORA = [f"{HOME}/data/corpus_merged_25.txt",
           f"{HOME}/data/corpus_merged_50.txt",
           f"{HOME}/data/corpus_merged_dedup.txt"]
ARMS = [
    ("25%", f"{HOME}/checkpoints/arm_s25/final.pt"),
    ("50%", f"{HOME}/checkpoints/arm_s50/final.pt"),
    ("100%", f"{HOME}/checkpoints/arm_a_data/final.pt"),
]


def load_trainer(path):
    spec = spec_from_file_location("clm_trainer", path)
    mod = module_from_spec(spec)
    sys.modules["clm_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def split(path):
    with open(path, "rb") as f:
        data = f.read()
    chunk = min(CHUNK, max(1, len(data) // 10))
    parts = [data[s:s + chunk] for s in range(0, len(data), chunk)]
    return (b"".join(p for i, p in enumerate(parts) if i % 10 != 9),
            b"".join(p for i, p in enumerate(parts) if i % 10 == 9))


def ce_of(model, x, y, device):
    total, ntok = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(x), BATCH):
            xb, yb = x[b:b + BATCH].to(device), y[b:b + BATCH].to(device)
            logits, _, _ = model(xb)
            total += F.cross_entropy(logits.view(-1, model.vocab_size),
                                     yb.reshape(-1), reduction="sum").item()
            ntok += yb.numel()
    return total / ntok


def main():
    out_path = sys.argv[1]
    clm = load_trainer(TRAINER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(SEED)

    trains, full_val = [], None
    for c in CORPORA:
        tr, va = split(c)
        trains.append(tr)
        if c.endswith("corpus_merged_dedup.txt"):
            full_val = va

    windows = []
    for start in range(0, len(full_val) - BLOCK - 2, STRIDE):
        if len(windows) >= TARGET_WINDOWS:
            break
        w = full_val[start:start + BLOCK + 1]
        probes = (w[:PROBE], w[(BLOCK - PROBE) // 2:(BLOCK - PROBE) // 2 + PROBE],
                  w[BLOCK - PROBE:BLOCK])
        if all(tr.find(p) == -1 for p in probes for tr in trains):
            windows.append(w)
    print(f"[span] {len(windows)} novelty-controlled windows", flush=True)

    x = torch.tensor([list(w[:BLOCK]) for w in windows], dtype=torch.long)
    y = torch.tensor([list(w[1:BLOCK + 1]) for w in windows], dtype=torch.long)
    # shuffled: permute each window's context independently; targets stay as they are,
    # so the task is identical and only the readable order is destroyed.
    perm = torch.stack([torch.randperm(BLOCK, generator=g) for _ in range(len(x))])
    x_shuf = torch.gather(x, 1, perm)
    x_unif = torch.randint(0, 256, x.shape, generator=g)

    results = {}
    for name, ckpt_path in ARMS:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        model = clm.ConsciousLM(vocab_size=256, d_model=int(cfg["dim"]),
                                n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                                block_size=BLOCK, dropout=0.0)
        model.load_state_dict(ck["model_state"], strict=False)
        model.to(device).eval()
        true_ce = ce_of(model, x, y, device)
        shuf_ce = ce_of(model, x_shuf, y, device)
        unif_ce = ce_of(model, x_unif, y, device)
        results[name] = {"true_bpc": true_ce / LN2, "shuffled_bpc": shuf_ce / LN2,
                         "uniform_bpc": unif_ce / LN2,
                         "context_gain_bpc": (shuf_ce - true_ce) / LN2,
                         "ckpt_step": ck.get("step")}
        print(f"[{name}] true={true_ce / LN2:.4f} · shuffled={shuf_ce / LN2:.4f} · "
              f"uniform-ctx={unif_ce / LN2:.4f} · context gain={(shuf_ce - true_ce) / LN2:+.4f} BPC",
              flush=True)
        del ck, model

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
