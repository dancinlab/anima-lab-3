#!/usr/bin/env python3
"""Is aiden's 300M run learning language, or is its reported BPC unmeasurable?

The nf9 run has been training for 10+ hours toward a 14.6-day finish, and its
reported validation BPC (0.65) was shown in DATA-3 to be unreadable: 82.5% of its
held-out span's 64-byte windows occur verbatim in its own train split, so the
number is part recall. That says nothing about whether the run is learning -- it
says the instrument is broken. This applies the instrument that works.

Same construction as measurement/novel_window_eval.py: a 257-byte window is kept
only when three 64-byte probes (head, middle, tail) are absent from the train
split, so what survives cannot be produced from memory. Scored against the train
split's own floors, which CLAUDE.md requires and which for corpus_v2 are unigram
5.9548 and bigram 3.4920 BPC -- beating unigram means the byte histogram was
learned, beating bigram is the gate a real LM must clear.

CPU only: the GPU is running the training this is asking about.
"""
import json
import math
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/aiden/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm_nf8.py"
CORPUS = f"{HOME}/data/corpus_v2.txt"
CKPT = f"{HOME}/checkpoints/clm_pure_300m_nf9/best.pt"
CHUNK = 1 << 20
BLOCK = 256
PROBE = 64
TARGET_WINDOWS = 256
STRIDE = 1021
BATCH = 4
LN2 = math.log(2)
TRAIN_UNIGRAM, TRAIN_BIGRAM = 5.9548, 3.4920


def main():
    out_path = sys.argv[1]
    spec = spec_from_file_location("nf8_trainer", TRAINER)
    nf8 = module_from_spec(spec)
    sys.modules["nf8_trainer"] = nf8
    spec.loader.exec_module(nf8)
    torch.set_num_threads(6)
    device = torch.device("cpu")

    with open(CORPUS, "rb") as f:
        data = f.read()
    parts = [data[s:s + CHUNK] for s in range(0, len(data), CHUNK)]
    train = b"".join(p for i, p in enumerate(parts) if i % 10 != 9)
    val = b"".join(p for i, p in enumerate(parts) if i % 10 == 9)
    print(f"[split] train={len(train):,} val={len(val):,}", flush=True)

    windows, tested = [], 0
    t0 = time.time()
    for start in range(0, len(val) - BLOCK - 2, STRIDE):
        if len(windows) >= TARGET_WINDOWS:
            break
        tested += 1
        w = val[start:start + BLOCK + 1]
        probes = (w[:PROBE], w[(BLOCK - PROBE) // 2:(BLOCK - PROBE) // 2 + PROBE],
                  w[BLOCK - PROBE:BLOCK])
        if all(train.find(p) == -1 for p in probes):
            windows.append(w)
    keep = len(windows) / max(1, tested)
    print(f"[select] kept {len(windows):,} of {tested:,} = {keep * 100:.1f}% "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not windows:
        print("[verdict] no novelty-controlled window exists in this corpus -- "
              "the run cannot be evaluated for language at this width", flush=True)
        with open(out_path, "w") as f:
            json.dump({"kept": 0, "tested": tested}, f, indent=2)
        return

    x = torch.tensor([list(w[:BLOCK]) for w in windows], dtype=torch.long)
    y = torch.tensor([list(w[1:BLOCK + 1]) for w in windows], dtype=torch.long)

    ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck.get("config", {}) or {}
    model = nf8.ConsciousLM(vocab_size=256, d_model=int(cfg["dim"]),
                            n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                            block_size=BLOCK, dropout=0.0)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    model.to(device).eval()
    print(f"[ckpt] step={ck.get('step')} cells={len(ck.get('mitosis_cells', []))} "
          f"· keys missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    total, ntok = 0.0, 0
    t = time.time()
    with torch.no_grad():
        for b in range(0, len(windows), BATCH):
            logits, _, _ = model(x[b:b + BATCH])
            total += F.cross_entropy(logits.view(-1, model.vocab_size),
                                     y[b:b + BATCH].reshape(-1), reduction="sum").item()
            ntok += y[b:b + BATCH].numel()
    ce = total / ntok
    bpc = ce / LN2
    gate = "PASS" if bpc < TRAIN_BIGRAM else "FAIL"
    print(f"[nf9] novelty-controlled BPC={bpc:.4f} (CE={ce:.4f} nats, {ntok:,} tokens, "
          f"{time.time() - t:.0f}s)", flush=True)
    print(f"[gate] vs train bigram {TRAIN_BIGRAM} = {bpc / TRAIN_BIGRAM * 100:.0f}% -> {gate} "
          f"· vs unigram {TRAIN_UNIGRAM} = {bpc / TRAIN_UNIGRAM * 100:.0f}%", flush=True)

    with open(out_path, "w") as f:
        json.dump({"bpc": bpc, "ce_nats": ce, "tokens": ntok, "ckpt_step": ck.get("step"),
                   "cells": len(ck.get("mitosis_cells", [])), "keep_rate": keep,
                   "kept": len(windows), "tested": tested, "gate": gate,
                   "train_unigram_bpc": TRAIN_UNIGRAM, "train_bigram_bpc": TRAIN_BIGRAM},
                  f, indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
