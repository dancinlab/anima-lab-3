#!/usr/bin/env python3
"""Measure the nf9 300M run's controls, so its rows stop being DIRECTIONAL.

gate.py leaves nf9 as DIRECTIONAL rather than FAIL because corpus_v2 has no
control measurement, and a missing control is not a passed control. That is the
one arm where it matters most: nf9's dashboard read 0.65 BPC for ten hours
while the honest span read 3.99 (DATA-7), so it is the run whose number was
least constrained and whose controls were never taken.

Same three controls as panel.py -- rho-init (identically-shaped model, random
weights), rho-shuffle (context positions permuted, targets untouched), rho-align
(windows re-selected at a different stride phase, reported as stability). Same
misattribution guard: the checkpoint's sha256, because a step number does not
identify a file.

Runs on CPU deliberately. The training job holds 8.9 GB of aiden's 12.2 GB card;
a 300M eval alongside it could OOM two days of training, and no measurement is
worth that. CPU costs minutes and risks nothing.

Novelty screening uses corpus_v2's own train split -- the span DATA-7 scored --
so the value here is directly comparable to its 3.9881 (step 12,000) and 3.8583
(step 14,000), and best.pt has since advanced to step 20,000.
"""
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HOME = "/home/aiden/anima-clm-pure"
CORPUS = f"{HOME}/data/corpus_v2.txt"
CKPT = f"{HOME}/checkpoints/clm_pure_300m_nf9/best.pt"
CHUNK = 1 << 20
BLOCK, PROBE = 256, 64
TARGET_WINDOWS, MAX_CANDIDATES = 256, 60_000
STRIDE = 1021
BATCH = 4
LN2 = math.log(2)
SHUFFLE_SEED, INIT_SEED = 20260727, 1337
BIGRAM_FLOOR = 3.4920      # corpus_v2 train split, CLAUDE.md


def model_class():
    """Import the fork's model. Deferred because the module lives on the host's
    path, not this file's -- resolving it at call time keeps the import legal
    without a linter suppression."""
    if HOME not in sys.path:
        sys.path.insert(0, HOME)
    from conscious_lm import ConsciousLM
    return ConsciousLM


def sha256_streaming(path, chunk=1 << 22):
    """Hash without holding the file. This checkpoint is 6.1 GB and aiden has
    ~7 GB free while training: reading it whole put the process into swap and it
    burned 40 minutes at 64s of CPU. Stream it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def split(path):
    data = open(path, "rb").read()
    chunk = min(CHUNK, max(1, len(data) // 10))
    parts = [data[s:s + chunk] for s in range(0, len(data), chunk)]
    return (b"".join(p for i, p in enumerate(parts) if i % 10 != 9),
            b"".join(p for i, p in enumerate(parts) if i % 10 == 9))


def select_windows(val, train, offset):
    windows, tested = [], 0
    for start in range(offset, len(val) - BLOCK - 2, STRIDE):
        if len(windows) >= TARGET_WINDOWS or tested >= MAX_CANDIDATES:
            break
        tested += 1
        w = val[start:start + BLOCK + 1]
        mid = (BLOCK - PROBE) // 2
        probes = (w[:PROBE], w[mid:mid + PROBE], w[BLOCK - PROBE:BLOCK])
        if all(train.find(p) == -1 for p in probes):
            windows.append(w)
    return windows, tested


def tensors(windows):
    return (torch.tensor([list(w[:BLOCK]) for w in windows], dtype=torch.long),
            torch.tensor([list(w[1:BLOCK + 1]) for w in windows], dtype=torch.long))


def score(model, x, y, label):
    total, ntok, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for b in range(0, len(x), BATCH):
            out = model(x[b:b + BATCH])
            logits = out[0] if isinstance(out, tuple) else out
            total += F.cross_entropy(logits.view(-1, 256),
                                     y[b:b + BATCH].reshape(-1),
                                     reduction="sum").item()
            ntok += y[b:b + BATCH].numel()
    bpc = total / ntok / LN2
    print(f"    {label}: {bpc:.4f} BPC ({time.time() - t0:.0f}s)", flush=True)
    return bpc


def build(cfg, state=None):
    m = model_class()(vocab_size=256, d_model=int(cfg["dim"]),
                      n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                      block_size=BLOCK, dropout=0.0)
    if state is not None:
        m.load_state_dict(state, strict=False)
    return m.eval()


def main():
    out_path = sys.argv[1]
    print("[device] cpu (deliberate: the training job owns the GPU)", flush=True)
    train, val = split(CORPUS)
    print(f"[split] corpus_v2 train={len(train):,} val={len(val):,}", flush=True)

    t0 = time.time()
    win, tested = select_windows(val, train, 0)
    win_alt, tested_alt = select_windows(val, train, STRIDE // 2)
    print(f"[select] primary {len(win)}/{tested} = {len(win)/max(1,tested)*100:.1f}% · "
          f"alt-phase {len(win_alt)}/{tested_alt} = {len(win_alt)/max(1,tested_alt)*100:.1f}% "
          f"({time.time()-t0:.0f}s). A window is kept only if 3x{PROBE}B probes are "
          f"absent from corpus_v2's train split.", flush=True)
    if not win:
        print("[verdict] no novelty-controlled window exists at this width -- "
              "the measurement cannot be made, and that is the result.", flush=True)
        json.dump({"kept": 0, "tested": tested}, open(out_path, "w"), indent=2)
        return

    x, y = tensors(win)
    x_alt, y_alt = tensors(win_alt)
    g = torch.Generator().manual_seed(SHUFFLE_SEED)
    x_shuf = torch.stack([row[torch.randperm(BLOCK, generator=g)] for row in x])

    print(f"[hash] streaming sha256 of {CKPT.split('/')[-1]} "
          f"({os.path.getsize(CKPT) / 1e9:.1f} GB)...", flush=True)
    sha = sha256_streaming(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]
    step = ck.get("step")
    print(f"[ckpt] step={step} sha={sha} dim={cfg['dim']} L={cfg['layers']} "
          f"H={cfg['heads']}", flush=True)

    model = build(cfg, ck["model_state"])
    value = score(model, x, y, "value")
    shuf = score(model, x_shuf, y, "ctrl rho-shuffle")
    align = score(model, x_alt, y_alt, "stability rho-align")
    del model, ck

    torch.manual_seed(INIT_SEED)
    init = score(build(cfg), x, y, "ctrl rho-init")

    worst = min(shuf, init)
    res = {"_select": {"kept": len(win), "tested": tested,
                       "keep_rate": len(win) / tested, "kept_alt": len(win_alt),
                       "probe_bytes": PROBE, "block": BLOCK,
                       "shuffle_seed": SHUFFLE_SEED, "init_seed": INIT_SEED,
                       "corpus": "corpus_v2.txt", "device": "cpu"},
           f"nf9_{step // 1000}k": {
               "bpc": value, "ckpt_step": step, "ckpt_sha256_16": sha,
               "ctrl_shuffle_bpc": shuf, "ctrl_init_bpc": init,
               "stability_alt_phase_bpc": align,
               "stability_delta": abs(align - value),
               "collapse_delta_shuffle": shuf - value,
               "collapse_delta_init": init - value,
               "ratio_over_worst_control": worst / value,
               "controls_collapsed": shuf > value and init > value,
               "bigram_floor": BIGRAM_FLOOR,
               "ratio_to_bigram": value / BIGRAM_FLOOR}}
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"[nf9] value={value:.4f} ({value / BIGRAM_FLOOR * 100:.0f}% of its "
          f"{BIGRAM_FLOOR} floor) · shuffle Δ{shuf - value:+.4f} · "
          f"init Δ{init - value:+.4f} · stability |Δ|{abs(align - value):.4f} · "
          f"ratio {worst / value:.2f}x", flush=True)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
