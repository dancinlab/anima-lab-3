#!/usr/bin/env python3
"""Run every arm as a PANEL AXIS, with the control set the sibling repo requires.

gate.py imported the shape of one instrument (rho-weave). This imports the
contract behind all seven of them (rho-form/fan/leap/weave/store/tether/self in
anima's cli/rho_axon.py):

    value + N controls that must ALL collapse + a ratio over the worst control

and the metalaw the panel states out loud: **the value is tunable, only the
collapse margin over the controls is earned**. Our floors are analytic --
computed from the corpus, never from a forward pass -- so nothing we have ever
reported measured what this architecture scores on this span WITHOUT training.
That is the ablation every anima axis carries and the one we lack.

Three controls, and what each one kills:

  A. rho-init      an identically-shaped model with random weights, same span,
     (ablation)    same forward path. This is the empirical no-learning reading.
                   An analytic bigram floor cannot see tokenizer, architecture
                   or evaluation-harness contributions; this does. If a trained
                   arm does not clearly beat it, the arm learned nothing that
                   this harness can detect.
  B. rho-shuffle   the context bytes inside each window are permuted while the
     (collapse)    targets stay put. A model that ignores context is unaffected,
                   so failing to degrade here means the score is a byte
                   histogram wearing a transformer.
  C. rho-align     the window set is re-selected at a different stride phase.
     (stability)   NOT a collapse control -- it must come out the SAME. A score
                   that moves with where the windows start is a property of the
                   selection, not of the model. Reported as stability, never as
                   a passed control, because calling a stability check a
                   collapse control is the unregistered-extra-hurdle mistake
                   rho-self's docstring warns about.

Also carries the misattribution guard from HYPOTHESES/CLAUDE.md lesson 2 --
"which ckpt (sha) was that number" -- by hashing every checkpoint file it reads.
This repo has already published one wrong checkpoint claim; a step number alone
does not identify a file.

Ratio bar stays PROSPECTIVE, as in gate.py: reported, never deciding.
"""
import hashlib
import json
import os
import math
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CHUNK = 1 << 20
BLOCK, PROBE = 256, 64
TARGET_WINDOWS, MAX_CANDIDATES = 256, 20_000
STRIDE = 1021          # prime, so candidates do not align with any 1MB structure
BATCH = 8
LN2 = math.log(2)
SHUFFLE_SEED, INIT_SEED = 20260727, 1337
RATIO = 3.0            # prospective only, see module docstring

# Corpora screened for novelty (every arm's train must be excluded) and the arms
# reported. Same convention as novel_window_eval.py: bare = best.pt, f = final.pt.
SCREEN_CORPORA = [f"{HOME}/data/corpus_merged_25.txt",
                  f"{HOME}/data/corpus_merged_50.txt",
                  f"{HOME}/data/corpus_merged_dedup.txt"]
VAL_CORPUS = f"{HOME}/data/corpus_merged_dedup.txt"
ARMS = {
    "s25":   f"{HOME}/checkpoints/arm_s25/best.pt",
    "v25":   f"{HOME}/checkpoints/arm_v25/best.pt",
    "s50":   f"{HOME}/checkpoints/arm_s50/best.pt",
    "v50":   f"{HOME}/checkpoints/arm_v50/best.pt",
    "s100":  f"{HOME}/checkpoints/arm_a_data/best.pt",
    "v100":  f"{HOME}/checkpoints/arm_v100/best.pt",
    "p50":   f"{HOME}/checkpoints/arm_p50/best.pt",
    "p50f":  f"{HOME}/checkpoints/arm_p50/final.pt",
    "p100":  f"{HOME}/checkpoints/arm_p100/best.pt",
    "p100f": f"{HOME}/checkpoints/arm_p100/final.pt",
    "e50":   f"{HOME}/checkpoints/arm_e50/best.pt",
    "e50f":  f"{HOME}/checkpoints/arm_e50/final.pt",
    "e100":  f"{HOME}/checkpoints/arm_e100/best.pt",
    "e100f": f"{HOME}/checkpoints/arm_e100/final.pt",
    "s25f":  f"{HOME}/checkpoints/arm_s25/final.pt",
    "v25f":  f"{HOME}/checkpoints/arm_v25/final.pt",
    "s50f":  f"{HOME}/checkpoints/arm_s50/final.pt",
    "v50f":  f"{HOME}/checkpoints/arm_v50/final.pt",
    "s100f": f"{HOME}/checkpoints/arm_a_data/final.pt",
    "v100f": f"{HOME}/checkpoints/arm_v100/final.pt",
    "p25":   f"{HOME}/checkpoints/arm_p25/best.pt",
    "p25f":  f"{HOME}/checkpoints/arm_p25/final.pt",
    "c50":   f"{HOME}/checkpoints/arm_50c/best.pt",
    "c50f":  f"{HOME}/checkpoints/arm_50c/final.pt",
}
SELECT = sys.argv[2:] or list(ARMS)


# The natural-corpus family. Selected with LAMBDA_FAMILY=natural rather than a
# fourth copy of this file: arm_nat trained on different bytes, so its honest
# span must be novel against ITS OWN train split, not against the constructed
# corpora it never saw. Screening the wrong corpus would score it on material it
# memorised and quietly invert the result.
if os.environ.get("LAMBDA_FAMILY") == "natural":
    NAT = f"{HOME}/data/corpus_natural_ko_dedup.txt"
    # All three subsets are screened, not just the one an arm trained on:
    # 25% is a subset of 50% is a subset of 100%, so a window novel against one
    # can sit inside another arm's train and score that arm on memory.
    SCREEN_CORPORA = [f"{HOME}/data/corpus_nat_25.txt",
                      f"{HOME}/data/corpus_nat_50.txt", NAT]
    VAL_CORPUS = NAT
    ARMS = {"nat": f"{HOME}/checkpoints/arm_nat/best.pt",
            "natf": f"{HOME}/checkpoints/arm_nat/final.pt",
            "nat25": f"{HOME}/checkpoints/arm_nat25/best.pt",
            "nat25f": f"{HOME}/checkpoints/arm_nat25/final.pt",
            "nat50": f"{HOME}/checkpoints/arm_nat50/best.pt",
            "nat50f": f"{HOME}/checkpoints/arm_nat50/final.pt"}
    SELECT = sys.argv[2:] or list(ARMS)


def load_trainer(path):
    spec = spec_from_file_location("clm_trainer", path)
    mod = module_from_spec(spec)
    sys.modules["clm_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def split(path):
    data = open(path, "rb").read()
    chunk = min(CHUNK, max(1, len(data) // 10))
    parts = [data[s:s + chunk] for s in range(0, len(data), chunk)]
    return (b"".join(p for i, p in enumerate(parts) if i % 10 != 9),
            b"".join(p for i, p in enumerate(parts) if i % 10 == 9))


def select_windows(val, trains, offset):
    """Windows whose 3 probes are absent from every train split, from `offset`."""
    windows, tested = [], 0
    for start in range(offset, len(val) - BLOCK - 2, STRIDE):
        if len(windows) >= TARGET_WINDOWS or tested >= MAX_CANDIDATES:
            break
        tested += 1
        w = val[start:start + BLOCK + 1]
        mid = (BLOCK - PROBE) // 2
        probes = (w[:PROBE], w[mid:mid + PROBE], w[BLOCK - PROBE:BLOCK])
        if all(tr.find(p) == -1 for p in probes for tr in trains):
            windows.append(w)
    return windows, tested


def tensors(windows):
    x = torch.tensor([list(w[:BLOCK]) for w in windows], dtype=torch.long)
    y = torch.tensor([list(w[1:BLOCK + 1]) for w in windows], dtype=torch.long)
    return x, y


def score(model, x, y, device):
    total, ntok = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(x), BATCH):
            xb, yb = x[b:b + BATCH].to(device), y[b:b + BATCH].to(device)
            logits, _, _ = model(xb)
            total += F.cross_entropy(logits.view(-1, model.vocab_size),
                                     yb.reshape(-1), reduction="sum").item()
            ntok += yb.numel()
    return total / ntok / LN2


def build(clm, cfg, device, state=None):
    model = clm.ConsciousLM(vocab_size=256, d_model=int(cfg["dim"]),
                            n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                            block_size=BLOCK, dropout=0.0)
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def main():
    out_path = sys.argv[1]
    clm = load_trainer(TRAINER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    trains = []
    for c in SCREEN_CORPORA:
        tr, va = split(c)
        trains.append(tr)
        if c == VAL_CORPUS:
            val = va
        print(f"[split] screening {c.split('/')[-1]}: train={len(tr):,}", flush=True)

    t0 = time.time()
    win, tested = select_windows(val, trains, 0)
    win_alt, tested_alt = select_windows(val, trains, STRIDE // 2)
    print(f"[select] primary {len(win)}/{tested} = {len(win)/tested*100:.1f}% · "
          f"alt-phase {len(win_alt)}/{tested_alt} = {len(win_alt)/tested_alt*100:.1f}% "
          f"({time.time()-t0:.0f}s)", flush=True)
    x, y = tensors(win)
    x_alt, y_alt = tensors(win_alt)

    # Context shuffle: permute the input positions, leave the targets alone. A
    # model that ignores context is unaffected -- that is the whole point.
    g = torch.Generator().manual_seed(SHUFFLE_SEED)
    x_shuf = torch.stack([row[torch.randperm(BLOCK, generator=g)] for row in x])

    results = {"_select": {"kept": len(win), "tested": tested,
                           "keep_rate": len(win) / tested,
                           "kept_alt": len(win_alt), "tested_alt": tested_alt,
                           "probe_bytes": PROBE, "block": BLOCK,
                           "shuffle_seed": SHUFFLE_SEED, "init_seed": INIT_SEED,
                           "ratio_prospective": RATIO}}

    # Control A is per-architecture, not per-arm: build it once from the first
    # arm's config and reuse it for every arm that shares that shape.
    init_cache = {}
    for name in SELECT:
        path = ARMS[name]
        if not os.path.exists(path):
            print(f"[{name}] checkpoint absent -- skipped, not scored", flush=True)
            continue
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        shape = (int(cfg["dim"]), int(cfg["heads"]), int(cfg["layers"]))

        model = build(clm, cfg, device, ck["model_state"])
        value = score(model, x, y, device)
        shuf = score(model, x_shuf, y, device)
        align = score(model, x_alt, y_alt, device)
        del model

        if shape not in init_cache:
            torch.manual_seed(INIT_SEED)
            im = build(clm, cfg, device)
            init_cache[shape] = score(im, x, y, device)
            del im
        init = init_cache[shape]

        worst = min(shuf, init)          # lower BPC = stronger control, so min
        results[name] = {
            "bpc": value, "ckpt_step": ck.get("step"), "ckpt_sha256_16": sha,
            "ctrl_shuffle_bpc": shuf, "ctrl_init_bpc": init,
            "stability_alt_phase_bpc": align,
            "stability_delta": abs(align - value),
            "collapse_delta_shuffle": shuf - value,
            "collapse_delta_init": init - value,
            "ratio_over_worst_control": worst / value if value > 0 else float("inf"),
            "controls_collapsed": shuf > value and init > value,
        }
        print(f"[{name}] value={value:.4f} · shuffle={shuf:.4f} (Δ{shuf-value:+.4f}) · "
              f"init={init:.4f} (Δ{init-value:+.4f}) · alt-phase={align:.4f} "
              f"(|Δ|{abs(align-value):.4f}) · ratio={worst/value:.2f}x · sha {sha}",
              flush=True)
        del ck

    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
