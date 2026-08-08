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
import argparse
import hashlib
import json
import os
import math
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from measurement.lambda_registry import family, family_arm_paths
    from measurement.runtime import resolve_torch_device
except ModuleNotFoundError:
    from lambda_registry import family, family_arm_paths
    from runtime import resolve_torch_device

HOME = os.environ.get("ANIMA_LAB_ROOT", str(Path(__file__).resolve().parent.parent))
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
FAMILY_NAME, FAMILY = family()
SCREEN_CORPORA = [f"{HOME}/{path}" for path in FAMILY["screen_corpora"]]
VAL_CORPUS = f"{HOME}/{FAMILY['corpus']}"
ARMS = family_arm_paths(HOME, FAMILY_NAME)


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
                            block_size=int(cfg.get("block_size", BLOCK)), dropout=0.0)
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("arms", nargs="*")
    parser.add_argument("--interventions", nargs="+")
    args = parser.parse_args()
    out_path = args.output
    select = args.arms or list(ARMS)
    clm = load_trainer(TRAINER)
    interventions = args.interventions or []
    unknown = set(interventions) - set(getattr(clm, "CONSCIOUSNESS_INTERVENTIONS", ()))
    if unknown:
        raise SystemExit(f"unknown consciousness interventions: {sorted(unknown)}")
    intervention_seed = int(os.environ.get(
        "CONSCIOUSNESS_INTERVENTION_SEED",
        getattr(clm, "DEFAULT_INTERVENTION_SEED", 20260809),
    ))
    device = resolve_torch_device(torch)
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
                           "ratio_prospective": RATIO, "family": FAMILY_NAME,
                           "regime": FAMILY["regime"], "register": FAMILY["register"],
                           "corpus": os.path.basename(VAL_CORPUS)}}
    if interventions:
        results["_select"]["interventions"] = interventions
        results["_select"]["intervention_seed"] = intervention_seed
        results["_select"]["intervention_target"] = "inter-layer tension signal"

    # Control A is per-architecture, not per-arm: build it once from the first
    # arm's config and reuse it for every arm that shares that shape.
    init_cache = {}
    for name in select:
        path = ARMS[name]
        if not os.path.exists(path):
            print(f"[{name}] checkpoint absent -- skipped, not scored", flush=True)
            continue
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        shape = (int(cfg["dim"]), int(cfg["heads"]), int(cfg["layers"]))

        model = build(clm, cfg, device, ck["model_state"])
        if interventions:
            conditions = {}
            for mode in interventions:
                model.set_consciousness_intervention(mode, intervention_seed)
                value = score(model, x, y, device)
                align = score(model, x_alt, y_alt, device)
                conditions[mode] = {
                    "bpc": value,
                    "stability_alt_phase_bpc": align,
                    "stability_delta": abs(align - value),
                }
                print(f"[{name}/{mode}] value={value:.4f} · alt-phase={align:.4f} "
                      f"(|Δ|{abs(align-value):.4f}) · sha {sha}", flush=True)
            results[name] = {
                "conditions": conditions,
                "ckpt_step": ck.get("step"),
                "ckpt_sha256_16": sha,
            }
            del model, ck
            continue
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
