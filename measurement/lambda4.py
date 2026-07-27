#!/usr/bin/env python3
"""λ4 RECOMBINATION — does the model handle atom pairs it never saw together?

This rung was name-reserved and left off the ladder because its prerequisite was
a NATURAL corpus, not an atom-pair corpus: any pair set built by hand is
synthetic by construction, so Λ REGIME would stay CONSTRUCTED and a pass would
certify the instrument rather than a faculty (p9). The sibling repo has the
measurement that makes this concrete -- its synthetic XBIND reached held-out
1.000 while the natural arm sat at 0.455 ≈ chance.

corpus_natural_ko_dedup removes that blocker, so the pairs here are FOUND rather
than authored: two words that each occur often enough in train to have been
learned, and never once on the same line. Wikipedia supplies them for free.

  value          BPC on held-out lines where such a pair DOES co-occur.
                 The atoms are familiar; only their combination is new.
  ctrl SEEN      BPC on held-out lines whose pair DID co-occur in train.
                 This is the retrieval ceiling -- the easiest the model can have
                 it while still being scored on held-out bytes.
  ctrl BROKEN    the same value lines with both atoms replaced by frequency-
                 matched words drawn at random. Surface statistics survive, the
                 composition does not. This is the no-composition floor.

  PASS = value < (SEEN + BROKEN) / 2  AND  value < BROKEN

No tuned constant: the bar is the midpoint between the model's own two controls,
so it moves with the model instead of with a number somebody picked. A retrieval
machine lands near BROKEN; a composing one lands near SEEN. Both controls must
be present or the rung is VOID, the same rule the other rungs run under.

Frequency band and pair count are frozen here before the first run.
"""
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CORPUS = f"{HOME}/data/corpus_natural_ko_dedup.txt"
CHUNK = 1 << 20
BLOCK = 256
BATCH = 8
LN2 = math.log(2)

# Frozen before the first run.
FREQ_LO, FREQ_HI = 50, 5000    # learned, but not ubiquitous
MIN_TOKEN_BYTES = 6            # skip particles and one-syllable noise
TARGET_LINES = 192             # per group
SEED = 20260728

ARMS = {
    "nat":   f"{HOME}/checkpoints/arm_nat/best.pt",
    "natf":  f"{HOME}/checkpoints/arm_nat/final.pt",
    "nat25": f"{HOME}/checkpoints/arm_nat25/best.pt",
    "nat50": f"{HOME}/checkpoints/arm_nat50/best.pt",
}
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


def build_pairs(train, val, rng):
    """Find pairs that never share a train line but do share a val line."""
    train_lines = [ln for ln in train.split(b"\n") if len(ln) > 40]
    val_lines = [ln for ln in val.split(b"\n") if len(ln) > 40]

    freq = Counter()
    for ln in train_lines:
        for t in set(ln.split()):
            if len(t) >= MIN_TOKEN_BYTES:
                freq[t] += 1
    band = {t for t, c in freq.items() if FREQ_LO <= c <= FREQ_HI}
    print(f"[atoms] {len(band):,} tokens in the {FREQ_LO}-{FREQ_HI} frequency band "
          f"of {len(freq):,} total", flush=True)

    # Which band tokens co-occur in train. Storing the pair set directly would
    # be quadratic in line length; restrict to band tokens first.
    seen_pairs = set()
    for ln in train_lines:
        toks = sorted({t for t in ln.split() if t in band})
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                seen_pairs.add((toks[i], toks[j]))
    print(f"[pairs] {len(seen_pairs):,} band pairs co-occur somewhere in train",
          flush=True)

    novel, seen = [], []
    for ln in val_lines:
        toks = sorted({t for t in ln.split() if t in band})
        if len(toks) < 2:
            continue
        got_novel = got_seen = None
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                p = (toks[i], toks[j])
                if p in seen_pairs:
                    got_seen = got_seen or p
                else:
                    got_novel = got_novel or p
        # A line counts as novel only if it carries a never-co-occurring pair
        # and no seen pair, so the two groups cannot be the same lines.
        if got_novel and not got_seen:
            novel.append((ln, got_novel))
        elif got_seen and not got_novel:
            seen.append((ln, got_seen))
    rng.shuffle(novel)
    rng.shuffle(seen)
    print(f"[lines] novel-cooccurrence {len(novel):,} · seen-cooccurrence {len(seen):,}",
          flush=True)
    return novel[:TARGET_LINES], seen[:TARGET_LINES], sorted(band)


def break_composition(lines, band, rng):
    """Replace both atoms with frequency-matched others: surface survives, the
    combination does not."""
    out = []
    for ln, (a, b) in lines:
        ra, rb = rng.choice(band), rng.choice(band)
        out.append(ln.replace(a, ra).replace(b, rb))
    return out


def to_tensor(lines):
    rows = []
    for ln in lines:
        pad = ln[:BLOCK + 1]
        if len(pad) < BLOCK + 1:
            pad = pad + b" " * (BLOCK + 1 - len(pad))
        rows.append(list(pad))
    x = torch.tensor([r[:BLOCK] for r in rows], dtype=torch.long)
    y = torch.tensor([r[1:BLOCK + 1] for r in rows], dtype=torch.long)
    return x, y


def score_per_line(model, x, y, device):
    """BPC for every line separately, so the spread between two conditions can be
    tested instead of eyeballed."""
    out = []
    with torch.no_grad():
        for b in range(0, len(x), BATCH):
            xb, yb = x[b:b + BATCH].to(device), y[b:b + BATCH].to(device)
            logits, _, _ = model(xb)
            ce = F.cross_entropy(logits.view(-1, model.vocab_size),
                                 yb.reshape(-1), reduction="none")
            out.extend((ce.view(yb.shape).mean(dim=1) / LN2).tolist())
    return out


def score(model, x, y, device):
    per = score_per_line(model, x, y, device)
    return sum(per) / len(per)


def spread_is_resolvable(seen_per_line, broken_per_line):
    """Is the instrument's dynamic range on this arm bigger than its own noise?

    The λ4 bar asks whether `value` sits below the midpoint of SEEN and BROKEN,
    which means resolving a distance of spread/2. If the SEEN-BROKEN gap is not
    itself distinguishable from zero across lines, that midpoint is a coin flip
    and the rung is VOID for that arm -- exactly the weakness the first run
    exposed, where the small arms' spread was half the large arms' and they
    passed for that reason rather than by composing.

    Tested as a paired comparison over the SAME lines (broken is those lines with
    the atoms swapped), so line difficulty cancels. The threshold is |t| >= 2,
    i.e. the gap is at least twice its own standard error -- a resolution
    requirement, not a bar on the result."""
    diffs = [b - s for s, b in zip(seen_per_line, broken_per_line)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf")
    return {"spread": mean, "se": se, "t": t, "resolvable": abs(t) >= 2.0}


def main():
    out_path = sys.argv[1]
    clm = load_trainer(TRAINER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    print(f"[device] {device}", flush=True)

    train, val = split(CORPUS)
    novel, seen, band = build_pairs(train, val, rng)
    if len(novel) < 32 or len(seen) < 32:
        print("[verdict] not enough lines in one group -- λ4 cannot be measured "
              "on this corpus, and that is the result.", flush=True)
        json.dump({"_error": "insufficient lines", "novel": len(novel),
                   "seen": len(seen)}, open(out_path, "w"))
        return

    broken = break_composition(novel, band, rng)
    xv, yv = to_tensor([ln for ln, _ in novel])
    xs, ys = to_tensor([ln for ln, _ in seen])
    xb, yb = to_tensor(broken)

    results = {"_setup": {"freq_band": [FREQ_LO, FREQ_HI], "min_token_bytes": MIN_TOKEN_BYTES,
                          "lines_per_group": len(novel), "seed": SEED,
                          "corpus": os.path.basename(CORPUS),
                          "regime": "NATURAL",
                          "bar": "value < (SEEN + BROKEN)/2 AND value < BROKEN",
                          "resolution_rule": "VOID unless the value-vs-BROKEN gap is at "
                                             "least 2x its own standard error across lines "
                                             "-- registered after run 1 exposed that a narrow "
                                             "spread passes the midpoint for free"}}
    for name in SELECT:
        path = ARMS[name]
        if not os.path.exists(path):
            print(f"[{name}] checkpoint absent -- skipped", flush=True)
            continue
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        m = clm.ConsciousLM(vocab_size=256, d_model=int(cfg["dim"]),
                            n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                            block_size=BLOCK, dropout=0.0)
        m.load_state_dict(ck["model_state"], strict=False)
        m.to(device).eval()
        v = score(m, xv, yv, device)
        s_per = score_per_line(m, xs, ys, device)
        b_per = score_per_line(m, xb, yb, device)
        s, b = sum(s_per) / len(s_per), sum(b_per) / len(b_per)
        mid = (s + b) / 2
        # BROKEN is built from the NOVEL lines, so the paired test compares the
        # value lines against themselves with the composition removed.
        res = spread_is_resolvable(score_per_line(m, xv, yv, device), b_per)
        ok = v < mid and v < b
        pos = (v - s) / (b - s) if b != s else float("nan")
        results[name] = {"value_novel_cooccurrence": v, "ctrl_seen_cooccurrence": s,
                         "ctrl_broken_composition": b, "midpoint": mid,
                         "normalised_position": pos,
                         "resolution": res, "lambda4": ok,
                         "lambda4_void": not res["resolvable"],
                         "ckpt_step": ck.get("step"), "ckpt_sha256_16": sha}
        verdict = ("VOID (spread not resolvable)" if not res["resolvable"]
                   else ("PASS" if ok else "FAIL"))
        print(f"[{name}] novel={v:.4f} · seen={s:.4f} · broken={b:.4f} · mid={mid:.4f} · "
              f"pos={pos:.3f} · Δ={res['spread']:+.4f} t={res['t']:.1f} → λ4 {verdict} "
              f"· sha {sha}", flush=True)
        del ck, m

    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
