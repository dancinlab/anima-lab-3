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
# Optional extra held-out pool: a Wikipedia slice disjoint from CORPUS, so the
# model never saw any of it. The clustered test's unit is the sentence, so the
# only way to resolve an arm sitting below threshold is more sentences -- and
# enlarging CORPUS's own val split would mean retraining, which changes the thing
# being measured. Fresh prose does not.
FRESH = f"{HOME}/data/corpus_natural_fresh.txt"
CHUNK = 1 << 20
BLOCK = 256
BATCH = 8
LN2 = math.log(2)

# Frozen before the first run.
FREQ_LO, FREQ_HI = 50, 5000    # learned, but not ubiquitous
MIN_TOKEN_BYTES = 6            # skip particles and one-syllable noise
TARGET_LINES = 192             # per group, for the unmatched run-1/2 controls
# The damage-matched comparison gets every available line and several swap draws
# per line. N is a POWER parameter, not a bar: the test (novel-swap <= seen-swap)
# is symmetric, so more samples cannot bias the answer toward either verdict.
# Raising it after a null is raising resolution, not moving a threshold -- and a
# result below resolution is not an answer, so leaving it there is not a finding.
MATCHED_LINES = 100000         # take every novel-cooccurrence line available
MATCHED_DRAWS = 4              # independent swap pairs per line
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
    # For the paired SEEN control: which band atoms each atom DID co-occur with.
    partners = {}
    for x, y in seen_pairs:
        partners.setdefault(x, []).append(y)
        partners.setdefault(y, []).append(x)
    rng.shuffle(novel)
    rng.shuffle(seen)
    print(f"[lines] novel-cooccurrence {len(novel):,} · seen-cooccurrence {len(seen):,} · "
          f"atoms with a seen partner {len(partners):,}", flush=True)
    return (novel[:TARGET_LINES], seen[:TARGET_LINES], sorted(band), partners,
            freq, novel[:MATCHED_LINES])


def break_composition(lines, band, rng):
    """Replace both atoms with frequency-matched others: surface survives, the
    combination does not."""
    out = []
    for ln, (a, b) in lines:
        ra, rb = rng.choice(band), rng.choice(band)
        out.append(ln.replace(a, ra).replace(b, rb))
    return out


def matched_swaps(lines, partners, freq, band_by_freq, rng, draws=1):
    """The control the first two designs both missed: ONE substitution on BOTH
    sides, differing only in whether the swapped-in atom co-occurred with the
    partner.

    Run 1 compared different sentences, so line difficulty was the confound.
    Run 2 compared an original sentence against one with a word replaced, so
    SUBSTITUTION DAMAGE was the confound -- and the numbers said so: two swaps
    (BROKEN) cost +0.17 and one swap cost +0.048, four arms all within 0.006 of
    each other, which is what a pure damage term looks like and not what a
    composition term looks like.

    Here both arms of the comparison get exactly one substitution of an atom of
    similar frequency into the same sentence slot. The ONLY difference is whether
    the new atom ever shared a train line with the untouched partner:
        SEEN-SWAP   A -> A_seen   where (A_seen, B) occurred in train
        NOVEL-SWAP  A -> A_novel  where (A_novel, B) never occurred
    Damage cancels. What is left is co-occurrence.

    λ4 PASS = BPC(NOVEL-SWAP) <= BPC(SEEN-SWAP): the model pays no penalty for a
    combination it has not seen. Frequency matching keeps the swapped atoms
    within a factor of two, so the comparison is not secretly about word rarity.
    """
    seen_out, novel_out, line_of = [], [], []
    for ln, (a, b) in lines:
        fa = freq.get(a, 0)
        if not fa:
            continue
        lo, hi = fa / 2, fa * 2
        pset = set(partners.get(b, ()))
        cands = band_by_freq(lo, hi)
        seen_c = [t for t in cands if t in pset and t != a and t != b]
        novel_c = [t for t in cands if t not in pset and t != a and t != b]
        if not seen_c or not novel_c:
            continue
        for _ in range(draws):
            seen_out.append(ln.replace(a, rng.choice(seen_c)))
            novel_out.append(ln.replace(a, rng.choice(novel_c)))
            line_of.append(len(line_of) // draws)   # which sentence this draw came from
    return seen_out, novel_out, line_of


def paired_seen(lines, partners, rng):
    """The SEEN control, on the SAME lines.

    Run 1 compared novel-cooccurrence lines against a different set of lines that
    happened to carry a seen pair. Those are different sentences, so line
    difficulty was never controlled and the headline gap (value - seen) carried
    that confound. This removes it: keep the line and one atom, swap the OTHER
    atom for one that DID co-occur with it in train. Same sentence, same length,
    same surrounding context -- the only thing that changes is whether the
    combination is one the model has seen.

    Returns (lines, kept) so a line with no available partner is dropped from
    BOTH sides rather than silently compared against itself."""
    out, keep = [], []
    for idx, (ln, (a, b)) in enumerate(lines):
        alts = partners.get(b)
        if not alts:
            continue
        alt = rng.choice(alts)
        if alt == a or alt not in ln and True:
            pass
        out.append(ln.replace(a, alt))
        keep.append(idx)
    return out, keep


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
    if os.path.exists(FRESH):
        fresh = open(FRESH, "rb").read()
        print(f"[fresh] {len(fresh):,} bytes of prose disjoint from the trained "
              f"corpus, appended to the held-out pool", flush=True)
        val = val + b"\n" + fresh
    novel, seen, band, partners, freq, novel_all = build_pairs(train, val, rng)
    if len(novel) < 32 or len(seen) < 32:
        print("[verdict] not enough lines in one group -- λ4 cannot be measured "
              "on this corpus, and that is the result.", flush=True)
        json.dump({"_error": "insufficient lines", "novel": len(novel),
                   "seen": len(seen)}, open(out_path, "w"))
        return

    broken = break_composition(novel, band, rng)
    pseen, keep = paired_seen(novel, partners, rng)
    print(f"[paired] SEEN control built on {len(pseen)} of {len(novel)} value lines "
          f"(same sentence, one atom swapped for a train co-occurring partner)", flush=True)
    xv, yv = to_tensor([ln for ln, _ in novel])
    xs, ys = to_tensor([ln for ln, _ in seen])
    xb, yb = to_tensor(broken)
    xp, yp = to_tensor(pseen)

    # The damage-matched pair of controls, one substitution on each side.
    by_freq = sorted(((freq[t], t) for t in band))
    def band_by_freq(lo, hi):
        return [t for f, t in by_freq if lo <= f <= hi]
    m_seen, m_novel, m_line = matched_swaps(novel_all, partners, freq, band_by_freq,
                                            rng, draws=MATCHED_DRAWS)
    print(f"[matched] damage-matched controls: {len(m_seen):,} paired samples from "
          f"{len(novel_all):,} lines x {MATCHED_DRAWS} draws (one swap each side, "
          f"frequency within 2x)", flush=True)
    xms, yms = to_tensor(m_seen)
    xmn, ymn = to_tensor(m_novel)
    # The paired comparison must use the same subset on both sides.
    xvk, yvk = to_tensor([novel[i][0] for i in keep])

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
        # Paired SEEN: same lines, one atom swapped to a train co-occurring one.
        v_k = score_per_line(m, xvk, yvk, device)
        p_per = score_per_line(m, xp, yp, device)
        paired = spread_is_resolvable(p_per, v_k)   # positive = novel costs more
        p_mean = sum(p_per) / len(p_per)
        ms_per = score_per_line(m, xms, yms, device)
        mn_per = score_per_line(m, xmn, ymn, device)
        # CLUSTERED by sentence. Several swap draws from one line are not
        # independent samples -- they share the sentence -- so testing across
        # draws underestimates the standard error and inflates t. Average the
        # draws within each line first, then test across LINES. This is strictly
        # more conservative and it is the number the verdict uses.
        by_line = {}
        for li, sv, nv in zip(m_line, ms_per, mn_per):
            by_line.setdefault(li, []).append(nv - sv)
        per_line = [sum(v) / len(v) for v in by_line.values()]
        matched = spread_is_resolvable([0.0] * len(per_line), per_line)
        matched["n_lines"] = len(per_line)
        matched["n_draws"] = len(ms_per)
        # Reported alongside so the inflation is visible rather than hidden.
        naive = spread_is_resolvable(ms_per, mn_per)
        matched["t_unclustered"] = naive["t"]
        ms_mean = sum(ms_per) / len(ms_per)
        mn_mean = sum(mn_per) / len(mn_per)
        lam4 = mn_mean <= ms_mean
        ok = v < mid and v < b
        pos = (v - s) / (b - s) if b != s else float("nan")
        results[name] = {"value_novel_cooccurrence": v, "ctrl_seen_cooccurrence": s,
                         "ctrl_broken_composition": b, "midpoint": mid,
                         "normalised_position": pos,
                         "resolution": res,
                         "ctrl_paired_seen": p_mean,
                         "paired_novelty_cost": paired["spread"],
                         "paired_t": paired["t"],
                         "paired_resolvable": paired["resolvable"],
                         "matched_seen_swap": ms_mean,
                         "matched_novel_swap": mn_mean,
                         "matched_novelty_cost": matched["spread"],
                         "matched_t": matched["t"],
                         "matched_resolvable": matched["resolvable"],
                         "lambda4_matched": (None if not matched["resolvable"] else lam4),
                         "lambda4_verdict": ("NULL" if not matched["resolvable"]
                                             else ("PASS" if lam4 else "FAIL")),
                         "lambda4_unmatched_run1": ok,
                         "lambda4_void": not res["resolvable"],
                         "ckpt_step": ck.get("step"), "ckpt_sha256_16": sha}
        # A knife-edge PASS/FAIL on an effect smaller than its own noise is not a
        # verdict. When the damage-matched cost is below resolution the honest
        # reading is NULL: the confounds are controlled and no penalty exists at
        # this resolution, in either direction.
        verdict = ("NULL (|cost| below resolution -- no penalty either way)"
                   if not matched["resolvable"] else ("PASS" if lam4 else "FAIL"))
        print(f"[{name}] novel={v:.4f} · seen={s:.4f} · broken={b:.4f} · mid={mid:.4f} · "
              f"pos={pos:.3f} · Δbroken={res['spread']:+.4f} t={res['t']:.1f} · "
              f"MATCHED seen-swap={ms_mean:.4f} novel-swap={mn_mean:.4f} "
              f"cost={matched['spread']:+.4f} t={matched['t']:+.1f} "
              f"(clustered over {matched['n_lines']} lines; unclustered would read "
              f"{matched['t_unclustered']:+.1f}) → λ4 {verdict} "
              f"· sha {sha}", flush=True)
        del ck, m

    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
