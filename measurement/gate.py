#!/usr/bin/env python3
"""Adjudicate every arm against a CONJUNCTIVE gate, not a single threshold.

Borrowed from the rho-weave instrument in the sibling anima repo
(HYPOTHESES/cards/H_9270, cli/rho_axon.py). Its shape is:

    PASS = value clears its bar
         AND every control stays collapsed
         AND value beats the worst control by a registered ratio

Our gate had only the first clause. Controls existed -- context shuffling
(DATA-6 3.5), span novelty (DATA-3), unigram/bigram floors -- but they were
reported in prose beside the verdict instead of being required by it. That gap
is exactly where the nf9 accident happened: a run reported 0.65 BPC for ten
hours while scoring 3.99 on material it could not recall (DATA-7). No single
number was wrong; nothing forced them to be read together.

Three conditions, each mechanism-derived so none of them is a bar tuned after
seeing the data (frozen-first, HYPOTHESES/CLAUDE.md lesson 3):

  C1 LANGUAGE   novelty-controlled BPC < the corpus's own TRAIN-SPLIT BIGRAM
                floor. Pre-registered in DATA-6. A model above it has not
                learned more than pair statistics.
  C2 CONTEXT    BPC < the corpus's own UNIGRAM floor, and shuffling the context
                inside the window must make it worse. A byte histogram scores
                exactly the unigram floor and loses exactly nothing to a
                shuffle, so both halves read zero for the thing being excluded.
                No tunable constant.
  C3 SPAN       the score came from windows whose 3x64B probes are absent from
                ALL train splits, with the keep rate recorded. Binary: either
                the strict selection ran or the number is not a language score.

  C4 MARGIN     value beats its WORST CONTROL by >= 3x -- the same quantity and
                the same constant rho-weave uses, now that panel.py measures the
                controls with forward passes instead of substituting an analytic
                floor for them. Read from panel_results.json where it exists.

                Still PROSPECTIVE: it binds runs from here on and is reported,
                never applied retroactively, because a bar confirmed after the
                numbers are in cannot also be the bar that judged them. Recorded
                because it matters: on the five arms measured it produces
                EXACTLY the C1 partition (s25 19.7x, s100 4.1x, e100 4.0x pass;
                s50 1.8x, e50 1.9x fail). Two independent paths -- corpus
                statistics for the floor, forward passes against controls for
                the ratio -- landing on the same split is corroboration the
                gate outcome is not an artefact of which floor was chosen.

                An earlier version of this file computed the margin as
                floor/bpc, which read 1.8x for the 100% arms and made them look
                like narrow passes. That was the wrong denominator: rho-weave
                measures against the worst CONTROL, not against a floor. With
                the matching quantity the 100% arms sit at ~4x.

Adding conditions to a conjunction can only turn PASS into FAIL, never the
reverse, so re-adjudicating settled arms cannot manufacture a pass. Any arm
that flips here was always failing and the old gate could not see it.

Tier follows the sibling's vocabulary: an arm missing a control measurement is
DIRECTIONAL, not a verdict -- absence of a control is not a passed control.
"""
import hashlib
import json
import sys
from pathlib import Path

# Each corpus's own train-split floors (measurement/build_scale_corpora.py and
# build_complement_half.py). A ratio against another corpus's floor is
# meaningless; the gate is a ratio.
FLOORS = {
    "25":   {"unigram": 6.0293, "bigram": 3.6140, "corpus": "corpus_merged_25.txt"},
    "50":   {"unigram": 6.0295, "bigram": 3.5925, "corpus": "corpus_merged_50.txt"},
    "50c":  {"unigram": 6.0275, "bigram": 3.5934, "corpus": "corpus_merged_50c.txt"},
    "100":  {"unigram": 6.0195, "bigram": 3.6010, "corpus": "corpus_merged_dedup.txt"},
    "v2":   {"unigram": 5.9548, "bigram": 3.4920, "corpus": "corpus_v2.txt"},
}

# arm key -> (corpus, human label). The suffix convention comes from
# novel_window_eval.py: bare = best.pt, trailing f = final.pt.
ARMS = {
    "s25":   ("25",  "25% seed 1337 best"),
    "v25":   ("25",  "25% seed 7331 best"),
    "s50":   ("50",  "50% seed 1337 best"),
    "v50":   ("50",  "50% seed 7331 best"),
    "s100":  ("100", "100% seed 1337 best"),
    "v100":  ("100", "100% seed 7331 best"),
    "p50":   ("50",  "50% phase-ablated best"),
    "p50f":  ("50",  "50% phase-ablated final"),
    "p100":  ("100", "100% phase-ablated best"),
    "p100f": ("100", "100% phase-ablated final"),
    "e50":   ("50",  "50% exposure-equalised best"),
    "e50f":  ("50",  "50% exposure-equalised final"),
    "e100":  ("100", "100% exposure-equalised best"),
    "e100f": ("100", "100% exposure-equalised final"),
    "s25f":  ("25",  "25% seed 1337 final"),
    "v25f":  ("25",  "25% seed 7331 final"),
    "s50f":  ("50",  "50% seed 1337 final"),
    "v50f":  ("50",  "50% seed 7331 final"),
    "s100f": ("100", "100% seed 1337 final"),
    "v100f": ("100", "100% seed 7331 final"),
    "p25":   ("25",  "25% phase-ablated best"),
    "p25f":  ("25",  "25% phase-ablated final"),
    "c50":   ("50c", "50% complement best"),
    "c50f":  ("50c", "50% complement final"),
    # The 300M nf9 run. corpus_v2 has no context-shuffle measurement, so these
    # come out DIRECTIONAL -- which is the point: the run whose dashboard was
    # 6.1x optimistic is exactly the one whose controls were never measured.
    "nf9_12k": ("v2", "nf9 300M @12,000"),
    "nf9_14k": ("v2", "nf9 300M @14,000"),
    "nf9_20k": ("v2", "nf9 300M @20,000 (controls measured on CPU)"),
}

# Context-shuffle measurements, keyed by the corpus they were run on
# (measurement/context_sensitivity.json). Arms whose corpus has no entry come
# out DIRECTIONAL rather than PASS/FAIL.
CONTEXT_JSON = "measurement/context_sensitivity.json"
CONTEXT_KEY = {"25": "25%", "50": "50%", "100": "100%"}

MARGIN_RATIO = 3.0  # prospective, see C4 above
PANEL_JSON = "measurement/panel_results.json"      # measured collapse ratios, when present
PANEL_NF9_JSON = "measurement/panel_nf9_results.json"  # the 300M run, measured on CPU


def load_scores(paths):
    """Merge the per-arm BPC tables, newest file wins on a repeated arm.

    The keep rate is carried PER SOURCE, not globally: a row's span receipt
    belongs to the file that produced it, and letting one file's rate stand in
    for another's is how a row ends up claiming a provenance it never had."""
    scores, keep_rate = {}, None
    for p in paths:
        blob = json.loads(Path(p).read_text())
        sel = blob.get("_select") or {}
        rate = sel.get("keep_rate")
        if rate is not None:
            keep_rate = rate
        for arm, row in blob.items():
            if arm.startswith("_") or not isinstance(row, dict) or "bpc" not in row:
                continue
            scores[arm] = {**row, "_src": Path(p).name, "_keep_rate": rate}
    return scores, keep_rate


def adjudicate(arm, row, ctx, keep_rate, panel=None):
    corpus_key, label = ARMS[arm]
    fl = FLOORS[corpus_key]
    bpc = row["bpc"]

    c1 = bpc < fl["bigram"]
    beats_unigram = bpc < fl["unigram"]
    # C2 is "beats the byte histogram AND actually uses context". panel.py
    # measures both per arm on the arm's own span, so prefer it; the older
    # context_sensitivity.json is per CORPUS at one step and is the fallback.
    prow_c2 = (panel or {}).get(arm)
    ctx_row = ctx.get(CONTEXT_KEY.get(corpus_key, ""))
    if prow_c2 and prow_c2.get("ctrl_shuffle_bpc") is not None:
        gain = prow_c2["ctrl_shuffle_bpc"] - bpc
        c2 = beats_unigram and gain > 0
        c2_note = (f"unigram {'<' if beats_unigram else '>='} · "
                   f"shuffle +{gain:.2f} BPC (panel, per arm)")
    elif ctx_row is not None:
        uses_context = ctx_row["shuffled_bpc"] > ctx_row["true_bpc"]
        c2 = beats_unigram and uses_context
        c2_note = (f"unigram {'<' if beats_unigram else '>='} · "
                   f"shuffle +{ctx_row['context_gain_bpc']:.2f} BPC (corpus-level)")
    else:
        c2, c2_note = None, "no context measurement for this arm or its corpus"
    # The strict 3x64B-probe selection is what produced these files at all; the
    # keep rate is its receipt. A table without one is not scored on a novel span.
    c3 = row.get("_keep_rate") is not None
    # The earned quantity is the collapse margin over the worst measured
    # control. Only fall back to the analytic floor when the panel has not run
    # this arm, and say so, because the two are not the same number.
    prow = (panel or {}).get(arm)
    if prow:
        margin, margin_src = prow["ratio_over_worst_control"], "worst control"
    else:
        margin, margin_src = (fl["bigram"] / bpc if bpc > 0 else float("inf")), "floor (no panel)"

    # Three tiers, because "not measured yet" and "can never be measured" are
    # different states and collapsing them either hides work or invents it.
    # UNMEASURABLE is still never a PASS -- the checkpoint being gone does not
    # promote a row, it only stops the clock on it.
    if c2 is None and row.get("ckpt_available") is False:
        tier, verdict = "UNMEASURABLE", "PASS" if (c1 and c3) else "FAIL"
    elif c2 is None:
        tier, verdict = "DIRECTIONAL", "PASS" if (c1 and c3) else "FAIL"
    else:
        tier = "TERMINAL"
        verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    return {
        "arm": arm, "label": label, "corpus": fl["corpus"], "bpc": bpc,
        "ckpt_step": row.get("ckpt_step"), "src": row["_src"],
        "bigram_floor": fl["bigram"], "unigram_floor": fl["unigram"],
        "ratio_to_bigram": bpc / fl["bigram"],
        "C1_language": c1, "C2_context": c2, "C2_note": c2_note,
        "C3_span": c3, "margin_x": margin, "margin_src": margin_src,
        "ctrl_shuffle_bpc": (prow or {}).get("ctrl_shuffle_bpc"),
        "ctrl_init_bpc": (prow or {}).get("ctrl_init_bpc"),
        "stability_delta": (prow or {}).get("stability_delta"),
        "C4_margin_prospective": margin >= MARGIN_RATIO,
        "tier": tier, "verdict": verdict,
    }


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "measurement/gate_verdicts.json"
    srcs = sys.argv[2:] or [
        "measurement/arm_gate_eval.json",
        "measurement/epoch_control_eval.json",
        "measurement/novel_window_epoch_final.json",
        "measurement/nf9_honest_eval.json",
        "measurement/phase_ablation_eval.json",
        "measurement/panel_results.json",
        "measurement/panel_nf9_results.json",
    ]
    srcs = [s for s in srcs if Path(s).exists()]
    scores, keep_rate = load_scores(srcs)
    ctx = json.loads(Path(CONTEXT_JSON).read_text()) if Path(CONTEXT_JSON).exists() else {}

    print(f"[src] {', '.join(srcs)}")
    print(f"[span] keep rate {keep_rate*100:.1f}% -- a window is kept only if 3x64B probes "
          f"are absent from every train split" if keep_rate else "[span] NO keep rate recorded")
    print(f"[ctx] {CONTEXT_JSON}: {', '.join(ctx) or 'absent'}\n")

    panel = {}
    for pj in (PANEL_JSON, PANEL_NF9_JSON):
        if Path(pj).exists():
            panel.update(json.loads(Path(pj).read_text()))
    uncovered = sorted(set(scores) - set(ARMS))
    if uncovered:
        print(f"[UNCOVERED] {len(uncovered)} arm(s) have a measurement but no ARMS entry, so "
              f"they are NOT adjudicated: {', '.join(uncovered)}. Add them or the verdict "
              f"below covers a subset.", flush=True)
    rows = [adjudicate(a, scores[a], ctx, keep_rate, panel) for a in ARMS if a in scores]
    hdr = f"{'arm':<6} {'BPC':>7} {'floor':>7} {'ratio':>7}  C1 C2 C3  {'tier':<12} verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = lambda v: " -" if v is None else (" Y" if v else " N")
        print(f"{r['arm']:<6} {r['bpc']:>7.4f} {r['bigram_floor']:>7.4f} "
              f"{r['ratio_to_bigram']*100:>6.0f}% {mark(r['C1_language'])}"
              f"{mark(r['C2_context'])}{mark(r['C3_span'])}  {r['tier']:<12} {r['verdict']}")

    flips = [r for r in rows if r["C1_language"] and r["verdict"] == "FAIL"]
    print(f"\n[conjunction] {len(flips)} arm(s) cleared the old single-threshold gate but fail "
          f"the conjunctive one" + (":" if flips else "."))
    for r in flips:
        print(f"    {r['arm']}: {r['C2_note']}")
    ds = [r for r in rows if r["tier"] == "DIRECTIONAL"]
    un = [r for r in rows if r["tier"] == "UNMEASURABLE"]
    print(f"[tier] {len(ds)} arm(s) DIRECTIONAL -- a missing control is not a passed control"
          + (": " + ", ".join(r["arm"] for r in ds) if ds else "."))
    if un:
        print(f"       {len(un)} arm(s) UNMEASURABLE -- checkpoint gone, controls can never be "
              f"taken, still not a PASS: " + ", ".join(r["arm"] for r in un))
    # A permanently unmeasurable row cannot have a measured margin either, so
    # counting it as outstanding work would keep the board red forever.
    fallback = [r for r in rows
                if r["margin_src"] != "worst control" and r["tier"] != "UNMEASURABLE"]
    ok = not ds and not fallback and not uncovered
    print(f"[validity] {'ALL MEASUREMENTS PASS' if ok else 'NOT YET'} -- "
          f"{len(uncovered)} uncovered, {len(ds)} measurable-but-unmeasured, "
          f"{len(fallback)} on a substituted floor"
          + (f", {len(un)} permanently unmeasurable (recorded, not counted)" if un else ""))
    measured = [r for r in rows if r["margin_src"] == "worst control"]
    print(f"[margin] prospective {MARGIN_RATIO:.0f}x bar, reported only. "
          f"Measured against the worst control ({len(measured)} arm(s)): "
          + ", ".join(f"{r['arm']} {r['margin_x']:.1f}x"
                      f"{'' if r['margin_x'] >= MARGIN_RATIO else ' (under bar)'}"
                      for r in measured))
    unmeasured = [r for r in rows if r["margin_src"] != "worst control" and r["C1_language"]]
    if unmeasured:
        print("          Falling back to floor/bpc, NOT the same quantity: "
              + ", ".join(f"{r['arm']} {r['margin_x']:.1f}x" for r in unmeasured))
    agree = all((r["margin_x"] >= MARGIN_RATIO) == r["C1_language"] for r in measured)
    print(f"[corroboration] measured ratio and C1 {'AGREE on every' if agree else 'DISAGREE on some'} "
          f"arm -- corpus statistics and forward-pass controls are independent paths")

    payload = {"_gate": {"conditions": ["C1 language", "C2 context", "C3 span"],
                         "C4_margin_prospective_ratio": MARGIN_RATIO,
                         "keep_rate": keep_rate, "sources": srcs,
                         "sources_sha256": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest()[:16]
                                            for s in srcs}},
               "arms": {r["arm"]: r for r in rows}}
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
