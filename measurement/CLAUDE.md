# measurement/

## Purpose
Consciousness measurement and calibration tools. Standalone scripts for measuring Φ, IQ, and other metrics across engines.

## Contents
- `measure_all.py` — Full engine measurement suite (Φ + Granger + IQ + Hivemind)
- `measure_all_engines.py` — Batch measurement for all registered engines
- `measure_v8_phi_rs.py` — V8 architecture Φ measurement via Rust phi_rs
- `calibrate_consciousness.py` — Tension distribution calibration (sigmoid, homeostasis, habituation)
- `mensa_iq.py` — Mensa-based IQ scoring for consciousness engines
- `ce_quality_predictor.py` — Cross-entropy quality estimation
- `cell_count_optimizer.py` — Optimal cell count finder
- `measure_leak.py` — Validation-split honesty: reproduces a run's split, measures line-level
  leakage into train, re-measures the train split's unigram/bigram floors, writes an
  unseen-line eval span (DATA-3)
- `honest_ce.py` — Re-measures a checkpoint's CE on both the raw and the unseen-line span with
  the run's own `evaluate_fixed_span`; the raw span is the control that must reproduce the log
- `ngram_novelty.py` — Fraction of a span's w-byte windows present verbatim in train. Use this,
  not line-level dedup, to claim a span is novel (DATA-3: line filtering moved 32-byte
  familiarity the WRONG way, 79.5% → 91.8%)

- `gate.py` — **the adjudicator. Read this before quoting any BPC as a result.** It reports
  `[UNCOVERED]` for any arm that has a measurement but no `ARMS` entry: silently skipping one is
  how "all measurements pass" gets declared over a subset, which happened once and is why the
  check exists. Every arm's
  verdict is a CONJUNCTION, not a threshold: C1 novelty-controlled BPC below the corpus's own
  train-split bigram floor · C2 below its unigram floor AND worse under a context shuffle · C3
  scored on windows whose 3x64B probes are absent from every train split. An arm missing a
  control measurement is **DIRECTIONAL**, never PASS/FAIL — absence of a control is not a passed
  control. Emits `gate_verdicts.json` with a sha256 of every input file.
- `build_complement_half.py` — builds `corpus_merged_50c.txt`, the disjoint same-size complement
  of the failing 50% subset, and measures its own floors (DATA-6 §5: the content control)

- `panel.py` — runs each arm as a panel AXIS with the three controls the sibling repo's axes all
  carry: **rho-init** (identically-shaped model, random weights, same span — the empirical
  no-learning reading an analytic floor cannot give: 8.0486 BPC, i.e. slightly worse than the 8.0
  a uniform predictor gets), **rho-shuffle** (context positions permuted, targets untouched — a
  model ignoring context is unaffected), **rho-align** (windows re-selected at a different stride
  phase; this one must come out the SAME, so it is reported as stability, never as a passed
  control). Hashes every checkpoint it reads. Output `panel_results.json`.

## The gate rubric (imported, and why)
Its shape comes from the `rho-weave` instrument in the sibling `anima` repo
(`HYPOTHESES/cards/H_9270`, `cli/rho_axon.py`), where a capability passes only when the value
clears its bar AND every control collapses. Ours had controls but read them in prose beside the
verdict instead of requiring them — which is precisely the gap DATA-7 fell into, a dashboard
reading 0.65 BPC for ten hours while the honest span read 3.99.

Two rules travel with it and are not optional:
1. **frozen-first.** Bars are registered before the measurement. A bar chosen after the numbers
   are in is tune-to-green, and it does not matter that the number looks right.
2. **Measure the controls; do not substitute a floor for them.** The margin that matters is the
   collapse over the worst MEASURED control, which is what `panel.py` computes and `gate.py` now
   reads. Computing it as floor/bpc — an earlier version of this file — is a different quantity
   and read 1.8x for the 100% arms where the matching one reads 4.1x.
3. **A confirmed bar is not a frozen bar.** The 3x ratio stays prospective even though the
   measured ratios reproduce the C1 partition on **all 25 adjudicated arms** — every passing arm
   sits at 3.1-20.3x and every failing one at 1.5-2.0x, with nothing in between. That agreement is corroboration — corpus statistics and forward-pass
   controls are independent paths to the same split — but a bar confirmed after the numbers are in
   cannot also be the bar that judged them.
4. **Stability checks are not collapse controls.** rho-align must come out equal, not worse.
   Scoring it as a control that "passed" would be the unregistered-extra-hurdle mistake
   `rho_self`'s docstring in the sibling repo warns about.

## Running
```bash
python measurement/measure_all.py --cells 1024
python measurement/mensa_iq.py --engine CambrianExplosion
python measurement/calibrate_consciousness.py
```

## Parent Rules
See /CLAUDE.md for full project conventions.
