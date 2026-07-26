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

## Running
```bash
python measurement/measure_all.py --cells 1024
python measurement/mensa_iq.py --engine CambrianExplosion
python measurement/calibrate_consciousness.py
```

## Parent Rules
See /CLAUDE.md for full project conventions.
