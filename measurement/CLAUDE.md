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

## λ-사다리 채점기 (자체 기준 · 바는 CLAUDE.md 에 동결)

- `gate.py` → **λ0 SPAN** + **λ1 SCREEN** (신규성통제 BPC · 통제 결합 · 25팔)
- `panel.py` / `panel_nf9.py` → λ0·λ1 의 통제 실측 (rho-init · rho-shuffle · rho-align)
- `g_gates.py` → **λ2 COHERENCE** + **λ3 NOVELTY** (행동 계측기 · anima G0/G2 에서 바 동결 이관)
- `corpus_regime.py` → **Λ REGIME** 전제 판정

λ2/λ3 의 통제 3종은 전부 필수이고, 하나라도 깨지면 그 등급 전 행이 무효다:

| 통제 | 무엇을 증명 | 실측 |
|---|---|---|
| positive (실제 held-out 텍스트) | 계측기가 읽히기는 하는가 | kwr **0.937** — VALID |
| before-backbone (동일구조 무학습망) | 지표가 모델을 읽는가 코퍼스를 읽는가 | kwr **0.000**, FAIL — VALID |
| retrieval (train 구간 복사) | '부재' 가 부재를 뜻하는가 | 부재 n-gram **0** — VALID |

전수 결과(24팔, summer 28M 로스터 = `gate.py` 가 판정하는 팔과 동일): **λ2 FAIL 0 · λ3 FAIL 0**.
kwr 0.770~0.976, 부재 4-gram 65~103. 300M `nf9_20k` 는 호스트가 달라 `g_gates_nf9.py` 로
같은 바·같은 통제에서 별도 채점한다 — 아래 등급이 위 등급보다 적은 팔을 덮으면 사다리가 아니다.

두 통제가 실제로 결함을 잡았고, 둘 다 내 코드였다:
1. **positive 통제가 없었다면 타입 버그가 결과로 읽혔다.** vocab 은 `bytes`, 생성 토큰은 `str`
   이라 kwr 이 전 팔 0.000 이었고, anti-Goodhart 통제도 0.000 이라(그게 정상이라서) 아무것도
   울리지 않았다. **구분할 수 없는 통제는 통제가 아니다** — 그래서 positive 통제를 넣었다.
2. **retrieval 통제가 부재 판정의 결함을 잡았다.** 질의는 공백을 하나로 접어 만들고 건초더미는
   원문 그대로라, 원래 구분자가 줄바꿈이던 n-gram 이 전부 '부재' 로 읽혀 통제가 0 대신 81 을
   냈다. 양쪽 모두 정규화해야 부재 판정이 공백 판정이 아니게 된다.

## Imported rule coverage (sibling `anima`, audited — not all of it transferred)

Source of the rules: `anima/CLAUDE.md` (p1-p9 + the pre-action hard-gate), `anima/cli/rho_axon.py`
(the seven axes), `anima/HYPOTHESES/CLAUDE.md` (the seven lessons). Audited line by line rather
than adopted by vibe, because two of them say things this repo's gate cannot deliver.

| rule | status here |
|---|---|
| value + N controls that must ALL collapse | **imported** — `panel.py`, bound into `gate.py` C2 |
| "read the signal as collapse-Δ vs **≥2 controls**, never a raw value" | **imported** — ρ-shuffle + ρ-init is exactly two; this is the floor, not comfort |
| ratio over the worst control | **imported**, prospective (`MARGIN_RATIO`) |
| frozen-first · no tune-to-green · a negative is a result | **imported** — CLAUDE.md `## Measurement validity` V5 |
| no self-judge (captured output is the evidence) | **already held** — every number here comes from a script's stdout |
| misattribution guard ("which ckpt *sha* was that") | **imported** — every row carries sha256 |
| DIRECTIONAL vs TERMINAL tier | **imported**, plus a third (UNMEASURABLE) this repo needed |
| multi-seed against sampler artefacts | **already held** — DATA-5 §5 |
| every H on **2 surfaces** | **partially** — analytic floors and forward-pass controls are two independent paths and agree 25/25, but that is corroboration of one instrument, not two instruments |
| **p7 — no perplexity verdict** | **CANNOT be satisfied.** BPC *is* perplexity in log₂ units |
| **p9 — natural corpus or the claim is off-standard** | **FAILS, measured.** `corpus_regime.py`: corpus_v2's three most common lines repeat **2,538× each** and are about this project itself |
| GREEN only when wired to a live engine | **not applicable** — no engine in this repo |
| the `HYPOTHESES/` registry + folder invariants | **deliberately not imported** — `docs/hypotheses/` + `ING.jsonl` already fill that role |

### What p7 and p9 cost, stated so nobody re-derives it as a surprise

Neither law invalidates the arm comparisons. DATA-6's PASS/FAIL/FAIL/PASS by train size is a real
measurement, it survived 25/25 across two independent paths, and it is reproducible. What the two
laws bound is what it may be **cited for**:

```
읽어도 되는 것                       │  읽으면 안 되는 것
──────────────────────────────      │  ──────────────────────────────
 이 코퍼스족 안에서 팔들이           │   "이 모델은 언어를 한다/못한다"
 pair 통계를 넘었는가                │   (p7: perplexity 는 verdict 아님)
 크기에 따라 비단조인가              │   "faculty 에 대한 증거"
 통제가 붕괴하는가                   │   (p9: 구성된 코퍼스 = off-standard)
──────────────────────────────      │  ──────────────────────────────
 = 계측기 판독 (instrument reading)  │   = 능력 주장 (faculty claim)
```

`gate.py` prints this scope on every run, so a PASS cannot be quoted as more than it is. Earning a
faculty-level verdict would need a **behavioural** instrument with its own ≥2 controls — anima's G0
(known-word-ratio) / G2 (corpus-absent novel n-grams with retrieval-control = 0) are the shape —
run on a **natural** corpus this repo does not currently have.

## Running
```bash
python measurement/measure_all.py --cells 1024
python measurement/mensa_iq.py --engine CambrianExplosion
python measurement/calibrate_consciousness.py
```

## Parent Rules
See /CLAUDE.md for full project conventions.
