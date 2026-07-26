<!-- @hypothesis-ok — CLAUDE.md designates docs/hypotheses/ as this repo's canonical hypothesis folder -->
# DATA-5 — the data-scaling claim does not survive a curve

Measured: 2026-07-26 23:36 – 2026-07-27 00:35, summer RTX5070 · 2 new arms x 12,000 steps + the DATA-4 100% arm reused
Common: 27,691,440 params (384d/6L/6H) · batch 32 · ctx 256 · lr 3e-4 · max-cells 16 · eval 250 · seed 1337 · dropout 0.1 · only `--data` differs
Related: [DATA-3](DATA-3-line-dedup-is-not-a-novelty-control.md) · [DATA-4](DATA-4-unique-data-beats-regularisation.md)

**Both pre-registered predictions FAILED, and DATA-4's headline claim does not survive.**

## 1. What was registered, before the run

DATA-4 concluded from ONE comparison (27MB vs 60MB, ratio to bigram floor 46.0% -> 24.0%)
that unique data is the binding constraint at 27.7M. A curve tests that. Subsets of the merged
corpus were taken by LINE INDEX MODULO, not by prefix — the corpus is a concatenation of sources,
so its first 25% is mostly one source, which would confound volume with distribution.

| corpus | train bytes | own unigram | own bigram | 12,000-step exposure |
|---|---|---|---|---|
| 25% | 15,656,455 | 6.0293 | 3.6140 | 6.3 epochs |
| 50% | 30,268,539 | 6.0295 | 3.5925 | 3.2 epochs |
| 100% (arm A, reused) | 60,518,025 | 6.0195 | 3.6010 | 1.6 epochs |

Floors within 0.6% of each other, so ratios are comparable.

- **P1** — if unique data is still binding: ratio(25%) > ratio(50%) > 24.0%.
- **P2** — overfitting starts near one epoch, so the best step scales with corpus size:
  ~1,900 / ~3,700 / 7,000 (the last observed).

## 2. Result — non-monotonic, and both predictions fail

| arm | best BPC @ step | ratio to own bigram | its span's novelty | P2 predicted peak |
|---|---|---|---|---|
| s25 (25%) | 0.5128 @ 11,750 | **14.2%** | 15.2% | ~1,900 → **11,750** (6.2x late) |
| s50 (50%) | 1.7390 @ 6,750 | **48.4%** | 11.2% | ~3,700 → 6,750 (1.8x late) |
| s100 (100%) | 0.8640 @ 7,000 | **24.0%** | 24.0% | 7,000 (observed, sets the prediction) |

```
ratio to own bigram floor — lower is better
  25% data   ███████░░░░░░░░░░░░░  14.2%   ← LEAST data, BEST score
 100% data   ████████████░░░░░░░░  24.0%
  50% data   ████████████████████████  48.4%   ← middle data, WORST score
```

**P1 fails twice over**: the 25% arm beats the 100% arm outright, and the relationship is not
monotonic in either direction — the middle point is the worst. **P2 fails**: the 25% arm ran 6.3
epochs and its validation number was still improving at the last evaluation before the step
budget ran out, so "overfitting begins near one epoch" is not what this architecture does.

## 3. Three checks before believing the shape

**(a) Is it the subsets?** No. Modulo sampling preserved composition:

| corpus | lines | unique-line fraction | empty lines | top speaker prefixes |
|---|---|---|---|---|
| 100% | 2,294,391 | 49.3% | 50.7% | VAD: 0.1% · 문제: 0.1% |
| 50% | 1,147,196 | 49.3% | 50.7% | VAD: 0.1% · 문제: 0.1% |
| 25% | 573,598 | 49.3% | 50.7% | VAD: 0.1% · 문제: 0.1% |

**(b) Is it that each arm was scored on its own span?** That is a real defect — three different
test sets are not a comparison, and their novelty came out non-monotonic too (15.2% / 11.2% /
24.0%). A shared span was built from held-out lines absent from ALL three train splits, and it
produced BPC 0.1232 / 0.1316 / 0.1501. **That result was discarded**: measured afterwards, 80-86%
of its 64-byte windows sit verbatim in every arm's train, and its own bigram floor is 3.32 — the
models were recalling, not predicting. This is DATA-3's finding a third time, and it caught the
author of DATA-3 building a span the same broken way.

**(c) Does the shape survive a real novelty control?** Yes. Selecting WINDOWS instead of lines —
a 257-byte window is kept only if three 64-byte probes (head, middle, tail) are each absent from
all three train splits — keeps 256 of 1,538 candidates (**16.6%**), 65,536 tokens:

| arm | novelty-controlled BPC |
|---|---|
| s25 (25%) | **0.4089** |
| s50 (50%) | **4.5001** |
| s100 (100%) | **1.9853** |

```
same ordering as the per-arm spans, on ONE test set the models cannot recall
  25%   █░░░░░░░░░░░░░░░░░░░  0.41
 100%   ████████░░░░░░░░░░░░  1.99
  50%   ██████████████████░░  4.50      (corpus bigram floor ≈ 3.6)
```

The U-shape is not a test-set artefact. It reproduces.

## 4. Finding

1. **DATA-4's headline does not survive.** "Unique data is the binding constraint at 27.7M" rested
   on a two-point comparison. A three-point curve is non-monotonic, so two points could not have
   established a direction — the same two points are equally consistent with the curve measured here.
2. **The spread is real, not noise — but it is not monotone in data.** The 11x spread in
   novelty-controlled BPC (0.41 / 4.50 / 1.99) is ~100x larger than run-to-run variation, measured
   directly in §6. What that licenses is "these arms genuinely differ", not "more data helps":
   the middle arm is the worst, so data volume is not the variable doing the work.
3. **Line-level span filtering has now failed three times** (DATA-3 on raw corpus_v2; the shared
   span here at 80-86%). Window-level probing is the construction that worked, and its keep rate
   (16.6%) is itself the number to report next to any BPC.

## 5. Variance control — the spread is not noise (measured 2026-07-27 00:54–01:58)

Registered before the control ran: if each repeat lands near its own seed-1337 twin, the arms
genuinely differ; if a repeat lands near ANOTHER arm's number, the curve was run-to-run noise and
no arm comparison in DATA-4 or DATA-5 stands. The 25% and 100% arms were repeated at **seed 7331**,
every other flag identical.

| arm | seed 1337 | seed 7331 | difference |
|---|---|---|---|
| 25% — best BPC @ step | 0.5128 @ 11,750 | 0.5103 @ **11,750** | 0.5%, same peak step |
| 25% — ratio to own bigram | 14.2% | 14.1% | 0.1pp |
| 25% — **novelty-controlled BPC** | 0.4089 | **0.4051** | **0.9%** |
| 100% — best BPC @ step | 0.8640 @ 7,000 | 0.8634 @ 5,250 | 0.07% |
| 100% — ratio to own bigram | 24.0% | 24.0% | 0.0pp |
| 100% — **novelty-controlled BPC** | 1.9853 | **2.0879** | **5.2%** |

```
novelty-controlled BPC — seed pairs sit on top of each other, arms do not
  25%   s25 0.409 · v25 0.405   ██░░░░░░░░░░░░░░░░░░
 100%   s100 1.985 · v100 2.088 █████████░░░░░░░░░░░
  50%   s50 4.500 (no replicate) ████████████████████
        └ within-arm spread 0.9-5.2%  ·  between-arm spread up to 11x
```

**Verdict: the arms genuinely differ.** Run-to-run variation is bounded at ~5% by two independent
pairs, and 5% cannot produce an 11x gap. The 25% pair reproduced even the peak step exactly
(11,750 both), so these runs are close to deterministic under a batch-order change.

Caveat kept explicit: the 50% arm — the extreme point — has **no replicate**. Its position rests on
the assumption that its variance resembles the two arms that were replicated.

## 6. Application

1. Retire the two-point data comparison as evidence for anything. DATA-4's regularisation result
   (dropout 0.3 worse than 0.1) rests on the same design; §6 shows run-to-run noise is small, so
   that result is likely real too — but it was not replicated either.
2. **The open question is now "what makes the 50% subset bad", not "does data help".** Data volume
   is out: the curve is non-monotone and the ordering is reproducible. Replicating the 50% arm is
   the next measurement.
3. Promote window-level novelty selection from a separate tool into the evaluation path, printing
   the keep rate beside the CE — `measurement/novel_window_eval.py` has the selector.
4. Reproduction: `measurement/build_scale_corpora.py` (subsets + floors),
   `measurement/subset_composition.py` (check a), `measurement/common_span_eval.py` (check b, the
   discarded one — kept because the discard is the lesson), `measurement/novel_window_eval.py`
   (check c).
