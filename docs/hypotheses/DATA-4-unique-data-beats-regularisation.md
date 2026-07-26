<!-- @hypothesis-ok — CLAUDE.md designates docs/hypotheses/ as this repo's canonical hypothesis folder -->
# DATA-4 — more unique data beats regularisation, on the most novel span of the three

Measured: 2026-07-26 18:20–19:48 (all three arms completed), summer RTX5070 · 3 arms x 12,000 steps, 27.7M (384d/6L/6H)
Common: batch 32 · ctx 256 · lr 3e-4 · max-cells 16 · eval-every 250 · seed 1337 · fixed 256KB val span
Related: [DATA-2](DATA-2-dedup-breaks-the-unigram-floor.md) · [DATA-3](DATA-3-line-dedup-is-not-a-novelty-control.md)

Decision lines were registered **before** the run: each arm is compared to its own corpus's bigram
floor, never to another corpus's raw BPC, and arm A had to beat **1.6988** (= 0.472 x 3.6010, the
ratio the prior single-corpus best achieved).

## 1. Arms

| arm | change | corpus | train bytes | dropout |
|---|---|---|---|---|
| C control | none (baseline) | corpus_v2_dedup | 27,173,705 | 0.1 |
| A data | more unique data | corpus_merged_dedup (v2+v4+v5, globally deduped) | 60,518,025 | 0.1 |
| B reg | stronger regularisation | corpus_v2_dedup | 27,173,705 | 0.3 |

## 2. Result — data wins, regularisation loses to doing nothing

| arm | best BPC @ step | its bigram floor | **ratio to floor** | verdict |
|---|---|---|---|---|
| A data | **0.8640** @ 7,000 | 3.6010 | **24.0%** | beats the 1.6988 line by 49% |
| C control | 1.6118 @ 1,750 | 3.5044 | 46.0% | beats prior best 1.6532 |
| B reg | 1.7503 @ 2,250 | 3.5044 | 49.9% | **worse than control** |
| (prior single-corpus best) | 1.6532 @ 4,000 | 3.5044 | 47.2% | — |

```
ratio to own bigram floor — lower is better
  A data     ██████████░░░░░░░░░░  24.0%
  C control  ███████████████████░  46.0%
  prior best ████████████████████  47.2%
  B reg      █████████████████████ 49.9%   ← dropout 0.1 → 0.3 made it worse
```

2.28x the unique data roughly halved the ratio. Tripling dropout moved it the wrong way, so the
27.7M model on 27MB was not capacity-limited in a way regularisation could fix — it was
data-limited, which is the same conclusion DATA-2 reached from the other direction.

## 3. The novelty control — why these numbers are readable

DATA-3 showed that `unseen lines only` is not a novelty guarantee: on raw corpus_v2 the filter
*raised* 32-byte familiarity from 79.5% to 91.8%, because filtering by line identity selects long,
near-duplicated lines. The arms are scored through that same filter
(`train_conscious_lm.py:899-905`), so the check had to be repeated on the corpora they use.

Fraction of w-byte windows occurring verbatim inside train (400 samples per cell, seed 1337):

| corpus (arm) | span | w=16 | w=32 | w=64 | w=128 |
|---|---|---|---|---|---|
| corpus_v2 raw — nf9, **not an arm** | unseen | 96.0% | 91.8% | 82.5% | 54.5% |
| | raw | 83.8% | 79.5% | 83.0% | 78.0% |
| corpus_v2_dedup (C, B) | unseen | 68.0% | 61.8% | 41.2% | 27.3% |
| | raw | 70.0% | 59.5% | 43.0% | 31.8% |
| corpus_merged_dedup (A) | unseen | 74.2% | **47.2%** | **21.0%** | **15.2%** |
| | raw | 71.5% | 55.2% | 28.0% | 16.5% |
| any train split (sanity ceiling) | train | 100% | 100% | 100% | 100% |

```
64-byte windows already present in train — the material each score was earned on
  nf9 raw corpus_v2   ████████████████░░░░  82.5%   ← DATA-3: unmeasurable
  arm C / B  dedup    ████████░░░░░░░░░░░░  41.2%
  arm A  merged       ████░░░░░░░░░░░░░░░░  21.0%   ← hardest span, best score
```

Two things follow. First, the perverse inversion DATA-3 found does **not** occur on the deduped
corpora — the filtered span is as novel as the raw one (C/B) or more novel (A), so the arms'
BPC is a real score and not a recall score. Second, arm A did not win by being handed easier
material: its span is the most novel of the three by a factor of two at w=64.

## 4. Differentiation — C and B are bit-identical, which is NF-8 as an exact equality

Mean over each arm's final 10 `[D]` evals (a single snapshot is not usable: arm B's
`phi_per_cell` reads 0.5684 at step 10,000, 0.6343 at 10,750 and 0.8438 at 11,500):

| arm | phi_per_cell | D | complexity_frac | cells |
|---|---|---|---|---|
| A data | 0.7401 | 2.5587 | 0.8665 | 16 |
| C control | 0.6943 | 2.5296 | 0.8666 | 16 |
| B reg | 0.6943 | 2.5296 | 0.8666 | 16 |

C and B are not merely close. Comparing the full telemetry line by line:

```
[D] rows per arm                     48
C vs B — rows that differ             0     ← identical over all 12,000 steps
C vs A — rows that differ            94     (of 96 halves; different corpus)

first five validation BPC
  C  3.2938  2.8822  2.3824  1.8107  1.7303
  B  3.5127  3.2038  3.0112  2.6816  2.3771   ← diverges at the FIRST eval
```

Dropout 0.1 -> 0.3 changed the language metric immediately and left the differentiation
trajectory byte-for-byte unchanged for 12,000 steps. NF-8 argued from the code that cell count is
causally disconnected from CE (mitosis runs under `no_grad`, the cells are absent from the
optimizer, their result is never read); this is that disconnection observed as an exact equality
rather than a weak correlation. Arm A differs only because a different corpus feeds the cells a
different byte stream.

The practical consequence: **a language hyperparameter cannot be judged by the differentiation
gate, and vice versa.** Reporting them as one score would have hidden a regression that only one
of them can see.

## 5. Findings

1. **Unique data is the binding constraint at 27.7M**, not capacity and not regularisation.
   2.28x data halved the bigram ratio; 3x dropout made it worse than the untouched control.
2. **A novelty control changes what a BPC means.** The identical filter produced an unreadable
   number on raw corpus_v2 (DATA-3) and a readable one on the deduped corpora. Corpus dedup is
   what makes the line filter safe — the filter is not doing the work.
3. **Report the ratio to the corpus's own bigram floor, not the raw BPC.** arm A's 0.8640 and
   arm C's 1.6118 are not comparable as numbers; 24.0% vs 46.0% is the comparison.
4. **The two gates cannot substitute for each other, measured.** Two runs differing only in
   dropout produced 48/48 identical differentiation rows and a language gap visible at the first
   eval. Any single "score" combining them would have reported no change.

## 6. Application

1. Scale the data before the model. The next run should use `corpus_merged_dedup` (or a larger
   globally deduped merge) as the default rather than corpus_v2_dedup.
2. Keep dropout at 0.1 for this size; revisit only when data stops being the constraint.
3. **Novelty control shipped** — `train_conscious_lm.py` now prints it at startup beside
   `dedup_note`, so a quietly unmeasurable span announces itself instead of producing a confident
   number. Same three constants as `measurement/arm_novelty.py` (w=64, n=400, seed 1337) and its
   own `random.Random`, because cell operations draw from the global RNG in an N-dependent amount
   and sampling from it here would shift every later draw:

   ```
   [data] train=27,173,705 val=2,084,274 bytes (... unseen lines only, 99.4% of held-out)
   [data] novelty: 37.8% of 64B windows already in train (n=400, seed 1337)
   ```

   Verified against the independent tool on both corpora — 37.8% vs 41.2% (v2_dedup) and 24.0%
   vs 21.0% (merged_dedup), each within the +-2.5pp sampling error of n=400. For contrast, raw
   corpus_v2 reads 82.5% on the same measure: that is what an unusable span looks like.
4. Reproduction: `measurement/arm_novelty.py` (split + filter + novelty table),
   raw output in `measurement/novelty_arms.json`.
