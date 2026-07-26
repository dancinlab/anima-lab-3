<!-- @hypothesis-ok — CLAUDE.md designates docs/hypotheses/ as this repo's canonical hypothesis folder -->
# DATA-3 — line-level dedup is not a novelty control: nf9's 0.65 BPC stays unmeasured

Measured: 2026-07-26 18:40–19:05 · split reproduced locally (byte-identical corpus), model CE re-measured on aiden CPU
Subject: `clm_pure_300m_nf9` on aiden — `train_conscious_lm_nf8.py`, 299,650,272 params (896d/12L/14H), corpus_v2.txt
Related: [DATA-1](DATA-1-corpus-duplication-and-free-bwd.md) · [DATA-2](DATA-2-dedup-breaks-the-unigram-floor.md) · [NF-9](NF-9-cell-persistence-hardware-ceiling.md)

**Pre-registered hypothesis: REFUTED.** The prediction was that re-measuring on held-out lines
absent from train would raise the BPC (CLAUDE.md's recorded pair for this corpus: 0.594 raw vs
1.401 unseen). It *fell* — 0.654 → 0.299. The reason turns out to matter more than the number.

## 1. The claim under test

The run reports validation BPC falling to **0.6544** by step 6,000, sitting at 0.66–0.67 after.
Its own train split has these MEASURED floors (172 distinct byte values):

```
BPC │8.0000 ── uniform
    │5.9548 ── unigram table        (CLAUDE.md's 5.95 for this corpus, confirmed)
    │3.4920 ── bigram context       (CLAUDE.md's 3.49, confirmed)
    │
    │0.6544 ●  nf9 reported = 18.7% of the bigram floor
    └───────────────────────────────────────────────────────────
```

A 300M model 5.3x better than order-1 context at step 6,000 (0.6 epochs) is the shape CLAUDE.md
warns about: *"the held-out span must also be DEDUPED against train, or the number measures
memorisation."*

## 2. The split, reproduced byte-exactly

`train_conscious_lm_nf8.py:988-995` holds out every 10th 1MB chunk with **no line-level dedup**:

```python
n = len(data); chunk = 1 << 20
for i, start in enumerate(range(0, n, chunk)):
    (val_parts if i % 10 == 9 else train_parts).append(data[start: start + chunk])
```

The corpus is byte-identical on both machines (`sha256` prefix `5ecc271ca4e61b5e3de8e1b7`,
70,507,316 B), so the split was reproduced locally and matches the run's own log to the byte:

| quantity | reproduction | aiden's log |
|---|---|---|
| train bytes | 64,215,860 | 64,215,860 |
| val bytes | 6,291,456 | 6,291,456 |

## 3. Line-level leakage — 57.1% of held-out lines are already in train

| quantity | value |
|---|---|
| distinct lines in train split | 501,180 |
| held-out lines, total | 103,224 |
| held-out lines occurring **verbatim** in train | **58,939 = 57.1%** |
| held-out **bytes** covered by those lines | **3,870,292 / 6,291,456 = 61.5%** |

```
held-out span, 6,291,456 B
  in lines present verbatim in train  ████████████░░░░░░░░  61.5%
  in lines absent from train          ░░░░░░░░░░░░████████  38.5%
```

Chunk granularity (1MB >> block_size 256) does bound *sequence* leakage across boundaries, which
is what the code comment claims. It does nothing about a concatenated corpus repeating whole lines
across chunks — DATA-1 measured 54.0% line duplication, 58.7% duplicate bytes inside this corpus.
Interleaving fixed the distribution mismatch it was written for (Hangul-free tail) and left
duplication untouched.

## 4. Model CE re-measured — and the control that makes it readable

Same `evaluate_fixed_span` the run itself calls (1,024 deterministic strided windows = 262,144
tokens), same checkpoint (`best.pt`, step 6,000, 4,359 cells), CPU only so the live GPU training
was not disturbed. `load_state_dict` reported **no missing and no unexpected keys**, so the model
class is the one that produced the checkpoint.

| span | CE (nats) | BPC | vs the run's log |
|---|---|---|---|
| raw held-out split — **control** | 0.4534 | **0.6541** | logged 0.6544 · **matches to 0.0003** |
| held-out lines absent from train | 0.2075 | **0.2993** | 2.2x *better*, not worse |

The control passing is what makes the second row meaningful: the harness reproduces the run's own
number, so the 0.2993 is not an artefact of re-measurement.

```
BPC │3.4920 ── bigram floor
    │
    │0.6541 ●───────── raw held-out split
    │0.2993       ●─── "unseen lines only"   ← EASIER, not harder
    └──────────────────────────────────────
      prediction was the arrow pointing UP past 3.49. It pointed down.
```

## 5. Why it fell — line-level absence is not novelty

The honest span is 262,144 B from 648 held-out lines absent from train. Those lines average
**404 B** — short lines are the ones that repeat, so filtering by line identity selects for long
lines. Measuring how many fixed-width windows of each span occur verbatim inside train (400
samples per cell, seed 1337) shows what that selection actually did:

| span | w=16 | w=32 | w=64 | w=128 |
|---|---|---|---|---|
| held-out lines absent from train | **96.0%** | **91.8%** | 82.5% | 54.5% |
| raw held-out split | 83.8% | 79.5% | 83.0% | 78.0% |
| train itself (sanity ceiling) | 100.0% | 100.0% | 100.0% | 100.0% |

```
fraction of 32-byte windows present verbatim in train
  "unseen lines"  ██████████████████░  91.8%   ← MORE familiar than the raw span
  raw split       ████████████████░░░  79.5%
  train ceiling   ████████████████████ 100.0%
```

A 404-byte line differing from a training line by a few characters is absent as a **line** while
being almost entirely present as **substrings**. At block_size 256 the model predicts from local
context, so a span with more familiar local substrings scores better — which is exactly what
happened. The line-level filter made the span *less* novel, not more.

## 6. Finding

1. **Line-level verbatim dedup is not a novelty control on this corpus.** It selects long lines,
   which are the most near-duplicated material. Any run whose honesty rests on
   `unseen lines only` is resting on a filter that here moved 32-byte familiarity from 79.5% to
   91.8% in the wrong direction.
2. **nf9's 0.6544 is not shown to be memorisation — and is not shown to be language either.**
   Both spans available from this corpus are 79–92% locally present in train, so neither can
   separate prediction from local copying. The number is *unmeasured*, which is a different and
   more accurate verdict than the "inflated by memorisation" one this document set out to confirm.
3. **The corpus is the defect, not the split.** No split-side filter fixes a source that is
   58.7% duplicate bytes; DATA-2 already measured the working path (dedup the corpus, then
   re-measure the floors on the deduped train split).
4. Independent of all of the above, gate 1 on this run is flat — `phi_per_cell` 0.76 → 1.27 while
   cells go 2 → 4,359 and Phi goes 1.5 → 5,518, i.e. the population's shadow (NF-8: cell count is
   causally disconnected from CE, so neither gate explains the other).

## 7. Application

1. **Retire `unseen lines only` as an honesty claim.** Where a novelty-controlled number is
   needed, control at substring level: reject validation windows whose w-byte n-grams
   (w >= 64) occur in train, and report the rejection rate next to the BPC.
2. **nf9 cannot produce a trustworthy language number on raw corpus_v2 at all.** Point the 300M
   config at the deduped corpus and restart from step 0 with a fresh checkpoint dir — changing the
   validation span changes what every number means, so `--resume` across it would compare two
   different metrics.
3. **Never print a BPC without its own train split's floors beside it.** 0.65 reads as excellent;
   0.65-on-a-79.5%-familiar-span reads as unmeasured. The second is the truth.
4. Reproduction (in `measurement/`): `measure_leak.py` (split + line leakage + floors + writes the
   honest span), `honest_ce.py` (model CE on both spans with the control — run on the host holding
   the checkpoint), `ngram_novelty.py` (the §5 table). Raw outputs kept beside them as
   `unseen_val_span.json` and `ngram_novelty.json`.
