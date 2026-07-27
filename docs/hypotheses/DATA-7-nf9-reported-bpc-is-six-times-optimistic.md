<!-- @hypothesis-ok — CLAUDE.md designates docs/hypotheses/ as this repo's canonical hypothesis folder -->
# DATA-7 — the 300M run's dashboard says 0.65 BPC; on material it cannot recall it is 3.99

Measured: 2026-07-27 03:30, aiden (CPU only — the GPU is running the job being measured)
Subject: `clm_pure_300m_nf9`, 299,650,272 params (896d/12L/14H) on raw `corpus_v2.txt`, checkpoint at step 12,000 (4,359 cells)
Related: [DATA-3](DATA-3-line-dedup-is-not-a-novelty-control.md) · [DATA-6](DATA-6-the-50-percent-arm-fails-the-language-gate.md)

DATA-3 showed this run's reported BPC is unreadable — 82.5% of its held-out span's 64-byte windows
sit verbatim in its own train split. That was a statement about the instrument. This is the
measurement the working instrument gives.

## 1. Result

Windows kept only when three 64-byte probes (head, middle, tail) are absent from the train split —
the same construction as DATA-6, so the numbers are comparable.

| quantity | value |
|---|---|
| candidate windows tested | 5,399 |
| kept (novelty-controlled) | 256 = **4.7%** |
| tokens scored | 65,536 |
| **novelty-controlled BPC** | **3.9881** |
| the run's own reported BPC | 0.65 |
| train unigram floor | 5.9548 → it reaches 67% |
| train bigram floor | 3.4920 → it reaches **114% = FAIL** |

```
BPC, corpus_v2 train-split floors
  5.95 ── unigram (byte histogram)
  3.99 ●  nf9 on material it cannot recall  ← above the gate line
  3.49 ── bigram — the gate
  0.65 ○  nf9 as its own dashboard reports it   ← 6.1x optimistic
```

**The reported number is 6.1x optimistic, and the honest one has not cleared the gate.** The
dashboard reads "excellent"; the model is not yet beating order-1 context on unseen material.

## 2. The corpus is a poor test bed, quantified

The keep rate is the number to compare: **4.7%** here against **16.6%** on the deduped merged
corpus (DATA-6). Raw `corpus_v2.txt` yields a third as much material a model cannot recall, which
is the same defect DATA-3 measured from the other direction (58.7% duplicate bytes, 57.1% of
held-out lines verbatim in train).

## 3. What this does and does not license

- It **does** say the progress signal this run is steering by is wrong by a factor of six, and
  that the run has 13+ days left on that signal.
- It **does not** say "300M loses to 27.7M". At step 12,000 with batch 4, nf9 has seen 12.3M bytes
  — 0.19 epochs — against the DATA-6 arms' 98.3M bytes at batch 32. It is far earlier in its data
  exposure, not merely worse. Comparing the two as if step counts were comparable would repeat
  exactly the kind of error DATA-5 made.
- It **does not** predict step 200,000. Failing the gate at 6% of the budget is a status, not a
  verdict.

## 3.5 Second measurement (17:55) — the model is improving, its metric has stopped

Re-checked as §4 recommends, five hours and 7,300 training steps later:

| checkpoint step | novelty-controlled BPC | vs its 3.4920 floor |
|---|---|---|
| 12,000 | 3.9881 | 114% FAIL |
| **14,000** | **3.8583** | **110% FAIL** |

It is improving on material it cannot recall — 3.3% in 2,000 checkpoint steps. Still below the gate,
but moving toward it rather than stalling.

**The problem is what the run keeps.** `best.pt` holds step 14,000 and is the only checkpoint left
(step files rotate away, `KEEP_STEP_CHECKPOINTS=2`). Its own validation `best=` has read 0.4088
unchanged across the last five evaluations while the run advanced to step 19,300 — so the selector
has not fired in ~5,300 steps.

```
run progress   ├────────── 12,000 ──── 14,000 ──────────────── 19,300 ──→ 200,000
honest BPC     │           3.9881      3.8583                  unmeasured
best.pt        │                       ●─────────────────────────────── frozen here
own val best   │                       0.4088 ─────────────────────────── unchanged
```

That selector runs on a span whose 64-byte windows are 82.5% recallable (§2), so a stalled `best=`
is not evidence the model stopped improving — the two measurements above show it did not. The
consequence is operational: **the run can train for another 13 days and never save a checkpoint
from any of it**, because the only thing that writes a checkpoint is a metric that has stopped
moving.

Limit of this measurement, stated: there is no checkpoint past step 14,000, so whether the honest
number kept improving after it is unmeasured. `--save-every 10000` will produce step 20,000 shortly;
that is the next readable point.

## 4. Application

1. **Novelty line ported into `train_conscious_lm_nf8.py`** — done. The canonical trainer already
   printed it; the fork did not, which is why a 6x-optimistic number went unchallenged for ten
   hours. Verified by running the patched fork:

   ```
   [data] train=64,215,860 val=6,291,456 bytes (interleaved 1MB chunks, every 10th held out)
   [data] novelty: 77.0% of 64B windows already in train (n=400, seed 1337)
   ```

   (The independent measurement of the same span read 83.0%; both are n=400 draws of the same
   quantity and both say the span is mostly recallable, which is the flag's job. It is a warning,
   not an estimate to quote.) The edit does not disturb the running job — Python holds the module
   in memory, so the line appears on the next start; `.py.bak` is beside it.
2. **Re-check nf9 at intervals with this script rather than reading its log.** One CPU minute per
   check, no GPU contention — `measurement/nf9_gate_check.py`. Done once (§3.5); it showed the
   model improving while its checkpoint selector had stopped.
3. **Fix what the run keeps, not just what it prints.** `best.pt` is written only when a
   82.5%-recallable metric improves, and that metric has been flat for 5,300 steps. Either select
   on the novelty-controlled number or keep periodic step checkpoints beyond the current
   rotation — otherwise 13 more days of training can leave nothing behind.
4. **A run on raw corpus_v2 cannot be evaluated well.** With 4.7% of windows usable, any span drawn
   from it is mostly recall. DATA-2's path (dedup the corpus) applies to this run too.
5. Reproduction: `measurement/nf9_gate_check.py`, raw output `measurement/nf9_gate_check.json`
   and `measurement/nf9_gate_check2.json`.
