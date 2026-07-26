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
   check, no GPU contention — `measurement/nf9_gate_check.py`.
3. **A run on raw corpus_v2 cannot be evaluated well.** With 4.7% of windows usable, any span drawn
   from it is mostly recall. DATA-2's path (dedup the corpus) applies to this run too.
4. Reproduction: `measurement/nf9_gate_check.py`, raw output `measurement/nf9_gate_check.json`.
