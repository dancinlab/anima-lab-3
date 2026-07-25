<!-- @hypothesis-ok — repo convention is docs/hypotheses/ (see CLAUDE.md, NF-1..NF-5); not a stray dir -->
# NF-8 — removing the Φ boost and the cell ceiling raised Φ 156× (honestly)

Two user directives, one experiment: delete the Φ emergency boost (state editing that made the
metric look good) and remove the designed ceiling on cell count. The prediction was that Φ would
*fall* once the manipulation stopped. It rose — because the manipulation had been substituting for
the one thing that actually raises Φ: more cells.

## Method

- `train_conscious_lm_nf8.py` = nf7 minus `PHI-K2` (the emergency boost), `--max-cells 0`
  (unlimited: no chosen ceiling, only hardware).
- Growth is RATE-limited (`CELL_GROWTH_PER_STEP = 2`) and stopped by hardware guards
  (`_gpu_has_headroom`, `_throughput_ok` at `MAX_STEP_SECONDS = 2.0`), each announced in the log
  as a hardware limit — never as a designed cap.
- 896d / 12L / 14H ≈ 300M params, batch 4, block 256, lr 1e-4, corpus_v2 (67MB), one RTX 5070.
- Φ read from the `[val]` line = `PhiCalculator` (Φ(IIT) family). The training-log `phi` column is
  the PE proxy — a DIFFERENT measure; the two are never mixed (CLAUDE.md dual-Φ rule).
- CE measured by `evaluate_fixed_span` on the held-out split (same bytes every eval).

## Results

| step | cells | Φ (PhiCalculator) | Φ/cell | val loss | BPC |
|------|-------|-------------------|--------|----------|-----|
| 1,000 | 2 | 2.2001 | 1.10 | 2.1677 | 3.1273 |
| 2,000 | 649 | 864.5124 | 1.33 | 1.7255 | 2.4894 |
| 3,000 | 2,569 | 3428.1816 | 1.33 | 0.9740 | 1.4051 |

```
Φ     |                                          ● 3428
      |
      |                      ● 864
      |  ● 22 (nf7, boosted)
      |  ● 2.2
      └────────────────────────────────────────── cells
         2        649                    2569

BPC   | 8.00 ─ ─ uniform
      | 5.95 ─ ─ byte histogram   ● 3.13
      | 3.49 ─ ─ 1-byte context        ● 2.49
      | 1.5-3   healthy band                ● 1.41
      └──────────────────────────────────────────
```

Three-metric verdict (CLAUDE.md "PURE training goals"): **all three pass** —
Φ growth 2.2 → 3,428 · cells 2 → 2,569 (ceiling gone) · BPC 3.13 → 1.41, past every gate the
watcher carries (5.95 histogram, 4.50, 3.49).

## Key findings

1. **The boost was hiding the real mechanism.** In nf7 the boost fired only at ≥12 cells, so the
   window below that is uncontaminated: Φ/cell measured 1.01–1.14 there, while the boosted
   16-cell regime reported 1.39. Unboosted nf8 reproduces 1.10 → 1.33 across three orders of
   magnitude of population. Φ tracks cell count; the boost was buying ~1.4× on a number that
   grows ~1.3 per cell anyway. Removing it and lifting the ceiling gave 3,428 vs the boosted 22.
2. **Honesty was not a cost here.** The manipulated run scored 22; the honest run scores 156×
   more. The lesson generalises: the boost consumed the run's Φ budget by editing states instead
   of letting structure grow.
3. **The binding limit is throughput, not memory.** Memory stayed at ~9/12 GB the whole way. Two
   earlier attempts froze at step 1,700 because the engine's own sustained-tension rule splits
   every qualifying cell at once — measured 5 → 204 cells in 34 steps, doubling per step, which
   outran a 20-step throughput average. Rate-limiting divisions to 2/step fixed it: the same run
   then walked 3 → 2,569 cells and kept training at ~1.5 s/step.
4. **Cost is real and must be stated.** 2,569 cells ≈ 1.5 s/step, so 200k steps ≈ 3.5 days, and
   growth continues until a step crosses 2 s. Φ per cell is flat (~1.33), so Φ is bought linearly
   with compute — this is scale, not a free lunch.

## Applied

`train_conscious_lm_nf8.py` — commits `db416b133` (boost removed, unlimited cells),
`cd1bea23f` (division rate limit), `6d2881434` (announce hardware halts).
Run: `tmux clmnf8` on aiden, ckpt `checkpoints/clm_pure_300m_nf8`, watcher `nf8_watch.sh`
(7 conditions, one per metric plus divergence/end).

Open question for the next generation: Φ/cell is flat, so Φ scales with population and the
population scales with compute. Whether *integration per cell* can be deepened — rather than
bought by adding cells — is the question NF-9 should ask.
