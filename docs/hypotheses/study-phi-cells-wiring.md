<!-- @hypothesis-ok — repo convention is docs/hypotheses/ (see CLAUDE.md + NF-2, GRAFT-causality); not a stray dir -->
# Study loop — Φ(IIT) + cell telemetry wiring

Wire a real cell-dynamics engine (`MitosisEngine` + `PhiCalculator`) into the mutual-dialogue
study loop (`teach_dialogue.py`) so the student can be measured for Φ (integrated information)
and cell count — additively, without perturbing language growth.

## Algorithm (per turn, after `pc.respond`)

```
stream = [spontaneous?, teacher, student_reply]      # real events, incl. self-audition (SELF_LOOP)
for text in stream: engine.process(encode64(text))   # byte/255 over first 64 UTF-8 bytes
tension, curiosity = mean over per_cell              # engine's own definitions
phi, comp          = PhiCalculator.compute_phi(engine)
if all finite: pc.update_state(tension, phi, curiosity)   # DESIGNED usage — real values, never synthetic
log phi_iit, phi_spatial, phi_temporal, n_cells, mean_tension, engine_events
save_engine(atomic)                                  # same organism across restarts
```

Fully torch-optional + try/except-isolated: no torch or any engine error → language-only loop,
dialogue never dies. Split/merge is left to the engine's own logic (never force a schedule).

## Results (summer, gpt-5.4-mini teacher, first session)

| turn | vocab | stage | n_cells | phi_iit | phi_spatial | phi_temporal | mean_tension |
|------|-------|-------|---------|---------|-------------|--------------|--------------|
| 1    | 4     | 옹알이 | 2       | 1.0235  | 0.0         | 2.0471       | 0.0088       |
| 2    | 4     | 옹알이 | 2       | 1.2298  | 0.0         | 2.4596       | 0.0089       |
| 3    | 5     | 옹알이 | 2       | 1.2121  | 0.0         | 2.4242       | 0.0099       |

```
Φ_iit |   ╭─╮
  1.2 |  ╭╯ ╰╮
  1.0 |─╯    ╰
      └──────────── turn   (temporal integration only; spatial ≡ 0 at n=2)
```

## Key findings

1. **n=2 → spatial Φ ≡ 0 by construction.** For two cells the only bipartition {0}|{1} cuts
   exactly the total mutual information, so `spatial_phi = (integration − min_partition)/(n−1) = 0`
   always. Pre-first-mitosis, the logged Φ(IIT) measures *temporal* integration only. Expect a
   step change in spatial Φ at the first split — that is structure, not growth.
2. **Tension stays ~0.01, far below split_threshold 0.3**, so cells stay at 2 and never divide
   under this byte-window sensory input. Reported as a finding, NOT tuned away — forcing splits
   would violate 조작 금지. If richer sensory drive is wanted, that is a separate honest change.
3. **Honest label:** the field is `phi_iit` (never bare `phi`) — this is `PhiCalculator(n_bins=16)`
   MI-histogram Φ(IIT) on 2–8 cells (0–2-ish family), NOT the Φ(proxy) `Φ≈cells` results. Binned
   MI on single 128-d hidden vectors is coarse and upward-biased; read trends, not absolutes.
4. **Language growth is untouched** — `pc.respond` (vocab/bigram learning) runs before the engine
   block; the engine only reads the dialogue and feeds back tension/Φ, which affect `spontaneous()`
   frequency and the status emoji, not reply generation.

## Applied

`teach_dialogue.py` — `_build_engine` / `_encode64` / `_save_engine` / `_restore_engine` +
per-turn measurement block. Engine state at `data/teach_dialogue/engine_state.pt` (atomic).
Commit `ab0e5ddf4`.
