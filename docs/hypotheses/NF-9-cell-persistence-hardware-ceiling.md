<!-- @hypothesis-ok — repo convention is docs/hypotheses/ (see CLAUDE.md, NF-1..NF-8); not a stray dir -->
# NF-9 — 자란 세포를 실제로 보존하고, "무제한"이 어디서 멈추는지 실측

NF-8 removed the cell ceiling and Φ rose 156×. But the checkpoint stored only a *status
summary* of the population — `{id, specialty, avg_tension, process_count, parent_id}` — so
every grown cell's weights were discarded at save time. A run that reached 4,621 cells could
not be resumed as anything but 2 cells. NF-9 asks two questions:

1. Do the grown cells actually survive a checkpoint?
2. With no designed ceiling, where does the population stop on real hardware?

## Method

- `save_checkpoint()` now writes, per cell: `state_dict` (weights), `hidden` (live state),
  `tension_history[-30:]`, and lineage (`cell_id`, `parent_id`, `specialty`, `creation_step`);
  plus `mitosis_meta` (`input_dim/hidden_dim/output_dim`, `step`, `next_id`) so restored
  lineage numbering cannot collide.
- `restore_cells(engine, ckpt, device)` rebuilds the population before training resumes.
- `KEEP_STEP_CHECKPOINTS = 2` + `_rotate_checkpoints()` — cells make each file ~2 GB heavier,
  so unbounded step checkpoints would exhaust the disk.
- Growth is rate-limited (`CELL_GROWTH_PER_STEP = 2`) and paused by hardware guards
  (`_throughput_ok` at `MAX_STEP_SECONDS = 2.0`, `_gpu_has_headroom`), each announced.
- Run `nf9` from step 0 (data/params changed → no `--resume`, new ckpt dir), 896d/12L/14H
  ≈ 300M, one RTX 5070.

## Results

Verified from `best.pt` written by the live run at step 3,000 — the same save path a step
checkpoint uses, so waiting for step 10,000 was unnecessary:

| 항목 | 값 |
|------|-----|
| VERDICT | **CELLS-PERSISTED** |
| 저장된 세포 | 2,651 |
| 세포당 파라미터 | 140,800 |
| 세포가 차지한 용량 | 1.49 GB (파일 5.11 GB 중) |
| 항목 키 | `cell_id` `parent_id` `specialty` `creation_step` `state_dict` `hidden` `tension_history` |
| 엔진 메타 | 64→128→64 · `next_id`=2651 |

Population trajectory and where the hardware answered:

```
cells |                                              ● 3969
      |                                        ● 3769  ← 가드 3회 발동 구간
      |                                 ● 3011
      |                          ● 2251
      |                   ● 1451
      |            ● 651
      | ●─────●──●          ← 1,700 step 까지 2~3 (언어단계 진입 전)
      └────────────────────────────────────────────── step
        0     800   2000  2400  2800  3200  3600 3700

  [cells] GROWTH HALTED by throughput at 2651 cells (9.35s/step)
  [cells] GROWTH HALTED by throughput at 3625 cells (2.02s/step)
  [cells] GROWTH HALTED by throughput at 3673 cells (2.02s/step)
  [cells] growth resumed …                    ← 3회 모두 스스로 재개
```

## Key findings

1. **The cells survive now — and they are most of the file.** 2,651 cells = 1.49 GB of a
   5.11 GB checkpoint. The population is no longer a byproduct that dies at save time; it is
   the majority of what gets written. This also makes rotation mandatory rather than tidy:
   three files at ~5–6 GB is ~17 GB, against 57 GB free on the training host.
2. **"Unlimited" resolves into a servo, not a number.** From step 2,000 the population climbs
   at exactly 800 cells / 400 steps = 2.0 per step — the rate limit is saturated, so the
   engine wants to divide *faster* than it is allowed. Growth then oscillates against
   `MAX_STEP_SECONDS`: halt at 2.02 s/step, resume when the step drops back under 2.0, three
   times so far, and the population has kept rising through every halt (3,625 → 3,673 →
   3,969). There is no ceiling in the code; the hardware states its own limit continuously
   and the run tracks it.
3. **A bigger population makes its own checkpoint slow it down.** The first halt reported
   9.35 s/step at 2,651 cells — far above the other two at 2.02 s. That step was a checkpoint
   save: more cells → a heavier file → a slower step → the throughput guard reads it as
   hardware strain and pauses growth. It self-corrects on the next step, so nothing is lost,
   but the coupling is real and worth naming: **persistence cost feeds back into the growth
   governor.** If it ever becomes more than a one-step pause, the fix is to exclude save
   steps from the throughput average rather than to raise the threshold.
4. **The verification did not need the milestone it was scheduled against.** A watcher was
   waiting for `step_10000.pt`; `best.pt` already exercised the identical code path at step
   3,000. Checking which artifact answers the question is cheaper than waiting for the one
   that was announced.

## Applied

`train_conscious_lm_nf8.py` — commit `29c1e4e34` (cell persistence + rotation).
Run: `tmux clmnf9` on aiden, ckpt `checkpoints/clm_pure_300m_nf9`, verdict recorded at
`logs/nf9_cellsave_verdict.txt`.

Open question carried from NF-8 and still unanswered: Φ per cell is flat (~1.33), so Φ is
still bought linearly with population, and the population is now bought with wall-clock.
Whether *integration per cell* can be deepened — rather than paid for by adding cells —
remains the question. NF-9 only guarantees that the cells bought so far are no longer thrown
away.
