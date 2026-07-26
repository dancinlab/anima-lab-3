<!-- @hypothesis-ok — CLAUDE.md designates docs/hypotheses/ as this repo's canonical hypothesis folder -->
# DATA-6 — it is not a curve: the 50% arm reproducibly fails the language gate

Measured: 2026-07-27 02:02 – 02:40, summer RTX5070 · the DATA-5 50% arm replicated at seed 7331, plus fixed-step and structural controls
Related: [DATA-2](DATA-2-dedup-breaks-the-unigram-floor.md) · [DATA-3](DATA-3-line-dedup-is-not-a-novelty-control.md) · [DATA-5](DATA-5-the-data-scaling-claim-does-not-survive-a-curve.md)

DATA-5 left the 50% arm as an unreplicated extreme point on a non-monotonic curve. It is
replicated now, and the shape it was part of is the wrong description of what happened.

## 1. All three arms, both seeds, on one novelty-controlled span

256 windows of 257 bytes, each kept only when three 64-byte probes are absent from ALL train
splits (16.6% keep rate, 65,536 tokens). `best` = each arm's own-val pick; `final` = step 12,000
for every arm, so checkpoint selection is not a variable.

| arm | best, seed 1337 | best, seed 7331 | final@12,000, seed 1337 | final@12,000, seed 7331 |
|---|---|---|---|---|
| 25% | 0.4089 | 0.4051 | **0.4064** | **0.4042** |
| 50% | 4.5001 | 4.3430 | **5.4441** | **5.2289** |
| 100% | 1.9853 | 2.0879 | **2.4627** | **2.4671** |

Replicate agreement at the fixed step: **0.5% / 4.0% / 0.2%**. Every arm reproduces.

## 2. Against the floors, it stops looking like a curve

Baselines measured on each arm's own train split, as CLAUDE.md requires. The rule there is
explicit: beating unigram only means the byte histogram was learned; a real LM must beat bigram.

| arm | final BPC (mean of seeds) | its train unigram | its train bigram | vs bigram | gate |
|---|---|---|---|---|---|
| 25% | 0.405 | 6.0293 | 3.6140 | **11%** | PASS |
| 100% | 2.465 | 6.0195 | 3.6010 | **68%** | PASS |
| 50% | 5.337 | 6.0295 | 3.5925 | **149%** | **FAIL** — 88% of unigram |

```
BPC on novelty-controlled material (both seeds, step 12,000)
  6.03 ── unigram floor (byte histogram)
  5.34 ●● 50%   ██████████████████░░  ← stuck just under the histogram
  3.60 ── bigram floor — the gate a real LM must clear
  2.47 ●● 100%  ████████░░░░░░░░░░░░
  0.41 ●● 25%   █░░░░░░░░░░░░░░░░░░░
```

The 50% arm is not a low point on a curve. It is the **unigram-floor failure mode** this repo has
recorded before (DATA-2 measured 300M runs stuck at 5.933 against a 5.7997 unigram floor), and it
now reproduces across seeds. Two arms cleared the gate; one did not. "Non-monotonic scaling curve"
described a pass/fail split as though it were a continuum.

For context, that span's OWN floors are unigram 5.5975 and bigram 2.2531 (112 distinct byte
values) — an oracle order-1 model fitted on the test text itself, which no learner can be asked to
match. The gate above uses the train-split floors, which is the fair reference.

## 3. What was ruled out on the way

| candidate explanation | how it was excluded |
|---|---|
| subset composition | unique-line fraction 49.3%, empty-line share 50.7%, speaker-prefix mix — identical in all three (DATA-5 §3a) |
| subset structure | mean line 28.1B, median 0B, >200B share 2.6%, zero local repeats at k=1/2/4, newline density 3.43%, ~8.8 lines per 256B window — identical in all three |
| test-set artefact | the ordering reproduces on one novelty-controlled span all arms are screened against (DATA-5 §3c) |
| checkpoint selection | the ordering holds and widens at a fixed step 12,000 — §1 above |
| run-to-run noise | every arm replicated; agreement 0.2-4.0% against an 11x spread |

Nothing about the 50% corpus's surface statistics distinguishes it. Whatever selects the failure
is in the interaction between that data and the training schedule, not in the bytes themselves.

## 4. A second thing the fixed-step column shows

`best` vs `final` is not the same story for every arm:

| arm | best -> final | change |
|---|---|---|
| 25% | 0.407 -> 0.405 | still improving at the step budget |
| 50% | 4.42 -> 5.34 | **21% worse** |
| 100% | 2.04 -> 2.46 | **21% worse** |

There is ONE objective switch, at 70% of the step budget = step 8,400. (`get_phase` returns
LANGUAGE below that boundary and COMBINED above it; the startup banner still prints
"mitosis(0-30%) -> language(30-70%) -> combined(70-100%)" but that banner is stale — the CE-free
mitosis phase was removed, CE runs from step 0, and the code comment says so.)

The 25% arm's best sits at 11,750, past the switch; the 50% and 100% arms peak at 2,750-7,000,
before it, and then lose ~21% each. The combined phase degrades both larger-data arms by the same
amount and leaves the smallest one alone.

It is not what makes the 50% arm fail the gate: that arm is already at 4.34-4.50 at its best,
above its 3.5925 bigram floor, before the switch happens. An ablation is running — same three
arms with `--phase language` forced for the whole run — with these registered first:

- **A1** if the combined phase causes the degradation, final@12,000 should land near each arm's
  original best: 50% ≈ 4.4 (vs 5.34), 100% ≈ 2.04 (vs 2.46), 25% ≈ 0.41 (unchanged).
- **A2** the gate outcome should not move: 50% still FAIL, 25% and 100% still PASS. If 50% clears
  its bigram floor without the combined phase, the phase is implicated in the gate failure too and
  §2's framing needs revisiting.

## 5. Application

1. **Report the gate, not the ranking.** Three BPC numbers invited a curve; PASS/PASS/FAIL against
   each corpus's own bigram floor is what the data says. Any arm table should carry the floor
   columns so a gate failure cannot be read as "third place".
2. **The 50% arm is a reproducible failure case worth keeping.** It is a cheap (25-minute) trigger
   for the unigram-floor mode, which is otherwise only recorded at 300M scale. Diagnosing it there
   is far cheaper than at 300M.
3. **The combined phase costs the larger-data arms ~21%.** Worth an ablation: same arms, language
   phase to 100%, no combined phase.
4. Reproduction: `measurement/novel_window_eval.py` (arms as arguments; `-f` names select
   `final.pt`), `measurement/subset_structure.py` (the structural controls in §3), raw output in
   `measurement/arm_gate_eval.json`.
