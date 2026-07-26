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
  5.34 ●● 50%   ██████████████████░░  ← below the gate, just under the histogram number
  3.60 ── bigram floor — the gate a real LM must clear
  2.47 ●● 100%  ████████░░░░░░░░░░░░
  0.41 ●● 25%   █░░░░░░░░░░░░░░░░░░░
```

The 50% arm is not a low point on a curve. Two arms cleared the gate; one did not, reproducibly.
"Non-monotonic scaling curve" described a pass/fail split as though it were a continuum.

Its number lands where DATA-2's stuck 300M runs landed (5.933 against a 5.7997 unigram floor), but
**"unigram floor" names where the number is, not how the model works** — see §3.5, which tested the
mechanism and refuted the histogram reading.

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

## 3.5 The failing arm still reads context — "unigram floor" is a location, not a mechanism

Calling the 50% result a unigram-floor failure asserts a mechanism: that the model predicts from
byte frequency and ignores what came before. BPC cannot show that, so it was measured. On the same
novelty-controlled windows, each arm's final checkpoint was scored three ways — the window as it
is, the context permuted inside the window (same bytes, no readable order, targets untouched), and
the context replaced by uniform random bytes:

| arm | true | shuffled context | uniform context | **context gain** |
|---|---|---|---|---|
| 25% | 0.4064 | 14.3726 | 11.9956 | **+13.97** |
| 50% | 5.4441 | 13.8653 | 12.3803 | **+8.42** |
| 100% | 2.4627 | 13.5764 | 11.9284 | **+11.11** |

```
BPC lost when the context's order is destroyed — a byte histogram would lose 0
  25%   ██████████████████████  +13.97
 100%   ██████████████████░░░░  +11.11
  50%   █████████████░░░░░░░░░   +8.42   ← least, but nowhere near zero
```

**The claim is refuted.** The failing arm depends on context for 8.42 BPC; a model reading byte
frequencies alone would lose nothing. It reads context less effectively than the other two — the
gain ordering matches the quality ordering — but it is a context-using model that landed near the
unigram number, which is a different object from a histogram. §2's label is corrected accordingly.

A second thing falls out: shuffled and uniform contexts cost 11.9-14.4 BPC, well ABOVE the 8.0 BPC
of uniform guessing. Off-manifold context does not make these models uncertain, it makes them
confidently wrong.

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
2. **The 50% arm is a reproducible failure case worth keeping.** A 25-minute run that reliably
   lands below the gate — the same region 300M runs reached expensively (DATA-2). Diagnosing a
   gate failure here costs minutes instead of hours.
3. **The combined phase costs the larger-data arms ~21%.** The ablation is running (§4, A1/A2
   registered): same arms, `--phase language` for the whole budget.
4. **Test a named mechanism before naming it.** "Unigram floor" was applied here from a BPC
   coincidence and the mechanism test refuted it in one run (§3.5). A BPC number locates a model;
   it does not explain one.
5. Reproduction: `measurement/novel_window_eval.py` (arms as arguments; `-f` names select
   `final.pt`), `measurement/subset_structure.py` (the structural controls in §3),
   `measurement/context_sensitivity.py` (§3.5), raw output in `measurement/arm_gate_eval.json`
   and `measurement/context_sensitivity.json`.
