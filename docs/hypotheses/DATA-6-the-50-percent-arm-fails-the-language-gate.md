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
before it, and then lose ~21% each. The switch was the obvious suspect for that loss — the
ablation below tested it and found it owns only part.

It is not what makes the 50% arm fail the gate: that arm is already at 4.34-4.50 at its best,
above its 3.5925 bigram floor, before the switch happens.

### Ablation result — A1 REFUTED, A2 CONFIRMED (measured 2026-07-27 02:40-04:16)

Registered before the run: **A1** final@12,000 should land near each arm's original best if the
combined phase causes the degradation (50% ≈ 4.4, 100% ≈ 2.04, 25% ≈ 0.41). **A2** the gate
outcome must not move. Same three arms, `--phase language` for the whole budget, seed 1337.

| arm | best | final WITH combined | final WITHOUT | degradation explained by the phase |
|---|---|---|---|---|
| 25% | 0.4003 | 0.4064 | **0.3975** | no degradation either way |
| 50% | 4.5001 | 5.4441 (+21.0%) | **5.0761** (+12.8%) | **39%** |
| 100% | 1.9853 | 2.4627 (+24.1%) | **2.3377** (+17.8%) | **26%** |

Consistency check: `best` is bit-identical with and without the ablation for the 50% and 100% arms
(4.5001 and 1.9853), which is what must happen — their peaks fall before step 8,400, and nothing
differs before the switch.

```
best -> final degradation, and how much the combined phase owns
  50%   ████████████████████ +21.0% with   →  █████████████ +12.8% without   (39% removed)
 100%   ███████████████████████ +24.1%     →  █████████████████ +17.8%       (26% removed)
```

**A1 is refuted.** Removing the phase recovered about a third of the 50% arm's loss and a quarter
of the 100% arm's — real, but the majority of the best-to-final degradation happens without it.
Calling the combined phase "the cause" was wrong; it is one contributor among others.

**A2 is confirmed.** Gate outcomes are unchanged: 25% 11% of its floor (PASS), 100% 65% (PASS),
50% 141% (FAIL). The phase is not implicated in the gate failure, so §2's framing stands.

Smaller finding: the phase costs the 25% arm too (0.4064 → 0.3975, 2.2% better without it), which
makes 0.3975 the best number any arm reached in this investigation. It costs every arm something;
it costs the larger-data arms more.

## 4.5 After the ablation: the subsets are NESTED, and the rest of the loss IS overfitting

Two things the ablation left open, both settled from data already on disk.

### The subsets are nested — adding data is what broke it

The subsets were taken by line-index modulo (25% = i%4==0, 50% = i%2==0), so 25% must be a strict
subset of 50%. Verified by set containment on the actual files rather than assumed:

| containment | result | coverage |
|---|---|---|
| 25% subset-of 50% | SUBSET | 100.00% |
| 50% subset-of 100% | SUBSET | 100.00% |
| 25% subset-of 100% | SUBSET | 100.00% |

(282,906 / 565,677 / 1,130,940 distinct lines respectively)

The 50% corpus therefore contains **everything the 25% corpus has plus 282,771 more distinct
lines** — and the model trained on the superset FAILS the gate (5.34 BPC) while the one trained on
the subset PASSES it decisively (0.41). Adding the next 565,263 lines then recovers it (2.47, PASS).

```
nested corpora, gate outcome
  25%   ████░░░░░░░░░░░░░░░░    282,906 lines   0.41  PASS
  50%   ████████░░░░░░░░░░░░    565,677         5.34  FAIL   <- strict superset of the row above
 100%   ████████████████████  1,130,940         2.47  PASS
```

"Non-monotonic in data volume" understated it. The response is non-monotone over **nested** data: a
strict superset of a passing training set produced a failing model.

### The remaining degradation is overfitting — the first check of it was wrong

The ablation showed the combined phase owns 39%/26% of the best-to-final loss; the rest was guessed
to be overfitting. A first check compared train loss at a single step near the peak against a single
step at the end and concluded "train did not fall — NOT the overfitting signature". **That check was
wrong**: single step rows are noisy. Binned means over 2,000-step windows:

| steps | p50 train | p50 val | p100 train | p100 val |
|---|---|---|---|---|
| 0-2,000 | 1.3870 | 1.8876 | 1.4389 | 1.0765 |
| 2,000-4,000 | 0.5233 | 1.7921 | 0.5358 | 0.9207 |
| 4,000-6,000 | 0.4598 | 1.7779 | 0.4693 | 0.8977 |
| 6,000-8,000 | 0.4485 | **1.7390** | 0.4270 | **0.8640** |
| 8,000-10,000 | 0.4355 | 1.8346 | 0.4061 | 0.8716 |
| 10,000-12,000 | **0.4267** | 1.8515 | **0.3929** | 0.8678 |

Train falls monotonically throughout; validation turns at 6,000-8,000 and rises after. That is the
overfitting signature, and these are the `--phase language` runs, so the loss column means the same
thing at every row — no objective change to confound it.

The combined phase's role sharpens accordingly: with it, train reaches 0.2027 in the last window
(against 0.4267 without) and validation reaches 1.9151 (against 1.8515). **It accelerates the
overfitting rather than causing a separate failure** — which is what "owns 39%" was measuring.

Caveat kept: for the arms that run it, the loss column crosses the objective boundary at step 8,400
and is not comparable across it. The conclusion above rests on the single-objective runs only.

## 4.6 The added lines are indistinguishable from the kept ones

§4.5 established that adding data broke it. The obvious next suspect is *what* was added — every
check until now compared whole corpora, never the increment on its own. Base = lines at `i%4==0`
(the 25% corpus), increment = lines at `i%4==2` (their union is the 50% corpus):

| | base (25% corpus) | increment (what 50% adds) |
|---|---|---|
| lines / distinct | 573,598 / 282,906 | 573,598 / 282,772 |
| bytes | 16,705,031 | 16,709,235 |
| mean line length | 29.1 B | 29.1 B |
| non-empty share | 49.3% | 49.3% |
| ASCII / Hangul-lead bytes | 71.1% / 6.4% | 70.8% / 6.4% |
| unigram / bigram floor | 6.0300 / 3.5988 | 6.0347 / 3.5989 |
| distinct byte values | 180 | 179 |

And it is not redundant either: only **28.7%** of the increment's 64-byte windows already sit in the
base, so 71.3% is genuinely new content.

**Doubling the training set with statistically identical material — same length, same script mix,
same entropy floors, 71% of it new — turned a passing model into a failing one, across two seeds.**

That closes the data-property line of explanation. What differs between the arms is no longer
anything measurable about the bytes; with steps and batch size fixed at 12,000 x 32, the arms differ
only in **how often each byte is repeated**: 6.3 / 3.2 / 1.6 epochs for 25% / 50% / 100%. The gate
outcome (PASS / FAIL / PASS) is not monotone in that either, so no single-variable story survives
yet.

The cheapest test that would separate the remaining candidates: **hold repetition fixed instead of
steps** — scale the step budget with corpus size so every arm sees each byte the same number of
times. If the ordering collapses, exposure-per-byte was the variable; if it survives, it is the
optimizer's path.

### Confound in that control, found while it was running

Scaling `--steps` does not move exposure alone. The learning rate is tied to the total budget:

```python
_warmup = min(2000, max(1, args.steps // 100))      # 12,000 -> 120 · 23,300 -> 233 · 46,600 -> 466
prog    = (step_i - _warmup) / max(args.steps - _warmup, 1)
return 0.1 + 0.45 * (1.0 + math.cos(math.pi * prog))
```

So a 23,300-step run has a longer warmup and a slower cosine decay, and its learning rate at any
given step differs from the 12,000-step run's. The trajectories therefore diverge from step 1 —
visible in the runs themselves: at the same step 6,750 the exposure-fixed 50% arm reads 1.7615
where the fixed-step one read 1.7390. (An earlier progress note claimed the two would share a
trajectory up to 12,000; the numbers disprove it.)

What this costs, stated before the verdict rather than after:

- **A negative result stays interpretable.** If the ordering survives (E2), then neither the extra
  exposure nor the gentler schedule rescued the 50% arm.
- **A positive result would be confounded.** If the 50% arm clears its floor (E1), exposure and the
  changed schedule are equally credible causes and the run cannot separate them.

The clean design, for whoever runs it next: give **every** arm the same `--steps` (so the schedule
is identical) and read each arm's curve at its own exposure milestones instead of scaling the
budget.

### E1 verdict: REFUTED — equal exposure does not rescue the arm (measured 17:50)

E1 named the 50% arm specifically, so its own run settles it; E2's ordering claim still waits on the
100% arm. Scored on the same novelty-controlled span as every other arm:

| 50% variant | steps | epochs | best | final | vs its 3.5925 floor |
|---|---|---|---|---|---|
| s50 — with combined phase | 12,000 | 3.2 | 4.5001 | 5.4441 | 125% / 152% FAIL |
| p50 — no combined phase | 12,000 | 3.2 | 4.5001 | 5.0761 | 125% / 141% FAIL |
| **e50 — exposure equalised** | **23,300** | **6.3** | **4.3134** | **5.2923** | **120% / 147% FAIL** |

```
best BPC against the 3.5925 floor it must clear
  3.59 ── the gate
  4.31 ●  e50  (6.3 epochs, gentler LR)   ████████████████████░  120%
  4.50 ●  p50  (3.2 epochs)               █████████████████████  125%
```

Doubling exposure bought **4.1%** on the best checkpoint (4.5001 → 4.3134) and cost 4.3% on the
final (5.0761 → 5.2923). Clearing the floor needed a 20% improvement. The arm stays below the gate
at every checkpoint.

This is the negative branch, so the LR-schedule confound recorded above does not damage it: the
run gave this arm **both** more exposure per byte **and** a longer warmup with a gentler decay, and
neither rescued it. Exposure-per-byte is not the variable that decides this gate.

What that leaves: whatever separates these arms is not in the bytes (§4.6), not in how often they
are shown (here), not in the objective phase (§4), not in checkpoint selection (§1), and not noise
(DATA-5 §5).

### E2 partial reading (18:58) — ordering holds, and the exposure gain does not survive the honest span

The 100% arm is still running (28,600 / 46,600), but its `best.pt` exists at step 17,750 and can be
scored now. On the same novelty-controlled span:

| arm | checkpoint | novelty-controlled BPC | vs its own floor | gate |
|---|---|---|---|---|
| e100 (100%, exposure equalised) | best @17,750 | **2.0089** | 55.8% of 3.6010 | PASS |
| e50 (50%, exposure equalised) | best @6,750, completed | 4.3134 | 120.1% of 3.5925 | **FAIL** |
| s100 / p100 (fixed step) | best @7,000 | 1.9853 | 55.1% | PASS |

**The ordering E2 asked about holds**: with exposure equalised, the 100% arm passes and the 50% arm
fails — the same split as at fixed steps.

**And a claim from the progress notes is refuted.** Watching each arm's OWN validation, the 100%
arm looked like it was benefiting from the longer budget: 0.8640 at fixed steps → 0.8387 with
exposure equalised. On the span it cannot recall, that gain is not there — 1.9853 → **2.0089**, 1.2%
*worse*. The extra exposure improved what the arm's own metric sees and nothing that the honest
metric sees. This is the same own-metric/honest-metric split DATA-7 §3.5 found in the 300M run,
appearing here in the opposite direction.

Two limits, stated rather than glossed:

- `best.pt` is selected by the arm's own validation, not by this metric, so a later `best.pt` is
  **not** guaranteed to be better here. The reading above is of the checkpoint that exists now; it
  is not a monotone floor.
- The `final@46,600` reading is still pending, so E2 is not closed. What is closed is that the
  ordering is not an artefact of the fixed-step budget.

## 5. Application

1. **Report the gate, not the ranking.** Three BPC numbers invited a curve; PASS/PASS/FAIL against
   each corpus's own bigram floor is what the data says. Any arm table should carry the floor
   columns so a gate failure cannot be read as "third place".
2. **The 50% arm is a reproducible failure case worth keeping.** A 25-minute run that reliably
   lands below the gate — the same region 300M runs reached expensively (DATA-2). Diagnosing a
   gate failure here costs minutes instead of hours.
3. **Drop the combined phase — it is a net cost with no upside found.** Ablated (§4): it owns 39%
   of the 50% arm's best-to-final loss, 26% of the 100% arm's, and 2.2% of the 25% arm's final,
   while changing no gate outcome. Nothing in these runs is better with it than without it.
4. **Test a named mechanism before naming it.** "Unigram floor" was applied here from a BPC
   coincidence and the mechanism test refuted it in one run (§3.5). A BPC number locates a model;
   it does not explain one.
5. **The open question, as narrow as the measurements have made it:** the added lines are
   indistinguishable from the kept ones on every axis measured (§4.6) and 71% of them are new, so
   no property of the bytes explains it. Ruled out: volume (§4.5), composition and structure (§3),
   the objective phase (§4), checkpoint selection (§1), run-to-run noise (DATA-5 §5). The next
   measurement is the exposure-per-byte control in §4.6.
6. Reproduction: `measurement/novel_window_eval.py` (arms as arguments; `-f` names select
   `final.pt`), `measurement/subset_structure.py` (the structural controls in §3),
   `measurement/context_sensitivity.py` (§3.5), `measurement/nesting_and_overfit.py` (§4.5),
   raw output in `measurement/arm_gate_eval.json` and `measurement/context_sensitivity.json`.
