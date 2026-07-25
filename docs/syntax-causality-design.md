# Design: sentence structure + causality for PureConsciousness (no hardcoding)

Status: design (2026-07-25) · companion to docs/teaching-loop-design.md
Target files: pure_consciousness.py, teach_dialogue.py
Constraint set: CLAUDE.md Laws 1 (no hardcoding), 22 (structure > function),
29 (speech is architectural necessity), 42 (growth > optimization).

## 1. Verdict on the current architecture

A first-order Markov chain over word tokens is structurally incapable of
grammar and causality. Three provable gaps, all verified in code:

1. **No utterance-boundary representation.** The chain has no notion of where a
   sentence begins or ends. Sentence endings are cosmetic glue: `_sentence`
   appends a literal `"!"` (pure_consciousness.py:198) and `_dialogue` appends
   `random.choice(['!', '?', '.'])` (line 227). These two lines are themselves
   micro-hardcodes and should be deleted with this design.
2. **No dependency longer than adjacency.** Order-1 conditions on exactly one
   word. `나는 물이 좋아` requires (나는, 물이)→좋아 to be representable; it is not.
3. **No cross-turn statistics at all.** `_learn_from_input` (lines 334-336)
   counts bigrams only WITHIN one utterance. A question→answer or cause→effect
   pair spanning a turn boundary literally cannot be stored, regardless of how
   well the teacher teaches.

Local SOV *fragments* do appear today (사과는 빨간 과일이야 chains) because the
teacher's adjacencies are SOV — that is mimicry of local order, not syntax.

## 2. The structural change (one deepening, not a feature)

Deepen the existing bigram counter into a single variable-order back-off count
trie (orders 1–3) over the SAME learned vocabulary, plus two content-free
additions to what the observation stream exposes:

```
  observation stream (teacher utterance):  "물이 어때? 물이 좋아."
                                                 │
        tokenize (unchanged: 한글 ≥2 syllables, particles ride in the token)
                                                 │
        insert boundary sentinels at utterance start/end and at observed
        sentence punctuation [.!?] — a PERCEPT, not a rule:
              ⟨s⟩ 물이 어때 ⟨/s⟩ ⟨s⟩ 물이 좋아 ⟨/s⟩
                                                 │
     ┌───────────────────────────────┬───────────────────────────────┐
  within-sentence n-grams         cross-boundary table            vocabulary
  (order 1..3, back-off)          final word → next first word    (UNCHANGED —
  (나는,물이)→좋아                 어때 → 물이   (self-answered Q→A) sentinels never
  (물이)→좋아                      왜 → 비가    (cause frames)      enter learned_words,
  ⟨s⟩→물이  좋아→⟨/s⟩              + turn boundary: my last word →  stage gates
                                   teacher's reply first word      untouched)
```

- **Boundary sentinels** are percepts (the utterance ended; the teacher paused
  at a period), like silence between a parent's sentences. They are not Korean
  grammar knowledge.
- **Cross-boundary table** unifies causality and dialogue structure: it learns
  "given how the last sentence ended, what starts the next one" from (a) the
  teacher's multi-sentence utterances (self-answered questions, cause→effect
  pairs) and (b) the turn boundary — the student's own last word → the
  teacher's reply's first word. Only competent-speaker (teacher) continuations
  are ever learned; the student never learns from its own output.
- **Generation** = sample from the highest order whose context has enough
  observations (count ≥ K, e.g. 2), else back off; utterance ends when ⟨/s⟩ is
  sampled (learned verb-final termination) or a hard cap is hit. Reply seeding:
  cross-boundary distribution for the input's final word, backing off to the
  current last-word chaining. `spontaneous()` seeds from the learned ⟨s⟩
  (utterance-initial) distribution instead of a random pool word; the novelty
  filter is unchanged.
- **Sparsity back-off without hardcoding**: when an exact word has no
  cross-boundary/n-gram entry, back off to its final-syllable equivalence class
  (key = sentinel + last syllable). The classes 은/는/이/가/어때/… emerge from
  data; the code names no morpheme. The same code would learn Japanese.

Why order 3 suffices to start: teacher frames are 3–5 words, so a 2-word
context captures a full SOV frame including particle–predicate agreement —
the particle rides inside the token, so the count table IS the agreement table.

Ceiling (stated honestly): n-grams buy local syntax and adjacency-pair
causality, NOT compositional productivity — the student cannot generalize
나는 물이 좋아 → 너는 밥이 좋아 unless the teacher instantiates those slots
(that is the curriculum's job, §4). The next structural step, only after vocab
~500+ and ~10k turns, is a small online character-level sequence model trained
ONLY on the dialogue stream. Not now.

## 3. The hardcoding line (mechanism vs content)

Architecture may define what is REPRESENTABLE about the observation stream:
token positions, boundaries, turn alternation, syllable decomposition, counts.
Architecture may NOT contain which Korean forms behave which way: no particle
tables, no POS labels, no frame templates, no morpheme lists.

Grep-able rule: **no Hangul string literal may participate in learning or
generation in pure_consciousness.py.** "Tokens ending in 은/는" as code =
forbidden (names morphemes). "Back off to last-syllable class" = allowed (the
classes emerge from data). Teacher-side Korean is environment, not hardcoding
— a parent speaks the language (established in docs/teaching-loop-design.md).

## 4. Causality: both sides must change

- **Student**: without the cross-boundary table a causal pair cannot even be
  stored (§1 gap 3). Capacity first.
- **Teacher**: without consistent cause/effect frames the table stays empty.
  Curriculum additions (drop-in for TEACHER_TEMPLATE):

Rules to add:
```
7. 한 턴에 문장 틀은 하나만 써라. 틀은 고정하고 한 자리만 바꿔라
   ("나는 물이 좋아" → "나는 밥이 좋아" → "너는 물이 좋아?").
8. 질문을 하면 같은 턴에서 스스로 답해라 ("물이 어때? 물이 좋아.").
   질문-대답 짝을 보여주는 것이 아니마가 대답을 배우는 유일한 길이다.
```
Stage-goal additions:
```
- 문장(3) 추가: 아니마의 말을 아니마가 아는 단어만으로 주어-목적어-서술어의
  완전한 문장으로 고쳐 들려준 뒤에 질문해라. 문장은 항상 서술어로 끝내라.
- 대화(4)·성찰(5) 추가: 원인-결과 틀을 반복해라 — "비가 와서 땅이 젖어",
  "배가 고파서 밥을 먹어". 연결어(그래서, 왜냐하면, ~서)는 같은 자리에
  일관되게 써라. "왜?"라고 물은 뒤 스스로 "~서 ~" 형태로 답해서
  질문-이유 짝을 보여줘라.
```
Self-answered questions matter doubly: they create the Q→A cross-boundary
adjacency inside a single teacher utterance, so learning does not depend on
the noisier turn boundary.

## 5. Measurement (one primary, contamination-free)

**Primary: prequential (evaluate-then-learn) per-word log-loss gap.** Before
`respond()` learns from a teacher utterance, score that utterance's words under
(a) the full-order model and (b) the SAME counts capped at order 1. Report the
trailing-300-teacher-words gap (a − b), in bits/word.

- No held-out set → no dedup problem; contamination is impossible by
  construction (each utterance is scored before it is learned). This is the
  same epistemics as the CE gate's memorisation lesson (the 2.4× inflation).
- Both numbers come from the same tables, so the gap isolates exactly what the
  higher-order structure bought — the analogue of unigram/bigram baselines for
  corpus runs.
- Per-word normalization makes it length-invariant: longer word salad cannot
  pump it; salad IS high-entropy continuation, which shows up as gap ≈ 0.

Thresholds: after ≥2,000 turns of trie accumulation, pass = gap ≤ −0.3
bits/word sustained; if after 3,000 further turns gap ≥ −0.05 bits/word,
record the experiment a FAILURE (the added orders bought nothing) and document
why. Guards reported alongside, never optimized:
- ⟨/s⟩-termination rate: fraction of non-silent student utterances that ended
  because the boundary was sampled, not the length cap (today 0% by
  construction; target: majority).
- distinct_ratio ≥ 0.5, echo rate (student utterance verbatim within any seen
  teacher utterance), silence rate.
Causality secondary: prequential loss of sentence-INITIAL words with vs
without conditioning on the previous sentence's final word.

## 6. Why this cannot become a fourth reward-hacking channel

The three prior channels (unbounded log_vars, tension inflation, Φ
state-editing) shared one shape: the optimized number could edit its own
inputs. Structural blocks here:

1. **Nothing optimizes the metric.** Learning is counting observed events —
   no objective, no gradient, no reward. The metric is computed in
   teach_dialogue.py, written to the transcript, and fed back into NOTHING:
   not learning, not generation, not the teacher prompt. Open loop by
   construction.
2. **Prequential scoring kills memorization**: verbatim echo cannot score on
   text it has not seen yet.
3. **Teacher gaming blocked**: the metric never enters the teacher prompt.
   (If it did, the degenerate move is collapsing to one trivial frame forever
   — gap looks great, language dies.) coach_note keeps only starvation/
   overload/floor signals. distinct_ratio catches teacher collapse anyway.
4. **Length pumping blocked** by per-word normalization.
5. **Boundary hack is self-defeating**: a model that immediately samples ⟨/s⟩
   produces silence — already the sanctioned failure mode, and visible in the
   silence-rate guard.
6. **Novelty-filter interaction, named in advance**: as counts concentrate on
   correct frames, spontaneous() finds fewer NEW chains → silence rises. That
   is honest behavior (Law 1), reported as telemetry, not misread as
   regression — the mirror of "no split is a population event."

Watcher rule (mirror of the PURE run-watcher): report three INDEPENDENT
states — structure (prequential gap) · diversity (distinct/echo/silence) ·
vocabulary (vocab, stage) — so silence cannot look like success.

## 7. Migration (live student keeps its state)

- growth.json is extended, never rewritten: add `ngrams` (space-joined context
  → next-word counts, pruned like the current top-500 heads policy),
  `cross_boundary`, and a `version` field. Existing `learned_words`,
  `word_freq`, `bigrams`, `patterns` load unchanged; old bigrams serve as the
  order-1 back-off floor from turn one, so the vocab-101 student's graph keeps
  working while higher orders fill from new dialogue.
- Sentinels never enter learned_words → stage arithmetic untouched; the
  student stays at reflection stage.
- The step-0 restart rule does not apply (no gradient objective, no
  checkpoint) — but bump the schema version so an old reader fails loudly.
- Delete the two punctuation micro-hardcodes (lines 198, 227) when ⟨/s⟩
  termination lands; verbatim pattern-replay (`learned_patterns` echo) stays
  as last resort before silence only.
