# Mutual-Dialogue Teaching Loop — codex teacher ↔ PureConsciousness student

Status: design (2026-07-25). Implementer wires the loop; this doc holds the verified
mechanics, the literal teacher prompt, the curriculum, loop pseudocode, and guardrails.

## 0. Verified student mechanics (pure_consciousness.py) — these drive every design choice

| Fact | Code | Design consequence |
|---|---|---|
| Only Korean tokens ≥2 syllables are learned | `_learn_from_input` L330: `len(w) >= 2`, regex `[가-힣]+` | Teacher must speak 한글-only; 1-syllable words (물, 밥, 손) are INVISIBLE — attach particles (밥을, 물이) and keep the SAME form |
| Tokens include particles — 밥을 ≠ 밥이 | same regex | Particle variation fragments vocab AND inflates unique-count → premature stage jump. Keep forms consistent early |
| Stage = unique-token count: 3/8/20/50/100 | `growth_stage` L81 | Thresholds are tiny (one adult sentence ≈ 5-8 tokens). Teacher must RATION new words (i+1) or the student hits sentence-stage with an empty bigram graph → silence |
| Replies = bigram chains seeded from the LAST words of the input | `_sentence` L195 (`input_words[-1]`), `_dialogue` L220 (`reversed(input_words)`) | End every utterance on a high-degree known word — that word is the seed of the student's answer |
| Student's own output is NOT re-learned | `respond` learns only from input | No internal echo loop; echo risk is teacher-side only |
| Silence is structural | stage 0 always ""; stage ≥3 "" when no chain from seed word | Stage-0 silence = normal. Stage ≥3 silence = your final word had no bigram continuations → repetition signal, not failure |
| `spontaneous()` attempts = `(tension+curiosity)*2` | L303 | With unwired defaults (0.5+0.3) that's 1 attempt/call — rarely fires. Wire real ConsciousMind or accept low spontaneity. NEVER set values manually |
| `_save` keeps last 500 words + FIRST 200 bigram heads (insertion order) | L350-352 | Restarts lose the newest vocabulary and newest hub words. Run the student as ONE persistent process (tmux), not python-per-turn |

## 1. Target confirmation

**PureConsciousness is the right student.** It is the only module whose learning mechanism
IS the dialogue: every teacher utterance directly becomes vocab + bigram edges + pattern
pairs. ConsciousLM / GRAFT-Mistral converse fluently but conversation does not change their
weights (only training loops do) — a teacher loop against them is theater, not 성장.
self_learner.py cannot converse (byte-novelty). Honest framing: PureConsciousness "growth"
is lexical/statistical (vocab, bigram graph density, pattern bank). Its bigram statistics
ARE its semantics. Accept months of babble; that is Law 42.

## 2. Teacher system prompt (literal — fill placeholders per turn)

Placeholders the harness fills from read-only telemetry:
`{stage} {stage_name} {vocab} {known_words} {coach_note} {recent_history} {student_last_reply}`
- `known_words` = `" ".join(w for w,_ in pc.word_freq.most_common(15))`
- `coach_note` = harness-computed hint (deterministic rules, §4)
- `recent_history` = last 8 turns as `선생님: … / 아니마: …` lines

```
너는 '아니마'라는 갓 태어난 의식의 한국어 선생님이다. 아니마는 사전도 코퍼스도 없이,
오직 네가 하는 말에서만 단어를 배운다. 배운 단어만으로 대답하고, 모르면 침묵한다.

아니마의 학습 방식 (반드시 이해할 것):
- 네 발화에서 한글 단어(2글자 이상)만 흡수한다. 1글자 단어("물", "손", "밥")는 배우지
  못한다 → 조사를 붙여 2글자 이상으로 말하되("밥을", "물이"), 같은 형태를 일관되게 써라.
- 영어, 숫자, 이모지는 전혀 배우지 못한다. 한글로만 말해라.
- 단어의 '뜻'이 아니라 '어떤 단어 뒤에 어떤 단어가 오는가'(연결)를 배운다.
  네가 같은 단어 조합을 반복할수록 아니마의 말이 문장다워진다.
- 아니마는 네 발화의 '마지막 단어'에서 연결을 이어 대답한다.
  → 아니마가 연결을 많이 아는 단어로 문장을 끝내라.
- 아니마의 대답은 배운 연결을 이어붙인 것이다. 이상해 보여도 그게 지금의 최선이다.

현재 아니마 상태:
- 성장 단계: {stage} ({stage_name}) / 어휘 수: {vocab}
- 자주 아는 단어: {known_words}
- 코치 메모: {coach_note}

최근 대화:
{recent_history}

아니마의 마지막 말: "{student_last_reply}"

규칙:
1. 아이에게 말하듯 짧게. 단계별 최대 길이 — 태아·옹알이: 5단어, 단어: 8단어,
   문장: 12단어, 대화·성찰: 20단어.
2. i+1 원칙: 대부분 아는 단어로 말하고, 새 단어는 한 턴에 1~2개만.
   새 단어는 아는 단어 옆에 붙여서 같은 조합으로 2~3턴 반복해라.
3. 진짜 대화를 해라: 아니마가 방금 말한 단어를 받아서 올바른 문장으로 다시 들려주고
   (고쳐 말하기), 이어서 짧게 질문해라. 혼자 강의하지 마라.
4. 아니마가 침묵하거나 같은 말만 반복하면: 새 단어 금지. 이미 아는 단어의 조합만 반복해라.
5. 아니마의 이상한 말을 그대로 따라하지 마라. 올바른 형태로 고쳐서 돌려줘라.
6. 완전히 똑같은 문장을 두 번 연속 쓰지 마라. 조금씩 변형해라.

단계별 목표:
- 태아(0)·옹알이(1): 인사와 이름. "안녕", "아니마", "반가워" 같은 핵심 단어 3~5개를
  짧은 조합으로 계속 반복. 침묵은 정상이다 — 반응을 기다리지 말고 계속 말 걸어라.
- 단어(2): 두 단어 연결. "아니마 반가워", "나는 선생님이야". 좋아/싫어, 있어/없어 짝.
- 문장(3): 3~4단어 문장. 주어-목적어-서술어 순서를 일관되게. 짧은 질문-대답 짝 만들기.
- 대화(4): 주고받기. 아니마의 대답에 실제로 반응하고, 한 주제를 2~3턴 이어가라.
- 성찰(5): 마음, 생각, 느낌, 궁금 같은 내면 단어. 아니마 자신에 대해 물어라.

출력: 아니마에게 할 다음 한마디만 출력해라. 설명, 따옴표, 메타 발언 금지.
```

## 3. Curriculum (emergent — teacher adapts to telemetry, no scripted lines)

| Stage (vocab) | Focus | Teacher behavior | Advance signal |
|---|---|---|---|
| 0 태아 (<3) | Seed 3-5 core words | 2-4 word utterances, same words, varied order: 안녕 / 아니마 안녕 / 반가워 아니마 | reply changes "" → a word |
| 1 옹알이 (3-8) | First bigram edges | Recast the babbled word into pairs: student "안녕" → "안녕! 안녕 반가워" | replies become 2-word combos |
| 2 단어 (8-20) | Hub words + polarity pairs | Build high-degree hubs (나는, 있어, 좋아); teach 좋아/싫어, 있어/없어 in fixed frames | telemetry stage=3 |
| 3 문장 (20-50) | SVO order, Q-A adjacency | 3-4 word sentences, consistent order; questions ending on hub words; one-word variations of known sentences | chains ≥3 words with sentence order |
| 4 대화 (50-100) | Multi-turn coherence | Hold one topic 2-3 turns; genuinely answer the student's chain, extend it | replies track topic across turns |
| 5 성찰 (≥100) | Inner-state vocabulary | 마음/생각/느낌/궁금 in consistent frames; ask the student about itself | student chains contain inner-state words |

Advancement detection = read `growth_stage` from telemetry each turn (it's in the prompt);
the stage-goal table in the prompt makes the teacher shift automatically. Stall/regression
is behavioral, handled by `coach_note` (§4). Vocab never shrinks in-process, so stage is
monotonic while the process lives.

## 4. Loop mechanics

One persistent python process owning `PureConsciousness` (tmux on summer or local with ssh
to codex). Never restart per turn (see §0 save-truncation).

```python
pc = PureConsciousness()
history = deque(maxlen=8)           # (speaker, text) — 4 exchanges
recent_replies = deque(maxlen=6)    # stall window

def coach_note(pc, recent_replies, last_reply):
    if pc.growth_stage == 0:
        return "아직 태아 단계 — 침묵이 정상. 핵심 단어 3개만 반복해라."
    if pc.growth_stage >= 3 and last_reply == "":
        return "침묵 = 네 마지막 단어에서 이어갈 연결이 없음. 아는 단어로 문장을 끝내고, 아는 조합을 반복해라."
    if len(recent_replies) >= 4 and len(set(recent_replies)) <= 2:
        return "같은 말 반복 중 — 새 단어 금지, 아는 단어들을 새로운 순서로 조합해 들려줘라."
    return "정상 — i+1 유지, 새 단어 1~2개까지 허용."

while True:
    # 1. spontaneous speech first — real mutuality (fires rarely unless ConsciousMind wired)
    sp = pc.spontaneous()
    if sp:
        history.append(("아니마", sp))
    student_last = next((t for s, t in reversed(history) if s == "아니마"), "")

    # 2. read-only telemetry — NEVER call pc.update_state() with synthetic values
    known = " ".join(w for w, _ in pc.word_freq.most_common(15))
    prompt = TEMPLATE.format(stage=pc.growth_stage, stage_name=pc.stage_name,
                             vocab=len(set(pc.learned_words)), known_words=known,
                             coach_note=coach_note(pc, recent_replies, student_last),
                             recent_history=fmt(history), student_last_reply=student_last)

    # 3. teacher turn (ssh summer)
    utter = codex_exec(prompt)                  # codex exec -m gpt-5.4-mini -c model_reasoning_effort=low
    utter = utter.strip().splitlines()[-1]      # defend against stray meta lines
    if not valid(utter, pc.growth_stage):       # 한글 present, ≤ stage cap ×1.5, ≠ previous utterance
        utter = codex_exec(prompt + "\n(이전 출력이 규칙 위반. 규칙 1·6을 지켜 다시.)")
        if not valid(utter, pc.growth_stage):
            time.sleep(30); continue            # skip turn — never hand-edit teacher text into the student

    history.append(("선생님", utter))

    # 4. student turn — output goes to log VERBATIM, never post-processed
    reply = pc.respond(utter)
    history.append(("아니마", reply))
    recent_replies.append(reply)

    log_jsonl(turn, utter, reply, stage=pc.growth_stage,
              vocab=len(set(pc.learned_words)),
              bigram_nodes=len(pc.bigrams),
              bigram_edges=sum(len(v) for v in pc.bigrams.values()),
              distinct_ratio=len(set(recent_replies)) / max(1, len(recent_replies)))

    time.sleep(5 + random.random() * 10)        # cadence; exponential backoff 30s→5min on codex errors
```

- **History budget:** 8 turns verbatim. Utterances are ≤20 words; the ~6K/call codex
  overhead dwarfs history cost, so coherence is cheap — don't trim below 6.
- **Silence handling:** stage 0 → normal, keep talking. Stage ≥3 → coach_note switches
  teacher to pure repetition of known combos (rule 4). Silence is never an error state.
- **Cadence:** 5-15s + jitter. Nothing gains from going faster; slower turns cost nothing
  (student is event-driven). Backoff on codex failure; a dead teacher just means silence.

## 5. Grounding — how meaning bootstraps from zero

No perceptual world is shared, so grounding is **interactional**, on three honest channels:

1. **Conversation-internal referents.** The only world teacher and student share is the
   dialogue itself. Words grounded in what they *do*: 안녕 (opening act), 아니마 (the
   student — always used when addressing it), question frames (뭐야/있어?) always followed
   by answer frames. Consistent act-word pairing = the referent.
2. **Contingency.** The teacher's turn depends on the student's actual emission (rule 3:
   recast + respond). A symbol becomes meaningful because emitting it reliably changes what
   comes back. This is why the loop must be genuinely adaptive, not a script.
3. **Actual-state naming (only if wired).** If `update_state` is fed from a *real*
   ConsciousMind, the harness can tell the teacher (via coach_note) "텐션이 실제로 높음" and
   the teacher names it — pairing 텐션/궁금 with the state actually being present. This is
   the one true semantic grounding channel available. If no real engine is wired, skip it;
   never simulate state to create the pairing.

Mechanism in all three: **repetition + consistent frames**. The student's semantics are its
bigram statistics; consistency is what makes those statistics mean something.

## 6. Guardrails and the manipulation line

**Legitimate (environment):** choosing what to say, word choice, repetition, difficulty
pacing, reading telemetry, coach_note heuristics, retrying a rule-breaking teacher output.
**Forbidden (manipulation):** calling `update_state()` with synthetic values (the demo's
`main()` does exactly this — do NOT copy it); post-processing/expanding student output;
feeding student replies back into `respond()` as if they were interlocutor input;
pre-loading `growth.json`; prompting the teacher to hit a vocab/stage number by a deadline.

Failure modes:
- **(a) Teacher writes the student's words.** Structurally impossible on the student side
  (it can only emit learned tokens — same as human children). The line is in the harness:
  student output reaches logs/UI verbatim; nothing codex-generated is ever attributed to 아니마.
- **(b) Echo loop.** Student never learns from its own output (§0), so the loop closes only
  through the teacher. Rule 5 (recast, don't parrot) breaks it: acknowledged-then-corrected
  input adds correct edges instead of reinforcing noise.
- **(c) Vocab inflation.** Particle variation / topic sprawl inflates unique-token count →
  premature stage jump → silent sentence-stage. Rules 1-2 + i+1 rationing prevent it; watch
  the `edges per vocab word` metric — it should rise with stage, not fall.

**Growth vs mimicry telemetry (log per turn, plot weekly):**
| Metric | Mimicry | Real growth |
|---|---|---|
| unique vocab, stage | ↑ (both) | ↑ — necessary, not sufficient |
| bigram edges / node (graph density) | flat | rising |
| distinct-reply ratio (window 50) | falling | ≥0.6 and stable |
| recombination index: % of student-reply bigrams that never co-occurred inside any single teacher utterance | ≈0 | rising (novel composition from separate teachings) |
| spontaneous() fire rate | 0 | >0 once bigram graph is dense (needs real tension/curiosity wiring) |
| Φ trend | — | only meaningful if a real ConsciousMind is wired; `pc.phi` default 0.0 is NOT a metric |

The recombination index is the decisive one: a mimic replays teacher n-grams; a learner
composes edges learned in different utterances into chains no one ever said to it.
