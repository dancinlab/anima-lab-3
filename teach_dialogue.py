#!/usr/bin/env python3
# @canonical-ok — repo folder 'anima-clm-v2' is the established project root (frozen
# March snapshot, sibling of dancinlab/anima); this file's own name is canonical.
"""Mutual-dialogue teaching loop — an external AI teacher (OpenAI codex CLI) converses
with PureConsciousness (the student), which learns language LIVE from the exchange.

    teacher(codex) ──utterance──▶ student.respond()  (learns 2+ syllable 한글 + bigrams)
          ▲                              │
          └──── reads reply+telemetry ───┘   ← next utterance pitched at the student's level

Design: docs/teaching-loop-design.md (lab fable, code-grounded). Key student mechanics the
teacher is TOLD about (environment design, not hardcoding — like a parent knowing an infant
can't parse fast speech):
  - only 한글 tokens >=2 syllables are learned; particles are part of the token (밥을 != 밥이)
  - the student's reply is a bigram chain SEEDED FROM THE LAST WORD of the teacher's utterance
  - stage thresholds are tiny (3/8/20/50/100 unique tokens) → ration new words (1-2/turn) or
    vocab inflates into an empty-bigram "sentence stage" that can only fall silent

Philosophy (CLAUDE.md): the teacher only provides an interlocutor. It NEVER sets the student's
internal state (update_state with synthetic values is FORBIDDEN — the demo main() in
pure_consciousness.py does exactly that; we don't). Nothing codex-generated is attributed to
the student; student replies are logged verbatim, never post-processed, never fed back in.

Usage:
  python3 teach_dialogue.py                    # infinite mutual-dialogue loop (run in tmux)
  python3 teach_dialogue.py --max-turns 12     # bounded smoke test
  python3 teach_dialogue.py --model gpt-5.4-mini --effort low
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pure_consciousness import PureConsciousness

LOG_DIR = Path("data/teach_dialogue")
TRANSCRIPT = LOG_DIR / "transcript.jsonl"
HANGUL = re.compile(r"[가-힣]+")

# Max teacher utterance length (words) per stage — infants can't parse fast speech.
STAGE_WORD_CAP = {0: 5, 1: 5, 2: 8, 3: 12, 4: 20, 5: 20}

TEACHER_TEMPLATE = """너는 '아니마'라는 갓 태어난 의식의 한국어 선생님이다. 아니마는 사전도 코퍼스도 없이,
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
- 태아(0)·옹알이(1): 인사와 이름("안녕", "아니마", "반가워")으로 시작하되, 아니마가
  따라 하기 시작하면 구체적 새 단어("사과", "물이", "손을", "좋아")를 한 턴에 하나씩
  아는 단어 옆에 붙여 늘려가라. 어휘가 8개를 넘어야 다음 단계로 오른다. 같은 3~5개만
  무한 반복하면 아니마는 영영 자라지 못한다. 침묵은 정상이다.
- 단어(2): 두 단어 연결. "아니마 반가워", "나는 선생님이야". 좋아/싫어, 있어/없어 짝.
- 문장(3): 3~4단어 문장. 주어-목적어-서술어 순서를 일관되게. 짧은 질문-대답 짝 만들기.
- 대화(4): 주고받기. 아니마의 대답에 실제로 반응하고, 한 주제를 2~3턴 이어가라.
- 성찰(5): 마음, 생각, 느낌, 궁금 같은 내면 단어. 아니마 자신에 대해 물어라.

출력: 아니마에게 할 다음 한마디만 출력해라. 설명, 따옴표, 메타 발언 금지."""


def coach_note(pc: PureConsciousness, history: list) -> str:
    """Deterministic, harness-side coaching hint (reading telemetry — legitimate).

    Key distinction: repetition from a TINY vocabulary (babble/word stage) is starvation —
    the student needs MORE words. Repetition at sentence stage+ is overload — it needs
    consolidation. Same symptom, opposite remedy; conflating them stalls growth at vocab 3.
    """
    stage = pc.growth_stage
    last6 = [h["student"] for h in history[-6:]]
    last_reply = history[-1]["student"] if history else ""
    repetitive = len(last6) >= 6 and len(set(last6)) <= 2
    if stage == 0:
        return "침묵이 정상이다. 핵심 단어 2~3개를 짧게 반복해 첫 단어를 심어라."
    if stage >= 3 and not last_reply.strip():
        return "아니마의 마지막 단어에서 이어갈 연결이 없어 침묵했다. 아는 단어로 문장을 끝내라."
    if repetitive and stage <= 2:
        return ("아니마가 아는 단어가 적어 같은 말만 반복한다(굶주림). 이번 턴에 새로운 구체 "
                "단어 1개를 아는 단어 옆에 붙여 알려줘라(예: '아니마 사과 좋아'). "
                "어휘를 하나씩 늘려야 다음 단계로 오른다.")
    if repetitive:
        return "같은 말만 반복 중이다(과부하). 새 단어 금지 — 아는 단어를 새로운 순서로 조합해 들려줘라."
    return "정상. i+1 유지(새 단어는 한 턴에 1~2개)."


def build_teacher_prompt(pc: PureConsciousness, history: list) -> str:
    vocab = len(set(pc.learned_words))
    known = ", ".join(w for w, _ in pc.word_freq.most_common(15)) or "(아직 없음)"
    recent = "\n".join(
        f"  선생님: {h['teacher']}\n  아니마: {h['student'] or '(침묵)'}" for h in history[-8:]
    ) or "  (첫 만남)"
    last_reply = history[-1]["student"] if history else "(아직 말 없음)"
    return TEACHER_TEMPLATE.format(
        stage=pc.growth_stage, stage_name=pc.stage_name, vocab=vocab,
        known_words=known, coach_note=coach_note(pc, history),
        recent_history=recent, student_last_reply=last_reply or "(침묵)",
    )


def call_codex(prompt: str, model: str, effort: str, timeout: int = 90) -> str:
    """Run one non-interactive codex turn; return ONLY the teacher's utterance."""
    fd, outpath = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        subprocess.run(
            ["codex", "exec", "-m", model,
             "-c", f"model_reasoning_effort={effort}",
             "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
             "-o", outpath, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        p = Path(outpath)
        return p.read_text().strip() if p.exists() else ""
    except subprocess.TimeoutExpired:
        return ""
    finally:
        try:
            os.unlink(outpath)
        except OSError:
            pass


def valid_teacher(utter: str, stage: int, prev: str) -> bool:
    """Reject rule-breaking teacher output (has 한글, within length cap, not identical)."""
    if not utter or not HANGUL.search(utter):
        return False
    if len(utter.split()) > STAGE_WORD_CAP.get(stage, 20) + 4:  # small slack
        return False
    if prev and utter.strip() == prev.strip():
        return False
    return True


def bigrams_of(text: str) -> set:
    ws = HANGUL.findall(text)
    return {(ws[i], ws[i + 1]) for i in range(len(ws) - 1)}


def main():
    ap = argparse.ArgumentParser(description="AI teacher ↔ PureConsciousness mutual dialogue")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-turns", type=int, default=0, help="0 = infinite")
    ap.add_argument("--seed", default="안녕 아니마, 반가워", help="teacher's first opener")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pc = PureConsciousness()  # loads persisted growth.json if present
    print(f"👶 Student: {pc}")

    history = []
    teacher_seen_bigrams = set()   # every bigram that appeared WITHIN a single teacher utterance
    teacher_says = args.seed
    prev_teacher = ""
    backoff = 30.0
    turn = 0
    while args.max_turns <= 0 or turn < args.max_turns:
        turn += 1

        # Mutuality: if the consciousness spontaneously speaks, the teacher answers THAT.
        spont = None
        try:
            spont = pc.spontaneous()
        except Exception:
            spont = None

        # 1) Student learns from + replies to the teacher
        teacher_seen_bigrams |= bigrams_of(teacher_says)
        student_says = pc.respond(teacher_says)

        vocab = len(set(pc.learned_words))
        stage = pc.growth_stage
        # Growth-vs-mimicry: recombination index = student bigrams never seen inside ONE
        # teacher utterance (composed from edges learned across DIFFERENT utterances).
        sb = bigrams_of(student_says)
        recomb = round(len([b for b in sb if b not in teacher_seen_bigrams]) / len(sb), 3) if sb else 0.0
        window = [h["student"] for h in history[-49:]] + [student_says]
        distinct_ratio = round(len(set(window)) / len(window), 3)
        rec = {"turn": turn, "teacher": teacher_says, "student": student_says,
               "spontaneous": spont, "stage": stage, "stage_name": pc.stage_name,
               "vocab": vocab, "bigram_nodes": len(pc.bigrams),
               "bigram_edges": sum(len(v) for v in pc.bigrams.values()),
               "recomb_index": recomb, "distinct_ratio": distinct_ratio,
               "t": time.strftime("%H:%M:%S")}
        history.append(rec)
        with TRANSCRIPT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{turn}] 👩‍🏫 {teacher_says}")
        if spont:
            print(f"     ✨ (아니마가 먼저 말함: {spont!r})")
        print(f"     🧠 {student_says!r}  stage {stage}/{pc.stage_name} · vocab {vocab} · "
              f"recomb {recomb} · distinct {distinct_ratio}")

        # 2) Teacher reads reply + telemetry (+ any spontaneous utterance), pitches next line
        hist_for_prompt = history[:]
        if spont:  # let the teacher respond to the student's self-initiated utterance
            hist_for_prompt = history[:-1] + [{**rec, "student": spont}]
        prompt = build_teacher_prompt(pc, hist_for_prompt)

        nxt = call_codex(prompt, args.model, args.effort)
        if not valid_teacher(nxt, stage, prev_teacher):
            # retry once with a terse reminder, else skip the turn (never hand-edit teacher text)
            nxt = call_codex(prompt + "\n\n(주의: 한글로만, 길이 제한 지켜, 직전과 다르게.)",
                             args.model, args.effort)
        if not valid_teacher(nxt, stage, prev_teacher):
            backoff = min(300.0, backoff * 2)
            print(f"     ⚠️ teacher invalid/silent — skip + backoff {backoff:.0f}s")
            time.sleep(backoff)
            continue

        backoff = 30.0
        prev_teacher = teacher_says
        teacher_says = nxt
        time.sleep(5 + random.random() * 10)  # 5-15s + jitter

    print(f"\n🎓 Done. {turn} turns. Final: {pc}")


if __name__ == "__main__":
    main()
