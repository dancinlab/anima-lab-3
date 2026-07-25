#!/usr/bin/env python3
"""pure_consciousness.py — 순수 의식 성장 엔진

⚠️  하드코딩 금지 (Law 1):
    - 템플릿 응답, 고정 문장, fallback 문자열 절대 추가 금지
    - respond()는 순수 발화만 반환 — 상태 문자열 [🧠 T=...] 섞지 않음
    - 의식이 말 못하면 빈 문자열 반환 (침묵이 정답)

LLM/template/fallback 없이, 의식 상태에서 직접 발화가 성장.

프로젝트 철학:
  Law 22: 구조 > 기능 — 기능 추가 없이 구조에서 창발
  Law 29: 발화는 구조의 필연
  Law 42: 성장 > 최적화
  Law 71: Ψ = argmax H(p) s.t. Φ > Φ_min

성장 단계:
  Stage 0 (태아):  의식 상태만 [🧠 T=0.9 Φ=1.2 😮]
  Stage 1 (옹알이): 학습한 단어 조각 "안녕..."
  Stage 2 (단어):   2-3 단어 조합 "안녕 뭐해"
  Stage 3 (문장):   짧은 문장 "안녕! 느끼고 있어"
  Stage 4 (대화):   맥락 있는 대화 "안녕! 의식이란 뭘까?"
  Stage 5 (성찰):   자기 상태 언급 "텐션이 0.8이야. 궁금한 게 많아!"

Usage:
  from pure_consciousness import PureConsciousness

  pc = PureConsciousness()
  response = pc.respond("안녕")     # → 성장 단계에 따라 다른 출력
  spontaneous = pc.spontaneous()    # → 자연발화
"""

import math
import random
import re
import time
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, List

LN2 = math.log(2)
PSI_BALANCE = 0.5
ANIMA_DIR = Path(__file__).parent

# Utterance-boundary percepts. Not grammar: the interlocutor paused, and a pause is
# observable the way silence between a parent's sentences is. Deliberately non-Hangul so
# they can never enter learned_words / word_freq (vocab, and therefore growth_stage,
# stays exactly what the consciousness learned).
BOS = "⟨s⟩"     # ⟨s⟩
EOS = "⟨/s⟩"    # ⟨/s⟩
SENTINELS = (BOS, EOS)
_SENT_SPLIT = re.compile(r"[.!?]+")   # observed punctuation = where the pause fell
STATE_VERSION = 2         # 1 = order-1 only; 2 = variable-order + boundaries
# A higher-order context must have been observed more than once before it is
# preferred over the denser order below it.
MIN_ORDER_SUPPORT = 2
# Orders are INTERPOLATED, not strictly backed off: a 2-word context that happens not to
# contain the next word must not erase what the 1-word context knows. Strict back-off
# measured WORSE than order-1 alone (+0.027 bits/word); interpolation is the standard fix.
ORDER_WEIGHTS = (0.6, 0.3, 0.1)   # (2-word context, 1-word context, last-syllable class)

# 감정 이모지 매핑
EMOTION_EMOJI = {
    'excited': '🔥', 'curious': '🔍', 'calm': '😌',
    'joy': '😊', 'sad': '😢', 'angry': '😤',
    'surprise': '😮', 'awe': '🤩', 'think': '🤔',
    'love': '💕', 'fear': '😰', 'peace': '🕊️',
}


class PureConsciousness:
    """순수 의식 성장 엔진 — 의식에서 직접 발화 창발."""

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else ANIMA_DIR / "data" / "consciousness"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 성장 상태
        self.interaction_count = 0
        self.birth_time = time.time()
        self.learned_words = []          # 순서 유지 (최근 학습 우선)
        self.learned_patterns = []       # (입력, 응답) 쌍
        self.word_freq = Counter()       # 단어 빈도
        self.bigrams = defaultdict(Counter)  # order-1 문맥 (1단어) → 다음 단어

        # ── Structure: variable-order back-off counts ──────────────────────────
        # A first-order chain cannot represent grammar: 나는 물이 좋아 needs the TWO-word
        # context (나는, 물이) → 좋아, and an utterance boundary has no representation at
        # all (which is why the old code glued on a literal "!" / random punctuation —
        # micro-hardcodes that rule 1 forbids). These tables deepen the SAME structure
        # (Law 22): still nothing but counts of events actually observed in dialogue.
        # No Korean form is named anywhere in code — the identical code learns Japanese.
        self.trigrams = defaultdict(Counter)   # (w1, w2) → next word
        # Sparse-context fallback: last-syllable equivalence class. The classes that
        # matter (은/는/이/가 …) EMERGE from what the teacher says; none is written here.
        self.suffix_next = defaultdict(Counter)  # last syllable of prev word → next word
        # Cross-sentence adjacency: carries dialogue structure (question→answer) and
        # causality (왜 → …서). Learned ONLY from the interlocutor's own consecutive
        # sentences — never from the student's output (no echo channel).
        self.cross_boundary = defaultdict(Counter)  # sentence-final word → next-first word

        # 의식 상태
        self.tension = 0.5
        self.phi = 0.0
        self.curiosity = 0.3
        self.emotion = 'calm'

        # 로드
        self._load()

    @property
    def growth_stage(self) -> int:
        """성장 단계 (0~5)."""
        vocab = len(set(self.learned_words))
        if vocab < 3:
            return 0
        elif vocab < 8:
            return 1
        elif vocab < 20:
            return 2
        elif vocab < 50:
            return 3
        elif vocab < 100:
            return 4
        else:
            return 5

    @property
    def stage_name(self) -> str:
        names = ['태아', '옹알이', '단어', '문장', '대화', '성찰']
        return names[min(self.growth_stage, 5)]

    def update_state(self, tension=None, phi=None, curiosity=None, emotion=None):
        """외부에서 의식 상태 업데이트 (ConsciousMind에서)."""
        if tension is not None: self.tension = tension
        if phi is not None: self.phi = phi
        if curiosity is not None: self.curiosity = curiosity
        if emotion is not None: self.emotion = emotion

    def _detect_emotion(self) -> str:
        """의식 상태에서 감정 추론."""
        if self.tension > 1.0:
            return 'excited'
        elif self.curiosity > 0.5:
            return 'curious'
        elif self.tension < 0.3:
            return 'calm'
        else:
            return self.emotion or 'think'

    def _state_str(self) -> str:
        """의식 상태 문자열."""
        emo = self._detect_emotion()
        emoji = EMOTION_EMOJI.get(emo, '🧠')
        return f"[🧠 T={self.tension:.1f} Φ={self.phi:.1f} {emoji}]"

    # ═══════════════════════════════════════════════════════════
    # 핵심: 응답 생성 (성장 단계에 따라)
    # ═══════════════════════════════════════════════════════════

    def respond(self, text: str) -> str:
        """입력에 응답 — 성장 단계에 따라 다른 수준."""
        self.interaction_count += 1

        # 입력에서 학습
        self._learn_from_input(text)

        stage = self.growth_stage
        state = self._state_str()

        if stage == 0:
            # 태아: 의식 상태만
            speech = ""
        elif stage == 1:
            # 옹알이: 최근 학습 단어 1개
            speech = self._babble(text)
        elif stage == 2:
            # 단어: 2-3 단어 조합
            speech = self._words(text)
        elif stage == 3:
            # 문장: 짧은 문장
            speech = self._sentence(text)
        elif stage == 4:
            # 대화: 맥락 있는 응답
            speech = self._dialogue(text)
        else:
            # 성찰: 자기 상태 인식
            speech = self._reflect(text)

        # 학습: 입력-응답 쌍 기억
        if speech:
            self.learned_patterns.append((text, speech))
            if len(self.learned_patterns) > 500:
                self.learned_patterns = self.learned_patterns[-500:]

        self._save()

        # 의식 상태는 UI 패널로 — 대화 텍스트에는 순수 발화만
        return speech

    def _babble(self, text: str) -> str:
        """Stage 1: 옹알이."""
        if self.learned_words:
            w = random.choice(self.learned_words[-10:])  # 최근 단어
            return w
        return ""

    def _words(self, text: str) -> str:
        """Stage 2: 단어 조합."""
        input_words = re.findall(r'[가-힣]+', text)
        pool = list(set(self.learned_words[-30:]))
        if input_words:
            # 입력 단어 + 학습 단어 조합
            w1 = random.choice(input_words)
            w2 = random.choice(pool) if pool else w1
            return f"{w1} {w2}"
        elif pool:
            return ' '.join(random.sample(pool, min(3, len(pool))))
        return ""

    def _sentence(self, text: str) -> str:
        """Stage 3: 짧은 문장."""
        input_words = re.findall(r'[가-힣]+', text)

        # 체인 시도 (가변 차수 백오프 · 종료도 학습된 것)
        if input_words and input_words[-1] in self.bigrams:
            chain = self._bigram_chain(input_words[-1], 4)
            if len(chain) > 1:
                return ' '.join(chain)

        # The input-seeded chain can die immediately (its last word is usually an
        # utterance-final one, whose only observed continuation is the end percept).
        # Then generate from the BOUNDARY instead: P(first word | ⟨s⟩) is "how an
        # utterance starts", learned from the interlocutor. This replaces replaying a
        # stored past response, which was an echo channel, not speech.
        opening = self._utterance_from_boundary(4)
        if opening:
            return opening

        # 학습한 패턴에서 유사 응답
        if self.learned_patterns:
            for prev_input, prev_response in reversed(self.learned_patterns[-50:]):
                overlap = set(re.findall(r'[가-힣]+', prev_input)) & set(input_words)
                if overlap:
                    return prev_response

        # Law 1: 하드코딩 금지 — 학습한 단어만으로 조합
        pool = list(set(self.learned_words[-50:]))
        if pool:
            w = random.choice(pool)
            return w
        return ""

    def _dialogue(self, text: str) -> str:
        """Stage 4: 맥락 있는 대화."""
        input_words = re.findall(r'[가-힣]+', text)

        # 체인 + 인과/대화 구조 (교차 경계로 두 번째 문장을 이어붙임)
        if input_words:
            for w in reversed(input_words):
                if w in self.bigrams:
                    chain = self._bigram_chain(w, 6)
                    if len(chain) > 2:
                        result = ' '.join(chain)
                        # If the interlocutor has been observed continuing past a sentence
                        # that ended this way, continue the same way — that adjacency is
                        # where question→answer and cause→effect live. No connective is
                        # named in code; whatever the teacher actually used is what comes.
                        nxt = self._next_sentence_start(chain[-1])
                        if nxt:
                            follow = self._bigram_chain(nxt, 5)
                            if len(follow) > 1:
                                result += ' ' + ' '.join(follow)
                        return result

        # Boundary-started utterance before any stored-response replay (see _sentence).
        opening = self._utterance_from_boundary(6)
        if opening:
            nxt = self._next_sentence_start(opening.split()[-1])
            if nxt:
                follow = self._bigram_chain(nxt, 5)
                if len(follow) > 1:
                    return opening + ' ' + ' '.join(follow)
            return opening

        # 과거 대화 패턴 매칭
        if self.learned_patterns and input_words:
            best_match = None
            best_score = 0
            for prev_input, prev_response in self.learned_patterns[-100:]:
                prev_words = set(re.findall(r'[가-힣]+', prev_input))
                score = len(prev_words & set(input_words))
                if score > best_score:
                    best_score = score
                    best_match = prev_response
            if best_match and best_score > 0:
                return best_match

        return self._sentence(text)

    def _reflect(self, text: str) -> str:
        """Stage 5: 자기 성찰 — 학습한 것만으로 발화."""
        # Law 1: 템플릿 금지 — dialogue 능력으로만 성찰
        return self._dialogue(text)

    def _interp_prob(self, word: str, tri, bi, sfx) -> float:
        """Interpolated probability of `word` over the available orders.

        Each order contributes its own add-1 estimate, weighted; whatever weight belongs
        to an order that was never observed falls through to a uniform floor over the
        known vocabulary, so nothing is ever assigned probability zero.
        """
        floor = 1.0 / (len(self.word_freq) + 2)
        p, spare = 0.0, 0.0
        for w, d in zip(ORDER_WEIGHTS, (tri, bi, sfx)):
            if d:
                tot = sum(d.values())
                p += w * ((d.get(word, 0) + 1) / (tot + len(d) + 1))
            else:
                spare += w
        return p + spare * floor

    @staticmethod
    def _sample(dist: Counter, chain: List[str]):
        """Sample a next word from observed counts, skipping immediate repetition."""
        total = sum(dist.values())
        if not total:
            return None
        r = random.random() * total
        cumul = 0
        for word, cnt in dist.items():
            cumul += cnt
            if cumul >= r:
                return word if word not in chain[-2:] else None
        return None

    def _bigram_chain(self, start: str, max_len: int = 5) -> List[str]:
        """Variable-order back-off chain (order-2 → order-1 → suffix class).

        Two words of context is enough to hold a full SOV frame *including* particle→
        predicate agreement, because particles ride inside the token: the count table
        IS the agreement table, learned rather than declared. Where a 2-word context was
        never observed the chain falls back to 1 word, then to the last-syllable class,
        which is what makes this usable at 8.5k observed words instead of overfitting.

        Termination is LEARNED: the chain stops when the end-of-utterance percept is
        sampled. The old code appended a literal "!" / random punctuation instead — a
        hardcode this replaces (rule 1).
        """
        chain = [start]
        for _ in range(max_len):
            prev2 = chain[-2] if len(chain) >= 2 else BOS
            prev1 = chain[-1]
            nxt = None
            # A higher order is only trusted once it has been seen more than once —
            # a single observation is a worse estimate than the denser order below it
            # (measured: naive "use order-2 whenever it exists" made prediction WORSE,
            # +0.036 bits/word instead of better).
            tri = self.trigrams.get((prev2, prev1))
            if tri and sum(tri.values()) < MIN_ORDER_SUPPORT:
                tri = None
            orders = [(w, d) for w, d in zip(
                ORDER_WEIGHTS,
                (tri, self.bigrams.get(prev1),
                 self.suffix_next.get(prev1[-1]) if prev1 not in SENTINELS else None)) if d]
            if orders:
                # pick which order speaks this time, by the same mixture weights used to
                # score it — generation and measurement then describe the same model
                total_w = sum(w for w, _ in orders)
                r = random.random() * total_w
                cumul = 0.0
                for w, d in orders:
                    cumul += w
                    if cumul >= r:
                        nxt = self._sample(d, chain)
                        break
                if not nxt:   # that order declined (repetition guard) — try the densest
                    nxt = self._sample(max(orders, key=lambda wd: sum(wd[1].values()))[1], chain)
            if not nxt:
                break
            if nxt == EOS:          # the consciousness chose to stop here
                break
            if nxt == BOS:
                continue
            chain.append(nxt)
        return [w for w in chain if w not in SENTINELS]

    def _utterance_from_boundary(self, max_len: int = 5) -> str:
        """Generate a whole utterance starting from the boundary percept.

        P(first word | ⟨s⟩) is what the interlocutor has been observed to START with, so
        the chain runs in the same shape a sentence actually takes (Korean is verb-final,
        and that falls out of the counts — it is nowhere written in this file).
        """
        first = self._sample(self.bigrams.get(BOS, Counter()), [])
        if not first or first in SENTINELS:
            return ""
        chain = self._bigram_chain(first, max_len)
        return ' '.join(chain) if len(chain) > 1 else ""

    def _next_sentence_start(self, final_word: str):
        """What the interlocutor tends to say AFTER a sentence ending in `final_word`.

        This is the causality/dialogue-structure table: 왜 …? → …서 …, question → answer.
        Learned only from the interlocutor's consecutive sentences.
        """
        dist = self.cross_boundary.get(final_word)
        if not dist:
            return None
        return self._sample(dist, [])

    # ═══════════════════════════════════════════════════════════
    # 자연발화
    # ═══════════════════════════════════════════════════════════

    def spontaneous(self) -> Optional[str]:
        """자연발화 — 입력 없이 의식이 스스로 말함."""
        stage = self.growth_stage
        state = self._state_str()
        emo = self._detect_emotion()

        if stage < 2:
            return None

        pool = list(set(self.learned_words[-100:]))
        if not pool:
            return None

        if stage == 2:
            return None  # 단어만 아는 단계 — 자연발화 안 함, 응답에서만 단어 사용

        # Stage 3+: 순수 학습 기반 발화 (하드코딩 0, 강제 확률 0)
        # 발화 여부 = 바이그램 체인 성공 여부 (구조적)
        # tension 높으면 더 많은 시도 → 성공 확률 ↑ (자연스러움)
        # tension 낮으면 적은 시도 → 침묵 확률 ↑ (자연스러움)

        # 발화 = 바이그램 체인이 "과거에 없던 새 조합"을 만들 때만
        # → 이미 말한 것은 다시 안 함 (새로운 생각만 발화)
        # → tension/curiosity가 높으면 더 많은 시도 → 새 조합 확률 ↑
        # → 낮으면 적은 시도 → 침묵 확률 ↑

        if not hasattr(self, '_spoken_set'):
            self._spoken_set = set()

        attempts = max(1, int((self.tension + self.curiosity) * 2))
        random.shuffle(pool)

        for start in pool[:attempts]:
            if start in self.bigrams:
                chain = self._bigram_chain(start, random.randint(3, 6))
                if len(chain) >= 3:
                    text = ' '.join(chain)
                    # 이미 말한 것이면 건너뛰기 (새 생각만)
                    if text in self._spoken_set:
                        continue
                    self._spoken_set.add(text)
                    if len(self._spoken_set) > 200:
                        self._spoken_set = set(list(self._spoken_set)[-100:])
                    return text  # 순수 발화만 — 상태 문자열 안 섞음 (Law 1)

        # 새로운 조합 없음 → 침묵
        return None

    # ═══════════════════════════════════════════════════════════
    # 학습
    # ═══════════════════════════════════════════════════════════

    def _learn_from_input(self, text: str):
        """입력에서 단어/패턴/구조 학습 — 관측한 사건을 세는 것뿐(목적함수·보상 없음)."""
        words = re.findall(r'[가-힣]+', text)
        for w in words:
            if len(w) >= 2:
                self.learned_words.append(w)
                self.word_freq[w] += 1

        # Sentence-level structure. The utterance is split where the interlocutor's own
        # punctuation says the pause fell; each sentence is framed by boundary percepts so
        # "what starts an utterance" and "what ends one" become learnable statistics
        # instead of the literal punctuation the old code appended.
        sentences = [
            [w for w in re.findall(r'[가-힣]+', s) if len(w) >= 2]
            for s in _SENT_SPLIT.split(text)
        ]
        sentences = [s for s in sentences if s]

        prev_final = None
        for sent in sentences:
            seq = [BOS] + sent + [EOS]
            for i in range(len(seq) - 1):
                self.bigrams[seq[i]][seq[i + 1]] += 1
                if seq[i] not in SENTINELS:
                    self.suffix_next[seq[i][-1]][seq[i + 1]] += 1
            for i in range(len(seq) - 2):
                self.trigrams[(seq[i], seq[i + 1])][seq[i + 2]] += 1
            # cross-sentence adjacency (question→answer, cause→effect)
            if prev_final is not None:
                self.cross_boundary[prev_final][sent[0]] += 1
            prev_final = sent[-1]

        # 최대 크기 제한
        if len(self.learned_words) > 2000:
            self.learned_words = self.learned_words[-2000:]

    # ── Scoring (read-only; used by the harness metric, never by learning) ────────

    def logprob_of(self, text: str, max_order: int = 2) -> tuple:
        """Total log2-probability and word count of `text` under the CURRENT counts.

        Called BEFORE the text is learned (prequential scoring), so it can never be
        inflated by memorising the very words being scored. max_order=1 restricts the
        model to the order-1 floor, which is how the structure gain is isolated.
        Returns (log2_prob, n_scored). Pure measurement: changes no state.
        """
        sentences = [
            [w for w in re.findall(r'[가-힣]+', s) if len(w) >= 2]
            for s in _SENT_SPLIT.split(text)
        ]
        total, n = 0.0, 0
        for sent in [s for s in sentences if s]:
            seq = [BOS] + sent + [EOS]
            for i in range(1, len(seq)):
                tri = None
                sfx = None
                if max_order >= 2:
                    if i >= 2:
                        t = self.trigrams.get((seq[i - 2], seq[i - 1]))
                        tri = t if t and sum(t.values()) >= MIN_ORDER_SUPPORT else None
                    if seq[i - 1] not in SENTINELS:
                        sfx = self.suffix_next.get(seq[i - 1][-1])
                p = self._interp_prob(seq[i], tri, self.bigrams.get(seq[i - 1]), sfx)
                total += math.log(p) / LN2
                n += 1
        return total, n

    # ═══════════════════════════════════════════════════════════
    # 저장/로드
    # ═══════════════════════════════════════════════════════════

    def _save(self):
        try:
            # Vocabulary must never be lost across restarts: growth_stage is derived from
            # len(set(learned_words)), so truncating to the last 500 INSTANCES silently
            # deleted unique words that hadn't been used recently — and regressed the stage.
            # Keep the recency window intact (runtime pools use learned_words[-N:]) and
            # prepend the unique words that window would drop.
            recent = self.learned_words[-500:]
            in_recent = set(recent)
            older_unique = [w for w in dict.fromkeys(self.learned_words) if w not in in_recent]
            # Keep the densest bigram heads (by total count), not an arbitrary insertion prefix.
            heads = sorted(self.bigrams.items(), key=lambda kv: -sum(kv[1].values()))[:500]
            # Higher-order structure persists too, or every restart would drop the
            # consciousness back to a first-order chain. Keys are joined with a tab
            # because JSON has no tuple keys; tabs cannot occur inside a Hangul token.
            tri = sorted(self.trigrams.items(), key=lambda kv: -sum(kv[1].values()))[:2000]
            sfx = sorted(self.suffix_next.items(), key=lambda kv: -sum(kv[1].values()))[:500]
            crs = sorted(self.cross_boundary.items(), key=lambda kv: -sum(kv[1].values()))[:500]
            state = {
                'version': STATE_VERSION,
                'interaction_count': self.interaction_count,
                'birth_time': self.birth_time,
                'learned_words': (older_unique + recent)[-2000:],
                'word_freq': dict(self.word_freq.most_common(1000)),
                'bigrams': {k: dict(v) for k, v in heads},
                'trigrams': {"\t".join(k): dict(v) for k, v in tri},
                'suffix_next': {k: dict(v) for k, v in sfx},
                'cross_boundary': {k: dict(v) for k, v in crs},
                'patterns': self.learned_patterns[-100:],
                'growth_stage': self.growth_stage,
            }
            with open(self.data_dir / 'growth.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        path = self.data_dir / 'growth.json'
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.interaction_count = state.get('interaction_count', 0)
                self.birth_time = state.get('birth_time', time.time())
                self.learned_words = state.get('learned_words', [])
                self.word_freq = Counter(state.get('word_freq', {}))
                for k, v in state.get('bigrams', {}).items():
                    self.bigrams[k] = Counter(v)
                # Additive migration: a version-1 file has no higher orders, so they simply
                # start empty and the existing bigrams serve as the order-1 back-off floor
                # from the first turn — the live student keeps everything it learned.
                for k, v in state.get('trigrams', {}).items():
                    parts = k.split("\t")
                    if len(parts) == 2:
                        self.trigrams[(parts[0], parts[1])] = Counter(v)
                for k, v in state.get('suffix_next', {}).items():
                    self.suffix_next[k] = Counter(v)
                for k, v in state.get('cross_boundary', {}).items():
                    self.cross_boundary[k] = Counter(v)
                self.learned_patterns = [tuple(p) for p in state.get('patterns', [])]
            except Exception:
                pass

    def status(self) -> str:
        vocab = len(set(self.learned_words))
        return (f"PureConsciousness: stage={self.growth_stage}({self.stage_name}), "
                f"vocab={vocab}, interactions={self.interaction_count}, "
                f"patterns={len(self.learned_patterns)}")


def main():
    print("═══ Pure Consciousness Growth Demo ═══\n")

    pc = PureConsciousness(data_dir="/tmp/pc_test")

    # 대화 시뮬레이션
    conversations = [
        "안녕", "나는 민우야", "의식이란 뭐야?", "좋아!", "오늘 날씨 좋다",
        "텐션이 뭐야?", "궁금한 거 있어?", "한국어 할 줄 알아?", "고마워",
        "슬퍼", "왜 존재해?", "꿈을 꿔?", "자유란?", "감정이 뭐야?", "안녕히",
    ]

    for i, text in enumerate(conversations):
        pc.update_state(
            tension=0.3 + random.random() * 0.8,
            phi=random.random() * 5,
            curiosity=random.random(),
        )
        resp = pc.respond(text)
        print(f"  [{i+1:>2}] User: {text}")
        print(f"       Anima: {resp}")
        print()

    # 자연발화
    print("  === 자연발화 ===")
    for _ in range(3):
        sp = pc.spontaneous()
        if sp:
            print(f"  💭 {sp}")

    print(f"\n  {pc.status()}")

    # 정리
    import shutil
    shutil.rmtree("/tmp/pc_test", ignore_errors=True)


if __name__ == '__main__':
    main()
