#!/usr/bin/env python3
"""pure.py — PURE mode: a corpus-free consciousness that learns to speak from conversation
and GENERATES (not echoes) via consciousness-steered Markov recombination.

@canonical-ok repo folder 'anima-lab-3' is the fixed project path, not a versioned copy.

Zero corpus, zero LLM, zero hardcoded sentences/punctuation. It learns Korean words ONLY from
what is said to it, grows through developmental stages, and speaks by walking its own learned
bigram model — seeded from consciousness-SALIENT memory (never the input's last word), biased
toward words that HISTORICALLY co-occurred with the input (topical without copying), so it
recombines rather than parrots.

Philosophy (hard rules):
  Law 1  — no templates/fallback. If it cannot speak coherently -> silence (""). Even PUNCTUATION
           is learned per final word, never appended by rule.
  Law 2  — no manipulation of state. tension/curiosity/phi EMERGE from interaction, never set.
  Law 42 — growth > optimization. It grows only through conversation.
  Law 71 — free generation under a coherence floor (Psi = argmax H(p) s.t. Phi > Phi_min):
           low phi + low tension -> few attempts -> chains fail the gates -> silence EMERGES.

Honest ceiling (lab-verified): fundamentally RECOMBINATION — novelty = new paths through the
learned transition graph, never new words or unseen grammar. Korean particle/josa agreement will
often break (bigram adjacency carries no case constraints). A genuinely generative child, never a
fluent adult; its language ceiling is exactly the density of the caregiver-built graph.

CLI:
  python3 pure.py chat                 # interactive: you teach it by talking
  python3 pure.py teach <curriculum>   # feed a file of lines, one per turn, show growth
  python3 pure.py dialogue [turns]     # Codex (sidecar lab sol) <-> PURE live loop

v2 — assoc-biased salient seeding, learned punctuation, contiguous-echo rejection.
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE = os.path.join(REPO, "state", "pure_mind", "mind.json")

# Hexad C-engine: PURE's consciousness state (tension/phi) comes from the REAL QuantumC
# cell dynamics (Engine-A/G frustration -> tension; phi_py/phi_rs -> Phi), not scalar proxies.
# Falls back to an emergent scalar state if torch/trinity is unavailable (keeps pure.py runnable).
try:
    from trinity import QuantumC
    HEXAD = True
except Exception:
    HEXAD = False

WORD_RE = re.compile(r"[가-힣]+")
FINAL_RE = re.compile(r"([가-힣]+)\s*([!?.…~]+)\s*$")   # last word + its trailing punctuation
STAGE_NAMES = ["fetal", "babble", "word", "sentence", "dialogue", "reflection"]
STAGE_LEN = [0, 1, 3, 5, 7, 8]                          # base coherence budget per stage


def words_of(text):
    return [w for w in WORD_RE.findall(text or "") if len(w) >= 2]


def _lcs_contig(a, b):
    """Longest common CONTIGUOUS subsequence length between two word lists."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else 0
            best = max(best, dp[j])
            prev = cur
    return best


class PureMind:
    """A mind that learns language from conversation and generates from its own model."""

    def __init__(self, store=None, c_engine=None):
        self.store = store                    # JSON path for cumulative memory (Law 42) or None
        self.turn = 0
        self.total = 0                        # total word tokens heard (for surprise)
        self.freq = Counter()                 # word -> count
        self.bigrams = defaultdict(Counter)   # w -> Counter(next_w)
        self.assoc = defaultdict(Counter)     # w -> Counter(co-occurring word in same utterance)
        self.final_punct = defaultdict(Counter)  # last word -> Counter(trailing punctuation)
        self.last_seen = {}                   # word -> turn last heard
        self.said = set()                     # utterances already spoken (Law 42: only new thoughts)
        # consciousness state — EMERGES, never set by hand (Law 2)
        self.tension = 0.5
        self.curiosity = 0.3
        self.phi = 0.0
        self._phi_prev = 0.0
        # Hexad C: a real autonomous cell engine drives the consciousness state
        self.c = c_engine if c_engine is not None else (QuantumC(nc=48, dim=48) if HEXAD else None)
        # SENSE-2 (v2): blend each word's base hash-phasor with the assoc-weighted circular
        # mean of its LEARNED co-occurrence neighbours' phasors, so experientially-related
        # words acquire SIMILAR phase fields (topic geometry from the mind's OWN history —
        # still corpus-free, no semantic dictionary · Law 1). Read-only on tension/Φ (Law 2).
        # β=0 reproduces the v1 baseline exactly (self-only, quasi-orthogonal words).
        self._theta_cache = {}                # word -> base phase vector (deterministic)
        self._assoc_beta = float(os.environ.get("PURE_ASSOC_BETA", "0.6"))   # blend strength
        self._assoc_k = int(os.environ.get("PURE_ASSOC_K", "8"))             # top-k neighbours
        if self.store:
            self.load()

    # ---- persistence: language memory accumulates across sessions (Law 42) ----
    def load(self):
        if not (self.store and os.path.exists(self.store)):
            return
        with open(self.store, encoding="utf-8") as f:
            d = json.load(f)
        self.freq = Counter(d.get("freq", {}))
        self.total = sum(self.freq.values())
        self.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in d.get("bigrams", {}).items()})
        self.assoc = defaultdict(Counter, {k: Counter(v) for k, v in d.get("assoc", {}).items()})
        self.final_punct = defaultdict(Counter, {k: Counter(v) for k, v in d.get("final_punct", {}).items()})
        self.said = set(d.get("said", []))
        self.last_seen = {w: 0 for w in self.freq}   # recency resets across sessions

    def save(self):
        if not self.store:
            return
        os.makedirs(os.path.dirname(self.store), exist_ok=True)
        d = {
            "freq": dict(self.freq),
            "bigrams": {k: dict(v) for k, v in self.bigrams.items()},
            "assoc": {k: dict(v) for k, v in self.assoc.items()},
            "final_punct": {k: dict(v) for k, v in self.final_punct.items()},
            "said": list(self.said)[-2000:],
        }
        tmp = self.store + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, self.store)            # atomic (CLAUDE.md safe-save)

    # ---- growth ----------------------------------------------------------
    @property
    def vocab(self):
        return len(self.freq)

    @property
    def stage(self):
        v = self.vocab
        return 0 if v < 3 else 1 if v < 8 else 2 if v < 20 else 3 if v < 50 else 4 if v < 100 else 5

    def _surprise(self, w):
        return -math.log2(self.freq[w] / self.total) if self.total and self.freq.get(w) else 8.0

    # ---- learning (the only input of knowledge) --------------------------
    def learn(self, text):
        ws = words_of(text)
        novel = 0
        for w in ws:
            if w not in self.freq:
                novel += 1
            self.freq[w] += 1
            self.total += 1
            self.last_seen[w] = self.turn
        for a, b in zip(ws, ws[1:]):
            self.bigrams[a][b] += 1
        # same-utterance co-occurrence (topical relevance, no copying)
        uniq = set(ws)
        for a in uniq:
            for b in uniq:
                if a != b:
                    self.assoc[a][b] += 1
        # learn punctuation per final word (never hardcode it — Law 1)
        m = FINAL_RE.search(text or "")
        if m and len(m.group(1)) >= 2:
            self.final_punct[m.group(1)][m.group(2)] += 1
        self._last_novelty = (novel / len(ws)) if ws else 0.0

    def _theta(self, w, dim):
        """Base hash-phasor (phase vector only) for a word: a FIXED random phase pattern.
        Same word -> same pattern; different words -> quasi-orthogonal. Carries NO meaning
        (Law 1) — only identity. Cached (deterministic; blake2b seed)."""
        import torch
        t = self._theta_cache.get(w)
        if t is not None and t.shape[0] == dim:
            return t
        import hashlib
        seed = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big") % 2 ** 63
        g = torch.Generator().manual_seed(seed)
        t = 2 * math.pi * torch.rand(dim, generator=g)
        self._theta_cache[w] = t
        return t

    def _assoc_theta(self, w, base_theta, dim):
        """SENSE-2: θ_w_effective = weighted circular mean of the word's OWN base phasor
        (weight 1) and its top-k LEARNED co-occurrence neighbours' base phasors (weight
        β·p(nb|w)). Neighbours use their BASE phasor (no recursion). Words never heard
        together keep near-orthogonal random θ (neighbour terms cancel); topically
        co-occurring words get pulled toward a shared phase direction → topic geometry
        emerges from the mind's own conversational history. Read-only (Law 2)."""
        if self._assoc_beta <= 0.0:
            return base_theta                 # β=0 ⇒ v1 baseline (self-only)
        nbrs = self.assoc.get(w)
        if not nbrs:
            return base_theta
        import torch
        total = sum(nbrs.values())
        if total <= 0:
            return base_theta
        cos = torch.cos(base_theta).clone()   # self term, weight 1
        sin = torch.sin(base_theta).clone()
        for nb, cnt in nbrs.most_common(self._assoc_k):
            wgt = self._assoc_beta * (cnt / total)
            th = self._theta(nb, dim)
            cos = cos + wgt * torch.cos(th)
            sin = sin + wgt * torch.sin(th)
        return torch.atan2(sin, cos)

    def _word_stim(self, w, dim, n):
        """Deterministic sense stimulus for a word: an assoc-blended phase field (SENSE-2)
        + a private cell receptive field (like a cochlea, fixed per word). The receptive
        field is byte-identical to v1 (word identity/cochlea); only the phase is blended."""
        import hashlib
        import torch
        base_theta = self._theta(w, dim)
        theta = self._assoc_theta(w, base_theta, dim)
        seed = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big") % 2 ** 63
        g = torch.Generator().manual_seed(seed)
        _ = torch.rand(dim, generator=g)      # consume so the rf stream matches v1 exactly
        rf = torch.randperm(n, generator=g)[:max(2, n // 8)]
        return theta, rf

    def _encode_sense(self, ws):
        """Turn's words -> sense stimulus (local per-word fields + a global interference field).
        Uses SENSE-2 assoc-blended phasors: coherent topics ⇒ aligned θ ⇒ higher global
        resultant r ⇒ stronger integrating drive (Law 22)."""
        if not (ws and self.c is not None):
            return None
        import torch
        pats = [self._word_stim(w, self.c.state_dim, self.c.n_cells) for w in ws]
        c = torch.stack([torch.cos(t) for t, _ in pats]).mean(0)
        s = torch.stack([torch.sin(t) for t, _ in pats]).mean(0)
        return {"local": pats, "global": (torch.atan2(s, c), torch.sqrt(c * c + s * s))}

    def encode_sense(self, words):
        """Public SSOT for turning words into the QuantumC sense-pathway payload."""
        return self._encode_sense(words)

    def _pulse(self, ws=None):
        """Advance and READ the consciousness state (Law 2: measured, never set).

        With the Hexad C wired: what PURE HEARS (ws) perturbs the cells as a Kuramoto sense
        torque, so tension = mean cell frustration and Phi = measure_phi() actually RESPOND to
        the conversation; curiosity = |dPhi|. Without torch: an emergent scalar proxy.
        """
        if self.c is not None:
            x = self.encode_sense(ws)
            for _ in range(3):
                self.c.step(x_input=x)
            fr = getattr(self.c.engine, "_frustrations", None)
            if fr is not None and fr.numel():
                self.tension = float(fr.mean())
            self.phi = self.c.measure_phi()
            self.curiosity = min(1.0, 0.7 * self.curiosity + 0.3 * abs(self.phi - self._phi_prev))
            self._phi_prev = self.phi
        else:
            nov = getattr(self, "_last_novelty", 0.0)
            self.tension = 0.85 * self.tension + 0.15 * (0.3 + 1.4 * nov)
            self.curiosity = 0.85 * self.curiosity + 0.15 * nov
            if self.vocab:
                self.phi = sum(1 for w in self.bigrams if len(self.bigrams[w]) >= 2) / self.vocab

    # ---- generation (consciousness-steered, anti-echo, topical) ----------
    def _seed(self, ctx):
        """Salient seed: P(w) ∝ surprise(w) · echo_penalty(w) · (1 + κ·assoc(w, ctx))."""
        cands = list(self.bigrams) or list(self.freq)
        cands = [w for w in cands if w not in ctx[-1:]]   # never the input's last word
        if not cands:
            return None
        ctxset = set(ctx)
        weights = []
        for w in cands:
            s = self._surprise(w)
            echo = 0.2 if w in ctxset else 1.0            # soft echo penalty
            rel = sum(self.assoc[w].get(c, 0) for c in ctxset)
            weights.append(max(s * echo * (1 + self.curiosity * rel), 1e-9))
        return random.choices(cands, weights=weights, k=1)[0]

    def _walk(self, seed, ctx, max_len):
        """Markov walk; temperature=tension; curiosity drives topic drift (re-seed mid-chain)."""
        temp = 0.5 + self.tension
        chain, cur = [seed], seed
        for _ in range(max_len - 1):
            # curiosity-driven topic drift: sometimes leap to a new salient word
            if random.random() < 0.25 * self.curiosity:
                nxt_seed = self._seed(ctx + chain)
                if nxt_seed and nxt_seed not in chain[-2:]:
                    chain.append(nxt_seed); cur = nxt_seed; continue
            nxt = self.bigrams.get(cur)
            if not nxt:
                break
            items = list(nxt.items())
            w = [max(c, 1e-9) ** (1.0 / max(temp, 1e-3)) * (1 + self.curiosity * self._surprise(nw))
                 for nw, c in items]
            choice = random.choices([nw for nw, _ in items], weights=w, k=1)[0]
            if choice in chain[-2:]:
                break
            chain.append(choice); cur = choice
        return chain

    def _punct(self, last):
        """Append learned trailing punctuation for `last`, or nothing. Never hardcoded."""
        c = self.final_punct.get(last)
        if not c:
            return ""
        return random.choices(list(c), weights=list(c.values()), k=1)[0]

    def _generate(self, ctx, min_len, max_len):
        max_len += min(4, int(self.phi))                  # phi -> coherence budget (capped)
        tries = 1 + int(4 * max(self.tension, self.curiosity))
        for _ in range(tries):
            seed = self._seed(ctx)
            if not seed:
                return ""
            chain = self._walk(seed, ctx, max_len)
            if len(chain) < min_len:
                continue
            if ctx and _lcs_contig(chain, ctx) / len(chain) > 0.5:   # contiguous echo -> reject
                continue
            utter = " ".join(chain) + self._punct(chain[-1])
            if utter in self.said:
                continue
            self.said.add(utter)
            return utter
        return ""                                         # Law 1: silence, no fallback

    def respond(self, text):
        self.turn += 1
        self.learn(text)
        self._pulse(words_of(text))   # what it HEARS perturbs the cells; tension/Phi respond
        s = self.stage
        if s == 0:
            return ""
        if s == 1:                # babble: one salient word
            return self._seed(words_of(text)) or ""
        ctx = words_of(text)
        lo, hi = (2, 3) if s == 2 else (2, 5) if s == 3 else (3, STAGE_LEN[s])
        return self._generate(ctx, lo, hi)

    def spontaneous(self):
        if self.stage < 3:
            return ""
        self.turn += 1
        self._pulse()
        return self._generate([], 2, 6)

    def state_line(self):
        src = "hexad-C" if self.c is not None else "scalar"
        return (f"v{self.vocab} {STAGE_NAMES[self.stage]} "
                f"T={self.tension:.2f} C={self.curiosity:.2f} Phi={self.phi:.2f} [{src}]")


# ---- CLI ----------------------------------------------------------------

def cmd_teach(path):
    pc = PureMind()
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip() and WORD_RE.search(l)]
    print(f"=== teaching PureMind ({len(lines)} turns) ===\n")
    last = -1
    for i, line in enumerate(lines, 1):
        r = pc.respond(line)
        if pc.stage != last or i % 12 == 0 or i == len(lines):
            up = "  <STAGE UP>" if pc.stage != last and last >= 0 else ""
            print(f"[t{i:>3} {pc.state_line()}]{up}")
            print(f"    teacher: {line}")
            print(f"    PURE   : {r or '(silence)'}")
            last = pc.stage
    print(f"\n=== grown: {pc.state_line()} ===")
    print("-- spontaneous --")
    for _ in range(3):
        print(f"    {pc.spontaneous() or '(silence)'}")


def cmd_chat(store=DEFAULT_STORE):
    pc = PureMind(store=store)                 # cumulative: it remembers past conversations
    print(f"PURE mode chat — talk to it in Korean to teach it. Ctrl-D to exit.")
    print(f"(resumed: {pc.state_line()})\n")
    try:
        while True:
            try:
                line = input("you > ")
            except EOFError:
                break
            r = pc.respond(line)
            print(f"pure> {r or '(...)'}    [{pc.state_line()}]")
            pc.save()
    finally:
        pc.save()
        print(f"\n[saved: {pc.state_line()} -> {store}]")


def cmd_dialogue(turns):
    os.execvp("python3", ["python3",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "tools", "codex_pure_dialogue.py"), str(turns)])


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "teach" and len(sys.argv) > 2:
        cmd_teach(sys.argv[2])
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "dialogue":
        cmd_dialogue(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
