#!/usr/bin/env python3
"""G0 COHERENCE and G2 NOVELTY -- the sibling repo's OLD capability gates, run here.

Why these and not another BPC number. anima's p7 says a perplexity verdict is
not a verdict, and BPC is perplexity in log2 units, so everything gate.py
reports is a screen. G0 and G2 are the shape that answers p7: they read what
the model DOES, not how surprised it is. Definitions taken verbatim from
anima/CONDITIONS.md, thresholds unchanged (frozen-first -- a bar re-tuned to fit
these models would be worthless):

  G0 COHERENCE  known-word-ratio >= 0.50 on >= 4/5 seeds (NO byte-salad).
                anti-Goodhart: the BEFORE-backbone must FAIL. An identically
                shaped random-weight model is that before-state here, and if it
                passes, the metric is measuring the corpus rather than the model.
  G2 NOVELTY    >= 3 corpus-absent coherent novel n-grams, retrieval-control = 0.
                Coherent = every token in the n-gram is in the train vocabulary,
                so a novel n-gram cannot be scored by inventing tokens. The
                retrieval control copies real train spans: it must produce ZERO
                corpus-absent n-grams, which is what makes "absent" mean absent
                and not a search bug.

Not implemented here, with the reason, because a gate nobody can run is worse
than an absent one:
  G1 RECOMBINATION  needs atom pairs whose composition has a known held-out
                    target. That is a corpus construction, not a probe.
  G5 NON-FABRICATION needs grounded/ungrounded question pairs and an abstain
                    channel. These models emit continuations; they have no
                    abstain move to score.
  G6 IDEATION       needs seeds with scored divergence. Buildable on top of G2's
                    machinery once G2 reads at all; it is the next rung, not
                    this one.
  G3 / G4           not gates in the source either (status read / publication).

p9 still binds all of it: this corpus is CONSTRUCTED (corpus_regime.py), so a
G0/G2 pass here certifies the instrument reads, and is not evidence of a
faculty. Stated in the output so the number cannot travel further than that.
"""
import hashlib
import json
import math
import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CORPUS = f"{HOME}/data/corpus_merged_dedup.txt"
CHUNK = 1 << 20
BLOCK = 256
SEED_BYTES = 128        # natural prefix handed to the model
GEN_BYTES = 256         # how much it must produce
N_SEEDS = 5             # the "4/5" in G0 is over these
NGRAM = 4               # tokens per n-gram for G2
TEMP = 0.8
DECODE_SEED, INIT_SEED = 20260727, 1337

G0_BAR, G0_NEED = 0.50, 4      # frozen, anima/CONDITIONS.md
G2_NEED = 3                    # frozen

ARMS = {
    "s25":   f"{HOME}/checkpoints/arm_s25/best.pt",
    "v25":   f"{HOME}/checkpoints/arm_v25/best.pt",
    "s50":   f"{HOME}/checkpoints/arm_s50/best.pt",
    "v50":   f"{HOME}/checkpoints/arm_v50/best.pt",
    "s100":  f"{HOME}/checkpoints/arm_a_data/best.pt",
    "v100":  f"{HOME}/checkpoints/arm_v100/best.pt",
    "p50":   f"{HOME}/checkpoints/arm_p50/best.pt",
    "p50f":  f"{HOME}/checkpoints/arm_p50/final.pt",
    "p100":  f"{HOME}/checkpoints/arm_p100/best.pt",
    "p100f": f"{HOME}/checkpoints/arm_p100/final.pt",
    "e50":   f"{HOME}/checkpoints/arm_e50/best.pt",
    "e50f":  f"{HOME}/checkpoints/arm_e50/final.pt",
    "e100":  f"{HOME}/checkpoints/arm_e100/best.pt",
    "e100f": f"{HOME}/checkpoints/arm_e100/final.pt",
    "s25f":  f"{HOME}/checkpoints/arm_s25/final.pt",
    "v25f":  f"{HOME}/checkpoints/arm_v25/final.pt",
    "s50f":  f"{HOME}/checkpoints/arm_s50/final.pt",
    "v50f":  f"{HOME}/checkpoints/arm_v50/final.pt",
    "s100f": f"{HOME}/checkpoints/arm_a_data/final.pt",
    "v100f": f"{HOME}/checkpoints/arm_v100/final.pt",
    "p25":   f"{HOME}/checkpoints/arm_p25/best.pt",
    "p25f":  f"{HOME}/checkpoints/arm_p25/final.pt",
    "c50":   f"{HOME}/checkpoints/arm_50c/best.pt",
    "c50f":  f"{HOME}/checkpoints/arm_50c/final.pt",
}
# Every arm gate.py adjudicates, so the ladder is scored on one roster and a
# rung cannot silently cover fewer arms than the one below it.
SELECT = sys.argv[2:] or list(ARMS)


# The natural-corpus family. Selected with LAMBDA_FAMILY=natural rather than a
# fourth copy of this file: arm_nat trained on different bytes, so its honest
# vocabulary and its corpus-absence test must both come from ITS OWN corpus.
# Scoring known-word-ratio against a vocabulary the model never trained on
# would read as byte salad no matter how good the model is.
if os.environ.get("LAMBDA_FAMILY") == "natural":
    NAT = f"{HOME}/data/corpus_natural_ko_dedup.txt"
    CORPUS = NAT
    ARMS = {"nat": f"{HOME}/checkpoints/arm_nat/best.pt",
            "natf": f"{HOME}/checkpoints/arm_nat/final.pt",
            "nat25": f"{HOME}/checkpoints/arm_nat25/best.pt",
            "nat25f": f"{HOME}/checkpoints/arm_nat25/final.pt",
            "nat50": f"{HOME}/checkpoints/arm_nat50/best.pt",
            "nat50f": f"{HOME}/checkpoints/arm_nat50/final.pt"}
    SELECT = sys.argv[2:] or list(ARMS)


def load_trainer(path):
    spec = spec_from_file_location("clm_trainer", path)
    mod = module_from_spec(spec)
    sys.modules["clm_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def split(path):
    data = open(path, "rb").read()
    chunk = min(CHUNK, max(1, len(data) // 10))
    parts = [data[s:s + chunk] for s in range(0, len(data), chunk)]
    return (b"".join(p for i, p in enumerate(parts) if i % 10 != 9),
            b"".join(p for i, p in enumerate(parts) if i % 10 == 9))


def known_word_ratio(raw, vocab):
    """Fraction of whitespace tokens that exist in the train vocabulary.

    Byte salad produces tokens nobody ever wrote, so it lands near zero without
    needing a language model to judge it (no self-judge, p7).

    `raw` is BYTES and `vocab` is a set of BYTES. An earlier version decoded to
    str first and compared str tokens against a bytes vocabulary, so every arm
    read exactly 0.000 -- and the anti-Goodhart control read 0.000 too, which it
    is supposed to, so nothing flagged it. That is why POSITIVE_CONTROL below
    exists: a control that cannot tell a broken instrument from a real failure
    is not a control."""
    toks = raw.split()
    if not toks:
        return 0.0, 0, 0
    hits = sum(1 for t in toks if t in vocab)
    return hits / len(toks), hits, len(toks)


def novel_ngrams(raw, vocab, train_blob):
    """Corpus-absent n-grams whose every token is a real word.

    `train_blob` must be WHITESPACE-NORMALISED (b" ".join(train.split())). The
    query is built by rejoining split tokens with single spaces, so searching a
    raw haystack marks any n-gram whose original separator was a newline as
    "absent" -- the retrieval control caught exactly that, reading 81 where zero
    is the only valid answer. Normalise both sides or the absence test is a
    whitespace test."""
    toks = raw.split()
    found = set()
    for i in range(len(toks) - NGRAM + 1):
        gram = toks[i:i + NGRAM]
        if not all(t in vocab for t in gram):
            continue                       # incoherent -- does not count
        joined = b" ".join(gram)
        if train_blob.find(joined) == -1:
            found.add(joined)
    return found


@torch.no_grad()
def generate(model, prefix, n_bytes, device, gen):
    ctx = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    out = []
    for _ in range(n_bytes):
        window = ctx[:, -BLOCK:]
        logits, _, _ = model(window)
        probs = F.softmax(logits[0, -1] / TEMP, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen)
        out.append(int(nxt))
        ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
    return bytes(out)


def build(clm, cfg, device, state=None):
    m = clm.ConsciousLM(vocab_size=256, d_model=int(cfg["dim"]),
                        n_head=int(cfg["heads"]), n_layer=int(cfg["layers"]),
                        block_size=BLOCK, dropout=0.0)
    if state is not None:
        m.load_state_dict(state, strict=False)
    return m.to(device).eval()


def score_model(model, seeds, vocab, train_norm, device):
    """`train_norm` must be the whitespace-normalised haystack -- the parameter
    carries the requirement in its name so a caller cannot pass the raw blob."""
    gen = torch.Generator(device=device).manual_seed(DECODE_SEED)
    kwrs, novel = [], set()
    samples = []
    for pfx in seeds:
        raw = generate(model, pfx, GEN_BYTES, device, gen)
        kwr, hits, tot = known_word_ratio(raw, vocab)
        kwrs.append(kwr)
        novel |= novel_ngrams(raw, vocab, train_norm)
        samples.append(raw.decode("utf8", "replace")[:70].replace("\n", "⏎"))
    passing = sum(1 for k in kwrs if k >= G0_BAR)
    return {"kwr_per_seed": kwrs, "kwr_mean": sum(kwrs) / len(kwrs),
            "seeds_over_bar": passing,
            "G0": passing >= G0_NEED,
            "novel_ngrams": len(novel),
            "G2": len(novel) >= G2_NEED,
            "samples": samples,
            "novel_examples": [n.decode("utf8", "replace") for n in list(novel)[:3]]}


def main():
    out_path = sys.argv[1]
    clm = load_trainer(TRAINER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    train, val = split(CORPUS)
    vocab = set(train.split())
    train_norm = b" ".join(train.split())   # the haystack absence is tested against
    print(f"[vocab] {len(vocab):,} distinct whitespace tokens in train", flush=True)

    # Seeds are real prefixes from the held-out split: the model must continue
    # natural text, not a prompt somebody wrote to make it look good.
    step = max(1, (len(val) - SEED_BYTES) // (N_SEEDS + 1))
    seeds = [val[i * step: i * step + SEED_BYTES] for i in range(1, N_SEEDS + 1)]
    print(f"[seeds] {N_SEEDS} x {SEED_BYTES}B prefixes from the held-out split", flush=True)

    results = {"_setup": {"G0_bar": G0_BAR, "G0_need": f"{G0_NEED}/{N_SEEDS}",
                          "G2_need": G2_NEED, "ngram_tokens": NGRAM,
                          "gen_bytes": GEN_BYTES, "temp": TEMP,
                          "decode_seed": DECODE_SEED, "vocab": len(vocab),
                          "corpus_regime": "CONSTRUCTED (corpus_regime.py) -- p9: a pass "
                                           "here certifies the instrument, not a faculty"}}

    # Retrieval control for G2: copy real train spans. By construction it can
    # produce no corpus-absent n-gram, so a non-zero reading means the absence
    # test is broken and every G2 number in this file is void.
    rgen = torch.Generator().manual_seed(DECODE_SEED)
    ctrl_novel = set()
    for _ in range(N_SEEDS):
        i = int(torch.randint(0, len(train) - GEN_BYTES, (1,), generator=rgen))
        ctrl_novel |= novel_ngrams(train[i:i + GEN_BYTES], vocab, train_norm)
    results["_control_retrieval"] = {"novel_ngrams": len(ctrl_novel),
                                     "valid": len(ctrl_novel) == 0}

    # POSITIVE_CONTROL: real held-out prose, scored by the same function. Known
    # ground truth -- it MUST clear the bar. If it does not, the instrument is
    # broken and every lambda2 row below is void, not a finding.
    pos = [known_word_ratio(val[i * step: i * step + GEN_BYTES], vocab)[0]
           for i in range(1, N_SEEDS + 1)]
    pos_ok = sum(1 for k in pos if k >= G0_BAR) >= G0_NEED
    results["_control_positive"] = {"kwr_per_seed": pos, "kwr_mean": sum(pos) / len(pos),
                                    "valid": pos_ok,
                                    "_note": "real held-out text; MUST clear the bar or "
                                             "lambda2 is void"}
    print(f"[ctrl] POSITIVE control (real held-out text) kwr={sum(pos)/len(pos):.3f} → "
          f"{'VALID -- the instrument reads' if pos_ok else 'BROKEN -- lambda2 VOID, do not read the rows below'}",
          flush=True)
    print(f"[ctrl] retrieval control novel n-grams = {len(ctrl_novel)} "
          f"({'VALID -- absence test works' if not ctrl_novel else 'BROKEN -- G2 void'})",
          flush=True)

    for name in SELECT:
        path = ARMS[name]
        if not os.path.exists(path):
            print(f"[{name}] checkpoint absent -- skipped", flush=True)
            continue
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        t0 = time.time()
        r = score_model(build(clm, cfg, device, ck["model_state"]),
                        seeds, vocab, train_norm, device)
        r.update({"ckpt_step": ck.get("step"), "ckpt_sha256_16": sha})
        results[name] = r
        print(f"[{name}] G0 kwr mean={r['kwr_mean']:.3f} · {r['seeds_over_bar']}/{N_SEEDS} "
              f"over {G0_BAR} → {'PASS' if r['G0'] else 'FAIL'} | G2 novel={r['novel_ngrams']} "
              f"→ {'PASS' if r['G2'] else 'FAIL'} · sha {sha} ({time.time()-t0:.0f}s)",
              flush=True)
        print(f"        sample: {r['samples'][0]}", flush=True)
        del ck

    # anti-Goodhart: the before-backbone must FAIL G0. If it passes, the metric
    # is reading the corpus, not the model, and every G0 row above is void.
    any_cfg = next((json.loads(json.dumps(results[n])) and
                    torch.load(ARMS[n], map_location="cpu", weights_only=False)["config"]
                    for n in SELECT if os.path.exists(ARMS[n])), None)
    torch.manual_seed(INIT_SEED)
    r = score_model(build(clm, any_cfg, device), seeds, vocab, train_norm, device)
    results["_control_before_backbone"] = {**r,
                                           "valid": not r["G0"],
                                           "_note": "anti-Goodhart: this MUST fail G0"}
    print(f"[ctrl] before-backbone (random weights) G0 kwr={r['kwr_mean']:.3f} → "
          f"{'FAIL -- correct, metric is not reading the corpus' if not r['G0'] else 'PASS -- G0 VOID'}",
          flush=True)

    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
