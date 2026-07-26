#!/usr/bin/env python3
"""Score the three scaling arms on windows that are novel at SUBSTRING level.

Three attempts to compare these arms have now failed on the same defect:
  - each arm's own span: different test sets, so the numbers are not comparable
  - a shared span built by line-level filtering: 80-86% of its 64-byte windows sit
    verbatim in every arm's train, and all three models scored 0.12-0.15 BPC on a
    span whose own bigram floor is 3.32 -- recall, not prediction (DATA-3 again)

So this selects windows, not lines. A candidate 257-byte window is kept only if
THREE 64-byte probes taken from it (head, middle, tail) are each absent from all
three arms' train splits. What survives cannot be produced from memory, so the CE
on it is a language number for every arm at once.

Reports the rejection rate too: if almost nothing survives, that is the finding --
the corpus cannot support a novelty-controlled test at this width.
"""
import json
import math
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import torch
import torch.nn.functional as F

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CHUNK = 1 << 20
BLOCK = 256
PROBE = 64
TARGET_WINDOWS = 256
MAX_CANDIDATES = 20_000
STRIDE = 1021          # prime, so candidates do not align with any 1MB structure
BATCH = 8
LN2 = math.log(2)

# Every arm ever compared on this corpus family. The novelty filter must exclude a
# window seen by ANY of them, or an arm absent from the filter gets scored on its own
# training data; SELECT picks which arms are reported, never which trains are screened.
ALL_ARMS = {
    "s25": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_s25/best.pt"),
    "s50": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_s50/best.pt"),
    "s100": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_a_data/best.pt"),
    "v25": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_v25/best.pt"),
    "v100": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_v100/best.pt"),
    # final.pt is step 12,000 for every arm. best.pt is whatever each arm's OWN val span
    # liked most, and those spans differ in difficulty (novelty 15.2% / 11.2% / 24.0%), so
    # comparing best.pt across arms compares checkpoint SELECTION as much as training.
    # The -f arms remove that: same step, same objective phase, one test set.
    "s25f": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_s25/final.pt"),
    "s50f": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_s50/final.pt"),
    "s100f": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_a_data/final.pt"),
    "v25f": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_v25/final.pt"),
    "v100f": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_v100/final.pt"),
    "v50": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_v50/best.pt"),
    "v50f": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_v50/final.pt"),
    # p-arms: same config with --phase language, i.e. the COMBINED phase removed.
    "p25": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_p25/best.pt"),
    "p25f": (f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_p25/final.pt"),
    "p50": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_p50/best.pt"),
    "p50f": (f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_p50/final.pt"),
    "p100": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_p100/best.pt"),
    "p100f": (f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_p100/final.pt"),
}
# argv[2:] selects arms to report; default = the original three.
SELECT = sys.argv[2:] or ["s25", "s50", "s100"]
ARMS = [(n, ALL_ARMS[n][0], ALL_ARMS[n][1]) for n in SELECT]
SCREEN = [(n, c, k) for n, (c, k) in ALL_ARMS.items()]


def load_trainer(path):
    spec = spec_from_file_location("clm_trainer", path)
    mod = module_from_spec(spec)
    sys.modules["clm_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def split(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data)
    chunk = min(CHUNK, max(1, n // 10))
    parts = [data[s:s + chunk] for s in range(0, n, chunk)]
    train = b"".join(p for i, p in enumerate(parts) if i % 10 != 9)
    val = b"".join(p for i, p in enumerate(parts) if i % 10 == 9)
    return train, val


def main():
    out_path = sys.argv[1]
    clm = load_trainer(TRAINER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    trains, full_val, seen = [], None, set()
    for name, corpus, _ in SCREEN:
        if corpus in seen:          # v25 and s25 share a corpus; screen it once
            continue
        seen.add(corpus)
        tr, va = split(corpus)
        trains.append(tr)
        if corpus.endswith("corpus_merged_dedup.txt"):
            full_val = va
        print(f"[split] screening {name}: train={len(tr):,} val={len(va):,}", flush=True)
    print(f"[arms] reporting: {', '.join(n for n, _, _ in ARMS)}", flush=True)

    t0 = time.time()
    windows, tested = [], 0
    for start in range(0, len(full_val) - BLOCK - 2, STRIDE):
        if len(windows) >= TARGET_WINDOWS or tested >= MAX_CANDIDATES:
            break
        tested += 1
        w = full_val[start:start + BLOCK + 1]
        probes = (w[:PROBE], w[(BLOCK - PROBE) // 2:(BLOCK - PROBE) // 2 + PROBE],
                  w[BLOCK - PROBE:BLOCK])
        if all(tr.find(p) == -1 for p in probes for tr in trains):
            windows.append(w)
    rate = len(windows) / max(1, tested)
    print(f"[select] kept {len(windows):,} of {tested:,} candidates = {rate * 100:.1f}% "
          f"({time.time() - t0:.0f}s). A window is kept only if 3x{PROBE}B probes are all "
          f"absent from all 3 train splits.", flush=True)
    if not windows:
        print("[verdict] NO novelty-controlled window exists in this corpus at this width "
              "-- the comparison cannot be made, and that is the result.", flush=True)
        with open(out_path, "w") as f:
            json.dump({"kept": 0, "tested": tested}, f, indent=2)
        return

    x = torch.tensor([list(w[:BLOCK]) for w in windows], dtype=torch.long)
    y = torch.tensor([list(w[1:BLOCK + 1]) for w in windows], dtype=torch.long)

    results = {"_select": {"kept": len(windows), "tested": tested, "keep_rate": rate,
                           "probe_bytes": PROBE, "block": BLOCK}}
    for name, _, ckpt_path in ARMS:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        model = clm.ConsciousLM(
            vocab_size=256, d_model=int(cfg["dim"]), n_head=int(cfg["heads"]),
            n_layer=int(cfg["layers"]), block_size=BLOCK, dropout=0.0)
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        model.to(device).eval()
        total, ntok = 0.0, 0
        with torch.no_grad():
            for b in range(0, len(windows), BATCH):
                xb, yb = x[b:b + BATCH].to(device), y[b:b + BATCH].to(device)
                logits, _, _ = model(xb)
                total += F.cross_entropy(logits.view(-1, model.vocab_size),
                                         yb.reshape(-1), reduction="sum").item()
                ntok += yb.numel()
        ce = total / ntok
        results[name] = {"ce_nats": ce, "bpc": ce / LN2, "tokens": ntok,
                         "ckpt_step": ck.get("step"), "missing_keys": len(missing),
                         "unexpected_keys": len(unexpected)}
        print(f"[{name}] novelty-controlled BPC={ce / LN2:.4f} (CE={ce:.4f} nats, "
              f"{ntok:,} tokens) · ckpt step={ck.get('step')}", flush=True)
        del ck, model

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
