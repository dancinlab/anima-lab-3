#!/usr/bin/env python3
"""Score the three scaling arms on ONE common span, so the comparison is a comparison.

The curve came out non-monotonic (ratio to own bigram floor: 14.2% at 25% data,
48.4% at 50%, 24.0% at 100%), which data volume alone cannot produce. Subset
composition was checked first and is not the cause -- unique-line fraction is
49.3% in all three and the speaker-prefix mix is proportionally identical.

What remains is that each arm was scored on ITS OWN held-out span: three different
sets of bytes, whose measured familiarity to their own train split also came out
non-monotonic (15.2% / 11.2% / 24.0%). Comparing BPC across different test sets
is not a comparison at all. This scores every checkpoint on one span.

The span must be unseen by ALL THREE arms. The subsets are drawn by line modulo
from the whole corpus, so a line held out of the 100% split can still sit in the
25% split's train -- taking the 100% arm's held-out material alone would hand the
smaller arms an advantage. So: candidate lines come from the full corpus's
held-out chunks, then survive only if absent from all three train splits.
"""
import json
import math
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import torch

HOME = "/home/summer/anima-clm-pure"
TRAINER = f"{HOME}/train_conscious_lm.py"
CHUNK = 1 << 20
SPAN_BYTES = 262_144
BATCH = 8
LN2 = math.log(2)

ARMS = [
    ("s25", f"{HOME}/data/corpus_merged_25.txt", f"{HOME}/checkpoints/arm_s25/best.pt"),
    ("s50", f"{HOME}/data/corpus_merged_50.txt", f"{HOME}/checkpoints/arm_s50/best.pt"),
    ("s100", f"{HOME}/data/corpus_merged_dedup.txt", f"{HOME}/checkpoints/arm_a_data/best.pt"),
]


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

    train_lines = {}
    for name, corpus, _ in ARMS:
        tr, va = split(corpus)
        train_lines[name] = set(tr.split(b"\n"))
        if name == "s100":
            full_val = va
        print(f"[split] {name}: train={len(tr):,} val={len(va):,} "
              f"distinct train lines={len(train_lines[name]):,}", flush=True)

    kept, seen_by = [], 0
    for ln in full_val.split(b"\n"):
        if len(ln) <= 8:
            continue
        if any(ln in train_lines[n] for n, _, _ in ARMS):
            seen_by += 1
            continue
        kept.append(ln)
    span = (b"\n".join(kept))[:SPAN_BYTES]
    print(f"[span] {len(span):,} bytes from {len(kept):,} lines unseen by ALL arms "
          f"({seen_by:,} candidate lines rejected as present in some arm's train)", flush=True)
    if len(span) < SPAN_BYTES:
        print(f"[warn] span shorter than {SPAN_BYTES:,} -- reporting on what exists", flush=True)

    span_t = torch.frombuffer(bytearray(span), dtype=torch.uint8).long()
    results = {}
    for name, _, ckpt_path in ARMS:
        t0 = time.time()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        model = clm.ConsciousLM(
            vocab_size=256, d_model=int(cfg["dim"]), n_head=int(cfg["heads"]),
            n_layer=int(cfg["layers"]), block_size=int(cfg.get("block_size", 256)),
            dropout=0.0,
        )
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        model.to(device).eval()
        with torch.no_grad():
            ce = clm.evaluate_fixed_span(model, span_t, int(cfg.get("block_size", 256)),
                                         BATCH, device, min(SPAN_BYTES, len(span)))
        results[name] = {"ce_nats": ce, "bpc": ce / LN2, "ckpt_step": ck.get("step"),
                         "missing_keys": len(missing), "unexpected_keys": len(unexpected),
                         "seconds": round(time.time() - t0, 1)}
        print(f"[{name}] common-span BPC={ce / LN2:.4f} (CE={ce:.4f} nats) "
              f"· ckpt step={ck.get('step')} · keys missing={len(missing)} "
              f"unexpected={len(unexpected)} · {time.time() - t0:.0f}s", flush=True)
        del ck, model

    results["_span"] = {"bytes": len(span), "lines": len(kept),
                        "rejected_seen_lines": seen_by}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
