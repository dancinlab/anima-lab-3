#!/usr/bin/env python3
"""Re-measure the nf9 checkpoint's CE on material it cannot have memorised.

The run reports BPC on a validation split built by holding out every 10th 1MB
chunk with no line-level dedup. Measured on the byte-identical corpus, 57.1% of
those held-out LINES occur verbatim in train and they cover 61.5% of held-out
bytes, so the reported number is part recall.

Two measurements with the SAME evaluate_fixed_span the run itself uses, so the
comparison is apples-to-apples:

  raw    -- the run's own held-out split (control: must reproduce the logged BPC,
            otherwise this harness is wrong and the honest number means nothing)
  unseen -- 262,144 bytes assembled only from held-out lines ABSENT from train

CPU only, on purpose: the GPU is running a healthy 4h training job and a 0.65 vs
1.4 question is not worth risking it.
"""
import importlib.util
import json
import math
import sys
import time

import torch

HOME = "/home/aiden/anima-clm-pure"
CKPT = f"{HOME}/checkpoints/clm_pure_300m_nf9/best.pt"
CORPUS = f"{HOME}/data/corpus_v2.txt"
TRAINER = f"{HOME}/train_conscious_lm_nf8.py"
VAL_BYTES = 262_144
BATCH = 4
LN2 = math.log(2)


def load_trainer(path):
    """Import the training module by file path (the model class must be the one
    that produced the checkpoint, not a look-alike from another version)."""
    spec = importlib.util.spec_from_file_location("nf8_trainer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf8_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    unseen_path, out_path = sys.argv[1], sys.argv[2]
    torch.set_num_threads(8)
    device = torch.device("cpu")

    nf8 = load_trainer(TRAINER)

    t0 = time.time()
    try:
        ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
        how = "mmap"
    except Exception as exc:
        print(f"[ckpt] mmap load failed ({exc}); falling back to full load", flush=True)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        how = "full"
    cfg = ck.get("config", {}) or {}
    print(f"[ckpt] loaded via {how} in {time.time()-t0:.1f}s · step={ck.get('step')} "
          f"phase={ck.get('phase')} cells={len(ck.get('mitosis_cells', []))}", flush=True)
    print("[cfg] " + json.dumps({k: cfg.get(k) for k in
                                 ("dim", "layers", "heads", "block_size", "dropout")}),
          flush=True)

    block = int(cfg.get("block_size", 256))
    model = nf8.ConsciousLM(
        vocab_size=256,
        d_model=int(cfg["dim"]),
        n_head=int(cfg["heads"]),
        n_layer=int(cfg["layers"]),
        block_size=block,
        dropout=0.0,
    )
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch — missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        print(f"       missing[:5]={list(missing)[:5]}", flush=True)
        print(f"       unexpected[:5]={list(unexpected)[:5]}", flush=True)
    else:
        print("[model] state_dict loaded exactly (no missing/unexpected keys)", flush=True)
    model.to(device).eval()
    del ck

    # rebuild the run's exact split
    with open(CORPUS, "rb") as f:
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()
    n = len(data)
    chunk = 1 << 20
    train_parts, val_parts = [], []
    for i, start in enumerate(range(0, n, chunk)):
        (val_parts if i % 10 == 9 else train_parts).append(data[start:start + chunk])
    val_raw = torch.cat(val_parts)
    del train_parts, val_parts, data
    print(f"[data] raw held-out split = {len(val_raw):,} bytes", flush=True)

    with open(unseen_path, "rb") as f:
        unseen = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()
    print(f"[data] unseen-line span   = {len(unseen):,} bytes", flush=True)

    results = {}
    for name, span in (("raw", val_raw), ("unseen", unseen)):
        t = time.time()
        with torch.no_grad():
            ce = nf8.evaluate_fixed_span(model, span, block, BATCH, device, VAL_BYTES)
        results[name] = {"ce_nats": ce, "bpc": ce / LN2, "span_bytes": len(span),
                         "seconds": round(time.time() - t, 1)}
        print(f"[{name}] CE={ce:.4f} nats  BPC={ce / LN2:.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    results["inflation_x"] = results["unseen"]["bpc"] / max(1e-9, results["raw"]["bpc"])
    results["train_floors"] = {"unigram_bpc": 5.9548, "bigram_bpc": 3.4920}
    results["leak"] = {"line_frac": 0.571, "byte_frac": 0.615}
    print(f"[verdict] unseen/raw = {results['inflation_x']:.2f}x · "
          f"bigram floor 3.4920 · unigram floor 5.9548", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] {out_path}", flush=True)


if __name__ == "__main__":
    main()
