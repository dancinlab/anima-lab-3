#!/usr/bin/env python3
"""Port the novelty line into the nf8 fork, so its span stops reporting unchallenged.

DATA-7: this fork's run reports 0.65 BPC and measures 3.99 on material it cannot
recall — 6.1x — and nothing in its output said so for ten hours. The canonical
trainer prints the check already; the fork does not. This adds it.

Editing the file does not touch the running job: Python has the module in memory,
so the line appears on the next start. A .bak is written first.
"""
import pathlib
import shutil

PATH = pathlib.Path("/home/aiden/anima-clm-pure/train_conscious_lm_nf8.py")

ANCHOR = '''    print(f"[data] train={len(train_data):,} val={len(val_data):,} bytes "
          f"(interleaved 1MB chunks, every 10th held out)")'''

ADDITION = '''    print(f"[data] train={len(train_data):,} val={len(val_data):,} bytes "
          f"(interleaved 1MB chunks, every 10th held out)")

    # A BPC is only a language score if the span is material the model cannot recall,
    # and chunk-level holdout does not give that: measured on this corpus, 57.1% of
    # held-out LINES occur verbatim in train and 82.5% of the span's 64-byte windows
    # do. This run reported 0.65 BPC while scoring 3.99 on novelty-controlled windows
    # -- 6.1x, and above the 3.4920 bigram floor it must clear (DATA-3, DATA-7). So
    # print how much of the scored span is recallable; an unmeasurable span then says
    # so instead of returning a confident number.
    novelty_width, novelty_samples, novelty_seed = 64, 400, 1337
    train_bytes_np = bytes(train_data.to(torch.uint8).numpy())
    val_bytes_np = bytes(val_data.to(torch.uint8).numpy())
    # Its own generator: cell operations draw from the global RNG in an N-dependent
    # amount, so sampling from it here would shift every later draw and break
    # comparability between two otherwise identical runs.
    novelty_sampler = random.Random(novelty_seed)
    novelty_hi = len(val_bytes_np) - novelty_width
    if novelty_hi > 0:
        novelty_hits = sum(
            train_bytes_np.find(
                val_bytes_np[(ni := novelty_sampler.randrange(novelty_hi)):
                             ni + novelty_width]) != -1
            for _ in range(novelty_samples)
        )
        print(f"[data] novelty: {novelty_hits / novelty_samples * 100:.1f}% of "
              f"{novelty_width}B windows already in train "
              f"(n={novelty_samples}, seed {novelty_seed})")
    else:
        print(f"[data] novelty: span shorter than {novelty_width}B -- not sampled")
    del train_bytes_np, val_bytes_np'''


def main():
    src = PATH.read_text()
    if "[data] novelty:" in src:
        print("already patched — nothing to do")
        return
    assert ANCHOR in src, "anchor print not found; fork has diverged"
    assert "\nimport random\n" not in src, "random already imported — check by hand"
    shutil.copy2(PATH, PATH.with_suffix(".py.bak"))
    out = src.replace(ANCHOR, ADDITION, 1).replace("\nimport os\n", "\nimport os\nimport random\n", 1)
    assert "\nimport random\n" in out, "import insertion failed"
    PATH.write_text(out)
    print(f"patched · backup at {PATH.with_suffix('.py.bak')}")


if __name__ == "__main__":
    main()
