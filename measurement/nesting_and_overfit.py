#!/usr/bin/env python3
"""Two gaps the phase ablation left open, both answerable from data already on disk.

(1) NESTING. The subsets were taken by line-index modulo: 25% = i%4==0, 50% = i%2==0.
    i%4==0 implies i%2==0, so 25% should be a strict SUBSET of 50%, and both of 100%.
    If that holds, DATA-6's result is sharper than "data volume is not the variable":
    the 50% corpus contains everything the 25% corpus has, plus more, and the model
    trained on the superset FAILS the gate (5.34 BPC) while the subset PASSES it
    decisively (0.41). Adding data would then have actively broken it.

(2) OVERFITTING. The ablation showed the combined phase owns only 39%/26% of the
    best->final degradation; the rest was guessed to be "plain overfitting past the
    peak" and never checked. If that guess is right, train loss keeps falling while
    validation rises after the peak. The logs already record both.
"""
import re
import sys
from pathlib import Path

CORPORA = {
    "25%": "corpus_merged_25.txt",
    "50%": "corpus_merged_50.txt",
    "100%": "corpus_merged_dedup.txt",
}
STEP_ROW = re.compile(r"^\s*(\d+)\s*\|\s*\w+\s*\|\s*([-\d.]+)\s*\|")
VAL_ROW = re.compile(r"\[val\].*?BPC=([\d.]+)")


def nesting(data_dir):
    sets = {}
    for name, fn in CORPORA.items():
        p = Path(data_dir) / fn
        lines = p.read_bytes().split(b"\n")
        sets[name] = set(lines)
        print(f"[{name}] {fn}: {len(lines):,} lines · {len(sets[name]):,} distinct", flush=True)
    for small, big in (("25%", "50%"), ("50%", "100%"), ("25%", "100%")):
        missing = sets[small] - sets[big]
        frac = 1 - len(missing) / max(1, len(sets[small]))
        verdict = "SUBSET" if not missing else f"NOT a subset ({len(missing):,} lines outside)"
        print(f"[nest] {small} ⊂ {big}? {verdict} · coverage {frac*100:.2f}%", flush=True)


def overfit(log_path):
    train, val = [], []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = STEP_ROW.match(line)
        if m:
            train.append((int(m.group(1)), float(m.group(2))))
            continue
        m = VAL_ROW.search(line)
        if m:
            val.append((train[-1][0] if train else 0, float(m.group(1))))
    if not train or not val:
        print(f"[{Path(log_path).name}] no rows parsed", flush=True)
        return
    best_i = min(range(len(val)), key=lambda i: val[i][1])
    peak_step = val[best_i][0]
    tr_at = [t for s, t in train if s <= peak_step]
    tr_after = [t for s, t in train if s > peak_step]
    v_after = [b for s, b in val if s > peak_step]
    if not tr_after or not v_after:
        print(f"[{Path(log_path).name}] peak at the end — nothing after", flush=True)
        return
    print(f"[{Path(log_path).name}] val peak {val[best_i][1]:.4f} @step{peak_step}", flush=True)
    print(f"    train loss  at peak {tr_at[-1]:.4f} → end {tr_after[-1]:.4f} "
          f"(mean after {sum(tr_after)/len(tr_after):.4f})", flush=True)
    print(f"    val   BPC   at peak {val[best_i][1]:.4f} → end {v_after[-1]:.4f} "
          f"(mean after {sum(v_after)/len(v_after):.4f})", flush=True)
    fell = tr_after[-1] < tr_at[-1]
    rose = v_after[-1] > val[best_i][1]
    print(f"    → train {'FELL' if fell else 'did not fall'} · "
          f"val {'ROSE' if rose else 'did not rise'} = "
          f"{'overfitting signature' if fell and rose else 'NOT the overfitting signature'}",
          flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "nest":
        nesting(sys.argv[2])
    else:
        for lg in sys.argv[2:]:
            overfit(lg)
