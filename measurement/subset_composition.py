#!/usr/bin/env python3
"""Check whether the 25%/50% subsets are "less of the same data" or different data.

The scaling curve came out non-monotonic (ratio to own bigram floor: 14.2% at 25%,
48.4% at 50%, 24.0% at 100%), which data volume alone cannot produce. Modulo line
sampling was chosen to avoid the prefix confound (the merged corpus is a
concatenation of sources, so its first 25% is mostly one source) -- but the corpus
is largely alternating dialogue, and taking every 2nd line of an A/B alternation
keeps only ONE speaker. That is not a smaller corpus; it is a different one.

Reports, per corpus: line count, unique-line fraction, and the most common
"<speaker>:" line prefixes. If the subsets' speaker mix differs sharply from the
full corpus, the curve measured sampling artefacts, not data volume.
"""
import re
import sys
from collections import Counter

PREFIX = re.compile(r"^([^\s:]{1,16}\s*:)\s")


def stats(path):
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    pref = Counter()
    for raw in lines:
        ln = raw.decode("utf-8", errors="replace")
        if not ln.strip():
            pref["(empty)"] += 1
            continue
        m = PREFIX.match(ln)
        pref[m.group(1) if m else "(no prefix)"] += 1
    total = len(lines)
    uniq = len(set(lines))
    print(f"{path}")
    print(f"  lines={total:,}  unique={uniq:,} ({uniq / total * 100:.1f}%)")
    print("  top prefixes: " + " · ".join(
        f"{k}={v:,}({v / total * 100:.1f}%)" for k, v in pref.most_common(6)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
