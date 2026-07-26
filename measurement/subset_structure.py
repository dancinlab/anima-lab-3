#!/usr/bin/env python3
"""Look past composition for what makes the 50% subset different.

DATA-5 left one question: the three subsets score 14.2% / 48.4% / 24.0% against
their own bigram floors, and the ordering reproduces under a novelty-controlled
test, so it is a real property of the runs. Composition is already ruled out --
unique-line fraction (49.3%), empty-line share (50.7%) and speaker-prefix mix are
proportionally identical across all three.

So this measures structure a composition histogram cannot see:

  local repetition  -- how often a line equals the line N before it. Modulo
                       sampling changes WHICH lines end up adjacent, so a corpus
                       that alternates can become one that repeats, or vice versa.
  line length       -- mean and median, and the share of lines over 200 bytes
  transitions       -- distinct (prev-line-prefix -> line-prefix) pairs, i.e. how
                       much conversational structure survived the sampling
  block homogeneity -- for 256-byte windows, how often the window stays inside a
                       single line vs straddles a boundary (the model trains on
                       windows, not lines)
"""
import sys
from collections import Counter


def stats(path):
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    n = len(lines)

    repeat = {k: sum(1 for i in range(k, n) if lines[i] and lines[i] == lines[i - k])
              for k in (1, 2, 4)}
    lens = sorted(len(x) for x in lines)
    nonempty = [x for x in lens if x]
    mean = sum(lens) / n
    median = lens[n // 2]
    long_share = sum(1 for x in lens if x > 200) / n

    trans = Counter()
    prev = b""
    for ln in lines:
        trans[(prev[:8], ln[:8])] += 1
        prev = ln
    newline_density = data.count(b"\n") / len(data)

    print(f"{path}")
    print(f"  lines={n:,} · mean len={mean:.1f}B · median={median}B · "
          f">200B share={long_share * 100:.1f}% · non-empty={len(nonempty):,}")
    print("  local repeats (line == line k before): " + " · ".join(
        f"k={k}: {v:,} ({v / n * 100:.2f}%)" for k, v in repeat.items()))
    print(f"  distinct 8-byte prefix transitions={len(trans):,} "
          f"({len(trans) / n * 100:.1f}% of lines) · newline density={newline_density * 100:.2f}%")
    print(f"  avg bytes between newlines={1 / newline_density:.1f} "
          f"(a 256B training window spans ~{256 * newline_density:.1f} lines)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
