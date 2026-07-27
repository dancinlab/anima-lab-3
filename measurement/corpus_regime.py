#!/usr/bin/env python3
"""Which corpus regime is this? natural / drill-installed / drill-only.

The sibling anima repo's p9 makes this a gate, not a footnote: a faculty claim
measured on a synthetic corpus is not weak, it is OFF-STANDARD and does not
enter the ledger as evidence for the faculty. Synthetic stays legal for exactly
one job -- certifying that the instrument reads at all.

We never stated a regime for corpus_v2/v4/v5. Measuring it rather than eyeballing
the A:/B: turn structure, because "looks generated" is not a reading.

Signals, each of which a hand-built corpus shows and a scraped one does not:
  - line-level repetition (how many times the average distinct line appears)
  - speaker-prefix regularity (a scrape has ragged formatting)
  - vocabulary growth vs size (a template exhausts its vocabulary early)
  - top-line concentration (how much of the file is its most common lines)
"""
import re
import sys
from collections import Counter

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    lines = [ln for ln in data.split(b"\n") if ln.strip()]
    c = Counter(lines)
    distinct = len(c)
    rep = len(lines) / max(1, distinct)
    prefixed = sum(1 for ln in lines if re.match(rb"^[AB]\s*:", ln))
    top10 = sum(n for _, n in c.most_common(10))
    # vocabulary growth: distinct whitespace tokens in the first 10% vs the whole
    head = b"\n".join(lines[:max(1, len(lines) // 10)])
    vocab_head = len(set(head.split()))
    vocab_all = len(set(data.split()))
    print(f"[{path.split('/')[-1]}]")
    print(f"  lines={len(lines):,} distinct={distinct:,} · repetition={rep:.2f}x")
    print(f"  speaker-prefixed 'A:'/'B:' = {prefixed/len(lines)*100:.1f}%")
    print(f"  top-10 lines cover {top10/len(lines)*100:.2f}% of the file")
    print(f"  vocab: first 10% has {vocab_head:,} of the full {vocab_all:,} "
          f"= {vocab_head/max(1,vocab_all)*100:.1f}% (a template exhausts early)")
    print(f"  most common: {[ (l[:40].decode('utf8','replace'), n) for l, n in c.most_common(3) ]}")
