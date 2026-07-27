#!/usr/bin/env python3
"""Turn the Korean Wikipedia dump into a natural-prose corpus, and measure it.

Why this exists. Every corpus this repo has trained on is CONSTRUCTED, measured
not assumed (measurement/corpus_regime.py): corpus_v2 repeats its three most
common lines 2,538 times each on the subject of this project, corpus_v4 repeats
"[합의 형성 중...]" 35,598 times, corpus_v5 repeats its own engine's telemetry
28,810 times. Under the p9 rule imported from the sibling anima repo, a faculty
claim measured on such a corpus is not weak but OFF-STANDARD, so every λ-ladder
pass so far is an instrument check. Flipping Λ REGIME to natural is the single
prerequisite that unblocks both λ4 and every faculty-level reading.

Wikipedia was chosen over the alternatives for reasons that are about failure
modes, not convenience:
  - Common Crawl derivatives (OSCAR, CulturaX, mC4-ko) are natural in origin but
    now carry machine-generated text. That contaminant violates p9 SILENTLY --
    the corpus would measure as natural and not be. Excluded.
  - 모두의 말뭉치 (국립국어원) and AI Hub are the highest-quality natural Korean
    available and are register-balanced, but both need an application and a human
    approval. They are the better corpus and cannot be fetched by a script.
  - 나무위키 is large and natural but CC BY-NC-SA (non-commercial) and written in
    a house style heavy with memes and nested footnotes.
  - Wikipedia is human-written prose, CC BY-SA, at a fixed reproducible URL, and
    big enough (1.32 GB compressed) that the 60 MB this repo needs is a sample
    rather than the whole thing.
Its known weakness is register: encyclopedic prose only. That is a bias to state,
not a defect to hide, and it is recorded in the output.

The cleaner is deliberately conservative. It drops anything it is not confident
is prose, because a corpus that keeps markup would fail the regime test for the
wrong reason and cost a re-run. Yield is not the goal; 60 MB out of ~1.3 GB is
plenty.
"""
import bz2
import html
import re
import sys

# Wiki markup, stripped in the order that avoids leaving fragments behind.
RE_COMMENT = re.compile(rb"<!--.*?-->", re.S)
RE_REF = re.compile(rb"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
RE_TAG = re.compile(rb"<[^>]+>")
RE_TEMPLATE = re.compile(rb"\{\{[^{}]*\}\}")          # applied repeatedly: nested
RE_TABLE = re.compile(rb"\{\|.*?\|\}", re.S)
RE_LINK_PIPED = re.compile(rb"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
RE_LINK_PLAIN = re.compile(rb"\[\[([^\[\]|]*)\]\]")
RE_EXTLINK = re.compile(rb"\[https?://[^\s\]]*\s?([^\]]*)\]")
RE_QUOTES = re.compile(rb"'{2,5}")
RE_HEADING = re.compile(rb"^=+.*?=+\s*$", re.M)
RE_WS = re.compile(rb"[ \t]+")
RE_TEXT = re.compile(rb"<text[^>]*>(.*?)</text>", re.S)

# A line is kept only if it looks like a sentence: enough Hangul, no leftover
# markup, not a list/table stub.
RE_HANGUL = re.compile(rb"[\xea-\xed]")               # UTF-8 lead bytes for 한글
BAD_CHARS = (b"{", b"}", b"[", b"]", b"|", b"=", b"<", b">")
MIN_LINE, MAX_LINE = 30, 400


def strip_markup(raw):
    raw = RE_COMMENT.sub(b" ", raw)
    raw = RE_REF.sub(b" ", raw)
    raw = RE_TABLE.sub(b" ", raw)
    for _ in range(6):                                 # templates nest
        raw, n = RE_TEMPLATE.subn(b" ", raw)
        if not n:
            break
    raw = RE_LINK_PIPED.sub(rb"\1", raw)
    raw = RE_LINK_PLAIN.sub(rb"\1", raw)
    raw = RE_EXTLINK.sub(rb"\1", raw)
    raw = RE_TAG.sub(b" ", raw)
    raw = RE_HEADING.sub(b" ", raw)
    raw = RE_QUOTES.sub(b"", raw)
    return raw


def keep(line):
    if not (MIN_LINE <= len(line) <= MAX_LINE):
        return False
    if line[:1] in (b"*", b"#", b":", b";", b"!"):      # list / table row
        return False
    if any(c in line for c in BAD_CHARS):               # markup survived
        return False
    hangul = len(RE_HANGUL.findall(line))
    if hangul * 3 < len(line) * 0.30:                   # ~30% of bytes Hangul
        return False
    return line.rstrip().endswith((b".", b"다", b"요", b"까", b"음"))


def main():
    dump, out_path, target_mb = sys.argv[1], sys.argv[2], float(sys.argv[3])
    target = int(target_mb * (1 << 20))
    written, pages, kept_lines = 0, 0, 0
    buf = b""
    with bz2.open(dump, "rb") as f, open(out_path, "wb") as out:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            buf += chunk
            while True:
                m = RE_TEXT.search(buf)
                if not m:
                    break
                body = strip_markup(m.group(1))
                buf = buf[m.end():]
                pages += 1
                for line in body.split(b"\n"):
                    line = RE_WS.sub(b" ", html.unescape(
                        line.decode("utf8", "replace")).encode("utf8")).strip()
                    if keep(line):
                        out.write(line + b"\n")
                        written += len(line) + 1
                        kept_lines += 1
                if written >= target:
                    break
            if written >= target:
                break
            if len(buf) > (1 << 26):                    # no <text> in 64 MB: resync
                buf = buf[-(1 << 20):]
            if pages and pages % 20000 == 0:
                print(f"[scan] pages={pages:,} kept={kept_lines:,} lines "
                      f"({written/1e6:.1f} MB)", flush=True)
    print(f"[done] {pages:,} pages scanned · {kept_lines:,} prose lines kept · "
          f"{written/1e6:.1f} MB -> {out_path}", flush=True)
    print("[regime] now run: python3 corpus_regime.py " + out_path, flush=True)
    print("[bias] register is encyclopedic prose only -- a natural corpus, but not "
          "a register-balanced one. State it in any claim built on this.", flush=True)


if __name__ == "__main__":
    main()
