#!/usr/bin/env python3
"""Merge an isolated scorer shard into its canonical result JSON.

Metadata keys (``_...``) are experimental receipts, so a shard is accepted only
when every receipt matches the canonical file.  Arm rows are then replaced by
name, making a retry idempotent without allowing a run from another setup to be
mixed into the board.
"""
import argparse
import json
from pathlib import Path


def merge(canonical, shard, expected_arms=None):
    canonical_arms = {key for key in canonical if not key.startswith("_")}
    shard_arms = {key for key in shard if not key.startswith("_")}
    expected = set(expected_arms or shard_arms)
    if shard_arms != expected:
        raise ValueError(
            f"shard roster mismatch: expected {sorted(expected)}, got {sorted(shard_arms)}"
        )

    canonical_meta = {key: value for key, value in canonical.items() if key.startswith("_")}
    shard_meta = {key: value for key, value in shard.items() if key.startswith("_")}
    if canonical_meta != shard_meta:
        differing = sorted(set(canonical_meta) | set(shard_meta))
        differing = [key for key in differing if canonical_meta.get(key) != shard_meta.get(key)]
        raise ValueError(f"experimental receipt mismatch: {', '.join(differing)}")

    merged = dict(canonical)
    merged.update((arm, shard[arm]) for arm in sorted(shard_arms))
    return merged, len(canonical_arms), len(shard_arms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("arms", nargs="*")
    args = parser.parse_args()

    canonical = json.loads(args.canonical.read_text())
    shard = json.loads(args.shard.read_text())
    merged, before, added = merge(canonical, shard, args.arms)
    # Scorer JSONs use readable UTF-8 samples as their canonical format.
    args.canonical.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    after = len([key for key in merged if not key.startswith("_")])
    print(f"merged {added} arm(s): {before} -> {after} in {args.canonical}")


if __name__ == "__main__":
    main()
