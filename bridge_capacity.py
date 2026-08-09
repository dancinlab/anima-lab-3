#!/usr/bin/env python3
"""Run the registered ThalamicBridge bottleneck-width comparison."""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from measurement.bridge_capacity_registry import BRIDGE_CAPACITY_SPEC, spec_sha256
from state_survival import run_seed


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def run_width(hub_dim: int) -> dict:
    spec = deepcopy(BRIDGE_CAPACITY_SPEC)
    spec["bridge"] = {
        "hub_dim": hub_dim,
        "output_dim": BRIDGE_CAPACITY_SPEC["bridge"]["output_dim"],
        "readout": BRIDGE_CAPACITY_SPEC["bridge"]["readout"],
    }
    return {
        "hub_dim": hub_dim,
        "pooling": BRIDGE_CAPACITY_SPEC["bridge"]["pooling"],
        "seeds": [run_seed(seed, spec=spec) for seed in BRIDGE_CAPACITY_SPEC["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/bridge_capacity_results.json")
    args = parser.parse_args()
    payload = {
        "experiment": BRIDGE_CAPACITY_SPEC["experiment"],
        "spec": BRIDGE_CAPACITY_SPEC,
        "spec_sha256": spec_sha256(),
        "widths": [run_width(width) for width in BRIDGE_CAPACITY_SPEC["bridge"]["hub_dims"]],
    }
    output = Path(args.output)
    _atomic_json(output, payload)
    print(f"[results] {output}")


if __name__ == "__main__":
    main()
