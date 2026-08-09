#!/usr/bin/env python3
"""WORKSPACE-1: recurrently combine two hidden modules through the GRAFT path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from graft_behavior import sha256_file
from measurement.workspace_information_gate import adjudicate as adjudicate_information
from measurement.workspace_registry import (
    WORKSPACE_CONTROL_SEED_REPAIR_SPEC,
    experiment,
    spec_sha256,
)
from synergy import HFDecoder, audit_examples, run_seed


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_map(spec: dict) -> dict:
    results_path = Path(spec["source_map_results"])
    verdict_path = Path(spec["source_map_verdict"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if adjudicate_information(results) != verdict:
        raise ValueError("committed WORKSPACE-1 information-map verdict does not reproduce")
    return {
        "results": results,
        "verdict": verdict,
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "verdict_path": str(verdict_path),
        "verdict_sha256": sha256_file(verdict_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/workspace_results.json")
    parser.add_argument("--verdict", default="measurement/workspace_verdict.json")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/workspace1_control_seed_repair",
    )
    parser.add_argument(
        "--experiment",
        default=WORKSPACE_CONTROL_SEED_REPAIR_SPEC["experiment"],
    )
    parser.add_argument("--model")
    parser.add_argument("--seeds")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    spec = experiment(args.experiment)
    if args.train_steps is not None:
        spec["train_steps"] = args.train_steps
    seeds = spec["seeds"] if args.seeds is None else [
        int(item) for item in args.seeds.split(",") if item
    ]
    model = args.model or spec["model"]
    output = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    decoder = HFDecoder(
        model, lora=False, freeze_base=True,
        gate_strength=spec["gate_strength"], gate_rms_max=spec["gate_rms_max"],
    )
    decoder.model.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    seed_rows = [run_seed(decoder, seed, checkpoint_dir, spec) for seed in seeds]
    audits = [row.pop("dataset_audit") for row in seed_rows]
    if any(audit != audits[0] for audit in audits[1:]):
        raise RuntimeError("seed-specific split-cue audits disagree")
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "model": model,
        "dataset_audit": audits[0],
        "source_map": _source_map(spec),
        "seeds": seed_rows,
    }
    _atomic_json(output, payload)
    if seeds == spec["seeds"] and args.train_steps is None:
        from measurement.workspace_gate import adjudicate
        verdict = adjudicate(payload)
        _atomic_json(Path(args.verdict), verdict)
        print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
