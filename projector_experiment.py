#!/usr/bin/env python3
"""PROJECTOR-1: separate calibration-state and training-randomness seeds."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from capacity import build_capacity_episodes
from capacity2 import _run_condition_count
from episode_control import build_reference_splits
from graft_behavior import sha256_file
from key_stability import collect_calibration_states, train_projector
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from measurement.projector_registry import (
    PROJECTOR_SPEC,
    evaluation_name,
    projector_name,
    spec_sha256,
)
from measurement.seedmap_registry import SEEDMAP_SPEC, combination_name, spec_sha256 as seedmap_spec_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _sources(spec: dict = PROJECTOR_SPEC) -> tuple[dict, dict, dict]:
    seedmap_results_path = Path(spec["source_results"])
    seedmap_verdict_path = Path(spec["source_verdict_path"])
    key_results_path = Path(spec["source_key_results"])
    key_verdict_path = Path(spec["source_key_verdict_path"])
    seedmap_results = json.loads(seedmap_results_path.read_text())
    seedmap_verdict = json.loads(seedmap_verdict_path.read_text())
    key_results = json.loads(key_results_path.read_text())
    key_verdict = json.loads(key_verdict_path.read_text())
    if (
        seedmap_results.get("experiment") != spec["source_experiment"]
        or seedmap_results.get("spec") != SEEDMAP_SPEC
        or seedmap_results.get("spec_sha256") != seedmap_spec_sha256(SEEDMAP_SPEC)
        or seedmap_verdict.get("verdict") != spec["source_verdict"]
        or key_results.get("experiment") != spec["source_key_experiment"]
        or key_results.get("spec") != KEY_SPEC
        or key_results.get("spec_sha256") != key_spec_sha256(KEY_SPEC)
        or key_verdict.get("verdict") != spec["source_key_verdict"]
    ):
        raise RuntimeError("registered SEEDMAP-1 or KEY-1 source changed")
    source = {
        "seedmap_results": _receipt(seedmap_results_path),
        "seedmap_verdict": _receipt(seedmap_verdict_path),
        "key_results": _receipt(key_results_path),
        "key_verdict": _receipt(key_verdict_path),
        "seedmap_spec_sha256": seedmap_spec_sha256(SEEDMAP_SPEC),
        "key_spec_sha256": key_spec_sha256(KEY_SPEC),
        "projector_checkpoints": seedmap_results["projector_checkpoints"],
        "prototype_checkpoints": seedmap_results["prototype_checkpoints"],
    }
    for receipts in (source["projector_checkpoints"], source["prototype_checkpoints"]):
        for receipt in receipts.values():
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError("inherited checkpoint changed")
    return seedmap_results, key_results, source


def _state_dict_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/projector_results.json")
    parser.add_argument("--verdict", default="measurement/projector_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/projector1")
    args = parser.parse_args()
    spec = PROJECTOR_SPEC
    seedmap_results, key_results, source = _sources(spec)
    key_rows = {row["seed"]: row for row in key_results["seeds"]}
    source_combinations = {row["name"]: row["result"] for row in seedmap_results["combinations"]}
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    calibration_episodes = splits[spec["calibration_split"]]
    states = {}
    labels = {}
    state_audits = {}
    for calibration_seed in spec["factor_seeds"]:
        states[calibration_seed], labels[calibration_seed], state_audits[calibration_seed] = (
            collect_calibration_states(calibration_episodes, calibration_seed, KEY_SPEC)
        )
    projectors = {}
    training_audits = {}
    native_matches = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_receipts = {}
    for combination in spec["training_combinations"]:
        name = projector_name(combination)
        calibration_seed = combination["calibration_seed"]
        training_seed = combination["training_seed"]
        projector, audit = train_projector(
            states[calibration_seed], labels[calibration_seed], training_seed, False, KEY_SPEC
        )
        projector.eval()
        projector.requires_grad_(False)
        projectors[name] = projector
        training_audits[name] = audit
        native_match = None
        if calibration_seed == training_seed:
            native_checkpoint = torch.load(
                source["projector_checkpoints"][str(training_seed)]["path"],
                map_location="cpu",
                weights_only=True,
            )
            native_match = _state_dict_equal(projector.state_dict(), native_checkpoint["projector"])
        native_matches[name] = native_match
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save({
            "experiment": spec["experiment"],
            "spec_sha256": spec_sha256(spec),
            **combination,
            "model_class": spec["model_class"],
            "projector": projector.state_dict(),
            "calibration_audit": state_audits[calibration_seed],
            "training_audit": audit,
            "native_checkpoint_match": native_match,
        }, checkpoint_path)
        checkpoint_receipts[name] = _receipt(checkpoint_path)

    episodes = build_capacity_episodes(spec["event_count"])
    condition = {"name": "settled", "updates": spec["settling_updates"], "disabled": []}
    projector_rows = []
    for combination in spec["training_combinations"]:
        name = projector_name(combination)
        before = {key: value.detach().clone() for key, value in projectors[name].state_dict().items()}
        evaluations = []
        for evaluation in spec["evaluation_combinations"]:
            eval_name = evaluation_name(evaluation)
            if combination["calibration_seed"] == combination["training_seed"]:
                source_name = combination_name({
                    "projector_seed": combination["training_seed"],
                    **evaluation,
                })
                result = source_combinations[source_name]
                reused = True
            else:
                prototype_checkpoint = torch.load(
                    source["prototype_checkpoints"][str(evaluation["prototype_seed"])]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                result, _ = _run_condition_count(
                    evaluation["engine_seed"],
                    spec["event_count"],
                    episodes,
                    condition,
                    projectors[name],
                    prototype_checkpoint["prototypes"]["quantum"],
                    spec,
                )
                reused = False
            evaluations.append({"name": eval_name, **evaluation, "source_reused": reused, "result": result})
        projector_rows.append({
            "name": name,
            **combination,
            "calibration_audit": state_audits[combination["calibration_seed"]],
            "training_audit": training_audits[name],
            "checkpoint": checkpoint_receipts[name],
            "native_checkpoint_match": native_matches[name],
            "projector_frozen": not any(parameter.requires_grad for parameter in projectors[name].parameters()),
            "projector_unchanged": _state_dict_equal(before, projectors[name].state_dict()),
            "evaluations": evaluations,
        })
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": key_results["dataset_audit"],
        "capacity_dataset_audit": seedmap_results["dataset_audit"],
        "source": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "projectors": projector_rows,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.projector_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
