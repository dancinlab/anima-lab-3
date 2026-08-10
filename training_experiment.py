#!/usr/bin/env python3
"""TRAINING-1: separate address initialization and minibatch-order seeds."""
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
from measurement.key_registry import KEY_SPEC
from measurement.projector_registry import PROJECTOR_SPEC, evaluation_name, projector_name, spec_sha256 as projector_spec_sha256
from measurement.training_registry import TRAINING_SPEC, spec_sha256, training_name


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source(spec: dict = TRAINING_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = projector_spec_sha256(PROJECTOR_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != PROJECTOR_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered PROJECTOR-1 source changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "projector_checkpoints": {
            row["name"]: row["checkpoint"] for row in results["projectors"]
        },
        "prototype_checkpoints": results["source"]["prototype_checkpoints"],
    }


def _state_dict_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/training_results.json")
    parser.add_argument("--verdict", default="measurement/training_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/training1")
    args = parser.parse_args()
    spec = TRAINING_SPEC
    source_results, source = _source(spec)
    source_rows = {row["name"]: row for row in source_results["projectors"]}
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    states, labels, calibration_audit = collect_calibration_states(
        splits[spec["calibration_split"]], spec["calibration_seed"], KEY_SPEC
    )
    projectors, training_audits, diagonal_matches, receipts = {}, {}, {}, {}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for combination in spec["training_combinations"]:
        name = training_name(combination)
        projector, audit = train_projector(
            states,
            labels,
            combination["initialization_seed"],
            False,
            KEY_SPEC,
            batch_seed=combination["batch_seed"],
        )
        projector.eval()
        projector.requires_grad_(False)
        projectors[name] = projector
        training_audits[name] = audit
        diagonal_match = None
        if combination["initialization_seed"] == combination["batch_seed"]:
            source_name = projector_name({
                "calibration_seed": spec["calibration_seed"],
                "training_seed": combination["initialization_seed"],
            })
            checkpoint = torch.load(
                source["projector_checkpoints"][source_name]["path"],
                map_location="cpu",
                weights_only=True,
            )
            diagonal_match = _state_dict_equal(projector.state_dict(), checkpoint["projector"])
        diagonal_matches[name] = diagonal_match
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save({
            "experiment": spec["experiment"],
            "spec_sha256": spec_sha256(spec),
            **combination,
            "calibration_seed": spec["calibration_seed"],
            "model_class": spec["model_class"],
            "projector": projector.state_dict(),
            "calibration_audit": calibration_audit,
            "training_audit": audit,
            "diagonal_checkpoint_match": diagonal_match,
        }, checkpoint_path)
        receipts[name] = _receipt(checkpoint_path)

    episodes = build_capacity_episodes(spec["event_count"])
    condition = {"name": "settled", "updates": spec["settling_updates"], "disabled": []}
    public = []
    for combination in spec["training_combinations"]:
        name = training_name(combination)
        before = {key: value.detach().clone() for key, value in projectors[name].state_dict().items()}
        evaluations = []
        diagonal = combination["initialization_seed"] == combination["batch_seed"]
        for evaluation in spec["evaluation_combinations"]:
            eval_name = evaluation_name(evaluation)
            if diagonal:
                source_name = projector_name({
                    "calibration_seed": spec["calibration_seed"],
                    "training_seed": combination["initialization_seed"],
                })
                result = next(
                    row["result"] for row in source_rows[source_name]["evaluations"]
                    if row["name"] == eval_name
                )
                reused = True
            else:
                prototype_checkpoint = torch.load(
                    source["prototype_checkpoints"][str(evaluation["prototype_seed"])]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                result, _ = _run_condition_count(
                    evaluation["engine_seed"], spec["event_count"], episodes, condition,
                    projectors[name], prototype_checkpoint["prototypes"]["quantum"], spec,
                )
                reused = False
            evaluations.append({"name": eval_name, **evaluation, "source_reused": reused, "result": result})
        public.append({
            "name": name,
            **combination,
            "calibration_seed": spec["calibration_seed"],
            "calibration_audit": calibration_audit,
            "training_audit": training_audits[name],
            "checkpoint": receipts[name],
            "diagonal_checkpoint_match": diagonal_matches[name],
            "projector_frozen": not any(parameter.requires_grad for parameter in projectors[name].parameters()),
            "projector_unchanged": _state_dict_equal(before, projectors[name].state_dict()),
            "evaluations": evaluations,
        })
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": source_results["dataset_audit"],
        "capacity_dataset_audit": source_results["capacity_dataset_audit"],
        "source_projector": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "training_combinations": public,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.training_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
