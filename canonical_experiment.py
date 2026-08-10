#!/usr/bin/env python3
"""CANONICAL-1: evaluate deterministic ridge stable-address maps."""
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
from key_stability import collect_calibration_states, fit_canonical_projector
from measurement.canonical_registry import CANONICAL_SPEC, spec_sha256
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.key_registry import KEY_SPEC
from measurement.projector_registry import evaluation_name
from measurement.training_registry import TRAINING_SPEC, spec_sha256 as training_spec_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source(spec: dict = CANONICAL_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = training_spec_sha256(TRAINING_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != TRAINING_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered TRAINING-1 source changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "prototype_checkpoints": results["source_projector"]["prototype_checkpoints"],
    }


def _state_dict_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _maximum_delta(left: dict, right: dict) -> float:
    return max(float((left[name] - right[name]).abs().max()) for name in left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/canonical_results.json")
    parser.add_argument("--verdict", default="measurement/canonical_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/canonical1")
    args = parser.parse_args()
    spec = CANONICAL_SPEC
    source_results, source = _source(spec)
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    states, labels, audits = {}, {}, {}
    for seed in spec["factor_seeds"]:
        states[seed], labels[seed], audits[seed] = collect_calibration_states(
            splits[spec["calibration_split"]], seed, KEY_SPEC
        )
    projectors, fit_audits, repeat_equal, order_deltas, receipts = {}, {}, {}, {}, {}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for arm in spec["calibration_arms"]:
        name = arm["name"]
        arm_states = torch.cat([states[seed] for seed in arm["calibration_seeds"]])
        arm_labels = torch.cat([labels[seed] for seed in arm["calibration_seeds"]])
        projector, fit_audit = fit_canonical_projector(arm_states, arm_labels, KEY_SPEC)
        repeated, _ = fit_canonical_projector(arm_states, arm_labels, KEY_SPEC)
        repeat_equal[name] = _state_dict_equal(projector.state_dict(), repeated.state_dict())
        order_delta = None
        if len(arm["calibration_seeds"]) > 1:
            reversed_projector, _ = fit_canonical_projector(
                torch.cat([states[seed] for seed in reversed(arm["calibration_seeds"])]),
                torch.cat([labels[seed] for seed in reversed(arm["calibration_seeds"])]),
                KEY_SPEC,
            )
            order_delta = _maximum_delta(projector.state_dict(), reversed_projector.state_dict())
        order_deltas[name] = order_delta
        projector.requires_grad_(False)
        projectors[name] = projector
        fit_audits[name] = fit_audit
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save({
            "experiment": spec["experiment"],
            "spec_sha256": spec_sha256(spec),
            "name": name,
            "calibration_seeds": arm["calibration_seeds"],
            "model_class": spec["model_class"],
            "projector": projector.state_dict(),
            "source_audits": {str(seed): audits[seed] for seed in arm["calibration_seeds"]},
            "fit_audit": fit_audit,
            "repeat_equal": repeat_equal[name],
            "reverse_order_max_abs_delta": order_delta,
        }, checkpoint_path)
        receipts[name] = _receipt(checkpoint_path)

    episodes = build_capacity_episodes(spec["event_count"])
    condition = {"name": "settled", "updates": spec["settling_updates"], "disabled": []}
    rows = []
    for arm in spec["calibration_arms"]:
        name = arm["name"]
        before = {key: value.detach().clone() for key, value in projectors[name].state_dict().items()}
        evaluations = []
        for evaluation in spec["evaluation_combinations"]:
            checkpoint = torch.load(
                source["prototype_checkpoints"][str(evaluation["prototype_seed"])]["path"],
                map_location="cpu",
                weights_only=True,
            )
            result, _ = _run_condition_count(
                evaluation["engine_seed"], spec["event_count"], episodes, condition,
                projectors[name], checkpoint["prototypes"]["quantum"], spec,
            )
            evaluations.append({"name": evaluation_name(evaluation), **evaluation, "result": result})
        rows.append({
            "name": name,
            "calibration_seeds": arm["calibration_seeds"],
            "source_audits": {str(seed): audits[seed] for seed in arm["calibration_seeds"]},
            "fit_audit": fit_audits[name],
            "repeat_equal": repeat_equal[name],
            "reverse_order_max_abs_delta": order_deltas[name],
            "checkpoint": receipts[name],
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
        "source_training": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "canonical_projectors": rows,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.canonical_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
