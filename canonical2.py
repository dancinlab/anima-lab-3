#!/usr/bin/env python3
"""CANONICAL-2: integrate and regress the canonical stable-address default."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from capacity import build_capacity_episodes, count_spec
from capacity2 import _run_condition_count
from episode2 import _load_frozen_projector
from episode_control import build_reference_splits
from graft_behavior import sha256_file
from key_stability import collect_calibration_states, fit_stable_key_projector, train_projector
from measurement.canonical2_registry import CANONICAL2_SPEC, spec_sha256
from measurement.canonical_registry import CANONICAL_SPEC, spec_sha256 as canonical_spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.key_registry import KEY_SPEC
from measurement.projector_registry import evaluation_name
from separation import dataset_audit


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _state_dict_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _source(spec: dict = CANONICAL2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = canonical_spec_sha256(CANONICAL_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CANONICAL_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered CANONICAL-1 source changed")
    pooled = next(row for row in results["canonical_projectors"] if row["name"] == "pooled")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "pooled_checkpoint": pooled["checkpoint"],
        "prototype_checkpoints": results["source_training"]["prototype_checkpoints"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/canonical2_results.json")
    parser.add_argument("--verdict", default="measurement/canonical2_verdict.json")
    parser.add_argument("--checkpoint", default="checkpoints/canonical2/default_pooled.pt")
    args = parser.parse_args()
    spec = CANONICAL2_SPEC
    source_results, source = _source(spec)
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    state_rows, label_rows, source_audits = [], [], {}
    for seed in spec["calibration_seeds"]:
        states, labels, audit = collect_calibration_states(
            splits[spec["calibration_split"]], seed, KEY_SPEC
        )
        state_rows.append(states)
        label_rows.append(labels)
        source_audits[str(seed)] = audit
    states, labels = torch.cat(state_rows), torch.cat(label_rows)
    projector, fit_audit = fit_stable_key_projector(
        states, labels, KEY_SPEC, method=spec["fit_method"]
    )
    source_checkpoint = torch.load(
        source["pooled_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    source_match = _state_dict_equal(projector.state_dict(), source_checkpoint["projector"])
    projector.requires_grad_(False)

    historical_results = json.loads(Path(source_results["source_training"]["results"]["path"]).read_text())
    projector_results = json.loads(Path(historical_results["source_projector"]["results"]["path"]).read_text())
    legacy_receipt = projector_results["source"]["projector_checkpoints"][str(spec["calibration_seeds"][0])]
    legacy_loaded = _load_frozen_projector(spec["calibration_seeds"][0], legacy_receipt, EPISODE2_SPEC)
    legacy_default, _ = train_projector(state_rows[0], label_rows[0], spec["calibration_seeds"][0], False, KEY_SPEC)
    legacy_explicit, _ = train_projector(
        state_rows[0], label_rows[0], spec["calibration_seeds"][0], False, KEY_SPEC,
        batch_seed=spec["calibration_seeds"][0],
    )
    legacy_compatibility = {
        "checkpoint_loaded": _state_dict_equal(legacy_loaded.state_dict(), torch.load(
            legacy_receipt["path"], map_location="cpu", weights_only=True
        )["projector"]),
        "default_matches_explicit": _state_dict_equal(legacy_default.state_dict(), legacy_explicit.state_dict()),
        "checkpoint": legacy_receipt,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "fit_method": spec["fit_method"],
        "calibration_seeds": spec["calibration_seeds"],
        "model_class": spec["model_class"],
        "projector": projector.state_dict(),
        "source_audits": source_audits,
        "fit_audit": fit_audit,
        "canonical1_pooled_match": source_match,
    }, checkpoint_path)
    before = {key: value.detach().clone() for key, value in projector.state_dict().items()}
    condition = {"name": "settled", "updates": spec["settling_updates"], "disabled": []}
    counts = []
    dataset_audits = {}
    for count in spec["event_counts"]:
        episodes = build_capacity_episodes(count)
        dataset_audits[str(count)] = dataset_audit(episodes, count_spec(count))
        evaluations = []
        for evaluation in spec["evaluation_combinations"]:
            prototype_checkpoint = torch.load(
                source["prototype_checkpoints"][str(evaluation["prototype_seed"])]["path"],
                map_location="cpu",
                weights_only=True,
            )
            result, _ = _run_condition_count(
                evaluation["engine_seed"], count, episodes, condition, projector,
                prototype_checkpoint["prototypes"]["quantum"], spec,
            )
            evaluations.append({"name": evaluation_name(evaluation), **evaluation, "result": result})
        counts.append({"event_count": count, "evaluations": evaluations})
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_canonical": source,
        "source_audits": source_audits,
        "fit_audit": fit_audit,
        "canonical1_pooled_match": source_match,
        "legacy_compatibility": legacy_compatibility,
        "checkpoint": _receipt(checkpoint_path),
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": _state_dict_equal(before, projector.state_dict()),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "dataset_audit": dataset_audits,
        "counts": counts,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.canonical2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
