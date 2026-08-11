#!/usr/bin/env python3
"""CUE-ROBUST-1: refit the existing component readout with partial-cue rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from address_center2 import _receipt
from component2 import balance_components, collect_states
from conjunction import _atomic_json, build_episodes, dataset_audit
from context_settle2 import _load_components
from cue_mechanism import run_evaluation
from graft_behavior import sha256_file
from key_stability import fit_stable_key_projector
from measurement.component2_gate import adjudicate as adjudicate_component
from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component_spec_sha256
from measurement.completion_registry import COMPLETION_SPEC, cue_mask_indices, mask_plan_audit
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_mechanism_gate import adjudicate as adjudicate_cue
from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256 as cue_spec_sha256
from measurement.cue_robust_registry import (
    CUE_ROBUST_SPEC,
    spec_sha256,
    training_examples_per_component,
    training_mask_indices,
    training_mask_plan_audit,
)
from measurement.projector_registry import evaluation_name
from value2 import _atomic_torch, _canonical_rows


def _source_receipt(spec: dict = CUE_ROBUST_SPEC) -> tuple[dict, dict]:
    cue_results_path = Path(spec["source_results"])
    cue_verdict_path = Path(spec["source_verdict_path"])
    cue_results = json.loads(cue_results_path.read_text())
    cue_verdict = json.loads(cue_verdict_path.read_text())
    cue_sha = cue_spec_sha256(CUE_MECHANISM_SPEC)
    if (
        cue_results.get("experiment") != spec["source_experiment"]
        or cue_results.get("spec") != CUE_MECHANISM_SPEC
        or cue_results.get("spec_sha256") != cue_sha
        or cue_verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_cue(cue_results) != cue_verdict
    ):
        raise RuntimeError("registered CUE-MECHANISM-1 source changed")

    component_results_path = Path(spec["source_component_results"])
    component_verdict_path = Path(spec["source_component_verdict_path"])
    component_results = json.loads(component_results_path.read_text())
    component_verdict = json.loads(component_verdict_path.read_text())
    component_sha = component_spec_sha256(COMPONENT2_SPEC)
    if (
        component_results.get("experiment") != COMPONENT2_SPEC["experiment"]
        or component_results.get("spec") != COMPONENT2_SPEC
        or component_results.get("spec_sha256") != component_sha
        or component_verdict.get("verdict") != spec["source_component_verdict"]
        or adjudicate_component(component_results) != component_verdict
        or cue_results["source"]["component_checkpoint"] != component_results["checkpoint"]
    ):
        raise RuntimeError("registered COMPONENT-2 source changed")

    inherited = cue_results["source"]
    receipts = [
        component_results["checkpoint"], inherited["value_checkpoint"],
        *inherited["prototype_checkpoints"].values(),
    ]
    if any(
        not Path(row["path"]).is_file()
        or sha256_file(Path(row["path"])) != row["sha256"]
        for row in receipts
    ):
        raise RuntimeError("registered CUE-ROBUST-1 checkpoint changed")
    return cue_results, {
        "cue_results": _receipt(cue_results_path),
        "cue_verdict": _receipt(cue_verdict_path),
        "cue_spec_sha256": cue_sha,
        "component_results": _receipt(component_results_path),
        "component_verdict": _receipt(component_verdict_path),
        "component_spec_sha256": component_sha,
        "component_checkpoint": dict(component_results["checkpoint"]),
        "value_checkpoint": dict(inherited["value_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in inherited["prototype_checkpoints"].items()
        },
    }


def _masked_rows(states: torch.Tensor, component: str,
                 spec: dict = CUE_ROBUST_SPEC) -> torch.Tensor:
    if (
        states.dim() != 2
        or states.shape != (training_examples_per_component(spec), spec["state_dim"])
        or not torch.isfinite(states).all()
    ):
        raise ValueError("registered training states changed shape or became non-finite")
    masked = states.detach().clone()
    for index in range(len(masked)):
        masked[index, list(training_mask_indices(index, component, spec))] = 0.0
    return masked


def _state_dict_equal(left, right) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _fit_component(states: torch.Tensor, labels: torch.Tensor, component: str,
                   source_model, spec: dict = CUE_ROBUST_SPEC):
    fit_spec = dict(COMPONENT2_SPEC)
    full_refit, full_audit = fit_stable_key_projector(
        states, labels, fit_spec, method=spec["fit_method"]
    )
    masked = _masked_rows(states, component, spec)
    augmented_states, augmented_labels = _canonical_rows(
        torch.cat((states, masked)), torch.cat((labels, labels))
    )
    robust, robust_audit = fit_stable_key_projector(
        augmented_states, augmented_labels, fit_spec, method=spec["fit_method"]
    )
    repeated, _ = fit_stable_key_projector(
        augmented_states, augmented_labels, fit_spec, method=spec["fit_method"]
    )
    reversed_states, reversed_labels = _canonical_rows(
        augmented_states.flip(0), augmented_labels.flip(0)
    )
    reordered, _ = fit_stable_key_projector(
        reversed_states, reversed_labels, fit_spec, method=spec["fit_method"]
    )
    fake_labels = (augmented_labels + spec["fake_label_offset"]) % spec[f"{component}s"]
    fake, fake_audit = fit_stable_key_projector(
        augmented_states, fake_labels, fit_spec, method=spec["fit_method"]
    )
    deterministic = (
        _state_dict_equal(robust.state_dict(), repeated.state_dict())
        and _state_dict_equal(robust.state_dict(), reordered.state_dict())
    )
    return robust, fake, {
        "full_examples": len(states),
        "masked_examples": len(masked),
        "total_examples": len(augmented_states),
        "full_state_sha256": hashlib.sha256(states.numpy().tobytes()).hexdigest(),
        "masked_state_sha256": hashlib.sha256(masked.numpy().tobytes()).hexdigest(),
        "augmented_state_sha256": hashlib.sha256(
            augmented_states.numpy().tobytes()
        ).hexdigest(),
        "label_sha256": hashlib.sha256(augmented_labels.numpy().tobytes()).hexdigest(),
        "label_counts": {
            str(label): int((augmented_labels == label).sum())
            for label in range(spec[f"{component}s"])
        },
        "full_refit": full_audit,
        "robust_fit": robust_audit,
        "fake_fit": fake_audit,
        "full_refit_matches_source": _state_dict_equal(
            full_refit.state_dict(), source_model.state_dict()
        ),
        "deterministic": deterministic,
    }


def _mask_overlap_audit(spec: dict = CUE_ROBUST_SPEC) -> dict:
    result = {}
    for component in spec["training_mask_components"]:
        training = {
            training_mask_indices(index, component, spec)
            for index in range(training_examples_per_component(spec))
        }
        evaluation = {
            cue_mask_indices(
                index, component, spec["training_missing_fraction"], COMPLETION_SPEC
            )
            for index in range(spec["eval_episodes"])
        }
        result[component] = {
            "training_unique_masks": len(training),
            "evaluation_unique_masks": len(evaluation),
            "exact_overlap": len(training & evaluation),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/cue_robust_results.json")
    parser.add_argument("--verdict", default="measurement/cue_robust_verdict.json")
    args = parser.parse_args()
    spec = CUE_ROBUST_SPEC
    cue_results, source = _source_receipt(spec)

    calibration_spec = {
        **CONJUNCTION2_SPEC,
        "eval_episodes": spec["calibration_episodes"],
        "data_seed": spec["calibration_data_seed"],
    }
    calibration = balance_components(build_episodes(calibration_spec), COMPONENT2_SPEC)
    context_states, context_labels, key_states, key_labels, state_audit = collect_states(
        calibration, COMPONENT2_SPEC
    )
    source_context, source_key = _load_components(source["component_checkpoint"], spec)
    context, fake_context, context_fit = _fit_component(
        context_states, context_labels, "context", source_context, spec
    )
    key, fake_key, key_fit = _fit_component(
        key_states, key_labels, "key", source_key, spec
    )

    checkpoint_path = Path(spec["checkpoint_path"])
    _atomic_torch(checkpoint_path, {
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "context_projector": context.state_dict(),
        "key_projector": key.state_dict(),
        "fake_context_projector": fake_context.state_dict(),
        "fake_key_projector": fake_key.state_dict(),
        "context_fit": context_fit,
        "key_fit": key_fit,
        "deterministic": context_fit["deterministic"] and key_fit["deterministic"],
    })
    checkpoint = _receipt(checkpoint_path)

    episodes = build_episodes(CONJUNCTION2_SPEC)
    evaluations = [
        {
            "name": evaluation_name(row),
            **run_evaluation(
                row["prototype_seed"], row["engine_seed"], episodes, source,
                cue_results, spec,
                context_projector_override=context,
                key_projector_override=key,
                fake_context_projector=fake_context,
                fake_key_projector=fake_key,
            ),
        }
        for row in spec["evaluation_combinations"]
    ]
    calibration_fingerprints = {row.fingerprint() for row in calibration}
    evaluation_fingerprints = {row.fingerprint() for row in episodes}
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "checkpoint": checkpoint,
        "calibration_dataset_audit": dataset_audit(calibration, calibration_spec),
        "calibration_state_audit": state_audit,
        "calibration_evaluation_overlap": len(
            calibration_fingerprints & evaluation_fingerprints
        ),
        "training_mask_plan_audit": training_mask_plan_audit(spec),
        "evaluation_mask_plan_audit": mask_plan_audit(COMPLETION_SPEC),
        "mask_overlap_audit": _mask_overlap_audit(spec),
        "context_fit": context_fit,
        "key_fit": key_fit,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.cue_robust_gate import adjudicate
    verdict = adjudicate(payload, cue_results=cue_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
