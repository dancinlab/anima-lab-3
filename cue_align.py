#!/usr/bin/env python3
"""CUE-ALIGN-1: align query-time context states to storage-time coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from address_center2 import _receipt
from component2 import balance_components
from conjunction import _atomic_json, build_episodes, dataset_audit
from cue_context import (
    _classification, _collect_pairs, _load_source_context, _mask_rows,
    _reference_metric_match,
)
from measurement.completion_registry import COMPLETION_SPEC, cue_mask_indices
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_align_registry import CUE_ALIGN_SPEC, calibration_pairs, spec_sha256
from measurement.cue_context_gate import adjudicate as adjudicate_context
from measurement.cue_context_registry import (
    CUE_CONTEXT_SPEC, spec_sha256 as context_spec_sha256, training_mask_indices,
)
from measurement.projector_registry import evaluation_name
from measurement.component2_registry import COMPONENT2_SPEC
from value2 import _atomic_torch


def _source_receipt(spec: dict = CUE_ALIGN_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    source_sha = context_spec_sha256(CUE_CONTEXT_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CUE_CONTEXT_SPEC
        or results.get("spec_sha256") != source_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_context(results) != verdict
    ):
        raise RuntimeError("registered CUE-CONTEXT-1 source changed")
    receipts = [results["checkpoint"], *(
        results["source"][name] for name in (
            "results", "verdict", "robust_checkpoint", "component_checkpoint",
            "value_checkpoint",
        )
    ), *results["source"]["prototype_checkpoints"].values()]
    if any(
        not Path(row["path"]).is_file()
        or hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]
        for row in receipts
    ):
        raise RuntimeError("registered CUE-ALIGN-1 source checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": source_sha,
        "context_checkpoint": dict(results["checkpoint"]),
        "upstream": {name: value for name, value in results["source"].items()},
    }


def _canonical_pair_order(inputs: torch.Tensor, targets: torch.Tensor) -> list[int]:
    if inputs.shape != targets.shape or inputs.dim() != 2:
        raise ValueError("alignment pairs changed shape")
    return sorted(range(len(inputs)), key=lambda index: hashlib.sha256(
        inputs[index].detach().cpu().numpy().tobytes()
        + targets[index].detach().cpu().numpy().tobytes()
    ).digest())


def _fit_affine(inputs: torch.Tensor, targets: torch.Tensor, ridge: float) -> tuple[dict, dict]:
    if not 0 < ridge < 1 or not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("alignment fit inputs are invalid")
    order = _canonical_pair_order(inputs, targets)
    x = inputs[order].to(torch.float64)
    y = targets[order].to(torch.float64)
    design = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * ridge
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    state = {
        "weight": weights[:-1].T.to(torch.float32).contiguous(),
        "bias": weights[-1].to(torch.float32).contiguous(),
    }
    predicted = _transform(state, inputs)
    return state, {
        "method": "canonical_affine_ridge",
        "examples": len(inputs),
        "input_dim": inputs.shape[1],
        "target_dim": targets.shape[1],
        "ridge": ridge,
        "input_sha256": hashlib.sha256(inputs[order].numpy().tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(targets[order].numpy().tobytes()).hexdigest(),
        "mse": float(F.mse_loss(predicted, targets)),
    }


def _transform(state: dict, rows: torch.Tensor) -> torch.Tensor:
    if set(state) != {"weight", "bias"} or rows.dim() != 2:
        raise ValueError("alignment state changed shape")
    result = rows @ state["weight"].T + state["bias"]
    if result.shape != rows.shape or not torch.isfinite(result).all():
        raise ValueError("alignment produced invalid states")
    return result


def _same_state(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(torch.equal(first[k], second[k]) for k in first)


def _fit_deterministic(inputs: torch.Tensor, targets: torch.Tensor, spec: dict) -> tuple[dict, dict]:
    state, audit = _fit_affine(inputs, targets, spec["ridge"])
    repeated, _ = _fit_affine(inputs, targets, spec["ridge"])
    reversed_state, _ = _fit_affine(inputs.flip(0), targets.flip(0), spec["ridge"])
    return state, {**audit, "deterministic": _same_state(state, repeated)
                   and _same_state(state, reversed_state)}


def _wrong_targets(targets: torch.Tensor, labels: torch.Tensor, contexts: int) -> torch.Tensor:
    groups = [torch.where(labels == label)[0].tolist() for label in range(contexts)]
    if not groups or len({len(group) for group in groups}) != 1:
        raise ValueError("wrong-pair control requires balanced labels")
    result = torch.empty_like(targets)
    for label, indices in enumerate(groups):
        next_indices = groups[(label + 1) % contexts]
        result[indices] = targets[next_indices]
    return result


def _apply_oracle(states: dict[str, dict], rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(rows)
    for label in range(len(states)):
        selected = labels == label
        output[selected] = _transform(states[str(label)], rows[selected])
    return output


def _alignment_diagnostics(target: torch.Tensor, before: torch.Tensor,
                           after: torch.Tensor) -> dict:
    return {
        "before_similarity": float(F.cosine_similarity(target, before, dim=1).mean()),
        "after_similarity": float(F.cosine_similarity(target, after, dim=1).mean()),
        "before_mse": float(F.mse_loss(before, target)),
        "after_mse": float(F.mse_loss(after, target)),
    }


def _evaluate(source_model, states: dict, pairs: dict, masks: list[tuple[int, ...]]) -> dict:
    query_masked = _mask_rows(pairs["query"], masks)
    storage_masked = _mask_rows(pairs["storage"], masks)
    transformed = {
        "source": (pairs["query"], query_masked),
        "global_affine": (
            _transform(states["global_affine"], pairs["query"]),
            _transform(states["global_affine"], query_masked),
        ),
        "category_oracle": (
            _apply_oracle(states["category_oracle"], pairs["query"], pairs["labels"]),
            _apply_oracle(states["category_oracle"], query_masked, pairs["labels"]),
        ),
        "wrong_pair": (
            _transform(states["wrong_pair"], pairs["query"]),
            _transform(states["wrong_pair"], query_masked),
        ),
    }
    return {
        name: {
            "conditions": {
                "query_full": _classification(source_model, full, pairs["labels"]),
                "query_quarter_missing": _classification(source_model, damaged, pairs["labels"]),
            },
            "alignment": {
                "query_full": _alignment_diagnostics(pairs["storage"], pairs["query"], full),
                "query_quarter_missing": _alignment_diagnostics(
                    storage_masked, query_masked, damaged
                ),
            },
        }
        for name, (full, damaged) in transformed.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/cue_align_results.json")
    parser.add_argument("--verdict", default="measurement/cue_align_verdict.json")
    args = parser.parse_args()
    spec = CUE_ALIGN_SPEC
    source_results, source = _source_receipt(spec)
    source_model = _load_source_context(source_results["source"], CUE_CONTEXT_SPEC)

    calibration_spec = {
        **CONJUNCTION2_SPEC,
        "eval_episodes": spec["calibration_episodes"],
        "data_seed": spec["calibration_data_seed"],
    }
    calibration_episodes = balance_components(build_episodes(calibration_spec), COMPONENT2_SPEC)
    calibration = _collect_pairs(
        calibration_episodes, spec["calibration_engine_seeds"],
        spec["calibration_seed_base"], spec,
    )
    if len(calibration["labels"]) != calibration_pairs(spec):
        raise RuntimeError("registered alignment pair count changed")
    training_masks = [training_mask_indices(index, CUE_CONTEXT_SPEC)
                      for index in range(len(calibration["labels"]))]
    query_masked = _mask_rows(calibration["query"], training_masks)
    storage_masked = _mask_rows(calibration["storage"], training_masks)
    inputs = torch.cat((calibration["query"], query_masked))
    targets = torch.cat((calibration["storage"], storage_masked))

    states, fit_audits = {}, {}
    states["global_affine"], fit_audits["global_affine"] = _fit_deterministic(
        inputs, targets, spec
    )
    wrong = torch.cat((
        _wrong_targets(calibration["storage"], calibration["labels"], spec["contexts"]),
        _wrong_targets(storage_masked, calibration["labels"], spec["contexts"]),
    ))
    states["wrong_pair"], fit_audits["wrong_pair"] = _fit_deterministic(inputs, wrong, spec)
    states["category_oracle"], fit_audits["category_oracle"] = {}, {}
    doubled_labels = torch.cat((calibration["labels"], calibration["labels"]))
    for label in range(spec["contexts"]):
        selected = doubled_labels == label
        state, audit = _fit_deterministic(inputs[selected], targets[selected], spec)
        states["category_oracle"][str(label)] = state
        fit_audits["category_oracle"][str(label)] = audit

    deterministic = all(
        audit["deterministic"] for name, audit in fit_audits.items()
        if name != "category_oracle"
    ) and all(row["deterministic"] for row in fit_audits["category_oracle"].values())
    checkpoint_path = Path(spec["checkpoint_path"])
    _atomic_torch(checkpoint_path, {
        "experiment": spec["experiment"], "spec_sha256": spec_sha256(spec),
        "states": states, "fit_audits": fit_audits, "deterministic": deterministic,
    })
    checkpoint = _receipt(checkpoint_path)

    evaluation_episodes = build_episodes(CONJUNCTION2_SPEC)
    calibration_fingerprints = {row.fingerprint() for row in calibration_episodes}
    evaluation_fingerprints = {row.fingerprint() for row in evaluation_episodes}
    by_engine = {}
    for engine_seed in sorted({row["engine_seed"] for row in spec["evaluation_combinations"]}):
        pairs = _collect_pairs(evaluation_episodes, [engine_seed], spec["episode_seed_base"], spec)
        masks = [cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
                 for index in range(len(evaluation_episodes))]
        by_engine[engine_seed] = {
            "pair_audit": pairs["audit"], "models": _evaluate(source_model, states, pairs, masks),
        }

    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = by_engine[identity["engine_seed"]]
        reference = next(item for item in source_results["evaluations"]
                         if item["name"] == evaluation_name(identity))
        source_conditions = row["models"]["source"]["conditions"]
        evaluations.append({
            "name": evaluation_name(identity), **identity, **row,
            "source_reference_audit": {
                condition: _reference_metric_match(
                    source_conditions[condition],
                    reference["models"]["source"]["conditions"][condition],
                )
                for condition in spec["conditions"]
            },
        })

    evaluation_masks = {
        cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
        for index in range(len(evaluation_episodes))
    }
    payload = {
        "experiment": spec["experiment"], "spec": spec,
        "spec_sha256": spec_sha256(spec), "source": source, "checkpoint": checkpoint,
        "calibration_dataset_audit": dataset_audit(calibration_episodes, calibration_spec),
        "evaluation_dataset_audit": dataset_audit(evaluation_episodes, CONJUNCTION2_SPEC),
        "calibration_evaluation_overlap": len(calibration_fingerprints & evaluation_fingerprints),
        "calibration_pair_audit": calibration["audit"],
        "mask_overlap_audit": {
            "training_unique_masks": len(set(training_masks)),
            "evaluation_unique_masks": len(evaluation_masks),
            "exact_overlap": len(set(training_masks) & evaluation_masks),
        },
        "fit_audits": fit_audits,
        "label_use_audit": {
            "global_affine": False, "category_oracle": True, "wrong_pair_control": True,
        },
        "wrong_pair_audit": {
            "pairs": len(inputs), "mismatched_label_fraction": 1.0,
            "rule": spec["wrong_pair_rule"],
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": spec["device"]},
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.cue_align_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
