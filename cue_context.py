#!/usr/bin/env python3
"""CUE-CONTEXT-1: separate storage-time and query-time context states."""
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
from conjunction import _atomic_json, build_episodes, dataset_audit, trace_episode
from context_settle2 import _runtime_spec
from cue_robust import _state_dict_equal
from measurement.component2_registry import COMPONENT2_SPEC
from measurement.completion_registry import COMPLETION_SPEC, cue_mask_indices
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_context_registry import (
    CUE_CONTEXT_SPEC,
    calibration_pairs,
    spec_sha256,
    training_mask_indices,
    training_mask_plan_audit,
)
from measurement.cue_robust_gate import adjudicate as adjudicate_robust
from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
from measurement.projector_registry import evaluation_name
from key_stability import fit_stable_key_projector
from context_settle2 import _load_components
from value2 import _atomic_torch, _canonical_rows


def _source_receipt(spec: dict = CUE_CONTEXT_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    source_sha = robust_spec_sha256(CUE_ROBUST_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CUE_ROBUST_SPEC
        or results.get("spec_sha256") != source_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_robust(results) != verdict
    ):
        raise RuntimeError("registered CUE-ROBUST-1 source changed")
    receipts = [results["checkpoint"], results["source"]["component_checkpoint"],
                results["source"]["value_checkpoint"],
                *results["source"]["prototype_checkpoints"].values()]
    if any(
        not Path(row["path"]).is_file()
        or hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]
        for row in receipts
    ):
        raise RuntimeError("registered CUE-CONTEXT-1 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": source_sha,
        "robust_checkpoint": dict(results["checkpoint"]),
        "component_checkpoint": dict(results["source"]["component_checkpoint"]),
        "value_checkpoint": dict(results["source"]["value_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in results["source"]["prototype_checkpoints"].items()
        },
    }


def _load_source_context(source: dict, spec: dict = CUE_CONTEXT_SPEC):
    context, _ = _load_components(source["component_checkpoint"], spec)
    checkpoint = torch.load(
        source["robust_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    context.load_state_dict(checkpoint["context_projector"])
    context.eval()
    return context


def _collect_pairs(episodes, engine_seeds: list[int], seed_base: int,
                   spec: dict = CUE_CONTEXT_SPEC) -> dict:
    runtime_spec = _runtime_spec(spec["settled_context_steps"], spec)
    storage, query, labels, positions, seeds, sense_audits = [], [], [], [], [], []
    cell_counts = []
    for engine_seed in engine_seeds:
        base = seed_base + engine_seed * spec["seed_stride"]
        for index, episode in enumerate(episodes):
            trial_seed = base + index
            trace = trace_episode(episode, trial_seed, runtime_spec)
            storage_state = trace["contexts"][episode.query_position].mean(0)
            query_state = trace["query_context"].mean(0)
            if storage_state.shape != query_state.shape or storage_state.numel() != spec["state_dim"]:
                raise RuntimeError("paired context states changed shape")
            storage.append(storage_state)
            query.append(query_state)
            labels.append(episode.query_context)
            positions.append(episode.query_position)
            seeds.append(trial_seed)
            sense_audits.append(trace["sense_audit"])
            cell_counts.extend(trace["cell_counts"])
    storage_tensor = torch.stack(storage)
    query_tensor = torch.stack(query)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return {
        "storage": storage_tensor,
        "query": query_tensor,
        "labels": label_tensor,
        "positions": positions,
        "audit": {
            "pairs": len(labels),
            "same_label_pairs": len(labels),
            "unique_trial_seeds": len(set(seeds)),
            "trial_seed_sha256": hashlib.sha256("\n".join(map(str, seeds)).encode()).hexdigest(),
            "storage_sha256": hashlib.sha256(storage_tensor.numpy().tobytes()).hexdigest(),
            "query_sha256": hashlib.sha256(query_tensor.numpy().tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(label_tensor.numpy().tobytes()).hexdigest(),
            "label_counts": {
                str(label): int((label_tensor == label).sum())
                for label in range(spec["contexts"])
            },
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
            "context_step_calls": sum(row["context_step_calls"] for row in sense_audits),
            "key_step_calls": sum(row["key_step_calls"] for row in sense_audits),
            "value_step_calls": sum(row["value_step_calls"] for row in sense_audits),
            "distractor_step_calls": sum(row["distractor_step_calls"] for row in sense_audits),
        },
    }


def _mask_rows(states: torch.Tensor, masks: list[tuple[int, ...]]) -> torch.Tensor:
    if states.dim() != 2 or len(states) != len(masks) or not torch.isfinite(states).all():
        raise ValueError("context states or mask roster changed")
    masked = states.detach().clone()
    for index, dimensions in enumerate(masks):
        masked[index, list(dimensions)] = 0.0
    return masked


def _fit(states: torch.Tensor, labels: torch.Tensor, spec: dict = CUE_CONTEXT_SPEC):
    fit_spec = dict(COMPONENT2_SPEC)
    ordered_states, ordered_labels = _canonical_rows(states, labels)
    model, audit = fit_stable_key_projector(
        ordered_states, ordered_labels, fit_spec, method=spec["fit_method"]
    )
    repeated, _ = fit_stable_key_projector(
        ordered_states, ordered_labels, fit_spec, method=spec["fit_method"]
    )
    reversed_states, reversed_labels = _canonical_rows(
        ordered_states.flip(0), ordered_labels.flip(0)
    )
    reordered, _ = fit_stable_key_projector(
        reversed_states, reversed_labels, fit_spec, method=spec["fit_method"]
    )
    return model, {
        **audit,
        "deterministic": _state_dict_equal(model.state_dict(), repeated.state_dict())
        and _state_dict_equal(model.state_dict(), reordered.state_dict()),
        "state_sha256": hashlib.sha256(ordered_states.numpy().tobytes()).hexdigest(),
        "label_sha256": hashlib.sha256(ordered_labels.numpy().tobytes()).hexdigest(),
        "label_counts": {
            str(label): int((ordered_labels == label).sum())
            for label in range(spec["contexts"])
        },
    }


def _classification(projector, states: torch.Tensor, labels: torch.Tensor) -> dict:
    addresses = projector.address(states).detach()
    prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    similarities = addresses @ prototypes.T
    predictions = similarities.argmax(1)
    correct = similarities.gather(1, labels[:, None])[:, 0]
    wrong = similarities.clone()
    wrong[torch.arange(len(labels)), labels] = -torch.inf
    recalls = [
        float((predictions[labels == label] == label).float().mean())
        for label in range(len(prototypes))
    ]
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "per_class_recall": recalls,
        "minimum_class_recall": min(recalls),
        "correct_similarity_mean": float(correct.mean()),
        "closest_wrong_similarity_mean": float(wrong.max(1).values.mean()),
        "center_margin_mean": float((correct - wrong.max(1).values).mean()),
        "center_margin_minimum": float((correct - wrong.max(1).values).min()),
        "positive_center_margin_fraction": float((correct > wrong.max(1).values).float().mean()),
    }


def _pair_similarity(projector, storage: torch.Tensor, query: torch.Tensor) -> dict:
    state_similarity = F.cosine_similarity(storage, query, dim=1)
    storage_address = projector.address(storage).detach()
    query_address = projector.address(query).detach()
    address_similarity = F.cosine_similarity(storage_address, query_address, dim=1)
    return {
        "state_mean": float(state_similarity.mean()),
        "state_minimum": float(state_similarity.min()),
        "address_mean": float(address_similarity.mean()),
        "address_minimum": float(address_similarity.min()),
    }


def _evaluate_models(models: dict, pairs: dict, masks: list[tuple[int, ...]]) -> dict:
    storage_masked = _mask_rows(pairs["storage"], masks)
    query_masked = _mask_rows(pairs["query"], masks)
    conditions = {
        "storage_full": pairs["storage"],
        "storage_quarter_missing": storage_masked,
        "query_full": pairs["query"],
        "query_quarter_missing": query_masked,
    }
    return {
        name: {
            "conditions": {
                condition: _classification(model, states, pairs["labels"])
                for condition, states in conditions.items()
            },
            "paired_similarity": _pair_similarity(model, pairs["storage"], pairs["query"]),
        }
        for name, model in models.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/cue_context_results.json")
    parser.add_argument("--verdict", default="measurement/cue_context_verdict.json")
    args = parser.parse_args()
    spec = CUE_CONTEXT_SPEC
    source_results, source = _source_receipt(spec)
    source_context = _load_source_context(source, spec)

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
        raise RuntimeError("registered calibration pair count changed")
    training_masks = [training_mask_indices(index, spec) for index in range(len(calibration["labels"]))]
    storage_masked = _mask_rows(calibration["storage"], training_masks)
    query_masked = _mask_rows(calibration["query"], training_masks)
    combined = torch.stack([
        calibration["storage"][index] if index % 2 == 0 else calibration["query"][index]
        for index in range(len(calibration["labels"]))
    ])
    combined_masked = _mask_rows(combined, training_masks)
    fit_inputs = {
        "storage_only": torch.cat((calibration["storage"], storage_masked)),
        "query_only": torch.cat((calibration["query"], query_masked)),
        "combined": torch.cat((combined, combined_masked)),
    }
    fit_labels = torch.cat((calibration["labels"], calibration["labels"]))
    fitted, fit_audits = {}, {}
    for name, states in fit_inputs.items():
        fitted[name], fit_audits[name] = _fit(states, fit_labels, spec)
    fake_labels = (fit_labels + spec["fake_label_offset"]) % spec["contexts"]
    fitted["fake_query"], fit_audits["fake_query"] = _fit(
        fit_inputs["query_only"], fake_labels, spec
    )
    models = {"source": source_context, **fitted}

    checkpoint_path = Path(spec["checkpoint_path"])
    _atomic_torch(checkpoint_path, {
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "models": {name: model.state_dict() for name, model in fitted.items()},
        "fit_audits": fit_audits,
        "deterministic": all(row["deterministic"] for row in fit_audits.values()),
    })
    checkpoint = _receipt(checkpoint_path)

    evaluation_episodes = build_episodes(CONJUNCTION2_SPEC)
    evaluation_fingerprints = {row.fingerprint() for row in evaluation_episodes}
    calibration_fingerprints = {row.fingerprint() for row in calibration_episodes}
    by_engine = {}
    for engine_seed in sorted({row["engine_seed"] for row in spec["evaluation_combinations"]}):
        pairs = _collect_pairs(
            evaluation_episodes, [engine_seed], spec["episode_seed_base"], spec
        )
        masks = [
            cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
            for index in range(len(evaluation_episodes))
        ]
        by_engine[engine_seed] = {
            "pair_audit": pairs["audit"],
            "models": _evaluate_models(models, pairs, masks),
        }

    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = by_engine[identity["engine_seed"]]
        source_row = next(
            candidate for candidate in source_results["evaluations"]
            if candidate["name"] == evaluation_name(identity)
        )
        query_source = row["models"]["source"]["conditions"]
        evaluations.append({
            "name": evaluation_name(identity),
            "prototype_seed": identity["prototype_seed"],
            "engine_seed": identity["engine_seed"],
            **row,
            "source_reference_audit": {
                "query_full_match": query_source["query_full"]
                == source_row["component_metrics"]["full_cue"]["context"],
                "query_quarter_match": query_source["query_quarter_missing"]
                == source_row["component_metrics"]["context_quarter_missing"]["context"],
            },
        })

    evaluation_masks = {
        cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
        for index in range(len(evaluation_episodes))
    }
    training_masks_set = set(training_masks)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "checkpoint": checkpoint,
        "calibration_dataset_audit": dataset_audit(calibration_episodes, calibration_spec),
        "evaluation_dataset_audit": dataset_audit(evaluation_episodes, CONJUNCTION2_SPEC),
        "calibration_evaluation_overlap": len(calibration_fingerprints & evaluation_fingerprints),
        "calibration_pair_audit": calibration["audit"],
        "training_mask_plan_audit": training_mask_plan_audit(spec),
        "mask_overlap_audit": {
            "training_unique_masks": len(training_masks_set),
            "evaluation_unique_masks": len(evaluation_masks),
            "exact_overlap": len(training_masks_set & evaluation_masks),
        },
        "combined_audit": {
            "storage_rows": (len(combined) + 1) // 2,
            "query_rows": len(combined) // 2,
            "total_rows": len(combined),
        },
        "fit_audits": fit_audits,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.cue_context_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
