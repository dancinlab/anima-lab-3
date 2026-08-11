#!/usr/bin/env python3
"""ADDRESS-MARGIN-1: separate category errors from within-category address drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from conjunction import _atomic_json, _memory_outcome, build_episodes, dataset_audit, trace_episode
from conjunction2 import _load_value_projector
from context import _component_address
from context_settle2 import _load_components, _runtime_spec
from episode import _decode
from graft_behavior import sha256_file
from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC, spec_sha256
from measurement.context_settle2_gate import adjudicate as adjudicate_context_settle2
from measurement.context_settle2_registry import (
    CONTEXT_SETTLE2_SPEC,
    spec_sha256 as context_settle2_spec_sha256,
)
from measurement.projector_registry import evaluation_name
from separation import _arm_metrics
from trinity import VectorMemory
from value2 import StableValueTransform


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = ADDRESS_MARGIN_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = context_settle2_spec_sha256(CONTEXT_SETTLE2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CONTEXT_SETTLE2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_context_settle2(results) != verdict
    ):
        raise RuntimeError("registered CONTEXT-SETTLE-2 source changed")
    receipts = [results["source"]["component_checkpoint"]]
    conjunction_source = results["source"]["conjunction_source"]
    receipts.extend((
        conjunction_source["value_checkpoint"],
        *conjunction_source["prototype_checkpoints"].values(),
    ))
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered ADDRESS-MARGIN-1 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "component_checkpoint": dict(results["source"]["component_checkpoint"]),
        "value_checkpoint": dict(conjunction_source["value_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value)
            for key, value in conjunction_source["prototype_checkpoints"].items()
        },
    }


def _center(projector, label: int) -> torch.Tensor:
    prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    if label < 0 or label >= len(prototypes):
        raise ValueError("address center label is out of range")
    return prototypes[label].clone()


def _component(projector, state: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor, float]:
    address = _component_address(projector, state)
    label = int(projector(state.mean(0).unsqueeze(0)).argmax(1)[0])
    center = _center(projector, label)
    deviation = float(1.0 - F.cosine_similarity(address, center, dim=0))
    return address, label, center, deviation


def _join(context: torch.Tensor, key: torch.Tensor, spec: dict) -> torch.Tensor:
    address = torch.cat((context, key)) * spec["component_weight"]
    if address.dim() != 1 or address.numel() != spec["composite_address_dim"]:
        raise ValueError("composite address width changed")
    if not torch.isfinite(address).all():
        raise ValueError("composite address contains a non-finite value")
    return address


def _outcome(addresses, query, values, prototypes, value_projector):
    transform = StableValueTransform(value_projector)
    memory = VectorMemory(
        capacity=len(addresses), dim=len(addresses[0]), value_transform=transform
    )
    for address, value in zip(addresses, values):
        memory.store(address, value)
    outcome = _memory_outcome(memory, query, values, prototypes)
    return outcome, {"value_calls": transform.calls, "stores": len(memory.keys), "retrievals": 1}


def _classification(labels: list[int], predictions: list[int], classes: int) -> dict:
    expected = torch.tensor(labels, dtype=torch.long)
    predicted = torch.tensor(predictions, dtype=torch.long)
    return {
        "accuracy": float((expected == predicted).float().mean()),
        "per_class_recall": [
            float((predicted[expected == label] == label).float().mean())
            for label in range(classes)
        ],
        "confusion_matrix": torch.bincount(
            expected * classes + predicted, minlength=classes * classes
        ).reshape(classes, classes).tolist(),
    }


def _margin(addresses, query, position: int) -> tuple[float, float, float]:
    similarities = torch.stack([
        F.cosine_similarity(query, address, dim=0) for address in addresses
    ])
    correct = float(similarities[position])
    wrong = torch.cat((similarities[:position], similarities[position + 1:]))
    closest_wrong = float(wrong.max())
    return correct, closest_wrong, correct - closest_wrong


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   spec: dict = ADDRESS_MARGIN_SPEC) -> dict:
    context_projector, key_projector = _load_components(source["component_checkpoint"], spec)
    value_projector = _load_value_projector(source["value_checkpoint"], spec)
    before = {
        "context": {name: value.clone() for name, value in context_projector.state_dict().items()},
        "key": {name: value.clone() for name, value in key_projector.state_dict().items()},
        "value": {name: value.clone() for name, value in value_projector.state_dict().items()},
    }
    stable_prototypes = F.normalize(value_projector.prototypes.detach(), dim=-1)
    runtime_spec = _runtime_spec(spec["settled_context_steps"], spec)
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    distance_rows = {name: [] for name in ("continuous_frozen", "predicted_centers")}
    context_labels: list[int] = []
    context_predictions: list[int] = []
    key_labels: list[int] = []
    key_predictions: list[int] = []
    context_deviations: list[float] = []
    key_deviations: list[float] = []
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    episode_seeds, cell_counts, sense_audits, path_audits = [], [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, runtime_spec)
        cell_counts.extend(trace["cell_counts"])
        sense_audits.append(trace["sense_audit"])

        stored_components = []
        for context_state, key_state, context_label, key_label in zip(
            trace["contexts"], trace["keys"], episode.contexts, episode.keys
        ):
            context_address, context_prediction, context_center, context_deviation = _component(
                context_projector, context_state
            )
            key_address, key_prediction, key_center, key_deviation = _component(
                key_projector, key_state
            )
            stored_components.append((
                context_address, key_address, context_center, key_center,
                context_label, key_label,
            ))
            context_labels.append(context_label); context_predictions.append(context_prediction)
            key_labels.append(key_label); key_predictions.append(key_prediction)
            context_deviations.append(context_deviation); key_deviations.append(key_deviation)
        query_context, query_context_prediction, query_context_center, query_context_deviation = _component(
            context_projector, trace["query_context"]
        )
        query_key, query_key_prediction, query_key_center, query_key_deviation = _component(
            key_projector, trace["query"]
        )
        context_labels.append(episode.query_context); context_predictions.append(query_context_prediction)
        key_labels.append(episode.query_key); key_predictions.append(query_key_prediction)
        context_deviations.append(query_context_deviation); key_deviations.append(query_key_deviation)

        continuous = [_join(row[0], row[1], spec) for row in stored_components]
        context_centered = [_join(row[2], row[1], spec) for row in stored_components]
        key_centered = [_join(row[0], row[3], spec) for row in stored_components]
        predicted_centers = [_join(row[2], row[3], spec) for row in stored_components]
        oracle_centers = [
            _join(_center(context_projector, row[4]), _center(key_projector, row[5]), spec)
            for row in stored_components
        ]
        query_position = episode.query_position
        shifted_position = (query_position + 1) % spec["events_per_episode"]
        queries = {
            "continuous_frozen": _join(query_context, query_key, spec),
            "context_centered": _join(query_context_center, query_key, spec),
            "key_centered": _join(query_context, query_key_center, spec),
            "predicted_centers": _join(query_context_center, query_key_center, spec),
            "oracle_centers": _join(
                _center(context_projector, episode.query_context),
                _center(key_projector, episode.query_key), spec,
            ),
            "shifted_center_control": oracle_centers[shifted_position],
        }
        addresses = {
            "continuous_frozen": continuous,
            "context_centered": context_centered,
            "key_centered": key_centered,
            "predicted_centers": predicted_centers,
            "oracle_centers": oracle_centers,
            "shifted_center_control": oracle_centers,
        }
        stable_content = int(value_projector(
            trace["values"][query_position].mean(0).unsqueeze(0)
        ).argmax(1)[0])
        for name in spec["arms"]:
            outcome, audit = _outcome(
                addresses[name], queries[name], trace["values"], stable_prototypes,
                value_projector,
            )
            record = records[name]
            record["predictions"].append(outcome[0]); record["selections"].append(outcome[1])
            record["contents"].append(stable_content); record["api"].append(outcome[2])
            record["margins"].append(outcome[3]); path_audits.append(audit)
        for name in distance_rows:
            distance_rows[name].append(_margin(addresses[name], queries[name], query_position))
        if (index + 1) % 128 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed}] "
                f"evaluated {index + 1}/{len(episodes)} episodes", flush=True,
            )

    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    margin_metrics = {}
    for name, rows in distance_rows.items():
        correct, wrong, margins = zip(*rows)
        margin_metrics[name] = {
            "correct_similarity_mean": sum(correct) / len(correct),
            "closest_wrong_similarity_mean": sum(wrong) / len(wrong),
            "retrieval_margin_mean": sum(margins) / len(margins),
            "retrieval_margin_minimum": min(margins),
            "positive_margin_fraction": sum(value > 0 for value in margins) / len(margins),
        }
    frozen = {
        name: all(torch.equal(values[key], model.state_dict()[key]) for key in values)
        for name, values, model in (
            ("context", before["context"], context_projector),
            ("key", before["key"], key_projector),
            ("value", before["value"], value_projector),
        )
    }
    event_queries = len(episodes) * (spec["events_per_episode"] + 1)
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "category_metrics": {
            "context": _classification(context_labels, context_predictions, spec["contexts"]),
            "key": _classification(key_labels, key_predictions, spec["keys"]),
        },
        "distance_metrics": margin_metrics,
        "deviation_metrics": {
            "context_mean": sum(context_deviations) / len(context_deviations),
            "context_maximum": max(context_deviations),
            "key_mean": sum(key_deviations) / len(key_deviations),
            "key_maximum": max(key_deviations),
        },
        "frozen_audit": frozen,
        "path_audit": {
            "arms_per_episode": len(spec["arms"]),
            "value_calls": sum(row["value_calls"] for row in path_audits),
            "stores": sum(row["stores"] for row in path_audits),
            "retrievals": sum(row["retrievals"] for row in path_audits),
            "address_width": spec["composite_address_dim"],
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256("\n".join(map(str, episode_seeds)).encode()).hexdigest(),
            "minimum_cells": min(cell_counts), "maximum_cells": max(cell_counts),
            "context_step_calls": sum(row["context_step_calls"] for row in sense_audits),
            "key_step_calls": sum(row["key_step_calls"] for row in sense_audits),
            "value_step_calls": sum(row["value_step_calls"] for row in sense_audits),
            "distractor_step_calls": sum(row["distractor_step_calls"] for row in sense_audits),
            "expected_context_states": event_queries,
            "expected_key_states": event_queries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/address_margin_results.json")
    parser.add_argument("--verdict", default="measurement/address_margin_verdict.json")
    args = parser.parse_args()
    spec = ADDRESS_MARGIN_SPEC
    source_results, source = _source_receipt(spec)
    episodes = build_episodes(CONTEXT_SETTLE2_SPEC)
    evaluations = [
        {"name": evaluation_name(row), **run_evaluation(
            row["prototype_seed"], row["engine_seed"], episodes, source, spec
        )}
        for row in spec["evaluation_combinations"]
    ]
    payload = {
        "experiment": spec["experiment"], "spec": spec,
        "spec_sha256": spec_sha256(spec), "source": source,
        "dataset_audit": dataset_audit(episodes, CONTEXT_SETTLE2_SPEC),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.address_margin_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
