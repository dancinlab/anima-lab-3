#!/usr/bin/env python3
"""CUE-MECHANISM-1: locate the first loss caused by partial memory cues."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from address_center2 import _receipt
from address_margin import _center, _classification, _join, _outcome
from completion import _partial_state
from conjunction import _atomic_json, _memory_outcome, build_episodes, dataset_audit, trace_episode
from conjunction2 import _load_value_projector
from context2 import CompositeStateTransform
from context_settle2 import _load_components, _runtime_spec
from graft_behavior import sha256_file
from measurement.completion_gate import adjudicate as adjudicate_completion
from measurement.completion_registry import (
    COMPLETION_SPEC, mask_plan_audit, spec_sha256 as completion_spec_sha256,
)
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256
from measurement.projector_registry import evaluation_name
from separation import _arm_metrics
from trinity import VectorMemory
from value2 import StableValueTransform


def _source_receipt(spec: dict = CUE_MECHANISM_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = completion_spec_sha256(COMPLETION_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != COMPLETION_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_completion(results) != verdict
    ):
        raise RuntimeError("registered COMPLETION-1 source changed")
    source = results["source"]
    receipts = [source["component_checkpoint"], source["value_checkpoint"]]
    receipts.extend(source["prototype_checkpoints"].values())
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered CUE-MECHANISM-1 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "component_checkpoint": dict(source["component_checkpoint"]),
        "value_checkpoint": dict(source["value_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in source["prototype_checkpoints"].items()
        },
    }


def _component_diagnostic(projector, state: torch.Tensor, label: int,
                          reference_address: torch.Tensor) -> dict:
    pooled = state.mean(0).unsqueeze(0)
    address = projector.address(pooled)[0].detach()
    prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    similarities = address @ prototypes.T
    if label < 0 or label >= len(prototypes):
        raise ValueError("component label is out of range")
    wrong = torch.cat((similarities[:label], similarities[label + 1:]))
    return {
        "prediction": int(similarities.argmax()),
        "correct_similarity": float(similarities[label]),
        "closest_wrong_similarity": float(wrong.max()),
        "center_margin": float(similarities[label] - wrong.max()),
        "full_address_similarity": float(F.cosine_similarity(
            address, reference_address, dim=0
        )),
    }


def _retrieval_distance(memory: VectorMemory, query: torch.Tensor,
                        position: int) -> dict:
    similarities = torch.stack([
        F.cosine_similarity(query, address, dim=0) for address in memory.keys
    ])
    wrong = torch.cat((similarities[:position], similarities[position + 1:]))
    return {
        "correct_similarity": float(similarities[position]),
        "closest_wrong_similarity": float(wrong.max()),
        "margin": float(similarities[position] - wrong.max()),
    }


def _integrated_diagnostic(trace: dict, episode, episode_index: int,
                           context_projector, key_projector, value_projector,
                           prototypes, spec: dict, *, context_missing: float = 0.0,
                           key_missing: float = 0.0):
    transform = CompositeStateTransform(
        context_projector, key_projector, spec, center_context=True,
    )
    value_transform = StableValueTransform(value_projector)
    memory = VectorMemory(
        capacity=spec["events_per_episode"], dim=spec["state_dim"],
        key_transform=transform, value_transform=value_transform,
    )
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], trace["values"]
    ):
        memory.store((context_state, key_state), value_state)

    original_context = trace["query_context"]
    original_key = trace["query"]
    query_context = (
        _partial_state(
            original_context, episode_index, "context", context_missing, COMPLETION_SPEC
        )
        if context_missing else original_context
    )
    query_key = (
        _partial_state(original_key, episode_index, "key", key_missing, COMPLETION_SPEC)
        if key_missing else original_key
    )
    full_context_address = context_projector.address(
        original_context.mean(0).unsqueeze(0)
    )[0].detach()
    full_key_address = key_projector.address(
        original_key.mean(0).unsqueeze(0)
    )[0].detach()
    outcome = _memory_outcome(
        memory, (query_context, query_key), trace["values"], prototypes
    )
    query_address = transform.outputs[-1]
    diagnostics = {
        "context": _component_diagnostic(
            context_projector, query_context, episode.query_context, full_context_address
        ),
        "key": _component_diagnostic(
            key_projector, query_key, episode.query_key, full_key_address
        ),
        "retrieval": _retrieval_distance(memory, query_address, episode.query_position),
    }
    audit = {
        "transform_calls": transform.calls,
        "minimum_components": min(transform.component_counts),
        "maximum_components": max(transform.component_counts),
        "minimum_address_width": min(transform.address_widths),
        "maximum_address_width": max(transform.address_widths),
        "value_calls": value_transform.calls,
        "stores": len(memory.keys),
        "retrievals": 1,
    }
    return outcome, diagnostics, audit


def _aggregate_component(rows: list[dict], labels: list[int], classes: int) -> dict:
    classification = _classification(
        labels, [row["prediction"] for row in rows], classes
    )
    return {
        **classification,
        "minimum_class_recall": min(classification["per_class_recall"]),
        "correct_similarity_mean": sum(row["correct_similarity"] for row in rows) / len(rows),
        "closest_wrong_similarity_mean": sum(
            row["closest_wrong_similarity"] for row in rows
        ) / len(rows),
        "center_margin_mean": sum(row["center_margin"] for row in rows) / len(rows),
        "center_margin_minimum": min(row["center_margin"] for row in rows),
        "positive_center_margin_fraction": sum(
            row["center_margin"] > 0 for row in rows
        ) / len(rows),
        "full_address_similarity_mean": sum(
            row["full_address_similarity"] for row in rows
        ) / len(rows),
        "full_address_similarity_minimum": min(
            row["full_address_similarity"] for row in rows
        ),
    }


def _aggregate_distance(rows: list[dict]) -> dict:
    return {
        "correct_similarity_mean": sum(row["correct_similarity"] for row in rows) / len(rows),
        "closest_wrong_similarity_mean": sum(
            row["closest_wrong_similarity"] for row in rows
        ) / len(rows),
        "retrieval_margin_mean": sum(row["margin"] for row in rows) / len(rows),
        "retrieval_margin_minimum": min(row["margin"] for row in rows),
        "positive_margin_fraction": sum(row["margin"] > 0 for row in rows) / len(rows),
    }


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   source_results: dict, spec: dict = CUE_MECHANISM_SPEC, *,
                   context_projector_override=None, key_projector_override=None,
                   fake_context_projector=None, fake_key_projector=None) -> dict:
    context_projector, key_projector = _load_components(source["component_checkpoint"], spec)
    if context_projector_override is not None:
        context_projector = context_projector_override
    if key_projector_override is not None:
        key_projector = key_projector_override
    value_projector = _load_value_projector(source["value_checkpoint"], spec)
    models = {"context": context_projector, "key": key_projector, "value": value_projector}
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    prototypes = F.normalize(value_projector.prototypes.detach(), dim=-1)
    runtime_spec = _runtime_spec(spec["settled_context_steps"], spec)
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    diagnostics = {
        name: {"context": [], "key": [], "retrieval": []}
        for name in spec["conditions"]
    }
    fake_diagnostics = (
        {
            name: {"context": [], "key": []}
            for name in spec["conditions"]
        }
        if fake_context_projector is not None and fake_key_projector is not None
        else None
    )
    if (fake_context_projector is None) != (fake_key_projector is None):
        raise ValueError("fake context and key projectors must be supplied together")
    call_audits = {name: [] for name in spec["conditions"]}
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    context_labels = [episode.query_context for episode in episodes]
    key_labels = [episode.query_key for episode in episodes]
    episode_seeds, cell_counts, sense_audits = [], [], []
    restore_context_equal = True
    restore_key_equal = True
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, runtime_spec)
        cell_counts.extend(trace["cell_counts"])
        sense_audits.append(trace["sense_audit"])
        outcomes = {}
        for name, (context_missing, key_missing) in spec["conditions"].items():
            outcome, diagnostic, audit = _integrated_diagnostic(
                trace, episode, index, context_projector, key_projector,
                value_projector, prototypes, spec,
                context_missing=context_missing, key_missing=key_missing,
            )
            outcomes[name] = outcome
            call_audits[name].append(audit)
            for component in diagnostics[name]:
                diagnostics[name][component].append(diagnostic[component])
            if fake_diagnostics is not None:
                query_context = (
                    _partial_state(
                        trace["query_context"], index, "context", context_missing,
                        COMPLETION_SPEC,
                    )
                    if context_missing else trace["query_context"]
                )
                query_key = (
                    _partial_state(
                        trace["query"], index, "key", key_missing, COMPLETION_SPEC,
                    )
                    if key_missing else trace["query"]
                )
                fake_context_reference = fake_context_projector.address(
                    trace["query_context"].mean(0).unsqueeze(0)
                )[0].detach()
                fake_key_reference = fake_key_projector.address(
                    trace["query"].mean(0).unsqueeze(0)
                )[0].detach()
                fake_diagnostics[name]["context"].append(_component_diagnostic(
                    fake_context_projector, query_context, episode.query_context,
                    fake_context_reference,
                ))
                fake_diagnostics[name]["key"].append(_component_diagnostic(
                    fake_key_projector, query_key, episode.query_key,
                    fake_key_reference,
                ))

        # Removing a mask must restore the exact full-cue path, not merely its average.
        restored_context, _, _ = _integrated_diagnostic(
            trace, episode, index, context_projector, key_projector,
            value_projector, prototypes, spec,
        )
        restored_key, _, _ = _integrated_diagnostic(
            trace, episode, index, context_projector, key_projector,
            value_projector, prototypes, spec,
        )
        restore_context_equal &= restored_context == outcomes["full_cue"]
        restore_key_equal &= restored_key == outcomes["full_cue"]

        oracle_rows = [
            _join(_center(context_projector, context_label),
                  _center(key_projector, key_label), spec)
            for context_label, key_label in zip(episode.contexts, episode.keys)
        ]
        query_oracle = _join(
            _center(context_projector, episode.query_context),
            _center(key_projector, episode.query_key), spec,
        )
        exact, _ = _outcome(
            oracle_rows, query_oracle, trace["values"], prototypes, value_projector
        )
        partner, _ = _outcome(
            oracle_rows, query_oracle,
            trace["values"][1:] + trace["values"][:1], prototypes, value_projector,
        )
        outcomes["exact_context_key_control"] = exact
        outcomes["exact_context_key_partner_swap"] = partner
        content = int(value_projector(
            trace["values"][episode.query_position].mean(0).unsqueeze(0)
        ).argmax(1)[0])
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
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
    component_metrics = {
        condition: {
            "context": _aggregate_component(rows["context"], context_labels, spec["contexts"]),
            "key": _aggregate_component(rows["key"], key_labels, spec["keys"]),
        }
        for condition, rows in diagnostics.items()
    }
    distance_metrics = {
        condition: _aggregate_distance(rows["retrieval"])
        for condition, rows in diagnostics.items()
    }
    source_row = next(
        row for row in source_results["evaluations"]
        if row["name"] == evaluation_name({
            "prototype_seed": prototype_seed, "engine_seed": engine_seed,
        })
    )
    event_queries = len(episodes) * (spec["events_per_episode"] + 1)
    result = {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "component_metrics": component_metrics,
        "distance_metrics": distance_metrics,
        "reference_audit": {
            "full_metric_match": arms["full_cue"] == source_row["arms"]["full_cue"],
            "both_quarter_metric_match": (
                arms["both_quarter_missing"]
                == source_row["arms"]["both_quarter_missing"]
            ),
        },
        "restoration_audit": {
            "context_restored_to_full": restore_context_equal,
            "key_restored_to_full": restore_key_equal,
        },
        "frozen_audit": {
            name: all(
                torch.equal(before[name][key], model.state_dict()[key])
                for key in before[name]
            )
            for name, model in models.items()
        },
        "memory_path_audit": {
            name: {
                "minimum_calls": min(row["transform_calls"] for row in rows),
                "maximum_calls": max(row["transform_calls"] for row in rows),
                "minimum_components": min(row["minimum_components"] for row in rows),
                "maximum_components": max(row["maximum_components"] for row in rows),
                "minimum_address_width": min(row["minimum_address_width"] for row in rows),
                "maximum_address_width": max(row["maximum_address_width"] for row in rows),
                "value_calls": sum(row["value_calls"] for row in rows),
                "stores": sum(row["stores"] for row in rows),
                "retrievals": sum(row["retrievals"] for row in rows),
            }
            for name, rows in call_audits.items()
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
            "context_step_calls": sum(row["context_step_calls"] for row in sense_audits),
            "key_step_calls": sum(row["key_step_calls"] for row in sense_audits),
            "value_step_calls": sum(row["value_step_calls"] for row in sense_audits),
            "distractor_step_calls": sum(row["distractor_step_calls"] for row in sense_audits),
            "expected_context_states": event_queries,
            "expected_key_states": event_queries,
        },
    }
    if fake_diagnostics is not None:
        result["fake_component_metrics"] = {
            condition: {
                "context": _aggregate_component(
                    rows["context"], context_labels, spec["contexts"]
                ),
                "key": _aggregate_component(rows["key"], key_labels, spec["keys"]),
            }
            for condition, rows in fake_diagnostics.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/cue_mechanism_results.json")
    parser.add_argument("--verdict", default="measurement/cue_mechanism_verdict.json")
    args = parser.parse_args()
    spec = CUE_MECHANISM_SPEC
    source_results, source = _source_receipt(spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    evaluations = [
        {"name": evaluation_name(row), **run_evaluation(
            row["prototype_seed"], row["engine_seed"], episodes, source,
            source_results, spec,
        )}
        for row in spec["evaluation_combinations"]
    ]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "mask_plan_audit": mask_plan_audit(COMPLETION_SPEC),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.cue_mechanism_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
