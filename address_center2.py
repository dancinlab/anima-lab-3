#!/usr/bin/env python3
"""ADDRESS-CENTER-2: route a fixed context-category center through shared memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from address_margin import _center, _join, _outcome
from conjunction import _atomic_json, _memory_outcome, build_episodes, dataset_audit, trace_episode
from conjunction2 import _load_value_projector
from context2 import CompositeStateTransform
from context_settle2 import _load_components, _runtime_spec
from graft_behavior import sha256_file
from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC, spec_sha256
from measurement.address_margin_gate import adjudicate as adjudicate_address_margin
from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC, spec_sha256 as margin_spec_sha256
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.projector_registry import evaluation_name
from separation import _arm_metrics
from trinity import VectorMemory
from value2 import StableValueTransform


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = ADDRESS_CENTER2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = margin_spec_sha256(ADDRESS_MARGIN_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != ADDRESS_MARGIN_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_address_margin(results) != verdict
    ):
        raise RuntimeError("registered ADDRESS-MARGIN-1 source changed")
    source = results["source"]
    receipts = [source["component_checkpoint"], source["value_checkpoint"]]
    receipts.extend(source["prototype_checkpoints"].values())
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered ADDRESS-CENTER-2 checkpoint changed")
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


def _integrated_outcome(trace, context_projector, key_projector, value_projector,
                        prototypes, spec, *, center_context: bool,
                        mask_context: bool = False, rotate: bool = False):
    key_transform = CompositeStateTransform(
        context_projector, key_projector, spec,
        center_context=center_context, mask_context=mask_context,
    )
    value_transform = StableValueTransform(value_projector)
    memory = VectorMemory(
        capacity=spec["events_per_episode"], dim=spec["state_dim"],
        key_transform=key_transform, value_transform=value_transform,
    )
    values = trace["values"][1:] + trace["values"][:1] if rotate else trace["values"]
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], values
    ):
        memory.store((context_state, key_state), value_state)
    outcome = _memory_outcome(
        memory, (trace["query_context"], trace["query"]), values, prototypes
    )
    return outcome, {
        "transform_calls": key_transform.calls,
        "minimum_components": min(key_transform.component_counts),
        "maximum_components": max(key_transform.component_counts),
        "minimum_address_width": min(key_transform.address_widths),
        "maximum_address_width": max(key_transform.address_widths),
        "value_calls": value_transform.calls,
        "stores": len(memory.keys),
        "retrievals": 1,
    }


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   source_results: dict, spec: dict = ADDRESS_CENTER2_SPEC) -> dict:
    context_projector, key_projector = _load_components(source["component_checkpoint"], spec)
    value_projector = _load_value_projector(source["value_checkpoint"], spec)
    before = {
        "context": {name: value.clone() for name, value in context_projector.state_dict().items()},
        "key": {name: value.clone() for name, value in key_projector.state_dict().items()},
        "value": {name: value.clone() for name, value in value_projector.state_dict().items()},
    }
    prototypes = F.normalize(value_projector.prototypes.detach(), dim=-1)
    runtime_spec = _runtime_spec(spec["settled_context_steps"], spec)
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    center_matches, disabled_matches = [], []
    call_audits = {name: [] for name in spec["arms"][:4]}
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    episode_seeds, cell_counts, sense_audits = [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, runtime_spec)
        cell_counts.extend(trace["cell_counts"])
        sense_audits.append(trace["sense_audit"])

        centered, centered_audit = _integrated_outcome(
            trace, context_projector, key_projector, value_projector, prototypes, spec,
            center_context=True,
        )
        disabled, disabled_audit = _integrated_outcome(
            trace, context_projector, key_projector, value_projector, prototypes, spec,
            center_context=False,
        )
        masked, masked_audit = _integrated_outcome(
            trace, context_projector, key_projector, value_projector, prototypes, spec,
            center_context=True, mask_context=True,
        )
        recovered, recovered_audit = _integrated_outcome(
            trace, context_projector, key_projector, value_projector, prototypes, spec,
            center_context=True,
        )
        context_rows = []
        continuous_rows = []
        oracle_rows = []
        for context_state, key_state, context_label, key_label in zip(
            trace["contexts"], trace["keys"], episode.contexts, episode.keys
        ):
            context_label_pred = int(context_projector(context_state.mean(0).unsqueeze(0)).argmax(1)[0])
            context_rows.append(_join(
                _center(context_projector, context_label_pred),
                key_projector.address(key_state.mean(0).unsqueeze(0))[0].detach(), spec,
            ))
            continuous_rows.append(_join(
                context_projector.address(context_state.mean(0).unsqueeze(0))[0].detach(),
                key_projector.address(key_state.mean(0).unsqueeze(0))[0].detach(), spec,
            ))
            oracle_rows.append(_join(
                _center(context_projector, context_label), _center(key_projector, key_label), spec
            ))
        query_context_label = int(
            context_projector(trace["query_context"].mean(0).unsqueeze(0)).argmax(1)[0]
        )
        query_centered = _join(
            _center(context_projector, query_context_label),
            key_projector.address(trace["query"].mean(0).unsqueeze(0))[0].detach(), spec,
        )
        query_continuous = _join(
            context_projector.address(trace["query_context"].mean(0).unsqueeze(0))[0].detach(),
            key_projector.address(trace["query"].mean(0).unsqueeze(0))[0].detach(), spec,
        )
        query_oracle = _join(
            _center(context_projector, episode.query_context),
            _center(key_projector, episode.query_key), spec,
        )
        center_reference, _ = _outcome(
            context_rows, query_centered, trace["values"], prototypes, value_projector
        )
        disabled_reference, _ = _outcome(
            continuous_rows, query_continuous, trace["values"], prototypes, value_projector
        )
        exact, _ = _outcome(
            oracle_rows, query_oracle, trace["values"], prototypes, value_projector
        )
        partner, _ = _outcome(
            oracle_rows, query_oracle,
            trace["values"][1:] + trace["values"][:1], prototypes, value_projector,
        )
        outcomes = {
            "integrated_context_center": centered,
            "integrated_center_disabled": disabled,
            "integrated_context_masked": masked,
            "integrated_context_center_recovered": recovered,
            "exact_context_key_control": exact,
            "exact_context_key_partner_swap": partner,
        }
        for name, audit in (
            ("integrated_context_center", centered_audit),
            ("integrated_center_disabled", disabled_audit),
            ("integrated_context_masked", masked_audit),
            ("integrated_context_center_recovered", recovered_audit),
        ):
            call_audits[name].append(audit)
        center_matches.append(centered[:2] == center_reference[:2])
        disabled_matches.append(disabled[:2] == disabled_reference[:2])
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
    source_row = next(
        row for row in source_results["evaluations"]
        if row["name"] == evaluation_name({
            "prototype_seed": prototype_seed, "engine_seed": engine_seed,
        })
    )
    event_queries = len(episodes) * (spec["events_per_episode"] + 1)
    frozen = {
        name: all(torch.equal(values[key], model.state_dict()[key]) for key in values)
        for name, values, model in (
            ("context", before["context"], context_projector),
            ("key", before["key"], key_projector),
            ("value", before["value"], value_projector),
        )
    }
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "reference_audit": {
            "center_prediction_selection_match": sum(center_matches) / len(center_matches),
            "disabled_prediction_selection_match": sum(disabled_matches) / len(disabled_matches),
            "center_metric_match": arms["integrated_context_center"] == source_row["arms"]["context_centered"],
            "disabled_metric_match": arms["integrated_center_disabled"] == source_row["arms"]["continuous_frozen"],
            "recovery_metric_match": arms["integrated_context_center_recovered"] == arms["integrated_context_center"],
        },
        "frozen_audit": frozen,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/address_center2_results.json")
    parser.add_argument("--verdict", default="measurement/address_center2_verdict.json")
    args = parser.parse_args()
    spec = ADDRESS_CENTER2_SPEC
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
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.address_center2_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
