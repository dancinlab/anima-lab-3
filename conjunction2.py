#!/usr/bin/env python3
"""CONJUNCTION-2: rerun pair retrieval with the frozen stable value path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from conjunction import (
    _atomic_json, _exact_addresses, _memory_outcome, _receipt,
    _source_receipt as conjunction_source_receipt, _wrong_value_states,
    build_episodes, dataset_audit, trace_episode,
)
from context import _composite, _load_key_projector
from context2 import CompositeStateTransform, _load_context_projector
from episode import _decode
from graft_behavior import sha256_file
from key_stability import StableKeyProjector
from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256
from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.value2_gate import adjudicate as adjudicate_value2
from measurement.value2_registry import VALUE2_SPEC, spec_sha256 as value2_spec_sha256
from separation import _arm_metrics
from trinity import VectorMemory
from value2 import StableValueTransform


def _source_receipt(spec: dict = CONJUNCTION2_SPEC) -> dict:
    value_results_path = Path(spec["source_value_results"])
    value_verdict_path = Path(spec["source_value_verdict_path"])
    value_results = json.loads(value_results_path.read_text())
    value_verdict = json.loads(value_verdict_path.read_text())
    value_sha = value2_spec_sha256(VALUE2_SPEC)
    if (
        value_results.get("experiment") != spec["source_value_experiment"]
        or value_results.get("spec") != VALUE2_SPEC
        or value_results.get("spec_sha256") != value_sha
        or value_verdict.get("verdict") != spec["source_value_verdict"]
        or value_verdict.get("spec_sha256") != value_sha
        or adjudicate_value2(value_results) != value_verdict
    ):
        raise RuntimeError("registered VALUE-2 source changed")
    value_checkpoint = value_results["checkpoint"]
    if not Path(value_checkpoint["path"]).is_file() or sha256_file(Path(value_checkpoint["path"])) != value_checkpoint["sha256"]:
        raise RuntimeError("registered VALUE-2 checkpoint changed")

    conjunction_results_path = Path(spec["source_conjunction_results"])
    conjunction_verdict_path = Path(spec["source_conjunction_verdict_path"])
    conjunction_results = json.loads(conjunction_results_path.read_text())
    conjunction_verdict = json.loads(conjunction_verdict_path.read_text())
    conjunction_sha = conjunction_spec_sha256(CONJUNCTION_SPEC)
    if (
        conjunction_results.get("experiment") != spec["source_conjunction_experiment"]
        or conjunction_results.get("spec") != CONJUNCTION_SPEC
        or conjunction_results.get("spec_sha256") != conjunction_sha
        or conjunction_verdict.get("verdict") != spec["source_conjunction_verdict"]
        or adjudicate_conjunction(conjunction_results) != conjunction_verdict
    ):
        raise RuntimeError("registered CONJUNCTION-1 source changed")
    inherited = conjunction_results["source_context2"]
    for receipt in (
        inherited["context_checkpoint"], inherited["canonical_checkpoint"],
        *inherited["prototype_checkpoints"].values(),
    ):
        if not Path(receipt["path"]).is_file() or sha256_file(Path(receipt["path"])) != receipt["sha256"]:
            raise RuntimeError("registered CONJUNCTION-1 checkpoint changed")
    return {
        "value_results": _receipt(value_results_path),
        "value_verdict": _receipt(value_verdict_path),
        "value_spec_sha256": value_sha,
        "value_checkpoint": dict(value_checkpoint),
        "conjunction_results": _receipt(conjunction_results_path),
        "conjunction_verdict": _receipt(conjunction_verdict_path),
        "conjunction_spec_sha256": conjunction_sha,
        "context_checkpoint": dict(inherited["context_checkpoint"]),
        "canonical_checkpoint": dict(inherited["canonical_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in inherited["prototype_checkpoints"].items()
        },
    }


def _load_value_projector(receipt: dict, spec: dict = CONJUNCTION2_SPEC):
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    model = StableKeyProjector(
        spec["state_dim"], spec["value_address_dim"], spec["values"],
        VALUE2_SPEC["temperature"], VALUE2_SPEC["bias"],
    )
    model.load_state_dict(checkpoint["projector"])
    model.eval().requires_grad_(False)
    return model


def _integrated(trace, prototypes, context_projector, key_projector, value_projector,
                spec, *, mask_context=False, mask_key=False, raw_value=False,
                stored_values=None):
    key_transform = CompositeStateTransform(
        context_projector, key_projector, spec,
        mask_context=mask_context, mask_key=mask_key,
    )
    value_transform = None if raw_value else StableValueTransform(value_projector)
    memory = VectorMemory(
        capacity=spec["events_per_episode"], dim=spec["state_dim"],
        key_transform=key_transform, value_transform=value_transform,
    )
    payloads = trace["values"] if stored_values is None else stored_values
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], payloads
    ):
        memory.store((context_state, key_state), value_state)
    outcome = _memory_outcome(
        memory, (trace["query_context"], trace["query"]), payloads, prototypes
    )
    return outcome, {
        "key_calls": key_transform.calls,
        "key_minimum_components": min(key_transform.component_counts),
        "key_maximum_components": max(key_transform.component_counts),
        "key_minimum_width": min(key_transform.address_widths),
        "key_maximum_width": max(key_transform.address_widths),
        "value_calls": 0 if value_transform is None else value_transform.calls,
        "value_minimum_width": 0 if value_transform is None else min(value_transform.output_widths),
        "value_maximum_width": 0 if value_transform is None else max(value_transform.output_widths),
        "stores": len(memory.keys), "retrievals": 1,
    }


def _external(trace, prototypes, context_projector, key_projector, value_projector, spec):
    memory = VectorMemory(capacity=spec["events_per_episode"], dim=spec["state_dim"])
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], trace["values"]
    ):
        memory.store(
            _composite(context_projector, key_projector, context_state, key_state, spec),
            value_projector.address(value_state.mean(0).unsqueeze(0))[0],
        )
    query = _composite(
        context_projector, key_projector, trace["query_context"], trace["query"], spec
    )
    return _memory_outcome(memory, query, memory.values, prototypes)


def _exact_stable(addresses, query, values, prototypes, value_projector, *, stored_values=None):
    transform = StableValueTransform(value_projector)
    memory = VectorMemory(
        capacity=len(addresses), dim=addresses[0].numel(), value_transform=transform
    )
    payloads = values if stored_values is None else stored_values
    for address, value in zip(addresses, payloads):
        memory.store(address, value)
    return _memory_outcome(memory, query, payloads, prototypes)


def run_evaluation(prototype_seed, engine_seed, episodes, source, spec=CONJUNCTION2_SPEC):
    context_projector = _load_context_projector(source["context_checkpoint"], spec)
    key_projector = _load_key_projector(source["canonical_checkpoint"], spec)
    value_projector = _load_value_projector(source["value_checkpoint"], spec)
    before = {
        "context": {name: value.clone() for name, value in context_projector.state_dict().items()},
        "key": {name: value.clone() for name, value in key_projector.state_dict().items()},
        "value": {name: value.clone() for name, value in value_projector.state_dict().items()},
    }
    raw_checkpoint = torch.load(
        source["prototype_checkpoints"][str(prototype_seed)]["path"],
        map_location="cpu", weights_only=True,
    )
    raw_prototypes = raw_checkpoint["prototypes"]["quantum"]
    stable_prototypes = F.normalize(value_projector.prototypes.detach(), dim=-1)
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    path_rows = {name: [] for name in (
        "integrated_stable_conjunction_normal", "integrated_stable_context_masked",
        "integrated_stable_key_masked", "integrated_stable_conjunction_recovered",
        "integrated_raw_value_control",
    )}
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    cell_counts, episode_seeds = [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, spec)
        cell_counts.extend(trace["cell_counts"])
        exact, exact_query = _exact_addresses(episode, spec=spec)
        context_only, context_query = _exact_addresses(episode, mask_key=True, spec=spec)
        key_only, key_query = _exact_addresses(episode, mask_context=True, spec=spec)
        normal, normal_path = _integrated(
            trace, stable_prototypes, context_projector, key_projector, value_projector, spec
        )
        context_masked, context_path = _integrated(
            trace, stable_prototypes, context_projector, key_projector, value_projector,
            spec, mask_context=True,
        )
        key_masked, key_path = _integrated(
            trace, stable_prototypes, context_projector, key_projector, value_projector,
            spec, mask_key=True,
        )
        recovered, recovered_path = _integrated(
            trace, stable_prototypes, context_projector, key_projector, value_projector, spec
        )
        raw, raw_path = _integrated(
            trace, raw_prototypes, context_projector, key_projector, value_projector,
            spec, raw_value=True,
        )
        outcomes = {
            "integrated_stable_conjunction_normal": normal,
            "external_stable_conjunction_reference": _external(
                trace, stable_prototypes, context_projector, key_projector, value_projector, spec
            ),
            "integrated_stable_context_masked": context_masked,
            "integrated_stable_key_masked": key_masked,
            "exact_stable_context_key_control": _exact_stable(
                exact, exact_query, trace["values"], stable_prototypes, value_projector
            ),
            "exact_stable_context_only_control": _exact_stable(
                context_only, context_query, trace["values"], stable_prototypes, value_projector
            ),
            "exact_stable_key_only_control": _exact_stable(
                key_only, key_query, trace["values"], stable_prototypes, value_projector
            ),
            "exact_stable_partner_swap": _exact_stable(
                exact, exact_query, trace["values"], stable_prototypes, value_projector,
                stored_values=_wrong_value_states(episode, trace),
            ),
            "integrated_stable_conjunction_recovered": recovered,
            "integrated_raw_value_control": raw,
        }
        stable_content = int(value_projector(
            trace["values"][episode.query_position].mean(0).unsqueeze(0)
        ).argmax(1)[0])
        raw_content = _decode(trace["values"][episode.query_position], raw_prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0]); row["selections"].append(outcome[1])
            row["contents"].append(raw_content if name == "integrated_raw_value_control" else stable_content)
            row["api"].append(outcome[2]); row["margins"].append(outcome[3])
        for name, audit in (
            ("integrated_stable_conjunction_normal", normal_path),
            ("integrated_stable_context_masked", context_path),
            ("integrated_stable_key_masked", key_path),
            ("integrated_stable_conjunction_recovered", recovered_path),
            ("integrated_raw_value_control", raw_path),
        ):
            path_rows[name].append(audit)
        if (index + 1) % 128 == 0:
            print(f"[prototype {prototype_seed} engine {engine_seed}] evaluated {index + 1}/{len(episodes)}", flush=True)
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        ) for name, row in records.items()
    }
    normal_records = records["integrated_stable_conjunction_normal"]
    reference_records = records["external_stable_conjunction_reference"]
    arms["integrated_stable_conjunction_normal"]["reference_prediction_match"] = float(
        normal_records["predictions"] == reference_records["predictions"]
    )
    arms["integrated_stable_conjunction_normal"]["reference_selection_match"] = float(
        normal_records["selections"] == reference_records["selections"]
    )
    arms["integrated_stable_conjunction_recovered"]["prediction_match"] = float(
        records["integrated_stable_conjunction_recovered"]["predictions"]
        == normal_records["predictions"]
    )
    def summarize(rows):
        return {
            key: (
                min(row[key] for row in rows)
                if "minimum" in key else max(row[key] for row in rows)
            )
            for key in (
                "key_calls", "key_minimum_components", "key_maximum_components",
                "key_minimum_width", "key_maximum_width", "value_calls",
                "value_minimum_width", "value_maximum_width", "stores", "retrievals",
            )
        }
    return {
        "prototype_seed": prototype_seed, "engine_seed": engine_seed,
        "arms": arms,
        "path_audit": {name: summarize(rows) for name, rows in path_rows.items()},
        "frozen_audit": {
            name: all(torch.equal(state[key], model.state_dict()[key]) for key in state)
            for name, state, model in (
                ("context", before["context"], context_projector),
                ("key", before["key"], key_projector),
                ("value", before["value"], value_projector),
            )
        },
        "state_audit": {
            "episodes": len(episodes), "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256("\n".join(map(str, episode_seeds)).encode()).hexdigest(),
            "minimum_cells": min(cell_counts), "maximum_cells": max(cell_counts),
        },
        "prototype_checkpoint": source["prototype_checkpoints"][str(prototype_seed)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/conjunction2_results.json")
    parser.add_argument("--verdict", default="measurement/conjunction2_verdict.json")
    args = parser.parse_args()
    spec = CONJUNCTION2_SPEC
    source = _source_receipt(spec)
    episodes = build_episodes(spec)
    payload = {
        "experiment": spec["experiment"], "spec": spec, "spec_sha256": spec_sha256(spec),
        "source": source, "dataset_audit": dataset_audit(episodes, spec),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "evaluations": [
            {"name": evaluation_name(row), **run_evaluation(
                row["prototype_seed"], row["engine_seed"], episodes, source, spec
            )} for row in spec["evaluation_combinations"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.conjunction2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
