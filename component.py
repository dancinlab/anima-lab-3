#!/usr/bin/env python3
"""COMPONENT-1: classify frozen context and key addresses by serial position."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from conjunction import _atomic_json, build_episodes, dataset_audit, trace_episode
from conjunction2 import _source_receipt as conjunction2_source_receipt
from context import _load_key_projector
from context2 import _load_context_projector
from key_stability import key_classification_metrics
from measurement.component_registry import COMPONENT_SPEC, spec_sha256


def _source_receipt(spec: dict = COMPONENT_SPEC) -> dict:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    from measurement.conjunction2_gate import adjudicate
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256 as source_sha
    expected_sha = source_sha(CONJUNCTION2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CONJUNCTION2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate(results) != verdict
    ):
        raise RuntimeError("registered CONJUNCTION-2 source changed")
    inherited = results["source"]
    return {
        "results": {"path": str(results_path), "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest()},
        "verdict": {"path": str(verdict_path), "sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest()},
        "source_spec_sha256": expected_sha,
        "context_checkpoint": dict(inherited["context_checkpoint"]),
        "canonical_checkpoint": dict(inherited["canonical_checkpoint"]),
    }


def run_engine(engine_seed, episodes, source, spec=COMPONENT_SPEC, *,
               context_projector_override=None, key_projector_override=None):
    context_projector = (
        _load_context_projector(source["context_checkpoint"], spec)
        if context_projector_override is None else context_projector_override
    )
    key_projector = (
        _load_key_projector(source["canonical_checkpoint"], spec)
        if key_projector_override is None else key_projector_override
    )
    before_context = {name: value.clone() for name, value in context_projector.state_dict().items()}
    before_key = {name: value.clone() for name, value in key_projector.state_dict().items()}
    context_states = [[] for _ in spec["positions"]]
    context_labels = [[] for _ in spec["positions"]]
    key_states = [[] for _ in spec["positions"]]
    key_labels = [[] for _ in spec["positions"]]
    query_context_states, query_context_labels = [], []
    query_key_states, query_key_labels = [], []
    cell_counts, used_seeds = [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        used_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, spec)
        cell_counts.extend(trace["cell_counts"])
        for position in spec["positions"]:
            context_states[position].append(trace["contexts"][position].mean(0))
            context_labels[position].append(episode.contexts[position])
            key_states[position].append(trace["keys"][position].mean(0))
            key_labels[position].append(episode.keys[position])
        query_context_states.append(trace["query_context"].mean(0))
        query_context_labels.append(episode.query_context)
        query_key_states.append(trace["query"].mean(0))
        query_key_labels.append(episode.query_key)
        if (index + 1) % 128 == 0:
            print(f"[engine {engine_seed}] evaluated {index + 1}/{len(episodes)}", flush=True)
    def metrics(model, states, labels, classes):
        return key_classification_metrics(
            model, torch.stack(states), torch.tensor(labels, dtype=torch.long), classes
        )
    return {
        "engine_seed": engine_seed,
        "positions": [
            {
                "position": position,
                "position_label": position + 1,
                "context": metrics(context_projector, context_states[position], context_labels[position], spec["contexts"]),
                "key": metrics(key_projector, key_states[position], key_labels[position], spec["keys"]),
            }
            for position in spec["positions"]
        ],
        "query": {
            "context": metrics(context_projector, query_context_states, query_context_labels, spec["contexts"]),
            "key": metrics(key_projector, query_key_states, query_key_labels, spec["keys"]),
        },
        "frozen_audit": {
            "context": all(torch.equal(before_context[name], context_projector.state_dict()[name]) for name in before_context),
            "key": all(torch.equal(before_key[name], key_projector.state_dict()[name]) for name in before_key),
        },
        "state_audit": {
            "episodes": len(episodes), "unique_episode_seeds": len(set(used_seeds)),
            "episode_seed_sha256": hashlib.sha256("\n".join(map(str, used_seeds)).encode()).hexdigest(),
            "minimum_cells": min(cell_counts), "maximum_cells": max(cell_counts),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/component_results.json")
    parser.add_argument("--verdict", default="measurement/component_verdict.json")
    args = parser.parse_args()
    spec = COMPONENT_SPEC
    source = _source_receipt(spec)
    episodes = build_episodes(spec)
    payload = {
        "experiment": spec["experiment"], "spec": spec, "spec_sha256": spec_sha256(spec),
        "source_conjunction2": source, "dataset_audit": dataset_audit(episodes, spec),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "engines": [run_engine(seed, episodes, source, spec) for seed in spec["engine_seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.component_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
