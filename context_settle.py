#!/usr/bin/env python3
"""CONTEXT-SETTLE-1: locate the minimum context transition settling count."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import torch

from component2 import balance_components
from conjunction import _atomic_json, build_episodes, dataset_audit
from episode import _new_engine
from graft_behavior import sha256_file
from key_stability import StableKeyProjector, key_classification_metrics
from measurement.component2_gate import adjudicate as adjudicate_component2
from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component2_spec_sha256
from measurement.component_registry import COMPONENT_SPEC
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256
from measurement.episode_registry import EPISODE_SPEC
from separation import _sense_separation_token


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = component2_spec_sha256(COMPONENT2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != COMPONENT2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_component2(results) != verdict
    ):
        raise RuntimeError("registered COMPONENT-2 source changed")
    checkpoint = dict(results["checkpoint"])
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path != Path(spec["source_checkpoint_path"]):
        raise RuntimeError("registered COMPONENT-2 checkpoint path changed")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError("registered COMPONENT-2 checkpoint changed")
    return {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "checkpoint": checkpoint,
        "source_spec_sha256": expected_sha,
    }


def _load_context_projector(receipt: dict, spec: dict = CONTEXT_SETTLE_SPEC):
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != COMPONENT2_SPEC["experiment"]
        or checkpoint.get("spec_sha256") != component2_spec_sha256(COMPONENT2_SPEC)
        or checkpoint.get("deterministic") is not True
    ):
        raise RuntimeError("registered context projector identity changed")
    projector = StableKeyProjector(
        spec["state_dim"], spec["address_dim"], spec["contexts"],
        spec["temperature"], spec["bias"],
    )
    projector.load_state_dict(checkpoint["context_projector"])
    projector.eval().requires_grad_(False)
    return projector


def build_evaluation_episodes(spec: dict = CONTEXT_SETTLE_SPEC):
    local = {
        **CONJUNCTION2_SPEC,
        "eval_episodes": spec["eval_episodes"],
        "data_seed": spec["data_seed"],
    }
    return balance_components(build_episodes(local), spec)


def _fingerprints(rows) -> set[str]:
    return {row.fingerprint() for row in rows}


def source_overlap(episodes, spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    component2_local = {
        **CONJUNCTION2_SPEC,
        "eval_episodes": COMPONENT2_SPEC["calibration_episodes"],
        "data_seed": COMPONENT2_SPEC["calibration_data_seed"],
    }
    sources = {
        "component2_calibration": balance_components(
            build_episodes(component2_local), COMPONENT2_SPEC
        ),
        "component1_evaluation": build_episodes(COMPONENT_SPEC),
        "conjunction2_evaluation": build_episodes(CONJUNCTION2_SPEC),
    }
    current = _fingerprints(episodes)
    return {name: len(current & _fingerprints(rows)) for name, rows in sources.items()}


def extended_dataset_audit(episodes, spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    local = {**CONJUNCTION2_SPEC, "eval_episodes": spec["eval_episodes"]}
    audit = dataset_audit(episodes, local)
    context_counts = Counter(value for row in episodes for value in row.contexts)
    key_counts = Counter(value for row in episodes for value in row.keys)
    audit["event_context_counts"] = {
        str(value): context_counts[value] for value in range(spec["contexts"])
    }
    audit["event_key_counts"] = {
        str(value): key_counts[value] for value in range(spec["keys"])
    }
    audit["source_overlap"] = source_overlap(episodes, spec)
    return audit


def trace_contexts(episode, trial_seed: int, context_steps: int,
                   spec: dict = CONTEXT_SETTLE_SPEC) -> tuple[list[torch.Tensor], list[int]]:
    c, encoder = _new_engine(trial_seed, EPISODE_SPEC)
    states, cell_counts = [], []
    for context, key, value in zip(episode.contexts, episode.keys, episode.values):
        context_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][context], context_steps, spec
        )
        key_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["key_words"][key], spec["key_steps"], spec
        )
        value_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["value_words"][value], spec["value_steps"], spec
        )
        states.append(context_state.mean(0))
        cell_counts.extend((context_state.shape[0], key_state.shape[0], value_state.shape[0]))
    return states, cell_counts


def run_candidate(engine_seed: int, context_steps: int, episodes,
                  projector, spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    states = [[] for _ in spec["positions"]]
    labels = [[] for _ in spec["positions"]]
    used_seeds, cell_counts = [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        used_seeds.append(trial_seed)
        episode_states, episode_cells = trace_contexts(
            episode, trial_seed, context_steps, spec
        )
        cell_counts.extend(episode_cells)
        for position in spec["positions"]:
            states[position].append(episode_states[position])
            labels[position].append(episode.contexts[position])
        if (index + 1) % 128 == 0:
            print(
                f"[engine {engine_seed} context-steps {context_steps}] "
                f"evaluated {index + 1}/{len(episodes)}",
                flush=True,
            )
    positions = []
    for position in spec["positions"]:
        metrics = key_classification_metrics(
            projector,
            torch.stack(states[position]),
            torch.tensor(labels[position], dtype=torch.long),
            spec["contexts"],
        )
        positions.append({
            "position": position,
            "position_label": position + 1,
            "context": metrics,
        })
    total_events = len(episodes) * spec["events_per_episode"]
    return {
        "context_steps": context_steps,
        "positions": positions,
        "state_audit": {
            "episodes": len(episodes),
            "states": total_events,
            "unique_episode_seeds": len(set(used_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, used_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
            "context_step_calls": total_events * context_steps,
            "key_step_calls": total_events * spec["key_steps"],
            "value_step_calls": total_events * spec["value_steps"],
        },
    }


def run_engine(engine_seed: int, episodes, source: dict,
               spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    projector = _load_context_projector(source["checkpoint"], spec)
    before = {name: value.clone() for name, value in projector.state_dict().items()}
    candidates = [
        run_candidate(engine_seed, steps, episodes, projector, spec)
        for steps in spec["context_steps"]
    ]
    after = projector.state_dict()
    return {
        "engine_seed": engine_seed,
        "candidates": candidates,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(
            torch.equal(before[name], after[name]) for name in before
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/context_settle_results.json")
    parser.add_argument("--verdict", default="measurement/context_settle_verdict.json")
    args = parser.parse_args()
    spec = CONTEXT_SETTLE_SPEC
    source = _source_receipt(spec)
    episodes = build_evaluation_episodes(spec)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_component2": source,
        "dataset_audit": extended_dataset_audit(episodes, spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "engines": [run_engine(seed, episodes, source, spec) for seed in spec["engine_seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.context_settle_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
