#!/usr/bin/env python3
"""VALUE-1: locate the event-count boundary of episodic value readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path

import torch

from conjunction import (
    ConjunctionEpisode,
    _direct_prediction,
    _exact_addresses,
    _receipt,
    _wrong_value_states,
    build_episodes,
    dataset_audit,
    trace_episode,
)
from episode import _decode
from graft_behavior import sha256_file
from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.value_registry import VALUE_SPEC, spec_sha256
from separation import _arm_metrics


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = VALUE_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = conjunction_spec_sha256(CONJUNCTION_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CONJUNCTION_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_conjunction(results) != verdict
    ):
        raise RuntimeError("registered CONJUNCTION-1 source changed")
    source = results["source_context2"]
    receipts = (
        source["context_checkpoint"], source["canonical_checkpoint"],
        *source["prototype_checkpoints"].values(),
    )
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered CONJUNCTION-1 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "prototype_checkpoints": {
            key: dict(value) for key, value in source["prototype_checkpoints"].items()
        },
    }


def _base_spec(spec: dict = VALUE_SPEC) -> dict:
    value = dict(CONJUNCTION_SPEC)
    for key in (
        "eval_episodes", "data_seed", "episode_seed_base", "event_counts",
    ):
        if key in spec:
            value[key] = spec[key]
    return value


def value_balanced_order(episode: ConjunctionEpisode) -> ConjunctionEpisode:
    """Put the query first and retain one occurrence of each active value per block."""
    labels = [episode.target, *sorted(value for value in episode.active_values if value != episode.target)]
    positions = {
        label: [index for index, value in enumerate(episode.values) if value == label]
        for label in labels
    }
    positions[episode.target].remove(episode.query_position)
    positions[episode.target].insert(0, episode.query_position)
    order = []
    for block in range(len(episode.active_values)):
        for label in labels:
            order.append(positions[label][block])
    if len(order) != len(episode.values) or len(set(order)) != len(order):
        raise RuntimeError("value-balanced event order is incomplete")
    return ConjunctionEpisode(
        contexts=tuple(episode.contexts[index] for index in order),
        keys=tuple(episode.keys[index] for index in order),
        values=tuple(episode.values[index] for index in order),
        active_contexts=episode.active_contexts,
        active_keys=episode.active_keys,
        active_values=episode.active_values,
        distractors=episode.distractors,
        query_position=0,
    )


def prefix_episode(episode: ConjunctionEpisode, event_count: int) -> ConjunctionEpisode:
    if event_count not in VALUE_SPEC["event_counts"]:
        raise ValueError("event count is not registered")
    ordered = value_balanced_order(episode)
    return ConjunctionEpisode(
        contexts=ordered.contexts[:event_count],
        keys=ordered.keys[:event_count],
        values=ordered.values[:event_count],
        active_contexts=ordered.active_contexts,
        active_keys=ordered.active_keys,
        active_values=ordered.active_values,
        distractors=ordered.distractors,
        query_position=0,
    )


def build_value_episodes(spec: dict = VALUE_SPEC) -> list[ConjunctionEpisode]:
    return [value_balanced_order(row) for row in build_episodes(_base_spec(spec))]


def value_dataset_audit(episodes: list[ConjunctionEpisode], spec: dict = VALUE_SPEC) -> dict:
    base = dataset_audit(episodes, _base_spec(spec))
    prefixes = {}
    for count in spec["event_counts"]:
        rows = [prefix_episode(row, count) for row in episodes]
        expected_per_value = count // spec["active_values_per_episode"]
        balanced = sum(
            all(Counter(row.values)[value] == expected_per_value for value in row.active_values)
            for row in rows
        )
        prefixes[str(count)] = {
            "episodes": len(rows),
            "query_included": sum(row.query_position == 0 for row in rows),
            "value_balanced": balanced,
            "minimum_unique_pairs": min(len(set(zip(row.contexts, row.keys))) for row in rows),
            "maximum_unique_pairs": max(len(set(zip(row.contexts, row.keys))) for row in rows),
            "fingerprint_set_sha256": hashlib.sha256(
                "\n".join(sorted(row.fingerprint() for row in rows)).encode()
            ).hexdigest(),
        }
    return {"base": base, "prefixes": prefixes}


def _run_count(prototype_seed: int, engine_seed: int, event_count: int,
               episodes: list[ConjunctionEpisode], prototypes: torch.Tensor,
               spec: dict = VALUE_SPEC) -> dict:
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [0] * len(episodes)
    episode_seeds, cell_counts = [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    local = _base_spec(spec)
    local["events_per_episode"] = event_count
    for index, full_episode in enumerate(episodes):
        episode = prefix_episode(full_episode, event_count)
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, local)
        cell_counts.extend(trace["cell_counts"])
        exact, query = _exact_addresses(episode, spec=local)
        outcomes = {
            "exact_value_normal": _direct_prediction(exact, trace["values"], query, prototypes),
            "exact_value_partner_swap": _direct_prediction(
                exact, trace["values"], query, prototypes,
                stored_values=_wrong_value_states(episode, trace),
            ),
            "exact_value_recovered": _direct_prediction(exact, trace["values"], query, prototypes),
        }
        content = _decode(trace["values"][0], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        if (index + 1) % 128 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed} events {event_count}] "
                f"evaluated {index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], local,
        )
        for name, row in records.items()
    }
    arms["exact_value_recovered"]["prediction_match"] = float(
        records["exact_value_recovered"]["predictions"]
        == records["exact_value_normal"]["predictions"]
    )
    return {
        "event_count": event_count,
        "arms": arms,
        "path_audit": {
            "stores_per_episode": event_count,
            "retrievals_per_episode": spec["retrievals_per_episode"],
            "address_width": spec["contexts"] + spec["keys"],
            "query_position": 0,
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
        },
    }


def run_evaluation(prototype_seed: int, engine_seed: int,
                   episodes: list[ConjunctionEpisode], source: dict,
                   spec: dict = VALUE_SPEC) -> dict:
    prototype_receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "counts": [
            _run_count(prototype_seed, engine_seed, count, episodes, prototypes, spec)
            for count in spec["event_counts"]
        ],
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/value_results.json")
    parser.add_argument("--verdict", default="measurement/value_verdict.json")
    args = parser.parse_args()
    spec = VALUE_SPEC
    _, source = _source_receipt(spec)
    episodes = build_value_episodes(spec)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_conjunction1": source,
        "dataset_audit": value_dataset_audit(episodes, spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": [
            {
                "name": evaluation_name(row),
                **run_evaluation(
                    row["prototype_seed"], row["engine_seed"], episodes, source, spec
                ),
            }
            for row in spec["evaluation_combinations"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.value_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
