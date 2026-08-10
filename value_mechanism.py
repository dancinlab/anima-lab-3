#!/usr/bin/env python3
"""VALUE-MECHANISM-1: isolate serial position in episodic value readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from conjunction import (
    ConjunctionEpisode,
    _direct_prediction,
    _exact_addresses,
    _receipt,
    _wrong_value_states,
    trace_episode,
)
from episode import _decode
from graft_behavior import sha256_file
from measurement.projector_registry import evaluation_name
from measurement.value_gate import adjudicate as adjudicate_value
from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256
from measurement.value_registry import VALUE_SPEC, spec_sha256 as value_spec_sha256
from separation import _arm_metrics
from value import build_value_episodes, value_dataset_audit


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = VALUE_MECHANISM_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = value_spec_sha256(VALUE_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != VALUE_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_value(results) != verdict
    ):
        raise RuntimeError("registered VALUE-1 source changed")
    inherited = results["source_conjunction1"]
    for receipt in inherited["prototype_checkpoints"].values():
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered VALUE-1 prototype changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "prototype_checkpoints": {
            key: dict(value) for key, value in inherited["prototype_checkpoints"].items()
        },
    }


def position_episode(episode: ConjunctionEpisode, query_position: int,
                     spec: dict = VALUE_MECHANISM_SPEC) -> ConjunctionEpisode:
    if query_position not in spec["query_positions"]:
        raise ValueError("query position is not registered")
    if episode.query_position != 0:
        raise ValueError("registered base episode must place the query first")
    order = list(range(len(episode.values)))
    order[0], order[query_position] = order[query_position], order[0]
    return ConjunctionEpisode(
        contexts=tuple(episode.contexts[index] for index in order),
        keys=tuple(episode.keys[index] for index in order),
        values=tuple(episode.values[index] for index in order),
        active_contexts=episode.active_contexts,
        active_keys=episode.active_keys,
        active_values=episode.active_values,
        distractors=episode.distractors,
        query_position=query_position,
    )


def position_dataset_audit(episodes: list[ConjunctionEpisode],
                           spec: dict = VALUE_MECHANISM_SPEC) -> dict:
    base = value_dataset_audit(episodes, VALUE_SPEC)
    positions = {}
    event_set_hashes = set()
    for position in spec["query_positions"]:
        rows = [position_episode(row, position, spec) for row in episodes]
        event_sets = [
            tuple(sorted(zip(row.contexts, row.keys, row.values))) for row in rows
        ]
        digest = hashlib.sha256(
            json.dumps(event_sets, separators=(",", ":")).encode()
        ).hexdigest()
        event_set_hashes.add(digest)
        positions[str(position)] = {
            "episodes": len(rows),
            "query_position_matches": sum(row.query_position == position for row in rows),
            "minimum_unique_pairs": min(len(set(zip(row.contexts, row.keys))) for row in rows),
            "maximum_unique_pairs": max(len(set(zip(row.contexts, row.keys))) for row in rows),
            "event_set_sha256": digest,
            "ordered_fingerprint_sha256": hashlib.sha256(
                "\n".join(row.fingerprint() for row in rows).encode()
            ).hexdigest(),
        }
    return {
        "base": base["base"],
        "positions": positions,
        "shared_event_set_sha256": next(iter(event_set_hashes)) if len(event_set_hashes) == 1 else None,
    }


def _run_position(prototype_seed: int, engine_seed: int, query_position: int,
                  episodes: list[ConjunctionEpisode], prototypes: torch.Tensor,
                  spec: dict = VALUE_MECHANISM_SPEC) -> dict:
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [query_position] * len(episodes)
    episode_seeds, cell_counts = [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, base_episode in enumerate(episodes):
        episode = position_episode(base_episode, query_position, spec)
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, spec)
        cell_counts.extend(trace["cell_counts"])
        exact, query = _exact_addresses(episode, spec=spec)
        outcomes = {
            "exact_value_normal": _direct_prediction(exact, trace["values"], query, prototypes),
            "exact_value_partner_swap": _direct_prediction(
                exact, trace["values"], query, prototypes,
                stored_values=_wrong_value_states(episode, trace),
            ),
            "exact_value_recovered": _direct_prediction(exact, trace["values"], query, prototypes),
        }
        content = _decode(trace["values"][query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        if (index + 1) % 128 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed} position {query_position + 1}] "
                f"evaluated {index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    arms["exact_value_recovered"]["prediction_match"] = float(
        records["exact_value_recovered"]["predictions"]
        == records["exact_value_normal"]["predictions"]
    )
    return {
        "query_position": query_position,
        "query_position_label": query_position + 1,
        "arms": arms,
        "path_audit": {
            "stores_per_episode": spec["stores_per_episode"],
            "retrievals_per_episode": spec["retrievals_per_episode"],
            "address_width": spec["contexts"] + spec["keys"],
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
                   spec: dict = VALUE_MECHANISM_SPEC) -> dict:
    receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "positions": [
            _run_position(prototype_seed, engine_seed, position, episodes, prototypes, spec)
            for position in spec["query_positions"]
        ],
        "prototype_checkpoint": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/value_mechanism_results.json")
    parser.add_argument("--verdict", default="measurement/value_mechanism_verdict.json")
    args = parser.parse_args()
    spec = VALUE_MECHANISM_SPEC
    _, source = _source_receipt(spec)
    episodes = build_value_episodes(VALUE_SPEC)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_value1": source,
        "dataset_audit": position_dataset_audit(episodes, spec),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
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
    from measurement.value_mechanism_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
