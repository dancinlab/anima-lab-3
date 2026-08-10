#!/usr/bin/env python3
"""CAPACITY-1: locate the stable-address event-count boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from episode import _decode
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from separation import (
    _arm_metrics,
    _direct_prediction,
    _source_receipt,
    build_episodes,
    dataset_audit,
    trace_similar_episode,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def count_spec(event_count: int, spec: dict = CAPACITY_SPEC) -> dict:
    if event_count not in spec["event_counts"]:
        raise ValueError("event count is not registered")
    value = dict(spec)
    value["events_per_episode"] = event_count
    value["eval_episodes"] = spec["eval_episodes_per_count"]
    value["data_seed"] = spec["data_seed_base"] + event_count * spec["event_count_seed_stride"]
    value["exact_marginal_balance"] = spec["balance_mode"] == "exact_marginals"
    return value


def build_capacity_episodes(event_count: int, spec: dict = CAPACITY_SPEC):
    return build_episodes(count_spec(event_count, spec))


def capacity_dataset_audit(event_count: int, spec: dict = CAPACITY_SPEC) -> dict:
    local = count_spec(event_count, spec)
    return dataset_audit(build_episodes(local), local)


def _exact_key_addresses(episode, spec: dict = CAPACITY_SPEC):
    addresses = [
        F.one_hot(torch.tensor(key), spec["keys"]).float()
        for key in episode.distinct_keys
    ]
    query = F.one_hot(
        torch.tensor(episode.distinct_keys[episode.query_position]), spec["keys"]
    ).float()
    return addresses, query


def _run_count(seed: int, event_count: int, episodes, projector, prototypes,
               spec: dict = CAPACITY_SPEC) -> dict:
    names = tuple(spec["arms"])
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in names
    }
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    calls, widths, episode_seeds, cell_counts = [], [], [], []
    base = (
        spec["episode_seed_base"]
        + event_count * spec["event_count_seed_stride"]
        + seed * spec["seed_stride"]
    )
    local = count_spec(event_count, spec)
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        traced = trace_similar_episode(
            episode, trial_seed, distinct=True, spec=local
        )
        cell_counts.extend(traced["cell_counts"])
        exact, exact_query = _exact_key_addresses(episode, spec)
        outcomes = {
            "stable_distinct_normal": _integrated_memory_prediction(
                traced["keys"], traced["values"], traced["query"], prototypes, projector
            ),
            "raw_distinct_control": _integrated_memory_prediction(
                traced["keys"], traced["values"], traced["query"], prototypes, None
            ),
            "exact_key_control": _direct_prediction(
                exact, traced["values"], exact_query, prototypes
            ),
            "exact_key_partner_swap": _direct_prediction(
                exact, traced["values"], exact_query, prototypes, rotate=True
            ),
            "exact_key_recovered": _direct_prediction(
                exact, traced["values"], exact_query, prototypes
            ),
        }
        content = _decode(traced["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            record = records[name]
            record["predictions"].append(outcome[0])
            record["selections"].append(outcome[1])
            record["contents"].append(content)
            record["api"].append(outcome[2])
            record["margins"].append(outcome[3])
        stable = outcomes["stable_distinct_normal"]
        calls.append(stable[4])
        widths.append(stable[5])
        if (index + 1) % 256 == 0:
            print(
                f"[seed {seed} events {event_count}] evaluated "
                f"{index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    arms = {
        name: _arm_metrics(
            expected,
            row["predictions"],
            row["selections"],
            positions,
            row["contents"],
            row["api"],
            row["margins"],
            local,
        )
        for name, row in records.items()
    }
    arms["exact_key_recovered"]["prediction_match"] = float(
        records["exact_key_recovered"]["predictions"]
        == records["exact_key_control"]["predictions"]
    )
    return {
        "event_count": event_count,
        "arms": arms,
        "integration_audit": {
            "stable_transform_calls": {
                "episodes": len(calls),
                "total": sum(calls),
                "minimum": min(calls),
                "maximum": max(calls),
            },
            "address_width_minimum": min(widths),
            "address_width_maximum": max(widths),
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


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = CAPACITY_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    counts = [
        _run_count(seed, event_count, episode_sets[event_count], projector, prototypes, spec)
        for event_count in spec["event_counts"]
    ]
    after = projector.state_dict()
    return {
        "seed": seed,
        "counts": counts,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/capacity_results.json")
    parser.add_argument("--verdict", default="measurement/capacity_verdict.json")
    args = parser.parse_args()
    spec = CAPACITY_SPEC
    _, source = _source_receipt(spec)
    episode_sets = {
        event_count: build_capacity_episodes(event_count, spec)
        for event_count in spec["event_counts"]
    }
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": {
            str(event_count): dataset_audit(
                episode_sets[event_count], count_spec(event_count, spec)
            )
            for event_count in spec["event_counts"]
        },
        "source_episode2": source,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.capacity_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
