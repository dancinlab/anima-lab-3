#!/usr/bin/env python3
"""CAPACITY-2: measure the stable-address boundary after autonomous settling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from capacity import _exact_key_addresses, build_capacity_episodes, count_spec
from episode import _decode
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from graft_behavior import sha256_file
from measurement.capacity2_registry import CAPACITY2_SPEC, spec_sha256
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from separation import _arm_metrics, _direct_prediction, dataset_audit, trace_similar_episode
from settle import _paired


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = CAPACITY2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = capacity_spec_sha256(CAPACITY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CAPACITY_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered CAPACITY-1 source is not the boundary-2 result")
    checkpoints, prototypes = {}, {}
    for row in results["seeds"]:
        seed = row["seed"]
        for receipt in (row["source_checkpoint"], row["prototype_checkpoint"]):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"CAPACITY-1 source checkpoint changed for seed {seed}")
        checkpoints[str(seed)] = dict(row["source_checkpoint"])
        prototypes[str(seed)] = dict(row["prototype_checkpoint"])
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": checkpoints,
        "prototype_checkpoints": prototypes,
    }


def _run_condition_count(seed: int, event_count: int, episodes, condition: dict,
                         projector, prototypes, spec: dict = CAPACITY2_SPEC):
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    expected = [episode.target for episode in episodes]
    positions = [episode.query_position for episode in episodes]
    calls, widths, episode_seeds, cell_counts = [], [], [], []
    before_digests, after_digests, query_rng_digests = [], [], []
    base = (
        spec["episode_seed_base"]
        + event_count * spec["event_count_seed_stride"]
        + seed * spec["seed_stride"]
    )
    local = count_spec(event_count, CAPACITY_SPEC)
    local["pre_query_updates"] = condition["updates"]
    local["pre_query_dynamics_ablation"] = condition["disabled"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        traced = trace_similar_episode(episode, trial_seed, distinct=True, spec=local)
        cell_counts.extend(traced["cell_counts"])
        update = traced["update_audit"]
        before_digests.append(update["state_before_sha256"])
        after_digests.append(update["state_after_sha256"])
        query_rng_digests.append(update["query_rng_sha256"])
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
        calls.append(outcomes["stable_distinct_normal"][4])
        widths.append(outcomes["stable_distinct_normal"][5])
        if (index + 1) % 256 == 0:
            print(
                f"[seed {seed} {condition['name']} events {event_count}] "
                f"evaluated {index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    expected_tensor = torch.tensor(expected)
    arms = {
        name: _arm_metrics(
            expected_tensor,
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
    public = {
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
        "update_audit": {
            "requested_updates": condition["updates"],
            "performed_updates_minimum": condition["updates"],
            "performed_updates_maximum": condition["updates"],
            "disabled": condition["disabled"],
            "unchanged_state_count": sum(
                before == after for before, after in zip(before_digests, after_digests)
            ),
            "state_before_sha256": hashlib.sha256(
                "\n".join(before_digests).encode()
            ).hexdigest(),
            "state_after_sha256": hashlib.sha256(
                "\n".join(after_digests).encode()
            ).hexdigest(),
            "query_rng_sha256": hashlib.sha256(
                "\n".join(query_rng_digests).encode()
            ).hexdigest(),
        },
    }
    return public, records


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = CAPACITY2_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    public_conditions, private = [], {}
    for condition in spec["conditions"]:
        counts = []
        for event_count in spec["event_counts"]:
            public, records = _run_condition_count(
                seed, event_count, episode_sets[event_count], condition,
                projector, prototypes, spec,
            )
            counts.append(public)
            private[(condition["name"], event_count)] = records
        public_conditions.append({
            "name": condition["name"],
            "updates": condition["updates"],
            "disabled": condition["disabled"],
            "counts": counts,
        })
    comparisons = []
    for event_count in spec["event_counts"]:
        expected = [episode.target for episode in episode_sets[event_count]]
        positions = [episode.query_position for episode in episode_sets[event_count]]
        settled = private[("settled", event_count)]["stable_distinct_normal"]
        blocked = private[("without_frustration_regulation", event_count)]["stable_distinct_normal"]
        comparisons.append({
            "event_count": event_count,
            "final": _paired(settled["predictions"], blocked["predictions"], expected),
            "selection": _paired(settled["selections"], blocked["selections"], positions),
        })
    after = projector.state_dict()
    return {
        "seed": seed,
        "conditions": public_conditions,
        "comparisons": comparisons,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/capacity2_results.json")
    parser.add_argument("--verdict", default="measurement/capacity2_verdict.json")
    args = parser.parse_args()
    spec = CAPACITY2_SPEC
    source_results, source = _source_receipt(spec)
    episode_sets = {
        event_count: build_capacity_episodes(event_count, CAPACITY_SPEC)
        for event_count in spec["event_counts"]
    }
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": {
            str(event_count): dataset_audit(
                episode_sets[event_count], count_spec(event_count, CAPACITY_SPEC)
            )
            for event_count in spec["event_counts"]
        },
        "source_capacity": source,
        "source_capacity_pass": json.loads(Path(spec["source_verdict_path"]).read_text())["capacity_pass"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.capacity2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
