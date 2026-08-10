#!/usr/bin/env python3
"""SEPARATION-2: retest similar episodes with the canonical stable address."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from episode import _decode
from episode2 import _integrated_memory_prediction
from graft_behavior import sha256_file
from key_stability import StableKeyProjector
from measurement.canonical2_registry import CANONICAL2_SPEC, spec_sha256 as canonical2_spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256
from separation import (
    _arm_metrics,
    _direct_prediction,
    _exact_addresses,
    build_episodes,
    dataset_audit,
    trace_similar_episode,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = SEPARATION2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = canonical2_spec_sha256(CANONICAL2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CANONICAL2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered CANONICAL-2 source changed")
    checkpoint = results["checkpoint"]
    prototypes = results["source_canonical"]["prototype_checkpoints"]
    for receipt in (checkpoint, *prototypes.values()):
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered CANONICAL-2 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "checkpoint": dict(checkpoint),
        "prototype_checkpoints": {key: dict(value) for key, value in prototypes.items()},
    }


def _load_projector(receipt: dict, spec: dict = SEPARATION2_SPEC) -> StableKeyProjector:
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != spec["source_experiment"]
        or checkpoint.get("spec_sha256") != canonical2_spec_sha256(CANONICAL2_SPEC)
        or checkpoint.get("fit_method") != spec["fit_method"]
        or checkpoint.get("model_class") != spec["model_class"]
    ):
        raise RuntimeError("canonical projector identity changed")
    model = StableKeyProjector(
        spec["state_dim"], spec["address_dim"], spec["keys"],
        CANONICAL2_SPEC["temperature"], CANONICAL2_SPEC["bias"],
    )
    model.load_state_dict(checkpoint["projector"])
    model.eval()
    model.requires_grad_(False)
    return model


def _digest_rows(rows: list[str]) -> str:
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   spec: dict = SEPARATION2_SPEC) -> dict:
    projector = _load_projector(source["checkpoint"], spec)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    similar_calls, distinct_calls, address_widths = [], [], []
    episode_seeds, cell_counts = [], []
    before_digests, after_digests, query_rng_digests = [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        similar = trace_similar_episode(episode, trial_seed, distinct=False, spec=spec)
        distinct = trace_similar_episode(episode, trial_seed, distinct=True, spec=spec)
        cell_counts.extend(similar["cell_counts"])
        cell_counts.extend(distinct["cell_counts"])
        for trace in (similar, distinct):
            before_digests.append(trace["update_audit"]["state_before_sha256"])
            after_digests.append(trace["update_audit"]["state_after_sha256"])
            query_rng_digests.append(trace["update_audit"]["query_rng_sha256"])
        exact, exact_query = _exact_addresses(episode, spec=spec)
        removed, removed_query = _exact_addresses(episode, remove_context=True, spec=spec)
        outcomes = {
            "stable_similar_normal": _integrated_memory_prediction(
                similar["keys"], similar["values"], similar["query"], prototypes, projector
            ),
            "raw_similar_normal": _integrated_memory_prediction(
                similar["keys"], similar["values"], similar["query"], prototypes, None
            ),
            "stable_distinct_key_control": _integrated_memory_prediction(
                distinct["keys"], distinct["values"], distinct["query"], prototypes, projector
            ),
            "exact_context_key_control": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "exact_context_key_partner_swap": _direct_prediction(
                exact, similar["values"], exact_query, prototypes, rotate=True
            ),
            "exact_context_key_recovered": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "context_removed_control": _direct_prediction(
                removed, similar["values"], removed_query, prototypes
            ),
        }
        content = _decode(similar["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            record = records[name]
            record["predictions"].append(outcome[0])
            record["selections"].append(outcome[1])
            record["contents"].append(content)
            record["api"].append(outcome[2])
            record["margins"].append(outcome[3])
        similar_calls.append(outcomes["stable_similar_normal"][4])
        distinct_calls.append(outcomes["stable_distinct_key_control"][4])
        address_widths.extend((
            outcomes["stable_similar_normal"][5],
            outcomes["stable_distinct_key_control"][5],
        ))
        if (index + 1) % 256 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed}] "
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
    arms["exact_context_key_recovered"]["prediction_match"] = float(
        records["exact_context_key_recovered"]["predictions"]
        == records["exact_context_key_control"]["predictions"]
    )
    expected_calls = spec["expected_stable_transform_calls_per_episode"]
    after = projector.state_dict()
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": _digest_rows(list(map(str, episode_seeds))),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
        },
        "update_audit": {
            "requested_updates": spec["settling_updates"],
            "performed_updates_minimum": spec["settling_updates"],
            "performed_updates_maximum": spec["settling_updates"],
            "disabled": list(spec["pre_query_dynamics_ablation"]),
            "state_before_sha256": _digest_rows(before_digests),
            "state_after_sha256": _digest_rows(after_digests),
            "query_rng_sha256": _digest_rows(query_rng_digests),
        },
        "integration_audit": {
            "similar_transform_calls": {
                "episodes": len(similar_calls), "total": sum(similar_calls),
                "minimum": min(similar_calls), "maximum": max(similar_calls),
            },
            "distinct_transform_calls": {
                "episodes": len(distinct_calls), "total": sum(distinct_calls),
                "minimum": min(distinct_calls), "maximum": max(distinct_calls),
            },
            "expected_calls_per_episode": expected_calls,
            "address_width_minimum": min(address_widths),
            "address_width_maximum": max(address_widths),
            "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
            "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        },
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/separation2_results.json")
    parser.add_argument("--verdict", default="measurement/separation2_verdict.json")
    args = parser.parse_args()
    spec = SEPARATION2_SPEC
    _, source = _source_receipt(spec)
    episodes = build_episodes(spec)
    evaluations = []
    for row in spec["evaluation_combinations"]:
        evaluations.append({
            "name": evaluation_name(row),
            **run_evaluation(row["prototype_seed"], row["engine_seed"], episodes, source, spec),
        })
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(episodes, spec),
        "source_canonical2": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.separation2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
