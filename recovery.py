#!/usr/bin/env python3
"""RECOVERY-1: densely replicate the three-candidate memory recovery curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from decay import (
    _exact_addresses,
    _trace,
    build_decay_episodes,
    decay_dataset_audit,
)
from episode import _decode
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from graft_behavior import sha256_file
from measurement.decay_registry import DECAY_SPEC, spec_sha256 as decay_spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.recovery_registry import RECOVERY_SPEC, spec_sha256
from separation import _arm_metrics, _direct_prediction


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _replicate_spec(replicate: int, spec: dict = RECOVERY_SPEC) -> dict:
    value = dict(spec)
    value["eval_episodes_per_delay"] = spec["episodes_per_replicate"]
    value["data_seed"] = spec["data_seed_base"] + replicate * spec["replicate_seed_stride"]
    return value


def build_recovery_episodes(replicate: int, spec: dict = RECOVERY_SPEC):
    if replicate not in spec["replicates"]:
        raise ValueError(f"unregistered replicate {replicate}")
    return build_decay_episodes(_replicate_spec(replicate, spec))


def recovery_dataset_audit(episode_sets: dict[int, list], spec: dict = RECOVERY_SPEC) -> dict:
    fingerprint_sets = {
        replicate: {row.fingerprint() for row in rows}
        for replicate, rows in episode_sets.items()
    }
    overlaps = {
        f"{left}:{right}": len(fingerprint_sets[left] & fingerprint_sets[right])
        for index, left in enumerate(spec["replicates"])
        for right in spec["replicates"][index + 1:]
    }
    audits = []
    for replicate in spec["replicates"]:
        audit = decay_dataset_audit(
            episode_sets[replicate], _replicate_spec(replicate, spec)
        )
        audit["replicate"] = replicate
        audits.append(audit)
    return {
        "replicates": audits,
        "cross_replicate_overlap": overlaps,
        "combined_unique_fingerprints": len(set().union(*fingerprint_sets.values())),
    }


def _source_receipt(spec: dict = RECOVERY_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = decay_spec_sha256(DECAY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != DECAY_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered DECAY-1 source is not the D5 result")
    checkpoints, prototypes = {}, {}
    for row in results["seeds"]:
        seed = row["seed"]
        for receipt in (row["source_checkpoint"], row["prototype_checkpoint"]):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"DECAY-1 source checkpoint changed for seed {seed}")
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


def _records(spec: dict = RECOVERY_SPEC) -> dict:
    return {
        name: {
            "predictions": [], "selections": [], "positions": [], "contents": [],
            "api": [], "margins": [],
        }
        for name in spec["arms"]
    }


def _extend_records(target: dict, source: dict) -> None:
    for name, fields in source.items():
        for field, values in fields.items():
            target[name][field].extend(values)


def _metrics(records: dict, expected: torch.Tensor, spec: dict = RECOVERY_SPEC) -> dict:
    return {
        name: _arm_metrics(
            expected,
            row["predictions"], row["selections"], row["positions"],
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }


def _similarities(keys, query, projector) -> list[float]:
    with torch.no_grad():
        query_address = projector.address(query.detach().float().mean(0))
        key_addresses = [
            projector.address(key.detach().float().mean(0)) for key in keys
        ]
    return [
        float(F.cosine_similarity(query_address, address, dim=0))
        for address in key_addresses
    ]


def _geometry(rows: list[dict]) -> dict:
    rank_counts = [0, 0, 0]
    selection_confusion = [[0, 0, 0] for _ in range(3)]
    for row in rows:
        ordered = sorted(range(3), key=lambda index: row["similarities"][index], reverse=True)
        rank_counts[ordered.index(row["target_position"])] += 1
        selection_confusion[row["target_position"]][row["selected_position"]] += 1
    return {
        "episodes": len(rows),
        "target_rank_counts": rank_counts,
        "selection_position_confusion": selection_confusion,
        "target_similarity_mean": sum(row["target_similarity"] for row in rows) / len(rows),
        "strongest_wrong_similarity_mean": sum(
            row["strongest_wrong_similarity"] for row in rows
        ) / len(rows),
        "third_candidate_similarity_mean": sum(
            row["third_candidate_similarity"] for row in rows
        ) / len(rows),
        "target_minus_strongest_wrong_mean": sum(
            row["target_minus_strongest_wrong"] for row in rows
        ) / len(rows),
        "target_minus_third_candidate_mean": sum(
            row["target_minus_third_candidate"] for row in rows
        ) / len(rows),
    }


def _run_replicate(seed: int, replicate: int, delay: int, episodes, projector, prototypes,
                   spec: dict = RECOVERY_SPEC) -> tuple[dict, dict, list[dict]]:
    records = _records(spec)
    geometry_rows = []
    call_rows = {name: [] for name in spec["stable_arms"]}
    widths, cell_counts, episode_seeds = [], [], []
    base = (
        spec["episode_seed_base"]
        + seed * spec["seed_stride"]
        + replicate * spec["replicate_seed_stride"]
    )
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = _trace(episode, trial_seed, 3, delay, spec)
        cell_counts.extend(trace["cell_counts"])
        exact, exact_query = _exact_addresses(episode, spec)
        stable_three = _integrated_memory_prediction(
            trace["keys"], trace["values"], trace["query"], prototypes, projector
        )
        outcomes = {
            "stable_three_candidates": stable_three,
            "stable_two_candidates": _integrated_memory_prediction(
                trace["keys"][:2], trace["values"][:2], trace["query"], prototypes, projector
            ),
            "exact_three_candidates": _direct_prediction(
                exact, trace["values"], exact_query, prototypes
            ),
            "exact_three_partner_swap": _direct_prediction(
                exact, trace["values"], exact_query, prototypes, rotate=True
            ),
            "exact_three_recovered": _direct_prediction(
                exact, trace["values"], exact_query, prototypes
            ),
        }
        content = _decode(trace["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["positions"].append(episode.query_position)
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
            if name in call_rows:
                call_rows[name].append(outcome[4])
                widths.append(outcome[5])
        similarities = _similarities(trace["keys"], trace["query"], projector)
        target = episode.query_position
        wrong = [value for index, value in enumerate(similarities) if index != target]
        geometry_rows.append({
            "similarities": similarities,
            "target_position": target,
            "selected_position": stable_three[1],
            "target_similarity": similarities[target],
            "strongest_wrong_similarity": max(wrong),
            "third_candidate_similarity": similarities[2],
            "target_minus_strongest_wrong": similarities[target] - max(wrong),
            "target_minus_third_candidate": similarities[target] - similarities[2],
        })
        if (index + 1) % 256 == 0:
            print(
                f"[seed {seed} replicate {replicate} delay {delay}] "
                f"evaluated {index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    expected = torch.tensor([episode.target for episode in episodes])
    metrics = _metrics(records, expected, spec)
    metrics["exact_three_recovered"]["prediction_match"] = float(
        records["exact_three_recovered"]["predictions"]
        == records["exact_three_candidates"]["predictions"]
    )
    return {
        "replicate": replicate,
        "arms": metrics,
        "geometry": _geometry(geometry_rows),
        "integration_audit": {
            "stable_transform_calls": {
                name: {
                    "episodes": len(values), "total": sum(values),
                    "minimum": min(values), "maximum": max(values),
                }
                for name, values in call_rows.items()
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
    }, records, geometry_rows


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = RECOVERY_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    delay_rows = []
    for delay in spec["distractor_steps"]:
        pooled_records = _records(spec)
        pooled_geometry = []
        replicate_rows = []
        for replicate in spec["replicates"]:
            public, records, geometry = _run_replicate(
                seed, replicate, delay, episode_sets[replicate], projector, prototypes, spec
            )
            replicate_rows.append(public)
            _extend_records(pooled_records, records)
            pooled_geometry.extend(geometry)
        pooled_expected = torch.tensor([
            episode.target
            for replicate in spec["replicates"]
            for episode in episode_sets[replicate]
        ])
        pooled_metrics = _metrics(pooled_records, pooled_expected, spec)
        pooled_metrics["exact_three_recovered"]["prediction_match"] = float(
            pooled_records["exact_three_recovered"]["predictions"]
            == pooled_records["exact_three_candidates"]["predictions"]
        )
        delay_rows.append({
            "distractor_steps": delay,
            "pooled": {"arms": pooled_metrics, "geometry": _geometry(pooled_geometry)},
            "replicates": replicate_rows,
        })
    after = projector.state_dict()
    return {
        "seed": seed,
        "delays": delay_rows,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/recovery_results.json")
    parser.add_argument("--verdict", default="measurement/recovery_verdict.json")
    args = parser.parse_args()
    spec = RECOVERY_SPEC
    _, source = _source_receipt(spec)
    episode_sets = {
        replicate: build_recovery_episodes(replicate, spec)
        for replicate in spec["replicates"]
    }
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": recovery_dataset_audit(episode_sets, spec),
        "source_decay": source,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.recovery_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
