#!/usr/bin/env python3
"""SETTLE-1: compare autonomous state evolution with a truly frozen state."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import torch

from episode2 import _load_frozen_projector
from graft_behavior import sha256_file
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.reset_registry import RESET_SPEC, spec_sha256 as reset_spec_sha256
from measurement.settle_registry import SETTLE_SPEC, spec_sha256
from recovery import _extend_records, _geometry, _metrics, _records
from reset_experiment import (
    _run_replicate,
    build_reset_episodes,
    reset_dataset_audit,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = SETTLE_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = reset_spec_sha256(RESET_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != RESET_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered RESET-1 source is not the mixed-mechanism result")
    recovery = results["source_recovery"]
    for receipts in (recovery["checkpoints"], recovery["prototype_checkpoints"]):
        for seed, receipt in receipts.items():
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"RESET-1 inherited checkpoint changed for seed {seed}")
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": recovery["checkpoints"],
        "prototype_checkpoints": recovery["prototype_checkpoints"],
    }


def _exact_p(to_correct: int, to_wrong: int) -> float:
    discordant = to_correct + to_wrong
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(to_correct, to_wrong) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _paired(values_a: list[int], values_b: list[int], expected: list[int]) -> dict:
    if not (len(values_a) == len(values_b) == len(expected)):
        raise RuntimeError("paired SETTLE-1 vectors changed length")
    both_correct = both_wrong = to_correct = to_wrong = 0
    for active, frozen, target in zip(values_a, values_b, expected):
        active_ok, frozen_ok = active == target, frozen == target
        if active_ok and frozen_ok:
            both_correct += 1
        elif active_ok:
            to_correct += 1
        elif frozen_ok:
            to_wrong += 1
        else:
            both_wrong += 1
    return {
        "episodes": len(expected),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "autonomous_only_correct": to_correct,
        "frozen_only_correct": to_wrong,
        "net_accuracy_delta": (to_correct - to_wrong) / len(expected),
        "exact_two_sided_p": _exact_p(to_correct, to_wrong),
    }


def _source_overlap(episode_sets: dict[int, list]) -> dict:
    old_sets = {
        replicate: {row.fingerprint() for row in build_reset_episodes(replicate, RESET_SPEC)}
        for replicate in RESET_SPEC["replicates"]
    }
    new_sets = {
        replicate: {row.fingerprint() for row in rows}
        for replicate, rows in episode_sets.items()
    }
    return {
        f"reset-{old}:settle-{new}": len(old_values & new_values)
        for old, old_values in old_sets.items()
        for new, new_values in new_sets.items()
    }


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = SETTLE_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    mode_rows = {mode: [] for mode in spec["update_modes"]}
    paired_rows = []
    expected = [
        episode.target for replicate in spec["replicates"]
        for episode in episode_sets[replicate]
    ]
    for updates in spec["update_steps"]:
        pooled_by_mode = {}
        replicate_public = {}
        replicate_records = {}
        for mode in spec["update_modes"]:
            pooled_records = _records(spec)
            pooled_geometry, public_rows, private_rows = [], [], {}
            for replicate in spec["replicates"]:
                public, records, geometry = _run_replicate(
                    seed, replicate, mode, updates, episode_sets[replicate],
                    projector, prototypes, spec,
                )
                public_rows.append(public)
                private_rows[replicate] = records
                _extend_records(pooled_records, records)
                pooled_geometry.extend(geometry)
            pooled_expected = torch.tensor(expected)
            pooled_metrics = _metrics(pooled_records, pooled_expected, spec)
            pooled_metrics["exact_three_recovered"]["prediction_match"] = float(
                pooled_records["exact_three_recovered"]["predictions"]
                == pooled_records["exact_three_candidates"]["predictions"]
            )
            mode_rows[mode].append({
                "update_steps": updates,
                "pooled": {"arms": pooled_metrics, "geometry": _geometry(pooled_geometry)},
                "replicates": public_rows,
            })
            pooled_by_mode[mode] = pooled_records
            replicate_public[mode] = {row["replicate"]: row for row in public_rows}
            replicate_records[mode] = private_rows
        active = pooled_by_mode["autonomous"]["stable_three_candidates"]
        frozen = pooled_by_mode["frozen"]["stable_three_candidates"]
        paired_rows.append({
            "update_steps": updates,
            "final": _paired(active["predictions"], frozen["predictions"], expected),
            "selection": _paired(active["selections"], frozen["selections"], [
                episode.query_position for replicate in spec["replicates"]
                for episode in episode_sets[replicate]
            ]),
            "replicates": [{
                "replicate": replicate,
                "final": _paired(
                    replicate_records["autonomous"][replicate]["stable_three_candidates"]["predictions"],
                    replicate_records["frozen"][replicate]["stable_three_candidates"]["predictions"],
                    [episode.target for episode in episode_sets[replicate]],
                ),
                "selection": _paired(
                    replicate_records["autonomous"][replicate]["stable_three_candidates"]["selections"],
                    replicate_records["frozen"][replicate]["stable_three_candidates"]["selections"],
                    [episode.query_position for episode in episode_sets[replicate]],
                ),
            } for replicate in spec["replicates"]],
        })
    after = projector.state_dict()
    return {
        "seed": seed,
        "modes": [{"mode": mode, "updates": mode_rows[mode]} for mode in spec["update_modes"]],
        "paired": paired_rows,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/settle_results.json")
    parser.add_argument("--verdict", default="measurement/settle_verdict.json")
    args = parser.parse_args()
    spec = SETTLE_SPEC
    _, source = _source_receipt(spec)
    episode_sets = {
        replicate: build_reset_episodes(replicate, spec) for replicate in spec["replicates"]
    }
    audit = reset_dataset_audit(episode_sets, spec)
    audit["source_overlap"] = _source_overlap(episode_sets)
    payload = {
        "experiment": spec["experiment"], "spec": spec, "spec_sha256": spec_sha256(spec),
        "dataset_audit": audit, "source_reset": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.settle_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
