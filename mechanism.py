#!/usr/bin/env python3
"""MECHANISM-1: ablate one active QuantumC settling component at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from episode2 import _load_frozen_projector
from graft_behavior import sha256_file
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.mechanism_registry import MECHANISM_SPEC, spec_sha256
from measurement.settle_registry import SETTLE_SPEC, spec_sha256 as settle_spec_sha256
from recovery import _extend_records, _geometry, _metrics, _records
from reset_experiment import _run_replicate, build_reset_episodes, reset_dataset_audit
from settle import _paired


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = MECHANISM_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = settle_spec_sha256(SETTLE_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != SETTLE_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered SETTLE-1 source is not the causal result")
    inherited = results["source_reset"]
    for receipts in (inherited["checkpoints"], inherited["prototype_checkpoints"]):
        for seed, receipt in receipts.items():
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"SETTLE-1 inherited checkpoint changed for seed {seed}")
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": inherited["checkpoints"],
        "prototype_checkpoints": inherited["prototype_checkpoints"],
    }


def _source_overlap(episode_sets: dict[int, list]) -> dict:
    old_sets = {
        replicate: {row.fingerprint() for row in build_reset_episodes(replicate, SETTLE_SPEC)}
        for replicate in SETTLE_SPEC["replicates"]
    }
    new_sets = {
        replicate: {row.fingerprint() for row in rows}
        for replicate, rows in episode_sets.items()
    }
    return {
        f"settle-{old}:mechanism-{new}": len(old_values & new_values)
        for old, old_values in old_sets.items()
        for new, new_values in new_sets.items()
    }


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = MECHANISM_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = [
        episode.target for replicate in spec["replicates"]
        for episode in episode_sets[replicate]
    ]
    positions = [
        episode.query_position for replicate in spec["replicates"]
        for episode in episode_sets[replicate]
    ]
    public_arms, private_arms = [], {}
    for intervention in spec["interventions"]:
        arm_spec = dict(spec)
        arm_spec["dynamics_ablation"] = intervention["disabled"]
        pooled_records = _records(spec)
        pooled_geometry, replicate_rows, replicate_records = [], [], {}
        for replicate in spec["replicates"]:
            public, records, geometry = _run_replicate(
                seed, replicate, intervention["mode"], spec["update_steps"][-1],
                episode_sets[replicate], projector, prototypes, arm_spec,
            )
            replicate_rows.append(public)
            replicate_records[replicate] = records
            _extend_records(pooled_records, records)
            pooled_geometry.extend(geometry)
        pooled_metrics = _metrics(pooled_records, torch.tensor(expected), spec)
        pooled_metrics["exact_three_recovered"]["prediction_match"] = float(
            pooled_records["exact_three_recovered"]["predictions"]
            == pooled_records["exact_three_candidates"]["predictions"]
        )
        public_arms.append({
            "name": intervention["name"],
            "mode": intervention["mode"],
            "disabled": intervention["disabled"],
            "pooled": {"arms": pooled_metrics, "geometry": _geometry(pooled_geometry)},
            "replicates": replicate_rows,
        })
        private_arms[intervention["name"]] = {
            "pooled": pooled_records,
            "replicates": replicate_records,
        }
    intact = private_arms["intact"]
    comparisons = []
    for intervention in spec["interventions"]:
        name = intervention["name"]
        if name == "intact":
            continue
        arm = private_arms[name]
        comparisons.append({
            "name": name,
            "final": _paired(
                intact["pooled"]["stable_three_candidates"]["predictions"],
                arm["pooled"]["stable_three_candidates"]["predictions"], expected,
            ),
            "selection": _paired(
                intact["pooled"]["stable_three_candidates"]["selections"],
                arm["pooled"]["stable_three_candidates"]["selections"], positions,
            ),
            "replicates": [{
                "replicate": replicate,
                "final": _paired(
                    intact["replicates"][replicate]["stable_three_candidates"]["predictions"],
                    arm["replicates"][replicate]["stable_three_candidates"]["predictions"],
                    [episode.target for episode in episode_sets[replicate]],
                ),
                "selection": _paired(
                    intact["replicates"][replicate]["stable_three_candidates"]["selections"],
                    arm["replicates"][replicate]["stable_three_candidates"]["selections"],
                    [episode.query_position for episode in episode_sets[replicate]],
                ),
            } for replicate in spec["replicates"]],
        })
    after = projector.state_dict()
    return {
        "seed": seed,
        "interventions": public_arms,
        "comparisons": comparisons,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/mechanism_results.json")
    parser.add_argument("--verdict", default="measurement/mechanism_verdict.json")
    args = parser.parse_args()
    spec = MECHANISM_SPEC
    _, source = _source_receipt(spec)
    episode_sets = {
        replicate: build_reset_episodes(replicate, spec) for replicate in spec["replicates"]
    }
    audit = reset_dataset_audit(episode_sets, spec)
    audit["source_overlap"] = _source_overlap(episode_sets)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": audit,
        "source_settle": source,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.mechanism_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
