#!/usr/bin/env python3
"""SEEDMAP-1: cross projector, value-prototype, and engine seed factors."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from capacity import build_capacity_episodes, count_spec
from capacity2 import _run_condition_count
from episode2 import _load_frozen_projector
from graft_behavior import sha256_file
from measurement.capacity2_registry import CAPACITY2_SPEC, spec_sha256 as capacity2_spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.seedmap_registry import SEEDMAP_SPEC, combination_name, spec_sha256
from separation import dataset_audit
from settle import _paired


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = SEEDMAP_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = capacity2_spec_sha256(CAPACITY2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CAPACITY2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered CAPACITY-2 source is not the seed-conditional result")
    inherited = results["source_capacity"]
    for receipts in (inherited["checkpoints"], inherited["prototype_checkpoints"]):
        for seed, receipt in receipts.items():
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"CAPACITY-2 inherited checkpoint changed for seed {seed}")
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": inherited["checkpoints"],
        "prototype_checkpoints": inherited["prototype_checkpoints"],
    }


def _single_swap_comparisons(private: dict, episodes, spec: dict = SEEDMAP_SPEC) -> list[dict]:
    low, high = spec["factor_seeds"]
    expected = [episode.target for episode in episodes]
    positions = [episode.query_position for episode in episodes]
    rows = []
    for factor in spec["factors"]:
        rescue = {name: low for name in spec["factors"]}
        rescue[factor] = high
        reverse = {name: high for name in spec["factors"]}
        reverse[factor] = low
        low_native = {name: low for name in spec["factors"]}
        high_native = {name: high for name in spec["factors"]}
        rescue_records = private[combination_name(rescue)]["stable_distinct_normal"]
        low_records = private[combination_name(low_native)]["stable_distinct_normal"]
        high_records = private[combination_name(high_native)]["stable_distinct_normal"]
        reverse_records = private[combination_name(reverse)]["stable_distinct_normal"]
        secondary = "contents" if factor == "prototype_seed" else "selections"
        secondary_expected = expected if factor == "prototype_seed" else positions
        rows.append({
            "factor": factor,
            "rescue_combination": combination_name(rescue),
            "reverse_combination": combination_name(reverse),
            "rescue": {
                "final": _paired(rescue_records["predictions"], low_records["predictions"], expected),
                "secondary": _paired(
                    rescue_records[secondary], low_records[secondary], secondary_expected
                ),
            },
            "reverse": {
                "final": _paired(high_records["predictions"], reverse_records["predictions"], expected),
                "secondary": _paired(
                    high_records[secondary], reverse_records[secondary], secondary_expected
                ),
            },
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/seedmap_results.json")
    parser.add_argument("--verdict", default="measurement/seedmap_verdict.json")
    args = parser.parse_args()
    spec = SEEDMAP_SPEC
    _, source = _source_receipt(spec)
    episodes = build_capacity_episodes(spec["event_count"])
    projectors, projector_before, prototypes = {}, {}, {}
    for seed in spec["factor_seeds"]:
        projectors[seed] = _load_frozen_projector(
            seed, source["checkpoints"][str(seed)], EPISODE2_SPEC
        )
        projector_before[seed] = {
            name: value.detach().clone()
            for name, value in projectors[seed].state_dict().items()
        }
        checkpoint = torch.load(
            source["prototype_checkpoints"][str(seed)]["path"],
            map_location="cpu",
            weights_only=True,
        )
        prototypes[seed] = checkpoint["prototypes"]["quantum"]
    condition = {"name": "settled", "updates": spec["settling_updates"], "disabled": []}
    public, private = [], {}
    for combination in spec["combinations"]:
        name = combination_name(combination)
        result, records = _run_condition_count(
            combination["engine_seed"],
            spec["event_count"],
            episodes,
            condition,
            projectors[combination["projector_seed"]],
            prototypes[combination["prototype_seed"]],
            spec,
        )
        public.append({"name": name, **combination, "result": result})
        private[name] = records
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(
            episodes, count_spec(spec["event_count"])
        ),
        "source_capacity2": source,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "combinations": public,
        "comparisons": _single_swap_comparisons(private, episodes, spec),
        "projectors_frozen": {
            str(seed): not any(parameter.requires_grad for parameter in projectors[seed].parameters())
            for seed in spec["factor_seeds"]
        },
        "projectors_unchanged": {
            str(seed): all(
                torch.equal(projector_before[seed][name], value)
                for name, value in projectors[seed].state_dict().items()
            )
            for seed in spec["factor_seeds"]
        },
        "projector_checkpoints": source["checkpoints"],
        "prototype_checkpoints": source["prototype_checkpoints"],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.seedmap_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
