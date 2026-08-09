#!/usr/bin/env python3
"""Fail-closed adjudication for CONTROL-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.episode_control_registry import CONTROL_SPEC, spec_sha256
except ModuleNotFoundError:
    from episode_control_registry import CONTROL_SPEC, spec_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _balanced(counts: dict, expected_categories: int, maximum_delta: int) -> bool:
    if set(counts) != {str(index) for index in range(expected_categories)}:
        return False
    values = list(counts.values())
    return all(isinstance(value, int) and value >= 0 for value in values) and (
        max(values) - min(values) <= maximum_delta
    )


def adjudicate(payload: dict) -> dict:
    spec = CONTROL_SPEC
    invalid = lambda reason: {
        "experiment": spec["experiment"], "verdict": "P0_INVALID", "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }
    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")
        audit = payload["dataset_audit"]
        if any(value != 0 for value in audit["overlap"].values()):
            return invalid("a dataset fingerprint appears in more than one split")
        maximum_delta = spec["thresholds"]["maximum_balance_delta"]
        for split, expected in spec["splits"].items():
            row = audit["splits"][split]
            if row["episodes"] != expected or row["unique_fingerprints"] != expected:
                return invalid(f"{split} contains missing or repeated episodes")
            if not _balanced(row["target_counts"], spec["values"], maximum_delta):
                return invalid(f"{split} targets are not balanced")
            if not _balanced(row["query_key_counts"], spec["keys"], maximum_delta):
                return invalid(f"{split} query keys are not balanced")
            if not _balanced(
                row["query_position_counts"], spec["relations_per_episode"], maximum_delta
            ):
                return invalid(f"{split} query positions are not balanced")
        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")
        judged = {}
        for seed in spec["seeds"]:
            row = rows[seed]
            if set(row["arms"]) != set(spec["arms"]):
                return invalid(f"seed {seed} arm roster changed")
            receipt = row["checkpoint"]
            path = Path(receipt["path"])
            if len(receipt["sha256"]) != 64 or not path.is_file():
                return invalid(f"seed {seed} checkpoint is missing")
            if _sha256_file(path) != receipt["sha256"]:
                return invalid(f"seed {seed} checkpoint SHA-256 changed")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("experiment") != spec["experiment"]
                or checkpoint.get("spec_sha256") != spec_sha256(spec)
                or checkpoint.get("seed") != seed
                or checkpoint.get("selected_step") != row["selected_step"]
            ):
                return invalid(f"seed {seed} checkpoint identity changed")
            arms = row["arms"]
            if arms["vector_memory"]["accuracy"] != spec["thresholds"]["vector_memory_accuracy"]:
                return invalid(f"seed {seed} exact memory control failed")
            ceiling = spec["thresholds"]["negative_control_max_accuracy"]
            if arms["no_memory"]["accuracy"] > ceiling:
                return invalid(f"seed {seed} no-memory control exceeded its ceiling")
            if arms["shuffled_labels"]["accuracy"] > ceiling:
                return invalid(f"seed {seed} shuffled-label control exceeded its ceiling")
            for arm in spec["arms"]:
                metrics = arms[arm]
                if (
                    len(metrics["confusion_matrix"]) != spec["values"]
                    or any(len(matrix_row) != spec["values"]
                           for matrix_row in metrics["confusion_matrix"])
                ):
                    return invalid(f"seed {seed} {arm} confusion matrix is incomplete")
                if len(metrics["per_value_recall"]) != spec["values"]:
                    return invalid(f"seed {seed} {arm} recall vector is incomplete")
            gru = arms["gru"]
            passed = (
                gru["accuracy"] >= spec["thresholds"]["gru_accuracy"]
                and min(gru["per_value_recall"]) >= spec["thresholds"]["gru_min_value_recall"]
            )
            judged[str(seed)] = {
                "passed": passed,
                "selected_step": row["selected_step"],
                "validation_accuracy": row["validation_accuracy"],
                "arms": arms,
                "checkpoint": receipt,
            }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return invalid(str(exc))
    passed = all(row["passed"] for row in judged.values())
    return {
        "experiment": spec["experiment"],
        "verdict": "P1_POSITIVE_CONTROL_VALID" if passed else "P2_TRAINING_PATH_INVALID",
        "reason": (
            "the standard GRU learned dynamic relations in both registered seeds"
            if passed else "the exact memory control passed but the standard GRU did not"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    args = parser.parse_args()
    verdict = adjudicate(json.loads(Path(args.results).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
