#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.memory_write_gate_registry import MEMORY_WRITE_GATE_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.memory_write_gate_registry import MEMORY_WRITE_GATE_SPEC, spec_sha256


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def adjudicate(payload: dict, spec: dict = MEMORY_WRITE_GATE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "G0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or not _finite(payload)
    ):
        return invalid("experiment, registered spec, digest, or finite-value check failed")
    rows = payload.get("seeds", [])
    if [row.get("seed") for row in rows] != spec["seeds"]:
        return invalid("registered seed roster changed")

    passed = True
    summaries = []
    thresholds = spec["thresholds"]
    for row in rows:
        audit = row.get("dataset_audit", {})
        if (
            audit.get("calibration_rows") != spec["calibration_rows"]
            or audit.get("calibration_unique") != spec["calibration_rows"]
            or audit.get("calibration_positive") != spec["calibration_rows"] // 2
            or audit.get("calibration_negative") != spec["calibration_rows"] // 2
            or audit.get("evaluation_episodes") != spec["evaluation_episodes"]
            or audit.get("evaluation_candidates") != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or audit.get("overlap") != 0
            or set(audit.get("fact_counts", {})) != set(spec["fact_kinds"])
            or any(count != spec["evaluation_episodes"] // len(spec["fact_kinds"])
                   for count in audit.get("fact_counts", {}).values())
            or any(len(audit.get(name, "")) != 64 for name in ("calibration_sha256", "evaluation_sha256"))
        ):
            return invalid(f"seed {row.get('seed')} dataset audit changed")
        fit = row.get("fit_audit", {})
        fake_fit = row.get("shuffled_fit_audit", {})
        if (
            fit.get("method") != spec["fit_method"]
            or fit.get("examples") != spec["calibration_rows"]
            or fit.get("feature_dim") != spec["feature_dim"]
            or fake_fit.get("method") != spec["fit_method"]
            or fake_fit.get("examples") != spec["calibration_rows"]
            or fake_fit.get("feature_dim") != spec["feature_dim"]
        ):
            return invalid(f"seed {row.get('seed')} fit audit changed")
        checkpoint = row.get("checkpoint", {})
        path = Path(checkpoint.get("path", ""))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint.get("sha256"):
            return invalid(f"seed {row.get('seed')} checkpoint changed")
        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {row.get('seed')} arm roster changed")
        for name, metrics in arms.items():
            if (
                set(metrics) != {
                    "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
                    "recall_at_3", "stored", "per_kind_recall", "records_sha256",
                }
                or set(metrics["per_kind_recall"]) != set(spec["fact_kinds"])
                or len(metrics["records_sha256"]) != 64
                or not 0 <= metrics["stored"] <= spec["evaluation_episodes"] * spec["candidates_per_episode"]
            ):
                return invalid(f"seed {row.get('seed')} arm {name} shape changed")
        selective = arms["selective_gate"]
        all_rows = arms["store_all"]
        oracle = arms["oracle_gate"]
        random_arm = arms["matched_random"]
        shuffled = arms["shuffled_gate"]
        none = arms["no_memory"]
        if (
            oracle["important_storage_rate"] < thresholds["oracle_important_storage_rate"]
            or oracle["recall_at_3"] < thresholds["oracle_recall_at_3"]
            or all_rows["recall_at_3"] < thresholds["store_all_recall_at_3"]
            or none["recall_at_3"] > thresholds["no_memory_max_recall_at_3"]
            or selective["recall_at_3"] - random_arm["recall_at_3"] < thresholds["minimum_fake_recall_gap"]
            or selective["recall_at_3"] - shuffled["recall_at_3"] < thresholds["minimum_fake_recall_gap"]
        ):
            return invalid(f"seed {row.get('seed')} positive, fake, or no-memory control failed")
        seed_pass = (
            selective["important_storage_rate"] >= thresholds["important_storage_rate"]
            and selective["recall_at_3"] >= thresholds["recall_at_3"]
            and selective["distractor_storage_rate"] <= thresholds["maximum_distractor_storage_rate"]
            and selective["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            and all_rows["recall_at_3"] - selective["recall_at_3"]
            <= thresholds["maximum_recall_drop_from_all"]
        )
        passed = passed and seed_pass
        summaries.append({
            "seed": row["seed"],
            "passed": seed_pass,
            "selective": selective,
            "store_all_recall_at_3": all_rows["recall_at_3"],
            "matched_random_recall_at_3": random_arm["recall_at_3"],
            "shuffled_recall_at_3": shuffled["recall_at_3"],
        })
    return {
        "experiment": spec["experiment"],
        "verdict": "G1_SELECTIVE_WRITE_VALID_NOT_UNIQUE" if passed else "G2_SELECTION_OR_RECALL_LOSS",
        "reason": (
            "the frozen write gate retained useful facts while shrinking the search index"
            if passed else "the controlled write gate missed a registered selection or recall threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("measurement/memory_write_gate_verdict.json"))
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
