#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-WRITE-CONTROL-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from measurement.balanced_natural_write_registry import (
    BALANCED_NATURAL_WRITE_SPEC,
    spec_sha256,
)


def _invalid(reason: str, spec: dict) -> dict:
    return {
        "experiment": spec["experiment"],
        "verdict": "GWC0_INVALID",
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }


def _rate_map(value: object, keys: list[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(keys)
        and all(
            isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1
            for rate in value.values()
        )
    )


def _checkpoint_valid(receipt: object) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
        return False
    path = Path(receipt["path"])
    return (
        isinstance(receipt["sha256"], str)
        and len(receipt["sha256"]) == 64
        and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    )


def adjudicate(
    payload: dict, spec: dict = BALANCED_NATURAL_WRITE_SPEC
) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") == "__PREREGISTRATION_COMMIT__"
    ):
        return _invalid("registration mismatch", spec)
    runtime = payload.get("runtime", {})
    if any(runtime.get(name) != spec["runtime"][name] for name in ("torch", "transformers")):
        return _invalid("runtime mismatch", spec)
    rows = payload.get("seeds")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != spec["seeds"]:
        return _invalid("seed roster mismatch", spec)

    expected_calibration_templates = len(spec["subject_heads"][spec["fact_kinds"][0]]) * len(
        spec["lexicons"]["calibration"]["subject_qualifiers"]
    ) * len(spec["lexicons"]["calibration"]["values"][spec["fact_kinds"][0]])
    expected_evaluation_templates = (
        len(spec["subject_heads"][spec["fact_kinds"][0]])
        * len(spec["lexicons"]["evaluation"][spec["replicates"][0]]["subject_qualifiers"])
        * len(spec["lexicons"]["evaluation"][spec["replicates"][0]]["values"][spec["fact_kinds"][0]])
    )
    template_keys = [
        f"{kind}:{index}"
        for kind in spec["fact_kinds"]
        for index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
    ]
    thresholds = spec["thresholds"]
    summaries = []
    normal_pass = True
    for row in rows:
        seed = row["seed"]
        audit = row.get("dataset_audit", {})
        if (
            audit.get("calibration_rows") != spec["calibration_rows"]
            or audit.get("calibration_unique") != spec["calibration_rows"]
            or audit.get("calibration_positive") != spec["calibration_rows"] // 2
            or audit.get("calibration_negative") != spec["calibration_rows"] // 2
            or set(audit.get("calibration_template_counts", {})) != set(template_keys)
            or any(
                count != expected_calibration_templates
                for count in audit["calibration_template_counts"].values()
            )
            or audit.get("synthetic_token_count") != 0
            or set(audit.get("evaluation_episodes", {})) != set(spec["replicates"])
            or any(value != spec["evaluation_episodes"] for value in audit["evaluation_episodes"].values())
            or any(
                value != spec["evaluation_episodes"] * spec["candidates_per_episode"]
                for value in audit.get("evaluation_candidates", {}).values()
            )
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or any(audit.get("calibration_evaluation_overlap", {}).values())
            or any(audit.get("cross_replicate_overlap", {}).values())
            or len(audit.get("calibration_sha256", "")) != 64
            or any(len(value) != 64 for value in audit.get("evaluation_sha256", {}).values())
        ):
            return _invalid(f"seed {seed} dataset audit failed", spec)
        evaluation_templates = audit.get("evaluation_fact_template_counts", {})
        if set(evaluation_templates) != set(spec["replicates"]):
            return _invalid(f"seed {seed} evaluation replicate audit changed", spec)
        for replicate in spec["replicates"]:
            if (
                set(evaluation_templates[replicate]) != set(template_keys)
                or any(
                    count != expected_evaluation_templates
                    for count in evaluation_templates[replicate].values()
                )
            ):
                return _invalid(f"seed {seed} replicate {replicate} template balance failed", spec)
        if not all(_checkpoint_valid(value) for value in row.get("checkpoints", {}).values()):
            return _invalid(f"seed {seed} checkpoint receipt failed", spec)
        threshold = row.get("selection_threshold")
        if (
            not isinstance(threshold, (int, float))
            or not math.isclose(
                threshold, thresholds["expected_selection_threshold"], rel_tol=0.0,
                abs_tol=thresholds["selection_threshold_tolerance"],
            )
        ):
            return _invalid(f"seed {seed} selection threshold changed", spec)
        replicates = row.get("replicates")
        if not isinstance(replicates, list) or [item.get("name") for item in replicates] != spec["replicates"]:
            return _invalid(f"seed {seed} replicate roster changed", spec)
        replicate_summaries = {}
        for replicate_row in replicates:
            replicate = replicate_row["name"]
            matching = replicate_row.get("matching_audit", {})
            counts = [
                matching.get("semantic_counts"),
                matching.get("matched_shuffled_counts"),
                matching.get("matched_random_counts"),
            ]
            if (
                any(
                    not isinstance(values, list)
                    or len(values) != spec["evaluation_episodes"]
                    or any(type(value) is not int or not 0 <= value <= spec["candidates_per_episode"] for value in values)
                    for values in counts
                )
                or counts[0] != counts[1]
                or counts[0] != counts[2]
                or any(
                    len(matching.get(name, "")) != 64
                    for name in (
                        "semantic_selection_sha256", "matched_shuffled_selection_sha256",
                        "matched_random_selection_sha256",
                    )
                )
            ):
                return _invalid(f"seed {seed} replicate {replicate} matching audit failed", spec)
            arms = replicate_row.get("arms", {})
            if set(arms) != {"semantic_gate", "matched_shuffled_gate", "matched_random"}:
                return _invalid(f"seed {seed} replicate {replicate} arm roster changed", spec)
            for arm_name, metrics in arms.items():
                if (
                    any(
                        not isinstance(metrics.get(name), (int, float))
                        or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1
                        for name in (
                            "important_storage_rate", "distractor_storage_rate", "search_size_ratio"
                        )
                    )
                    or type(metrics.get("stored")) is not int
                    or not _rate_map(metrics.get("per_kind_storage_rate"), spec["fact_kinds"])
                    or not _rate_map(metrics.get("per_template_storage_rate"), template_keys)
                    or not _rate_map(metrics.get("per_distractor_storage_rate"), spec["distractor_kinds"])
                    or len(metrics.get("selection_sha256", "")) != 64
                    or len(metrics.get("scores_sha256", "")) != 64
                    or metrics["stored"] != sum(counts[0 if arm_name == "semantic_gate" else 1 if arm_name == "matched_shuffled_gate" else 2])
                ):
                    return _invalid(f"seed {seed} replicate {replicate} arm {arm_name} metrics failed", spec)
            normal = arms["semantic_gate"]
            fake_max = max(
                arms["matched_shuffled_gate"]["important_storage_rate"],
                arms["matched_random"]["important_storage_rate"],
            )
            if (
                fake_max > thresholds["maximum_fake_important_storage_rate"]
                or normal["important_storage_rate"] - fake_max < thresholds["minimum_fake_storage_gap"]
            ):
                return _invalid(f"seed {seed} replicate {replicate} fake control failed", spec)
            passed = (
                normal["important_storage_rate"] >= thresholds["minimum_important_storage_rate"]
                and min(normal["per_kind_storage_rate"].values()) >= thresholds["minimum_per_kind_storage_rate"]
                and min(normal["per_template_storage_rate"].values()) >= thresholds["minimum_per_template_storage_rate"]
                and normal["distractor_storage_rate"] <= thresholds["maximum_distractor_storage_rate"]
                and max(normal["per_distractor_storage_rate"].values()) <= thresholds["maximum_per_distractor_storage_rate"]
                and normal["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            )
            normal_pass &= passed
            replicate_summaries[replicate] = {
                "passed": passed,
                "important_storage_rate": normal["important_storage_rate"],
                "minimum_kind_storage_rate": min(normal["per_kind_storage_rate"].values()),
                "minimum_template_storage_rate": min(normal["per_template_storage_rate"].values()),
                "distractor_storage_rate": normal["distractor_storage_rate"],
                "maximum_fake_important_storage_rate": fake_max,
            }
        summaries.append({"seed": seed, "replicates": replicate_summaries})
    return {
        "experiment": spec["experiment"],
        "verdict": (
            "GWC1_BALANCED_NATURAL_WRITE_VALID"
            if normal_pass else "GWC2_WRITE_SELECTION_LOSS"
        ),
        "reason": (
            "balanced templates and held-out natural words passed every registered write-selection threshold"
            if normal_pass else "at least one balanced template or natural-word replicate missed a registered write-selection threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", type=Path,
        default=Path("measurement/balanced_natural_write_results.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/balanced_natural_write_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(verdict["verdict"])


if __name__ == "__main__":
    main()
