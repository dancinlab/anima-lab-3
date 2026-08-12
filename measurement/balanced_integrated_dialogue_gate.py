#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-4."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from measurement.balanced_integrated_dialogue_registry import (
    BALANCED_INTEGRATED_DIALOGUE_SPEC,
    spec_sha256,
)
from measurement.balanced_natural_write_gate import _checkpoint_valid
from measurement.balanced_natural_write_registry import (
    BALANCED_NATURAL_WRITE_SPEC,
    spec_sha256 as write_spec_sha256,
)
from measurement.content_swap_retrieval_control_registry import (
    spec_sha256 as retrieval_spec_sha256,
)


def _invalid(reason: str, spec: dict) -> dict:
    return {
        "experiment": spec["experiment"],
        "verdict": "G4_0_INVALID",
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }


def _rate_map(value: object, keys: list[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(keys)
        and all(
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and math.isfinite(rate)
            and 0 <= rate <= 1
            for rate in value.values()
        )
    )


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def adjudicate(
    payload: dict,
    spec: dict = BALANCED_INTEGRATED_DIALOGUE_SPEC,
) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") == "__PREREGISTRATION_COMMIT__"
        or spec.get("write_spec_sha256") != write_spec_sha256()
        or spec.get("retrieval_spec_sha256") != retrieval_spec_sha256()
        or not _finite(payload)
    ):
        return _invalid("registration, inherited specification, or finite-value check failed", spec)
    runtime = payload.get("runtime", {})
    if any(runtime.get(name) != spec["runtime"][name] for name in ("torch", "transformers")):
        return _invalid("runtime mismatch", spec)
    rows = payload.get("seeds")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != spec["seeds"]:
        return _invalid("seed roster mismatch", spec)

    source = BALANCED_NATURAL_WRITE_SPEC
    template_keys = [
        f"{kind}:{index}"
        for kind in spec["fact_kinds"]
        for index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
    ]
    expected_calibration_template = (
        len(source["subject_heads"][spec["fact_kinds"][0]])
        * len(source["lexicons"]["calibration"]["subject_qualifiers"])
        * len(source["lexicons"]["calibration"]["values"][spec["fact_kinds"][0]])
    )
    expected_evaluation_template = spec["evaluation_episodes"] // len(template_keys)
    metric_keys = {
        "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
        "stored", "recall_at_1", "recall_at_3", "stored_fact_recall_at_1",
        "per_kind_storage_rate", "per_template_storage_rate",
        "per_kind_recall_at_1", "per_template_recall_at_1",
        "per_position_recall_at_1", "per_distractor_storage_rate",
        "mean_fact_rank", "mean_fact_margin", "rankings_sha256",
    }
    thresholds = spec["thresholds"]
    verdicts = []
    summaries = []
    for row in rows:
        seed = row["seed"]
        audit = row.get("dataset_audit", {})
        if (
            audit.get("calibration_rows") != spec["calibration_rows"]
            or audit.get("calibration_unique") != spec["calibration_rows"]
            or audit.get("calibration_positive") != spec["calibration_rows"] // 2
            or audit.get("calibration_negative") != spec["calibration_rows"] // 2
            or set(audit.get("calibration_template_counts", {})) != set(template_keys)
            or any(value != expected_calibration_template for value in audit["calibration_template_counts"].values())
            or audit.get("synthetic_token_count") != 0
            or audit.get("evaluation_episodes") != {
                name: spec["evaluation_episodes"] for name in spec["replicates"]
            }
            or audit.get("evaluation_candidates") != {
                name: spec["evaluation_episodes"] * spec["candidates_per_episode"]
                for name in spec["replicates"]
            }
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or any(audit.get("calibration_evaluation_overlap", {}).values())
            or any(audit.get("cross_replicate_overlap", {}).values())
            or len(audit.get("calibration_sha256", "")) != 64
            or any(len(value) != 64 for value in audit.get("evaluation_sha256", {}).values())
        ):
            return _invalid(f"seed {seed} dataset audit failed", spec)
        evaluation_templates = audit.get("evaluation_fact_template_counts", {})
        if set(evaluation_templates) != set(spec["replicates"]):
            return _invalid(f"seed {seed} replicate audit changed", spec)
        if any(
            set(evaluation_templates[name]) != set(template_keys)
            or any(value != expected_evaluation_template for value in evaluation_templates[name].values())
            for name in spec["replicates"]
        ):
            return _invalid(f"seed {seed} template balance failed", spec)
        if set(row.get("checkpoints", {})) != {"semantic", "shuffled"} or not all(
            _checkpoint_valid(receipt) for receipt in row["checkpoints"].values()
        ):
            return _invalid(f"seed {seed} checkpoint receipt failed", spec)
        threshold = row.get("selection_threshold")
        if (
            not isinstance(threshold, (int, float))
            or not math.isclose(
                threshold,
                thresholds["expected_selection_threshold"],
                rel_tol=0.0,
                abs_tol=thresholds["selection_threshold_tolerance"],
            )
        ):
            return _invalid(f"seed {seed} selection threshold changed", spec)
        replicates = row.get("replicates")
        if not isinstance(replicates, list) or [item.get("name") for item in replicates] != spec["replicates"]:
            return _invalid(f"seed {seed} replicate roster changed", spec)

        replicate_summaries = {}
        for replicate in replicates:
            name = replicate["name"]
            matching = replicate.get("matching_audit", {})
            counts = [
                matching.get("semantic_counts"), matching.get("matched_random_counts"),
                matching.get("matched_shuffled_counts"),
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
                    len(matching.get(key, "")) != 64
                    for key in (
                        "semantic_selection_sha256", "matched_random_selection_sha256",
                        "matched_shuffled_selection_sha256", "fake_scores_sha256",
                    )
                )
            ):
                return _invalid(f"seed {seed} replicate {name} matching audit failed", spec)
            preservation = replicate.get("preservation_audit", {})
            expected_candidates = spec["evaluation_episodes"] * spec["candidates_per_episode"]
            if (
                preservation.get("raw_candidate_count_before") != expected_candidates
                or preservation.get("raw_candidate_count_after") != expected_candidates
                or preservation.get("raw_text_sha256_before") != preservation.get("raw_text_sha256_after")
                or len(preservation.get("raw_text_sha256_before", "")) != 64
                or preservation.get("long_term_selection_is_separate") is not True
            ):
                return _invalid(f"seed {seed} replicate {name} raw transcript preservation failed", spec)
            address = replicate.get("address_audit", {})
            segments = spec["evaluation_episodes"] * (spec["candidates_per_episode"] // 2)
            if (
                address.get("query_topics") != spec["evaluation_episodes"]
                or address.get("candidate_topics") != expected_candidates
                or address.get("episode_segment_addresses") != segments
                or address.get("unique_episode_segment_addresses") != segments
                or any(
                    len(address.get(key, "")) != 64
                    for key in (
                        "query_address_sha256", "candidate_address_sha256", "address_scores_sha256",
                    )
                )
                or len(replicate.get("content_scores_sha256", "")) != 64
            ):
                return _invalid(f"seed {seed} replicate {name} address audit failed", spec)
            search = replicate.get("search_audit", {})
            if (
                search.get("rankings_subset_of_stored") is not True
                or search.get("pools_subset_of_stored") is not True
                or search.get("empty_exact_when_no_stored_candidates") is not True
                or set(search.get("pool_sha256", {})) != set(spec["arms"])
                or any(len(value) != 64 for value in search["pool_sha256"].values())
            ):
                return _invalid(f"seed {seed} replicate {name} stored-candidate search failed", spec)
            arms = replicate.get("arms", {})
            if set(arms) != set(spec["arms"]):
                return _invalid(f"seed {seed} replicate {name} arm roster changed", spec)
            for arm_name, metrics in arms.items():
                if (
                    set(metrics) != metric_keys
                    or type(metrics.get("stored")) is not int
                    or not 0 <= metrics["stored"] <= expected_candidates
                    or any(
                        not isinstance(metrics.get(key), (int, float))
                        or not 0 <= metrics[key] <= 1
                        for key in (
                            "important_storage_rate", "distractor_storage_rate",
                            "search_size_ratio", "recall_at_1", "recall_at_3",
                            "stored_fact_recall_at_1",
                        )
                    )
                    or not _rate_map(metrics.get("per_kind_storage_rate"), spec["fact_kinds"])
                    or not _rate_map(metrics.get("per_template_storage_rate"), template_keys)
                    or not _rate_map(metrics.get("per_kind_recall_at_1"), spec["fact_kinds"])
                    or not _rate_map(metrics.get("per_template_recall_at_1"), template_keys)
                    or not _rate_map(
                        metrics.get("per_position_recall_at_1"),
                        [str(position) for position in spec["fact_positions"]],
                    )
                    or not _rate_map(metrics.get("per_distractor_storage_rate"), spec["distractor_kinds"])
                    or len(metrics.get("rankings_sha256", "")) != 64
                ):
                    return _invalid(f"seed {seed} replicate {name} arm {arm_name} metrics failed", spec)
            expected_stored = sum(counts[0])
            if any(
                arms[arm]["stored"] != expected_stored
                for arm in (
                    "semantic_integrated", "matched_random_integrated",
                    "matched_shuffled_integrated",
                )
            ):
                return _invalid(f"seed {seed} replicate {name} matched storage totals changed", spec)

            semantic = arms["semantic_integrated"]
            store_all = arms["store_all_integrated"]
            oracle = arms["oracle_integrated"]
            random_arm = arms["matched_random_integrated"]
            shuffled = arms["matched_shuffled_integrated"]
            no_memory = arms["no_memory"]
            if not (
                store_all["recall_at_1"] >= thresholds["minimum_store_all_recall_at_1"]
                and store_all["recall_at_3"] >= thresholds["minimum_store_all_recall_at_3"]
                and oracle["recall_at_1"] >= thresholds["minimum_oracle_recall_at_1"]
                and no_memory["recall_at_1"] <= thresholds["maximum_no_memory_recall_at_1"]
                and semantic["recall_at_1"] - random_arm["recall_at_1"] >= thresholds["minimum_fake_recall_gap"]
                and semantic["recall_at_1"] - shuffled["recall_at_1"] >= thresholds["minimum_fake_recall_gap"]
            ):
                return _invalid(f"seed {seed} replicate {name} positive, fake, or no-memory control failed", spec)
            write_valid = (
                semantic["important_storage_rate"] >= thresholds["minimum_important_storage_rate"]
                and min(semantic["per_kind_storage_rate"].values()) >= thresholds["minimum_important_storage_rate"]
                and min(semantic["per_template_storage_rate"].values()) >= thresholds["minimum_important_storage_rate"]
                and semantic["distractor_storage_rate"] <= thresholds["maximum_distractor_storage_rate"]
                and max(semantic["per_distractor_storage_rate"].values()) <= thresholds["maximum_per_distractor_storage_rate"]
                and semantic["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            )
            retrieval_valid = semantic["stored_fact_recall_at_1"] >= thresholds["minimum_stored_fact_recall_at_1"]
            integrated_valid = (
                semantic["recall_at_1"] >= thresholds["minimum_integrated_recall_at_1"]
                and semantic["recall_at_3"] >= thresholds["minimum_integrated_recall_at_3"]
                and min(semantic["per_kind_recall_at_1"].values()) >= thresholds["minimum_per_kind_recall_at_1"]
                and min(semantic["per_template_recall_at_1"].values()) >= thresholds["minimum_per_template_recall_at_1"]
                and min(semantic["per_position_recall_at_1"].values()) >= thresholds["minimum_per_position_recall_at_1"]
                and store_all["recall_at_1"] - semantic["recall_at_1"] <= thresholds["maximum_recall_drop_from_store_all"]
            )
            verdict = (
                "G4A_BALANCED_INTEGRATED_MEMORY_VALID_NOT_UNIQUE"
                if write_valid and retrieval_valid and integrated_valid
                else "G4B_WRITE_SELECTION_LOSS"
                if not write_valid and retrieval_valid
                else "G4C_RETRIEVAL_LOSS"
            )
            verdicts.append(verdict)
            replicate_summaries[name] = {
                "verdict": verdict,
                "important_storage_rate": semantic["important_storage_rate"],
                "distractor_storage_rate": semantic["distractor_storage_rate"],
                "search_size_ratio": semantic["search_size_ratio"],
                "recall_at_1": semantic["recall_at_1"],
                "stored_fact_recall_at_1": semantic["stored_fact_recall_at_1"],
                "minimum_kind_recall_at_1": min(semantic["per_kind_recall_at_1"].values()),
                "minimum_template_recall_at_1": min(semantic["per_template_recall_at_1"].values()),
                "store_all_recall_at_1": store_all["recall_at_1"],
                "matched_random_recall_at_1": random_arm["recall_at_1"],
                "matched_shuffled_recall_at_1": shuffled["recall_at_1"],
            }
        summaries.append({"seed": seed, "replicates": replicate_summaries})

    final = (
        "G4C_RETRIEVAL_LOSS"
        if any(value == "G4C_RETRIEVAL_LOSS" for value in verdicts)
        else "G4B_WRITE_SELECTION_LOSS"
        if any(value == "G4B_WRITE_SELECTION_LOSS" for value in verdicts)
        else "G4A_BALANCED_INTEGRATED_MEMORY_VALID_NOT_UNIQUE"
    )
    reasons = {
        "G4A_BALANCED_INTEGRATED_MEMORY_VALID_NOT_UNIQUE": (
            "balanced natural write selection and stored-candidate retrieval passed every registered control"
        ),
        "G4B_WRITE_SELECTION_LOSS": (
            "stored-candidate retrieval passed, but balanced natural write selection missed a threshold"
        ),
        "G4C_RETRIEVAL_LOSS": (
            "write selection retained facts, but stored-candidate retrieval or final recall missed a threshold"
        ),
    }
    return {
        "experiment": spec["experiment"],
        "verdict": final,
        "reason": reasons[final],
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", type=Path,
        default=Path("measurement/balanced_integrated_dialogue_results.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/balanced_integrated_dialogue_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(verdict["verdict"])


if __name__ == "__main__":
    main()
