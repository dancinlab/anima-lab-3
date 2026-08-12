#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        spec_sha256,
        template_sha256,
    )
    from measurement.semantic_memory_write_gate import _checkpoint_valid
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        spec_sha256,
        template_sha256,
    )
    from measurement.semantic_memory_write_gate import _checkpoint_valid


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rate_map_valid(values: dict, names: list[str]) -> bool:
    return (
        set(values) == set(names)
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and 0 <= value <= 1 for value in values.values())
    )


def adjudicate(payload: dict, spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "G2R0_INVALID",
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
    runtime = payload.get("runtime", {})
    if (
        not str(runtime.get("python", "")).startswith(spec["runtime"]["python"] + ".")
        or str(runtime.get("torch", "")).split("+")[0] != spec["runtime"]["torch"]
        or runtime.get("transformers") != spec["runtime"]["transformers"]
        or runtime.get("device") != spec["encoder"]["device"]
    ):
        return invalid("registered runtime changed")
    rows = payload.get("seeds", [])
    if [row.get("seed") for row in rows] != spec["seeds"]:
        return invalid("registered seed roster changed")

    thresholds = spec["thresholds"]
    summaries = []
    passed = True
    for row in rows:
        seed = row.get("seed")
        audit = row.get("dataset_audit", {})
        expected_fact_count = spec["evaluation_episodes"] // len(spec["fact_kinds"])
        expected_position_count = spec["evaluation_episodes"] // len(spec["fact_positions"])
        if (
            set(audit) != {
                "calibration_rows", "calibration_unique", "calibration_positive",
                "calibration_negative", "evaluation_episodes", "evaluation_candidates",
                "evaluation_unique", "overlap", "fact_counts", "fact_position_counts",
                "distractor_counts", "topic_switch_counts", "template_sha256",
                "calibration_sha256", "evaluation_sha256",
            }
            or audit.get("calibration_rows") != spec["calibration_rows"]
            or audit.get("calibration_unique") != spec["calibration_rows"]
            or audit.get("calibration_positive") != spec["calibration_rows"] // 2
            or audit.get("calibration_negative") != spec["calibration_rows"] // 2
            or audit.get("evaluation_episodes") != spec["evaluation_episodes"]
            or audit.get("evaluation_candidates")
            != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or audit.get("overlap") != 0
            or audit.get("fact_counts")
            != {kind: expected_fact_count for kind in spec["fact_kinds"]}
            or audit.get("fact_position_counts")
            != {str(position): expected_position_count for position in spec["fact_positions"]}
            or audit.get("distractor_counts")
            != {kind: spec["evaluation_episodes"] for kind in spec["distractor_kinds"]}
            or audit.get("topic_switch_counts")
            != {str(spec["topic_switches_per_episode"]): spec["evaluation_episodes"]}
            or audit.get("template_sha256") != template_sha256(spec)
            or any(len(audit.get(name, "")) != 64
                   for name in ("calibration_sha256", "evaluation_sha256"))
        ):
            return invalid(f"seed {seed} dataset audit changed")

        encoder = row.get("encoder_audit", {})
        encoder_spec = spec["encoder"]
        if (
            encoder.get("model_id") != encoder_spec["model_id"]
            or encoder.get("requested_revision") != encoder_spec["revision"]
            or encoder.get("loaded_revision") != encoder_spec["revision"]
            or encoder.get("embedding_dim") != encoder_spec["embedding_dim"]
            or encoder.get("pooling") != encoder_spec["pooling"]
            or encoder.get("normalize") != encoder_spec["normalize"]
            or encoder.get("max_length") != encoder_spec["max_length"]
        ):
            return invalid(f"seed {seed} encoder audit changed")
        embedding = row.get("embedding_audit", {})
        for split, count in {
            "calibration": spec["calibration_rows"],
            "evaluation": spec["evaluation_episodes"] * spec["candidates_per_episode"],
        }.items():
            split_audit = embedding.get(split, {})
            if (
                split_audit.get("rows") != count
                or split_audit.get("feature_dim") != encoder_spec["feature_dim"]
                or not 0.999999 <= split_audit.get("sentence_norm_min", 0) <= 1.000001
                or not 0.999999 <= split_audit.get("sentence_norm_max", 0) <= 1.000001
                or len(split_audit.get("features_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} {split} embedding audit changed")
        for name in ("fit_audit", "shuffled_fit_audit"):
            fit = row.get(name, {})
            if (
                fit.get("method") != spec["fit_method"]
                or fit.get("examples") != spec["calibration_rows"]
                or fit.get("positives") != spec["calibration_rows"] // 2
                or fit.get("negatives") != spec["calibration_rows"] // 2
                or fit.get("feature_dim") != encoder_spec["feature_dim"]
                or fit.get("ridge") != spec["ridge"]
            ):
                return invalid(f"seed {seed} {name} changed")
        checkpoints = row.get("checkpoints", {})
        if set(checkpoints) != {"semantic", "shuffled"} or any(
            not _checkpoint_valid(receipt, encoder_spec) for receipt in checkpoints.values()
        ):
            return invalid(f"seed {seed} checkpoint changed")

        matching = row.get("matching_audit", {})
        expected_matching_keys = {
            "method", "semantic_counts", "matched_shuffled_counts",
            "matched_random_counts", "semantic_selection_sha256",
            "matched_shuffled_selection_sha256", "matched_random_selection_sha256",
            "fake_scores_sha256",
        }
        if set(matching) != expected_matching_keys or matching.get("method") != spec["matching"]:
            return invalid(f"seed {seed} matching audit changed")
        counts = [
            matching.get("semantic_counts"), matching.get("matched_shuffled_counts"),
            matching.get("matched_random_counts"),
        ]
        if any(
            not isinstance(values, list) or len(values) != spec["evaluation_episodes"]
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or not 0 <= value <= spec["candidates_per_episode"] for value in values)
            for values in counts
        ) or counts[0] != counts[1] or counts[0] != counts[2]:
            return invalid(f"seed {seed} per-episode storage counts do not match")
        if any(not isinstance(matching.get(name), str) or len(matching[name]) != 64 for name in (
            "semantic_selection_sha256", "matched_shuffled_selection_sha256",
            "matched_random_selection_sha256", "fake_scores_sha256",
        )):
            return invalid(f"seed {seed} matching digest changed")

        arms = row.get("arms", {})
        expected_metric_keys = {
            "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
            "recall_at_3", "stored", "per_kind_recall", "per_position_recall",
            "per_distractor_storage_rate", "records_sha256",
        }
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        for name, metrics in arms.items():
            if (
                set(metrics) != expected_metric_keys
                or not _rate_map_valid(metrics.get("per_kind_recall", {}), spec["fact_kinds"])
                or not _rate_map_valid(
                    metrics.get("per_position_recall", {}),
                    [str(position) for position in spec["fact_positions"]],
                )
                or not _rate_map_valid(
                    metrics.get("per_distractor_storage_rate", {}), spec["distractor_kinds"],
                )
                or len(metrics.get("records_sha256", "")) != 64
                or not isinstance(metrics.get("stored"), int)
                or not 0 <= metrics["stored"]
                <= spec["evaluation_episodes"] * spec["candidates_per_episode"]
            ):
                return invalid(f"seed {seed} arm {name} shape changed")
        expected_total = sum(counts[0])
        if any(arms[name]["stored"] != expected_total for name in (
            "semantic_gate", "matched_random", "matched_shuffled_gate",
        )):
            return invalid(f"seed {seed} stored totals do not match the episode audit")

        semantic = arms["semantic_gate"]
        all_rows = arms["store_all"]
        oracle = arms["oracle_gate"]
        random_arm = arms["matched_random"]
        shuffled = arms["matched_shuffled_gate"]
        none = arms["no_memory"]
        if (
            oracle["important_storage_rate"] < thresholds["oracle_important_storage_rate"]
            or oracle["recall_at_3"] < thresholds["oracle_recall_at_3"]
            or all_rows["recall_at_3"] < thresholds["store_all_recall_at_3"]
            or none["recall_at_3"] > thresholds["no_memory_max_recall_at_3"]
            or semantic["recall_at_3"] - random_arm["recall_at_3"]
            < thresholds["minimum_fake_recall_gap"]
            or semantic["recall_at_3"] - shuffled["recall_at_3"]
            < thresholds["minimum_fake_recall_gap"]
        ):
            return invalid(f"seed {seed} positive, fake, or no-memory control failed")
        seed_pass = (
            semantic["important_storage_rate"] >= thresholds["important_storage_rate"]
            and semantic["recall_at_3"] >= thresholds["recall_at_3"]
            and min(semantic["per_kind_recall"].values())
            >= thresholds["minimum_per_kind_recall"]
            and min(semantic["per_position_recall"].values())
            >= thresholds["minimum_per_position_recall"]
            and semantic["distractor_storage_rate"]
            <= thresholds["maximum_distractor_storage_rate"]
            and max(semantic["per_distractor_storage_rate"].values())
            <= thresholds["maximum_per_distractor_storage_rate"]
            and semantic["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            and all_rows["recall_at_3"] - semantic["recall_at_3"]
            <= thresholds["maximum_recall_drop_from_all"]
        )
        passed = passed and seed_pass
        summaries.append({
            "seed": seed,
            "passed": seed_pass,
            "semantic": semantic,
            "store_all_recall_at_3": all_rows["recall_at_3"],
            "matched_random_recall_at_3": random_arm["recall_at_3"],
            "matched_shuffled_recall_at_3": shuffled["recall_at_3"],
        })
    return {
        "experiment": spec["experiment"],
        "verdict": (
            "G2R1_REALISTIC_WRITE_VALID_NOT_UNIQUE"
            if passed else "G2R2_STYLE_OR_SHIFT_LOSS"
        ),
        "reason": (
            "the frozen semantic selector retained durable facts across realistic styles and topic shifts"
            if passed else "the realistic selector missed a registered overall, style, or position threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/realistic_memory_write_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
