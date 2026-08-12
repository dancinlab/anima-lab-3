#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-RETRIEVAL-CONTROL-1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.realistic_memory_write_registry import template_sha256
    from measurement.semantic_retrieval_control_registry import (
        SEMANTIC_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.realistic_memory_write_registry import template_sha256
    from measurement.semantic_retrieval_control_registry import (
        SEMANTIC_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rate_map(values: object, names: list[str]) -> bool:
    return (
        isinstance(values, dict)
        and set(values) == set(names)
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1 for value in values.values())
    )


def adjudicate(payload: dict, spec: dict = SEMANTIC_RETRIEVAL_CONTROL_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "GRC0_INVALID",
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
    seeds = payload.get("seeds", [])
    if [row.get("seed") for row in seeds] != spec["seeds"]:
        return invalid("registered seed roster changed")

    expected_count = spec["evaluation_episodes"] // len(spec["fact_kinds"])
    expected_position_count = spec["evaluation_episodes"] // len(spec["fact_positions"])
    summaries = []
    passed = True
    for row in seeds:
        seed = row.get("seed")
        audit = row.get("dataset_audit", {})
        if (
            set(audit) != {
                "evaluation_episodes", "evaluation_candidates", "evaluation_unique",
                "query_unique", "query_candidate_overlap", "fact_counts",
                "fact_position_counts", "topic_switch_counts", "template_sha256",
                "evaluation_sha256", "query_sha256",
            }
            or audit.get("evaluation_episodes") != spec["evaluation_episodes"]
            or audit.get("evaluation_candidates")
            != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or audit.get("query_unique") != spec["evaluation_episodes"]
            or audit.get("query_candidate_overlap") != 0
            or audit.get("fact_counts")
            != {kind: expected_count for kind in spec["fact_kinds"]}
            or audit.get("fact_position_counts")
            != {str(position): expected_position_count for position in spec["fact_positions"]}
            or audit.get("topic_switch_counts")
            != {str(spec["topic_switches_per_episode"]): spec["evaluation_episodes"]}
            or audit.get("template_sha256") != template_sha256()
            or any(len(audit.get(name, "")) != 64 for name in (
                "evaluation_sha256", "query_sha256",
            ))
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
        embeddings = row.get("embedding_audit", {})
        for name, count in {
            "queries": spec["evaluation_episodes"],
            "candidates": spec["evaluation_episodes"] * spec["candidates_per_episode"],
        }.items():
            item = embeddings.get(name, {})
            if (
                item.get("rows") != count
                or item.get("feature_dim") != encoder_spec["feature_dim"]
                or not 0.999999 <= item.get("sentence_norm_min", 0) <= 1.000001
                or not 0.999999 <= item.get("sentence_norm_max", 0) <= 1.000001
                or len(item.get("features_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} {name} embedding audit changed")
        feature = row.get("feature_audit", {})
        if (
            set(feature) != {
                "feature_dim", "query_features_sha256", "candidate_features_sha256",
            }
            or feature.get("feature_dim") != spec["retrieval"]["feature_dim"]
            or len(feature.get("query_features_sha256", "")) != 64
            or len(feature.get("candidate_features_sha256", "")) != 64
        ):
            return invalid(f"seed {seed} retrieval feature audit changed")
        arms = row.get("arms", {})
        expected_metrics = {
            "recall_at_3", "per_kind_recall", "per_position_recall",
            "mean_fact_rank", "mean_fact_margin", "rankings_sha256",
        }
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        for name, metrics in arms.items():
            if (
                set(metrics) != expected_metrics
                or not isinstance(metrics.get("recall_at_3"), (int, float))
                or isinstance(metrics.get("recall_at_3"), bool)
                or not 0 <= metrics["recall_at_3"] <= 1
                or not _rate_map(metrics.get("per_kind_recall"), spec["fact_kinds"])
                or not _rate_map(
                    metrics.get("per_position_recall"),
                    [str(position) for position in spec["fact_positions"]],
                )
                or not isinstance(metrics.get("mean_fact_rank"), (int, float))
                or not isinstance(metrics.get("mean_fact_margin"), (int, float))
                or len(metrics.get("rankings_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} arm {name} shape changed")

        semantic = arms["semantic_retrieval"]
        character = arms["character_retrieval"]
        oracle = arms["oracle_memory"]
        shuffled = arms["shuffled_query"]
        none = arms["no_memory"]
        thresholds = spec["thresholds"]
        if (
            oracle["recall_at_3"] < thresholds["oracle_recall_at_3"]
            or shuffled["recall_at_3"] > thresholds["maximum_shuffled_recall_at_3"]
            or none["recall_at_3"] > thresholds["maximum_no_memory_recall_at_3"]
            or semantic["recall_at_3"] - shuffled["recall_at_3"]
            < thresholds["minimum_shuffled_gap"]
        ):
            return invalid(f"seed {seed} positive, shuffled, or no-memory control failed")
        seed_pass = (
            semantic["recall_at_3"] >= thresholds["semantic_recall_at_3"]
            and min(semantic["per_kind_recall"].values())
            >= thresholds["minimum_per_kind_recall"]
            and min(semantic["per_position_recall"].values())
            >= thresholds["minimum_per_position_recall"]
            and character["recall_at_3"] - semantic["recall_at_3"]
            <= thresholds["maximum_drop_from_character"]
        )
        passed = passed and seed_pass
        summaries.append({
            "seed": seed,
            "passed": seed_pass,
            "semantic": semantic,
            "character_recall_at_3": character["recall_at_3"],
            "oracle_recall_at_3": oracle["recall_at_3"],
            "shuffled_recall_at_3": shuffled["recall_at_3"],
            "no_memory_recall_at_3": none["recall_at_3"],
        })
    return {
        "experiment": spec["experiment"],
        "verdict": (
            "GRC1_SEMANTIC_RETRIEVAL_VALID"
            if passed else "GRC2_SEMANTIC_RETRIEVAL_LOSS"
        ),
        "reason": (
            "the frozen semantic representation supplied a valid all-memory retrieval path"
            if passed else "the semantic retrieval path missed a registered overall, kind, or position threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_retrieval_control_adjudication.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    verdict = adjudicate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
