#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-RETRIEVAL-CONTROL-2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        template_sha256,
    )
    from measurement.split_retrieval_control_registry import (
        SPLIT_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        template_sha256,
    )
    from measurement.split_retrieval_control_registry import (
        SPLIT_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rate_map(value: object, names: list[str]) -> bool:
    return (
        isinstance(value, dict) and set(value) == set(names)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(item) and 0 <= item <= 1 for item in value.values())
    )


def adjudicate(payload: dict, spec: dict = SPLIT_RETRIEVAL_CONTROL_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "GRC2_0_INVALID",
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

    expected_kind = spec["evaluation_episodes"] // len(spec["fact_kinds"])
    expected_position = spec["evaluation_episodes"] // len(spec["fact_positions"])
    summaries = []
    verdicts = []
    expected_metric_keys = {
        "recall_at_1", "recall_at_3", "per_kind_recall_at_1",
        "per_position_recall_at_1", "mean_fact_rank", "mean_fact_margin",
        "rankings_sha256",
    }
    for row in seeds:
        seed = row.get("seed")
        audit = row.get("dataset_audit", {})
        if (
            audit.get("evaluation_episodes") != spec["evaluation_episodes"]
            or audit.get("evaluation_candidates")
            != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or audit.get("query_unique") != spec["evaluation_episodes"]
            or audit.get("query_candidate_overlap") != 0
            or audit.get("fact_counts")
            != {kind: expected_kind for kind in spec["fact_kinds"]}
            or audit.get("fact_position_counts")
            != {str(position): expected_position for position in spec["fact_positions"]}
            or audit.get("topic_switch_counts")
            != {str(spec["topic_switches_per_episode"]): spec["evaluation_episodes"]}
            or audit.get("template_sha256") != template_sha256()
        ):
            return invalid(f"seed {seed} dataset audit changed")
        encoder = row.get("encoder_audit", {})
        encoder_spec = spec["encoder"]
        if (
            encoder.get("model_id") != encoder_spec["model_id"]
            or encoder.get("requested_revision") != encoder_spec["revision"]
            or encoder.get("loaded_revision") != encoder_spec["revision"]
            or encoder.get("embedding_dim") != encoder_spec["embedding_dim"]
        ):
            return invalid(f"seed {seed} encoder audit changed")
        embeddings = row.get("embedding_audit", {})
        for name, count in {
            "calibration": REALISTIC_MEMORY_WRITE_SPEC["calibration_rows"],
            "queries": spec["evaluation_episodes"],
            "candidates": spec["evaluation_episodes"] * spec["candidates_per_episode"],
        }.items():
            item = embeddings.get(name, {})
            if (
                item.get("rows") != count
                or item.get("feature_dim") != encoder_spec["feature_dim"]
                or len(item.get("features_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} {name} embedding audit changed")
        address = row.get("address_audit", {})
        if (
            address.get("query_topics") != spec["evaluation_episodes"]
            or address.get("candidate_topics")
            != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or address.get("candidate_topic_unique") != spec["evaluation_episodes"] * 4
            or len(address.get("query_address_sha256", "")) != 64
            or len(address.get("candidate_address_sha256", "")) != 64
        ):
            return invalid(f"seed {seed} address audit changed")
        score_audit = row.get("score_audit", {})
        if (
            len(score_audit.get("content_scores_sha256", "")) != 64
            or len(score_audit.get("address_scores_sha256", "")) != 64
            or score_audit.get("normal_pool_fact_coverage") != 1.0
        ):
            return invalid(f"seed {seed} score audit changed")
        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        for name, metrics in arms.items():
            if (
                set(metrics) != expected_metric_keys
                or not 0 <= metrics.get("recall_at_1", -1) <= 1
                or not 0 <= metrics.get("recall_at_3", -1) <= 1
                or not _rate_map(metrics.get("per_kind_recall_at_1"), spec["fact_kinds"])
                or not _rate_map(
                    metrics.get("per_position_recall_at_1"),
                    [str(position) for position in spec["fact_positions"]],
                )
                or len(metrics.get("rankings_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} arm {name} shape changed")

        split = arms["split_topic_content"]
        topic = arms["topic_only"]
        shuffled_topic = arms["shuffled_topic"]
        shuffled_content = arms["shuffled_content"]
        oracle = arms["oracle_memory"]
        none = arms["no_memory"]
        thresholds = spec["thresholds"]
        if (
            oracle["recall_at_3"] < thresholds["oracle_recall_at_3"]
            or none["recall_at_3"] > thresholds["maximum_no_memory_recall_at_3"]
            or topic["recall_at_1"] > thresholds["maximum_topic_only_recall_at_1"]
        ):
            return invalid(f"seed {seed} positive, no-memory, or content-necessity control failed")
        topic_valid = (
            topic["recall_at_3"] >= thresholds["minimum_topic_only_recall_at_3"]
            and shuffled_topic["recall_at_3"]
            <= thresholds["maximum_shuffled_topic_recall_at_3"]
        )
        content_valid = (
            split["recall_at_3"] >= thresholds["split_recall_at_3"]
            and split["recall_at_1"] >= thresholds["split_recall_at_1"]
            and min(split["per_kind_recall_at_1"].values())
            >= thresholds["minimum_per_kind_recall_at_1"]
            and min(split["per_position_recall_at_1"].values())
            >= thresholds["minimum_per_position_recall_at_1"]
            and shuffled_content["recall_at_1"]
            <= thresholds["maximum_shuffled_content_recall_at_1"]
            and split["recall_at_1"] - shuffled_content["recall_at_1"]
            >= thresholds["minimum_shuffled_content_gap"]
        )
        verdict = (
            "GRC2A_SPLIT_RETRIEVAL_VALID" if topic_valid and content_valid
            else "GRC2C_TOPIC_ADDRESS_LOSS" if not topic_valid
            else "GRC2B_CONTENT_RANKING_LOSS"
        )
        verdicts.append(verdict)
        summaries.append({
            "seed": seed,
            "verdict": verdict,
            "split": split,
            "topic_only_recall_at_3": topic["recall_at_3"],
            "shuffled_topic_recall_at_3": shuffled_topic["recall_at_3"],
            "shuffled_content_recall_at_1": shuffled_content["recall_at_1"],
            "oracle_recall_at_3": oracle["recall_at_3"],
            "no_memory_recall_at_3": none["recall_at_3"],
        })
    final = (
        "GRC2A_SPLIT_RETRIEVAL_VALID"
        if all(value == "GRC2A_SPLIT_RETRIEVAL_VALID" for value in verdicts)
        else "GRC2C_TOPIC_ADDRESS_LOSS"
        if any(value == "GRC2C_TOPIC_ADDRESS_LOSS" for value in verdicts)
        else "GRC2B_CONTENT_RANKING_LOSS"
    )
    return {
        "experiment": spec["experiment"],
        "verdict": final,
        "reason": {
            "GRC2A_SPLIT_RETRIEVAL_VALID": "topic addressing and content ranking passed every registered control",
            "GRC2B_CONTENT_RANKING_LOSS": "topic addressing passed but content ranking missed a registered threshold",
            "GRC2C_TOPIC_ADDRESS_LOSS": "normal or shuffled topic addressing missed a registered threshold",
        }[final],
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/split_retrieval_control_adjudication.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    verdict = adjudicate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
