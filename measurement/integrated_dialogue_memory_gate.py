#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-3."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.content_swap_retrieval_control_registry import spec_sha256 as retrieval_spec_sha256
    from measurement.integrated_dialogue_memory_registry import (
        INTEGRATED_DIALOGUE_MEMORY_SPEC,
        spec_sha256,
    )
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        spec_sha256 as write_spec_sha256,
        template_sha256,
    )
    from measurement.semantic_memory_write_gate import _checkpoint_valid
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.content_swap_retrieval_control_registry import spec_sha256 as retrieval_spec_sha256
    from measurement.integrated_dialogue_memory_registry import (
        INTEGRATED_DIALOGUE_MEMORY_SPEC,
        spec_sha256,
    )
    from measurement.realistic_memory_write_registry import (
        REALISTIC_MEMORY_WRITE_SPEC,
        spec_sha256 as write_spec_sha256,
        template_sha256,
    )
    from measurement.semantic_memory_write_gate import _checkpoint_valid


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rate_map(value: object, names: list[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(names)
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and 0 <= item <= 1
            for item in value.values()
        )
    )


def adjudicate(payload: dict, spec: dict = INTEGRATED_DIALOGUE_MEMORY_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "G3_0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("write_spec_sha256") != write_spec_sha256()
        or spec.get("retrieval_spec_sha256") != retrieval_spec_sha256()
        or not _finite(payload)
    ):
        return invalid("experiment, registered spec, inherited spec, digest, or finite-value check failed")
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
    expected_side = expected_kind // 2
    metric_keys = {
        "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
        "stored", "recall_at_1", "recall_at_3", "stored_fact_recall_at_1",
        "per_kind_recall_at_1", "per_position_recall_at_1",
        "per_distractor_storage_rate", "mean_fact_rank", "mean_fact_margin",
        "rankings_sha256",
    }
    summaries = []
    verdicts = []
    thresholds = spec["thresholds"]
    for row in seeds:
        seed = row.get("seed")
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
            or audit.get("fact_counts") != {kind: expected_kind for kind in spec["fact_kinds"]}
            or audit.get("fact_position_counts") != {
                str(position): expected_position for position in spec["fact_positions"]
            }
            or audit.get("distractor_counts") != {
                kind: spec["evaluation_episodes"] for kind in spec["distractor_kinds"]
            }
            or audit.get("topic_switch_counts") != {
                str(spec["topic_switches_per_episode"]): spec["evaluation_episodes"]
            }
            or audit.get("template_sha256") != template_sha256(REALISTIC_MEMORY_WRITE_SPEC)
            or any(len(audit.get(name, "")) != 64 for name in (
                "calibration_sha256", "evaluation_sha256",
            ))
        ):
            return invalid(f"seed {seed} dataset audit changed")
        expected_balance = {
            kind: {"0": expected_side, "1": expected_side} for kind in spec["fact_kinds"]
        }
        if row.get("balance_audit") != {"fact_kind_side_counts": expected_balance}:
            return invalid(f"seed {seed} fact kind and side balance changed")

        encoder = row.get("encoder_audit", {})
        encoder_spec = spec["encoder"]
        if (
            encoder.get("model_id") != encoder_spec["model_id"]
            or encoder.get("requested_revision") != encoder_spec["revision"]
            or encoder.get("loaded_revision") != encoder_spec["revision"]
            or encoder.get("embedding_dim") != encoder_spec["embedding_dim"]
            or encoder.get("pooling") != encoder_spec["pooling"]
            or encoder.get("normalize") is not True
            or encoder.get("max_length") != encoder_spec["max_length"]
        ):
            return invalid(f"seed {seed} encoder audit changed")
        embeddings = row.get("embedding_audit", {})
        for name, count in {
            "calibration": spec["calibration_rows"],
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
        expected_matching = {
            "method", "semantic_counts", "matched_random_counts",
            "matched_shuffled_counts", "semantic_selection_sha256",
            "matched_random_selection_sha256", "matched_shuffled_selection_sha256",
            "fake_scores_sha256",
        }
        if set(matching) != expected_matching or matching.get("method") != spec["selection"]["matching"]:
            return invalid(f"seed {seed} matching audit changed")
        counts = [
            matching.get("semantic_counts"), matching.get("matched_random_counts"),
            matching.get("matched_shuffled_counts"),
        ]
        if any(
            not isinstance(values, list) or len(values) != spec["evaluation_episodes"]
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or not 0 <= value <= spec["candidates_per_episode"] for value in values)
            for values in counts
        ) or counts[0] != counts[1] or counts[0] != counts[2]:
            return invalid(f"seed {seed} per-episode storage counts do not match")
        if any(len(matching.get(name, "")) != 64 for name in (
            "semantic_selection_sha256", "matched_random_selection_sha256",
            "matched_shuffled_selection_sha256", "fake_scores_sha256",
        )):
            return invalid(f"seed {seed} selection digest changed")

        address = row.get("address_audit", {})
        segments = spec["evaluation_episodes"] * (spec["candidates_per_episode"] // 2)
        if (
            address.get("query_topics") != spec["evaluation_episodes"]
            or address.get("candidate_topics") != spec["evaluation_episodes"] * spec["candidates_per_episode"]
            or address.get("episode_segment_addresses") != segments
            or address.get("unique_episode_segment_addresses") != segments
            or any(len(address.get(name, "")) != 64 for name in (
                "query_address_sha256", "candidate_address_sha256", "address_scores_sha256",
            ))
            or len(row.get("content_scores_sha256", "")) != 64
        ):
            return invalid(f"seed {seed} address or content score audit changed")
        search = row.get("search_audit", {})
        if (
            set(search) != {
                "rankings_subset_of_stored", "pools_subset_of_stored",
                "empty_exact_when_no_stored_candidates", "pool_sha256",
            }
            or search.get("rankings_subset_of_stored") is not True
            or search.get("pools_subset_of_stored") is not True
            or search.get("empty_exact_when_no_stored_candidates") is not True
            or set(search.get("pool_sha256", {})) != set(spec["arms"])
            or any(len(value) != 64 for value in search["pool_sha256"].values())
        ):
            return invalid(f"seed {seed} stored-candidate search audit failed")

        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        for name, metrics in arms.items():
            if (
                set(metrics) != metric_keys
                or not isinstance(metrics.get("stored"), int)
                or not 0 <= metrics["stored"] <= spec["evaluation_episodes"] * spec["candidates_per_episode"]
                or any(not 0 <= metrics.get(key, -1) <= 1 for key in (
                    "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
                    "recall_at_1", "recall_at_3", "stored_fact_recall_at_1",
                ))
                or not _rate_map(metrics.get("per_kind_recall_at_1"), spec["fact_kinds"])
                or not _rate_map(
                    metrics.get("per_position_recall_at_1"),
                    [str(position) for position in spec["fact_positions"]],
                )
                or not _rate_map(metrics.get("per_distractor_storage_rate"), spec["distractor_kinds"])
                or len(metrics.get("rankings_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} arm {name} shape changed")
        expected_stored = sum(counts[0])
        if any(arms[name]["stored"] != expected_stored for name in (
            "semantic_integrated", "matched_random_integrated", "matched_shuffled_integrated",
        )):
            return invalid(f"seed {seed} matched arm storage totals changed")

        semantic = arms["semantic_integrated"]
        all_rows = arms["store_all_integrated"]
        oracle = arms["oracle_integrated"]
        random_arm = arms["matched_random_integrated"]
        shuffled = arms["matched_shuffled_integrated"]
        none = arms["no_memory"]
        positive_valid = (
            all_rows["recall_at_1"] >= thresholds["minimum_store_all_recall_at_1"]
            and all_rows["recall_at_3"] >= thresholds["minimum_store_all_recall_at_3"]
            and oracle["recall_at_1"] >= thresholds["minimum_oracle_recall_at_1"]
            and oracle["recall_at_3"] >= thresholds["minimum_oracle_recall_at_3"]
            and none["recall_at_1"] <= thresholds["maximum_no_memory_recall_at_1"]
            and none["recall_at_3"] <= thresholds["maximum_no_memory_recall_at_3"]
        )
        fake_valid = (
            semantic["recall_at_1"] - random_arm["recall_at_1"]
            >= thresholds["minimum_fake_recall_gap"]
            and semantic["recall_at_1"] - shuffled["recall_at_1"]
            >= thresholds["minimum_fake_recall_gap"]
        )
        if not positive_valid or not fake_valid:
            return invalid(f"seed {seed} positive, fake, or no-memory control failed")

        write_valid = (
            semantic["important_storage_rate"] >= thresholds["minimum_important_storage_rate"]
            and semantic["distractor_storage_rate"] <= thresholds["maximum_distractor_storage_rate"]
            and max(semantic["per_distractor_storage_rate"].values())
            <= thresholds["maximum_per_distractor_storage_rate"]
            and semantic["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            and min(semantic["per_kind_recall_at_1"].values())
            >= thresholds["minimum_per_kind_recall_at_1"]
            and min(semantic["per_position_recall_at_1"].values())
            >= thresholds["minimum_per_position_recall_at_1"]
        )
        retrieval_valid = (
            semantic["stored_fact_recall_at_1"]
            >= thresholds["minimum_stored_fact_recall_at_1"]
        )
        integrated_valid = (
            semantic["recall_at_1"] >= thresholds["minimum_integrated_recall_at_1"]
            and semantic["recall_at_3"] >= thresholds["minimum_integrated_recall_at_3"]
            and all_rows["recall_at_1"] - semantic["recall_at_1"]
            <= thresholds["maximum_recall_drop_from_store_all"]
        )
        verdict = (
            "G3A_INTEGRATED_DIALOGUE_MEMORY_VALID_NOT_UNIQUE"
            if write_valid and retrieval_valid and integrated_valid
            else "G3C_RETRIEVAL_LOSS"
            if write_valid and (not retrieval_valid or not integrated_valid)
            else "G3B_WRITE_SELECTION_LOSS"
            if retrieval_valid
            else "G3C_RETRIEVAL_LOSS"
        )
        verdicts.append(verdict)
        summaries.append({
            "seed": seed,
            "verdict": verdict,
            "semantic": semantic,
            "store_all_recall_at_1": all_rows["recall_at_1"],
            "oracle_recall_at_1": oracle["recall_at_1"],
            "matched_random_recall_at_1": random_arm["recall_at_1"],
            "matched_shuffled_recall_at_1": shuffled["recall_at_1"],
            "no_memory_recall_at_1": none["recall_at_1"],
        })

    final = (
        "G3C_RETRIEVAL_LOSS"
        if any(value == "G3C_RETRIEVAL_LOSS" for value in verdicts)
        else "G3B_WRITE_SELECTION_LOSS"
        if any(value == "G3B_WRITE_SELECTION_LOSS" for value in verdicts)
        else "G3A_INTEGRATED_DIALOGUE_MEMORY_VALID_NOT_UNIQUE"
    )
    return {
        "experiment": spec["experiment"],
        "verdict": final,
        "reason": {
            "G3A_INTEGRATED_DIALOGUE_MEMORY_VALID_NOT_UNIQUE": (
                "semantic write selection and stored-candidate retrieval passed every registered control"
            ),
            "G3B_WRITE_SELECTION_LOSS": (
                "retrieval recovered stored facts, but realistic semantic write selection missed a registered threshold"
            ),
            "G3C_RETRIEVAL_LOSS": (
                "write selection passed or retained facts, but stored-candidate retrieval missed a registered threshold"
            ),
        }[final],
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("measurement/integrated_dialogue_memory_adjudication.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
