#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-RETRIEVAL-CONTROL-4."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.balanced_retrieval_control_gate import adjudicate as adjudicate_baseline
    from measurement.balanced_retrieval_control_registry import (
        BALANCED_RETRIEVAL_CONTROL_SPEC,
        spec_sha256 as balanced_spec_sha256,
    )
    from measurement.content_swap_retrieval_control_registry import (
        CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.balanced_retrieval_control_gate import adjudicate as adjudicate_baseline
    from measurement.balanced_retrieval_control_registry import (
        BALANCED_RETRIEVAL_CONTROL_SPEC,
        spec_sha256 as balanced_spec_sha256,
    )
    from measurement.content_swap_retrieval_control_registry import (
        CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
        spec_sha256,
    )


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _metric_shape(value: object, spec: dict) -> bool:
    expected = {
        "recall_at_1", "recall_at_3", "per_kind_recall_at_1",
        "per_position_recall_at_1", "mean_fact_rank", "mean_fact_margin",
        "rankings_sha256",
    }
    return (
        isinstance(value, dict)
        and set(value) == expected
        and all(isinstance(value.get(name), (int, float)) for name in (
            "recall_at_1", "recall_at_3", "mean_fact_rank", "mean_fact_margin",
        ))
        and 0 <= value["recall_at_1"] <= 1
        and 0 <= value["recall_at_3"] <= 1
        and set(value.get("per_kind_recall_at_1", {})) == set(spec["fact_kinds"])
        and set(value.get("per_position_recall_at_1", {}))
        == set(map(str, spec["fact_positions"]))
        and all(0 <= item <= 1 for item in value["per_kind_recall_at_1"].values())
        and all(0 <= item <= 1 for item in value["per_position_recall_at_1"].values())
        and len(value.get("rankings_sha256", "")) == 64
    )


def adjudicate(payload: dict, spec: dict = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "GRC4_0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("balanced_spec_sha256") != balanced_spec_sha256()
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

    baseline_payload = {
        "experiment": BALANCED_RETRIEVAL_CONTROL_SPEC["experiment"],
        "spec": BALANCED_RETRIEVAL_CONTROL_SPEC,
        "spec_sha256": balanced_spec_sha256(),
        "runtime": runtime,
        "seeds": [row.get("baseline") for row in seeds],
    }
    baseline_verdict = adjudicate_baseline(baseline_payload)
    if baseline_verdict.get("verdict") == "GRC3_0_INVALID":
        return invalid("inherited balanced retrieval audit failed")

    thresholds = spec["thresholds"]
    summaries = []
    verdicts = []
    for row, base_summary in zip(seeds, baseline_verdict.get("seeds", [])):
        seed = row["seed"]
        baseline = row.get("baseline", {})
        if baseline.get("seed") != seed or base_summary.get("seed") != seed:
            return invalid(f"seed {seed} inherited result changed")
        if base_summary.get("verdict") == "GRC3C_EPISODE_ADDRESS_LOSS":
            verdicts.append("GRC4C_EPISODE_ADDRESS_LOSS")
            summaries.append({"seed": seed, "verdict": verdicts[-1]})
            continue
        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]) or not all(
            _metric_shape(value, spec) for value in arms.values()
        ):
            return invalid(f"seed {seed} swap arm roster or shape changed")
        normal = baseline.get("arms", {}).get("balanced_split", {})
        topic = baseline.get("arms", {}).get("topic_only", {})
        shuffled_address = baseline.get("arms", {}).get("shuffled_episode_address", {})
        oracle = baseline.get("arms", {}).get("oracle_memory", {})
        none = baseline.get("arms", {}).get("no_memory", {})
        swapped = arms["within_pool_content_swap"]
        restored = arms["restored_content"]
        audit = row.get("swap_audit", {})
        required_audit_keys = {
            "pool_rows", "pool_size", "swapped_pairs", "uses_labels",
            "uses_episode_order", "outside_pool_unchanged", "pair_exchange_exact",
            "score_multisets_preserved", "restored_scores_exact",
            "normal_pool_sha256", "normal_scores_sha256", "swapped_scores_sha256",
            "restored_scores_sha256", "restored_rankings_exact",
        }
        if (
            set(audit) != required_audit_keys
            or audit.get("pool_rows") != spec["evaluation_episodes"]
            or audit.get("pool_size") != spec["retrieval"]["address_pool"]
            or audit.get("swapped_pairs") != spec["evaluation_episodes"]
            or audit.get("uses_labels") is not False
            or audit.get("uses_episode_order") is not False
            or not all(audit.get(name) is True for name in (
                "outside_pool_unchanged", "pair_exchange_exact",
                "score_multisets_preserved", "restored_scores_exact",
                "restored_rankings_exact",
            ))
            or any(len(audit.get(name, "")) != 64 for name in (
                "normal_pool_sha256", "normal_scores_sha256", "swapped_scores_sha256",
                "restored_scores_sha256",
            ))
            or audit["normal_scores_sha256"] != audit["restored_scores_sha256"]
            or restored != normal
        ):
            return invalid(f"seed {seed} score swap or restoration audit failed")
        positive_valid = (
            oracle.get("recall_at_3", -1) >= thresholds["oracle_recall_at_3"]
            and none.get("recall_at_3", 2) <= thresholds["maximum_no_memory_recall_at_3"]
        )
        address_valid = (
            thresholds["minimum_topic_only_recall_at_1"]
            <= topic.get("recall_at_1", -1)
            <= thresholds["maximum_topic_only_recall_at_1"]
            and topic.get("recall_at_3", -1) >= thresholds["minimum_topic_only_recall_at_3"]
            and shuffled_address.get("recall_at_3", 2)
            <= thresholds["maximum_shuffled_episode_recall_at_3"]
        )
        normal_valid = (
            normal.get("recall_at_3", -1) >= thresholds["normal_recall_at_3"]
            and normal.get("recall_at_1", -1) >= thresholds["normal_recall_at_1"]
            and min(normal.get("per_kind_recall_at_1", {"": -1}).values())
            >= thresholds["minimum_per_kind_recall_at_1"]
            and min(normal.get("per_position_recall_at_1", {"": -1}).values())
            >= thresholds["minimum_per_position_recall_at_1"]
            and baseline.get("score_audit", {}).get("normal_pool_fact_coverage", -1)
            >= thresholds["minimum_normal_pool_fact_coverage"]
        )
        if not positive_valid:
            return invalid(f"seed {seed} positive or no-memory control failed")
        verdict = (
            "GRC4C_EPISODE_ADDRESS_LOSS"
            if not address_valid or not normal_valid
            else "GRC4A_CONTENT_ALIGNMENT_CAUSAL"
            if (
                swapped["recall_at_1"] <= thresholds["maximum_swapped_recall_at_1"]
                and normal["recall_at_1"] - swapped["recall_at_1"]
                >= thresholds["minimum_swap_gap"]
            )
            else "GRC4B_CONTENT_ALIGNMENT_NOT_CAUSAL"
        )
        verdicts.append(verdict)
        summaries.append({
            "seed": seed,
            "verdict": verdict,
            "normal_recall_at_1": normal["recall_at_1"],
            "normal_recall_at_3": normal["recall_at_3"],
            "swapped_recall_at_1": swapped["recall_at_1"],
            "swap_gap": normal["recall_at_1"] - swapped["recall_at_1"],
            "restored_recall_at_1": restored["recall_at_1"],
            "topic_only_recall_at_1": topic["recall_at_1"],
            "shuffled_episode_recall_at_3": shuffled_address["recall_at_3"],
        })

    final = (
        "GRC4C_EPISODE_ADDRESS_LOSS"
        if any(value == "GRC4C_EPISODE_ADDRESS_LOSS" for value in verdicts)
        else "GRC4A_CONTENT_ALIGNMENT_CAUSAL"
        if all(value == "GRC4A_CONTENT_ALIGNMENT_CAUSAL" for value in verdicts)
        else "GRC4B_CONTENT_ALIGNMENT_NOT_CAUSAL"
    )
    return {
        "experiment": spec["experiment"],
        "verdict": final,
        "reason": {
            "GRC4A_CONTENT_ALIGNMENT_CAUSAL": (
                "candidate-aligned content scores caused selection and exact restoration recovered it"
            ),
            "GRC4B_CONTENT_ALIGNMENT_NOT_CAUSAL": (
                "selection remained above the registered limit after within-pool score exchange"
            ),
            "GRC4C_EPISODE_ADDRESS_LOSS": (
                "normal or shuffled episode addressing missed a registered threshold"
            ),
        }[final],
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/content_swap_retrieval_control_adjudication.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
