#!/usr/bin/env python3
"""Fail-closed adjudication for CAPACITY-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from measurement.capacity2_registry import CAPACITY2_SPEC, spec_sha256
from measurement.capacity_gate import (
    _finite_tree,
    _metric_shape,
    _valid_receipt,
    _verify_projector,
    _verify_prototypes,
)
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
from measurement.recovery_gate import _close
from measurement.settle_gate import _pair_valid


def _passes(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["selection_accuracy"] >= thresholds["stable_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["stable_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["stable_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = CAPACITY2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CP0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_capacity"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CAPACITY-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = capacity_spec_sha256(CAPACITY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CAPACITY_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
            or payload["source_capacity_pass"] != source_verdict["capacity_pass"]
        ):
            return invalid("CAPACITY-1 source identity or verdict changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("CAPACITY-1 registered episodes changed")

        conditions = {row["name"]: row for row in spec["conditions"]}
        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if (
            set(source_rows) != set(spec["seeds"])
            or set(rows) != set(spec["seeds"])
            or len(rows) != len(payload["seeds"])
        ):
            return invalid("registered seeds are missing or duplicated")

        thresholds = spec["thresholds"]
        judged: dict[str, dict[str, dict]] = {
            str(count): {} for count in spec["event_counts"]
        }
        comparisons_by_seed = {}
        for seed in spec["seeds"]:
            row = rows[seed]
            source_row = source_rows[seed]
            projector_receipt = row["source_checkpoint"]
            prototype_receipt = row["prototype_checkpoint"]
            if (
                projector_receipt != source["checkpoints"].get(str(seed))
                or projector_receipt != source_row["source_checkpoint"]
                or prototype_receipt != source["prototype_checkpoints"].get(str(seed))
                or prototype_receipt != source_row["prototype_checkpoint"]
                or not _verify_projector(projector_receipt, seed, spec)
                or not _verify_prototypes(prototype_receipt, spec)
                or not row["projector_frozen"]
                or not row["projector_unchanged"]
            ):
                return invalid(f"seed {seed} source checkpoint changed")
            condition_rows = {item["name"]: item for item in row["conditions"]}
            if set(condition_rows) != set(conditions) or len(condition_rows) != len(row["conditions"]):
                return invalid(f"seed {seed} condition roster changed")
            source_counts = {item["event_count"]: item for item in source_row["counts"]}
            audit_signatures = {count: [] for count in spec["event_counts"]}
            public_by_condition = {}
            for name, condition_row in condition_rows.items():
                registered = conditions[name]
                if (
                    condition_row["updates"] != registered["updates"]
                    or condition_row["disabled"] != registered["disabled"]
                ):
                    return invalid(f"seed {seed} condition {name} changed")
                count_rows = {item["event_count"]: item for item in condition_row["counts"]}
                if set(count_rows) != set(spec["event_counts"]) or len(count_rows) != len(condition_row["counts"]):
                    return invalid(f"seed {seed} condition {name} event roster changed")
                public_by_condition[name] = count_rows
                for count, item in count_rows.items():
                    if set(item["arms"]) != set(spec["arms"]):
                        return invalid(f"seed {seed} condition {name} count {count} arm roster changed")
                    total = spec["eval_episodes_per_count"]
                    state = item["state_audit"]
                    if (
                        state["episodes"] != total
                        or state["unique_episode_seeds"] != total
                        or len(state["episode_seed_sha256"]) != 64
                        or not spec["minimum_cells"] <= state["minimum_cells"]
                        or state["minimum_cells"] > state["maximum_cells"]
                        or state["maximum_cells"] > spec["maximum_cells"]
                    ):
                        return invalid(f"seed {seed} condition {name} count {count} state stream changed")
                    integration = item["integration_audit"]
                    calls = integration["stable_transform_calls"]
                    expected_calls = count + 1
                    if (
                        calls != {
                            "episodes": total,
                            "total": total * expected_calls,
                            "minimum": expected_calls,
                            "maximum": expected_calls,
                        }
                        or integration["address_width_minimum"] != spec["address_dim"]
                        or integration["address_width_maximum"] != spec["address_dim"]
                    ):
                        return invalid(f"seed {seed} condition {name} count {count} memory path changed")
                    update = item["update_audit"]
                    if (
                        update["requested_updates"] != registered["updates"]
                        or update["performed_updates_minimum"] != registered["updates"]
                        or update["performed_updates_maximum"] != registered["updates"]
                        or update["disabled"] != registered["disabled"]
                        or not all(len(update[field]) == 64 for field in (
                            "state_before_sha256", "state_after_sha256", "query_rng_sha256",
                        ))
                    ):
                        return invalid(f"seed {seed} condition {name} count {count} update audit changed")
                    if name == "baseline" and (
                        update["unchanged_state_count"] != total
                        or update["state_before_sha256"] != update["state_after_sha256"]
                    ):
                        return invalid(f"seed {seed} count {count} baseline state changed")
                    audit_signatures[count].append((
                        state["episode_seed_sha256"],
                        update["state_before_sha256"],
                        update["query_rng_sha256"],
                    ))
                    arms = item["arms"]
                    for arm_name in spec["arms"]:
                        if (
                            not _metric_shape(arms[arm_name], spec["values"])
                            or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                        ):
                            return invalid(f"seed {seed} condition {name} count {count} metrics changed")
                    exact = arms["exact_key_control"]
                    if (
                        exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                        or exact["accuracy"] < thresholds["exact_final_accuracy"]
                        or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                        or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                        or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                    ):
                        return invalid(f"seed {seed} condition {name} count {count} control failed")
                    if name == "baseline" and (
                        item["arms"] != source_counts[count]["arms"]
                        or item["integration_audit"] != source_counts[count]["integration_audit"]
                        or item["state_audit"] != source_counts[count]["state_audit"]
                    ):
                        return invalid(f"seed {seed} count {count} CAPACITY-1 baseline did not replay")
            if any(len(set(values)) != 1 for values in audit_signatures.values()):
                return invalid(f"seed {seed} paired starts or question RNG differ")

            comparisons = {item["event_count"]: item for item in row["comparisons"]}
            if set(comparisons) != set(spec["event_counts"]) or len(comparisons) != len(row["comparisons"]):
                return invalid(f"seed {seed} comparison roster changed")
            comparison_summary = {}
            for count, comparison in comparisons.items():
                settled = public_by_condition["settled"][count]["arms"]["stable_distinct_normal"]
                blocked = public_by_condition["without_frustration_regulation"][count]["arms"]["stable_distinct_normal"]
                total = spec["eval_episodes_per_count"]
                if (
                    not _pair_valid(comparison["final"], total)
                    or not _pair_valid(comparison["selection"], total)
                    or not _close(
                        comparison["final"]["net_accuracy_delta"],
                        settled["accuracy"] - blocked["accuracy"],
                    )
                    or not _close(
                        comparison["selection"]["net_accuracy_delta"],
                        settled["selection_accuracy"] - blocked["selection_accuracy"],
                    )
                ):
                    return invalid(f"seed {seed} count {count} paired comparison changed")
                margin_drop = settled["key_margin_mean"] - blocked["key_margin_mean"]
                mechanism_specific = (
                    comparison["final"]["net_accuracy_delta"] >= thresholds["minimum_mechanism_accuracy_drop"]
                    and comparison["selection"]["net_accuracy_delta"] >= thresholds["minimum_mechanism_accuracy_drop"]
                    and comparison["final"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                    and comparison["selection"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                    and margin_drop > 0
                )
                comparison_summary[str(count)] = {
                    "mechanism_specific": mechanism_specific,
                    "final_drop": comparison["final"]["net_accuracy_delta"],
                    "selection_drop": comparison["selection"]["net_accuracy_delta"],
                    "final_exact_two_sided_p": comparison["final"]["exact_two_sided_p"],
                    "selection_exact_two_sided_p": comparison["selection"]["exact_two_sided_p"],
                    "margin_drop": margin_drop,
                }
                for condition_name in ("baseline", "settled", "without_frustration_regulation"):
                    metrics = public_by_condition[condition_name][count]["arms"]["stable_distinct_normal"]
                    judged[str(count)].setdefault(str(seed), {})[condition_name] = {
                        "passed": _passes(metrics, thresholds),
                        "selection_accuracy": metrics["selection_accuracy"],
                        "final_accuracy": metrics["accuracy"],
                        "minimum_value_recall": min(metrics["per_value_recall"]),
                        "content_accuracy": metrics["correct_content_accuracy"],
                        "key_margin_mean": metrics["key_margin_mean"],
                    }
            comparisons_by_seed[str(seed)] = comparison_summary

        per_seed_pass = {
            str(seed): [
                judged[str(count)][str(seed)]["settled"]["passed"]
                for count in spec["event_counts"]
            ]
            for seed in spec["seeds"]
        }
        joint_pass = [
            all(per_seed_pass[str(seed)][index] for seed in spec["seeds"])
            for index in range(len(spec["event_counts"]))
        ]
        non_monotonic = any(
            values[index] and not all(values[:index])
            for values in per_seed_pass.values()
            for index in range(1, len(values))
        )
        if non_monotonic:
            verdict = "CP5_NON_MONOTONIC"
            reason = "a larger event count passed after a smaller event count failed"
        else:
            boundaries = {
                seed: max(
                    [count for count, passed in zip(spec["event_counts"], values) if passed],
                    default=0,
                )
                for seed, values in per_seed_pass.items()
            }
            if len(set(boundaries.values())) != 1:
                verdict = "CP4_SEED_CONDITIONAL_CAPACITY"
                reason = "the post-settling capacity boundary differed between seeds"
            else:
                boundary = next(iter(boundaries.values()))
                if boundary <= 2:
                    verdict = "CP3_NO_CAPACITY_GAIN"
                    reason = "autonomous settling did not expand the validated two-event boundary"
                else:
                    newly_recovered = [
                        count for count in spec["event_counts"]
                        if count <= boundary and not payload["source_capacity_pass"][str(count)]
                    ]
                    specific = all(
                        comparisons_by_seed[str(seed)][str(count)]["mechanism_specific"]
                        for seed in spec["seeds"]
                        for count in newly_recovered
                    )
                    if not specific:
                        verdict = "CP6_GAIN_NOT_MECHANISM_SPECIFIC"
                        reason = "capacity increased but the new pass did not depend on the registered settling mechanism"
                    elif boundary >= 4:
                        verdict = "CP1_SETTLED_CAPACITY_AT_LEAST_4"
                        reason = "settling expanded the validated boundary through four events"
                    else:
                        verdict = "CP2_SETTLED_CAPACITY_BOUNDARY_3"
                        reason = "settling expanded the validated boundary to three events"
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "settled_capacity_pass": {
                str(count): value for count, value in zip(spec["event_counts"], joint_pass)
            },
            "counts": judged,
            "mechanism_comparisons": comparisons_by_seed,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/capacity2_results.json")
    parser.add_argument("--output", default="measurement/capacity2_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
