#!/usr/bin/env python3
"""Fail-closed adjudication for SEEDMAP-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from measurement.capacity2_gate import _passes
from measurement.capacity2_registry import CAPACITY2_SPEC, spec_sha256 as capacity2_spec_sha256
from measurement.capacity_gate import (
    _finite_tree,
    _metric_shape,
    _valid_receipt,
    _verify_projector,
    _verify_prototypes,
)
from measurement.recovery_gate import _close
from measurement.seedmap_registry import SEEDMAP_SPEC, combination_name, spec_sha256
from measurement.settle_gate import _pair_valid


def _tree_close(left, right, tolerance: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _tree_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _tree_close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _direction_passes(pair: dict, source_pass: bool, target_pass: bool, thresholds: dict) -> bool:
    return (
        source_pass
        and not target_pass
        and pair["final"]["net_accuracy_delta"] >= thresholds["minimum_factor_delta"]
        and pair["secondary"]["net_accuracy_delta"] >= thresholds["minimum_factor_delta"]
        and pair["final"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
        and pair["secondary"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
    )


def _classify(factors: dict[str, dict], crossed_passes: list[bool]) -> tuple[str, str]:
    causal = [name for name, row in factors.items() if row["rescue"] and row["reverse"]]
    if len(causal) == 1:
        return "SM1_SINGLE_FACTOR_CAUSAL", "exactly one factor passed both directional swaps"
    if len(causal) > 1:
        return "SM2_MULTIPLE_FACTORS_CAUSAL", "multiple factors passed both directional swaps"
    if any(row["rescue"] != row["reverse"] for row in factors.values()):
        return "SM3_ASYMMETRIC_FACTOR_EFFECT", "a factor passed the registered test in only one direction"
    if any(crossed_passes) and not all(crossed_passes):
        return "SM4_FACTOR_INTERACTION", "crossed combinations changed outcome without a bidirectional single-factor cause"
    return "SM5_NO_FACTOR_EFFECT", "crossed combinations did not isolate a registered factor effect"


def adjudicate(payload: dict, spec: dict = SEEDMAP_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "SM0_INVALID",
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

        source = payload["source_capacity2"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CAPACITY-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = capacity2_spec_sha256(CAPACITY2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CAPACITY2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("CAPACITY-2 source identity or verdict changed")
        if payload["dataset_audit"] != source_results["dataset_audit"][str(spec["event_count"])]:
            return invalid("registered CAPACITY-2 episodes changed")

        inherited = source_results["source_capacity"]
        if (
            payload["projector_checkpoints"] != source["checkpoints"]
            or payload["prototype_checkpoints"] != source["prototype_checkpoints"]
            or source["checkpoints"] != inherited["checkpoints"]
            or source["prototype_checkpoints"] != inherited["prototype_checkpoints"]
        ):
            return invalid("inherited checkpoint roster changed")
        for seed in spec["factor_seeds"]:
            if (
                not _verify_projector(source["checkpoints"][str(seed)], seed, spec)
                or not _verify_prototypes(source["prototype_checkpoints"][str(seed)], spec)
                or not payload["projectors_frozen"][str(seed)]
                or not payload["projectors_unchanged"][str(seed)]
            ):
                return invalid(f"seed {seed} frozen source changed")

        expected = {combination_name(row): row for row in spec["combinations"]}
        rows = {row["name"]: row for row in payload["combinations"]}
        if set(rows) != set(expected) or len(rows) != len(payload["combinations"]):
            return invalid("factor combination roster changed")
        total = spec["eval_episodes"]
        thresholds = spec["thresholds"]
        passes = {}
        summaries = {}
        signatures: dict[int, list[tuple]] = {seed: [] for seed in spec["factor_seeds"]}
        for name, row in rows.items():
            combination = expected[name]
            if any(row[factor] != combination[factor] for factor in spec["factors"]):
                return invalid(f"combination {name} factor identity changed")
            result = row["result"]
            if result["event_count"] != spec["event_count"] or set(result["arms"]) != set(spec["arms"]):
                return invalid(f"combination {name} memory arm roster changed")
            state = result["state_audit"]
            if (
                state["episodes"] != total
                or state["unique_episode_seeds"] != total
                or len(state["episode_seed_sha256"]) != 64
                or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
            ):
                return invalid(f"combination {name} state stream changed")
            integration = result["integration_audit"]
            calls = spec["event_count"] + 1
            if (
                integration["stable_transform_calls"] != {
                    "episodes": total,
                    "total": total * calls,
                    "minimum": calls,
                    "maximum": calls,
                }
                or integration["address_width_minimum"] != spec["address_dim"]
                or integration["address_width_maximum"] != spec["address_dim"]
            ):
                return invalid(f"combination {name} memory path changed")
            update = result["update_audit"]
            if (
                update["requested_updates"] != spec["settling_updates"]
                or update["performed_updates_minimum"] != spec["settling_updates"]
                or update["performed_updates_maximum"] != spec["settling_updates"]
                or update["disabled"] != []
                or not all(len(update[field]) == 64 for field in (
                    "state_before_sha256", "state_after_sha256", "query_rng_sha256",
                ))
            ):
                return invalid(f"combination {name} settling audit changed")
            signatures[combination["engine_seed"]].append((
                state["episode_seed_sha256"], update["state_before_sha256"], update["query_rng_sha256"],
            ))
            arms = result["arms"]
            if any(
                not _metric_shape(arms[arm], spec["values"])
                or arms[arm]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                for arm in spec["arms"]
            ):
                return invalid(f"combination {name} metrics changed")
            exact = arms["exact_key_control"]
            if (
                exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"combination {name} control failed")
            metrics = arms["stable_distinct_normal"]
            passes[name] = _passes(metrics, thresholds)
            summaries[name] = {
                "passed": passes[name],
                "selection_accuracy": metrics["selection_accuracy"],
                "content_accuracy": metrics["correct_content_accuracy"],
                "final_accuracy": metrics["accuracy"],
                "minimum_value_recall": min(metrics["per_value_recall"]),
            }
        if any(len(set(values)) != 1 for values in signatures.values()):
            return invalid("projector or prototype crossing changed paired engine starts")

        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        tolerance = thresholds["source_metric_tolerance"]
        for seed in spec["factor_seeds"]:
            native = combination_name({factor: seed for factor in spec["factors"]})
            settled = next(
                row for row in source_rows[seed]["conditions"] if row["name"] == "settled"
            )
            source_count = next(
                row for row in settled["counts"] if row["event_count"] == spec["event_count"]
            )
            if not _tree_close(rows[native]["result"], source_count, tolerance):
                return invalid(f"native seed {seed} did not replay CAPACITY-2")

        comparisons = {row["factor"]: row for row in payload["comparisons"]}
        if set(comparisons) != set(spec["factors"]) or len(comparisons) != len(payload["comparisons"]):
            return invalid("factor comparison roster changed")
        low, high = spec["factor_seeds"]
        low_native = combination_name({factor: low for factor in spec["factors"]})
        high_native = combination_name({factor: high for factor in spec["factors"]})
        factor_summary = {}
        for factor, comparison in comparisons.items():
            rescue = {name: low for name in spec["factors"]}
            rescue[factor] = high
            reverse = {name: high for name in spec["factors"]}
            reverse[factor] = low
            rescue_name = combination_name(rescue)
            reverse_name = combination_name(reverse)
            if (
                comparison["rescue_combination"] != rescue_name
                or comparison["reverse_combination"] != reverse_name
                or any(not _pair_valid(comparison[direction][metric], total)
                       for direction in ("rescue", "reverse") for metric in ("final", "secondary"))
            ):
                return invalid(f"factor {factor} paired comparison changed")
            secondary_metric = "correct_content_accuracy" if factor == "prototype_seed" else "selection_accuracy"
            expected_deltas = {
                "rescue": (
                    summaries[rescue_name]["final_accuracy"] - summaries[low_native]["final_accuracy"],
                    summaries[rescue_name]["content_accuracy" if factor == "prototype_seed" else "selection_accuracy"]
                    - summaries[low_native]["content_accuracy" if factor == "prototype_seed" else "selection_accuracy"],
                ),
                "reverse": (
                    summaries[high_native]["final_accuracy"] - summaries[reverse_name]["final_accuracy"],
                    summaries[high_native]["content_accuracy" if factor == "prototype_seed" else "selection_accuracy"]
                    - summaries[reverse_name]["content_accuracy" if factor == "prototype_seed" else "selection_accuracy"],
                ),
            }
            for direction in ("rescue", "reverse"):
                if (
                    not _close(comparison[direction]["final"]["net_accuracy_delta"], expected_deltas[direction][0])
                    or not _close(comparison[direction]["secondary"]["net_accuracy_delta"], expected_deltas[direction][1])
                ):
                    return invalid(f"factor {factor} {direction} effect changed")
            rescue_pass = _direction_passes(
                comparison["rescue"], passes[rescue_name], passes[low_native], thresholds
            )
            reverse_pass = _direction_passes(
                comparison["reverse"], passes[high_native], passes[reverse_name], thresholds
            )
            factor_summary[factor] = {
                "rescue": rescue_pass,
                "reverse": reverse_pass,
                "causal": rescue_pass and reverse_pass,
                "secondary_metric": secondary_metric,
                "rescue_combination": rescue_name,
                "reverse_combination": reverse_name,
                "rescue_final_delta": comparison["rescue"]["final"]["net_accuracy_delta"],
                "rescue_secondary_delta": comparison["rescue"]["secondary"]["net_accuracy_delta"],
                "reverse_final_delta": comparison["reverse"]["final"]["net_accuracy_delta"],
                "reverse_secondary_delta": comparison["reverse"]["secondary"]["net_accuracy_delta"],
            }
        crossed = [passed for name, passed in passes.items() if name not in {low_native, high_native}]
        verdict, reason = _classify(factor_summary, crossed)
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "combinations": summaries,
            "factors": factor_summary,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/seedmap_results.json")
    parser.add_argument("--output", default="measurement/seedmap_verdict.json")
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
