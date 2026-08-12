#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-WRITE-MECHANISM-1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from measurement.write_mechanism_registry import WRITE_MECHANISM_SPEC, spec_sha256


def _invalid(reason: str) -> dict:
    return {"verdict": "GWM0_INVALID", "reason": reason, "seeds": []}


def _rate_map(value: object, keys: list[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(keys)
        and all(isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1
                for rate in value.values())
    )


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _movement_fraction(baseline: float, peer: float, intervention: float) -> float:
    gap = peer - baseline
    if abs(gap) <= 1e-15:
        return 1.0 if _close(intervention, baseline) else float("-inf")
    return (intervention - baseline) / gap


def adjudicate(payload: dict, spec: dict = WRITE_MECHANISM_SPEC) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") != "25f626b3f"
    ):
        return _invalid("registration mismatch")
    runtime = payload.get("runtime", {})
    if any(runtime.get(name) != spec["runtime"][name] for name in ("torch", "transformers")):
        return _invalid("runtime mismatch")
    rows = payload.get("seeds")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != spec["seeds"]:
        return _invalid("seed roster mismatch")
    by_seed = {row["seed"]: row for row in rows}
    thresholds = spec["thresholds"]
    summaries = []

    for row in rows:
        seed = row["seed"]
        peer = next(value for value in spec["seeds"] if value != seed)
        if row.get("peer_seed") != peer or set(row.get("arms", {})) != set(spec["arms"]):
            return _invalid(f"seed {seed} peer or arm roster changed")
        for arm_name, arm in row["arms"].items():
            expected_sources = {
                factor: peer if factor in spec["arms"][arm_name] else seed
                for factor in spec["factors"]
            }
            factor_audit = arm.get("factor_audit", {})
            if (
                factor_audit.get("sources") != expected_sources
                or set(factor_audit.get("source_sha256", {})) != set(spec["factors"])
                or any(len(value) != 64 for value in factor_audit["source_sha256"].values())
            ):
                return _invalid(f"seed {seed} arm {arm_name} factor audit failed")
            dataset = arm.get("dataset_audit", {})
            if (
                dataset.get("calibration_rows") != spec["calibration_rows"]
                or dataset.get("calibration_unique") != spec["calibration_rows"]
                or dataset.get("evaluation_episodes") != spec["evaluation_episodes"]
                or dataset.get("evaluation_candidates")
                != spec["evaluation_episodes"] * spec["candidates_per_episode"]
                or dataset.get("evaluation_unique") != dataset.get("evaluation_candidates")
                or dataset.get("overlap") != 0
                or any(len(dataset.get(name, "")) != 64 for name in (
                    "calibration_sha256", "evaluation_sha256", "calibration_labels_sha256",
                ))
            ):
                return _invalid(f"seed {seed} arm {arm_name} dataset audit failed")
            threshold = arm.get("selection_threshold")
            if (
                not isinstance(threshold, (int, float))
                or not _close(
                    threshold,
                    thresholds["expected_selection_threshold"],
                    thresholds["selection_threshold_tolerance"],
                )
            ):
                return _invalid(f"seed {seed} arm {arm_name} threshold changed")
            checkpoint = arm.get("checkpoint", {})
            if not checkpoint.get("path") or len(checkpoint.get("sha256", "")) != 64:
                return _invalid(f"seed {seed} arm {arm_name} checkpoint audit failed")
            metrics = arm.get("metrics", {})
            if (
                any(not isinstance(metrics.get(name), (int, float))
                    or not 0 <= metrics[name] <= 1 for name in (
                        "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
                    ))
                or not isinstance(metrics.get("stored"), int)
                or not _rate_map(metrics.get("per_kind_storage_rate"), spec["fact_kinds"])
                or not _rate_map(
                    metrics.get("per_position_storage_rate"),
                    [str(position) for position in spec["fact_positions"]],
                )
                or not _rate_map(
                    metrics.get("per_distractor_storage_rate"), spec["distractor_kinds"]
                )
                or len(metrics.get("selection_sha256", "")) != 64
                or len(metrics.get("scores_sha256", "")) != 64
            ):
                return _invalid(f"seed {seed} arm {arm_name} metric shape failed")
            if (
                metrics["distractor_storage_rate"]
                > thresholds["maximum_distractor_storage_rate"]
                or max(metrics["per_distractor_storage_rate"].values())
                > thresholds["maximum_per_distractor_storage_rate"]
            ):
                return _invalid(f"seed {seed} arm {arm_name} distractor control failed")

        baseline = row["arms"]["baseline"]
        expected = spec["expected_baselines"][str(seed)]
        base_metrics = baseline["metrics"]
        if (
            not _close(base_metrics["important_storage_rate"], expected["important_storage_rate"])
            or not _close(base_metrics["distractor_storage_rate"], expected["distractor_storage_rate"])
            or not _close(
                base_metrics["per_kind_storage_rate"]["commitment"],
                expected["commitment_storage_rate"],
            )
            or base_metrics["selection_sha256"] != expected["selection_sha256"]
        ):
            return _invalid(f"seed {seed} baseline did not replay GATE-3")

    # Swapping every data-generation factor must reproduce the peer's data and selection.
    for seed, row in by_seed.items():
        peer = row["peer_seed"]
        swapped = row["arms"]["all_swap"]
        peer_base = by_seed[peer]["arms"]["baseline"]
        for name in ("calibration_sha256", "evaluation_sha256", "calibration_labels_sha256"):
            if swapped["dataset_audit"][name] != peer_base["dataset_audit"][name]:
                return _invalid(f"seed {seed} all-factor data did not reproduce peer")

    factor_results = {}
    causal_candidates = []
    for factor in spec["factors"]:
        arm_name = f"{factor}_swap"
        fractions = []
        changes = []
        for seed, row in by_seed.items():
            peer = row["peer_seed"]
            baseline = row["arms"]["baseline"]["metrics"]
            target = by_seed[peer]["arms"]["baseline"]["metrics"]
            arm = row["arms"][arm_name]["metrics"]
            for getter in (
                lambda value: value["important_storage_rate"],
                lambda value: value["per_kind_storage_rate"]["commitment"],
            ):
                fractions.append(_movement_fraction(getter(baseline), getter(target), getter(arm)))
                changes.append(abs(getter(arm) - getter(baseline)))
        factor_results[factor] = {
            "minimum_peer_gap_fraction": min(fractions),
            "maximum_absolute_change": max(changes),
            "fractions": fractions,
        }

    for factor in spec["factors"]:
        inactive = [
            factor_results[name]["maximum_absolute_change"]
            for name in spec["factors"] if name != factor
        ]
        if (
            factor_results[factor]["minimum_peer_gap_fraction"]
            >= thresholds["minimum_peer_gap_fraction"]
            and max(inactive) < thresholds["maximum_inactive_factor_change"]
        ):
            causal_candidates.append(factor)

    all_swap_reproduces = all(
        row["arms"]["all_swap"]["metrics"]["selection_sha256"]
        == by_seed[row["peer_seed"]]["arms"]["baseline"]["metrics"]["selection_sha256"]
        for row in rows
    )
    verdict_map = {
        "template": "GWM1_TEMPLATE_CAUSAL",
        "identifier": "GWM2_IDENTIFIER_CAUSAL",
        "layout": "GWM3_LAYOUT_CAUSAL",
    }
    verdict = (
        verdict_map[causal_candidates[0]]
        if len(causal_candidates) == 1
        else "GWM4_MULTIFACTOR"
        if all_swap_reproduces
        else "GWM5_UNEXPLAINED"
    )
    for row in rows:
        seed = row["seed"]
        summaries.append({
            "seed": seed,
            "baseline": row["arms"]["baseline"]["metrics"],
            "swaps": {
                name: arm["metrics"] for name, arm in row["arms"].items() if name != "baseline"
            },
        })
    return {
        "verdict": verdict,
        "reason": "registered seed-factor decomposition completed",
        "causal_candidates": causal_candidates,
        "all_swap_reproduces_peer": all_swap_reproduces,
        "factor_results": factor_results,
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result", nargs="?", type=Path,
        default=Path("measurement/write_mechanism_results.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/write_mechanism_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.result.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(verdict["verdict"])


if __name__ == "__main__":
    main()
