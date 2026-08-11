#!/usr/bin/env python3
"""Fail-closed adjudication for CUE-CONTEXT-1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

try:
    from measurement.cue_context_registry import (
        CUE_CONTEXT_SPEC, calibration_pairs, spec_sha256, training_mask_plan_audit,
    )
    from measurement.cue_robust_gate import adjudicate as adjudicate_robust
    from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.cue_context_registry import (
        CUE_CONTEXT_SPEC, calibration_pairs, spec_sha256, training_mask_plan_audit,
    )
    from measurement.cue_robust_gate import adjudicate as adjudicate_robust
    from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
    from measurement.projector_registry import evaluation_name


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(row) for row in value.values())
    if isinstance(value, list):
        return all(_finite(row) for row in value)
    return True


def _receipt_valid(receipt: dict) -> bool:
    try:
        path = Path(receipt["path"])
        import hashlib
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    except (KeyError, TypeError):
        return False


def _metric_shape(metric: dict, classes: int) -> bool:
    return (
        set(metric) == {
            "accuracy", "per_class_recall", "minimum_class_recall",
            "correct_similarity_mean", "closest_wrong_similarity_mean",
            "center_margin_mean", "center_margin_minimum",
            "positive_center_margin_fraction",
        }
        and len(metric["per_class_recall"]) == classes
        and 0.0 <= metric["accuracy"] <= 1.0
        and 0.0 <= metric["minimum_class_recall"] <= 1.0
    )


def _passes(metric: dict, thresholds: dict) -> bool:
    return (
        metric["accuracy"] >= thresholds["category_accuracy"]
        and metric["minimum_class_recall"] >= thresholds["minimum_category_recall"]
    )


def adjudicate(payload: dict, spec: dict = CUE_CONTEXT_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CC0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if (
            payload["experiment"] != spec["experiment"]
            or payload["spec"] != spec
            or payload["spec_sha256"] != spec_sha256(spec)
            or not _finite(payload)
        ):
            return invalid("experiment, registered spec, digest, or finite-value check failed")
        source = payload["source"]
        for name in ("results", "verdict", "robust_checkpoint", "component_checkpoint",
                     "value_checkpoint"):
            if not _receipt_valid(source[name]):
                return invalid(f"registered source {name} changed")
        if any(not _receipt_valid(row) for row in source["prototype_checkpoints"].values()):
            return invalid("registered prototype checkpoint changed")
        if not _receipt_valid(payload["checkpoint"]):
            return invalid("CUE-CONTEXT-1 checkpoint changed")

        if source_results is None:
            source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = robust_spec_sha256(CUE_ROBUST_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CUE_ROBUST_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_robust(source_results) != source_verdict
            or source["source_spec_sha256"] != source_sha
            or source["robust_checkpoint"] != source_results["checkpoint"]
        ):
            return invalid("registered CUE-ROBUST-1 identity changed")

        checkpoint = torch.load(
            payload["checkpoint"]["path"], map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("experiment") != spec["experiment"]
            or checkpoint.get("spec_sha256") != spec_sha256(spec)
            or checkpoint.get("deterministic") is not True
            or checkpoint.get("fit_audits") != payload["fit_audits"]
            or set(checkpoint.get("models", {})) != {"storage_only", "query_only", "combined", "fake_query"}
        ):
            return invalid("checkpoint identity, model roster, or fit audit changed")

        pairs = calibration_pairs(spec)
        expected_counts = {str(label): pairs // spec["contexts"] for label in range(spec["contexts"])}
        if (
            payload["calibration_evaluation_overlap"] != 0
            or payload["calibration_pair_audit"]["pairs"] != pairs
            or payload["calibration_pair_audit"]["same_label_pairs"] != pairs
            or payload["calibration_pair_audit"]["label_counts"] != expected_counts
            or payload["training_mask_plan_audit"] != training_mask_plan_audit(spec)
            or payload["mask_overlap_audit"]["exact_overlap"] != 0
            or payload["combined_audit"] != {
                "storage_rows": (pairs + 1) // 2,
                "query_rows": pairs // 2,
                "total_rows": pairs,
            }
        ):
            return invalid("calibration pairing, balance, overlap, or combined schedule changed")
        expected_fit_counts = {
            str(label): pairs * 2 // spec["contexts"] for label in range(spec["contexts"])
        }
        for name, audit in payload["fit_audits"].items():
            if (
                name not in {"storage_only", "query_only", "combined", "fake_query"}
                or audit["method"] != "ridge_fixed_orthogonal_targets"
                or audit["examples"] != pairs * 2
                or audit["label_counts"] != expected_fit_counts
                or audit["deterministic"] is not True
            ):
                return invalid(f"fit audit for {name} changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        profiles = []
        recoveries = []
        judged = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or row["pair_audit"]["pairs"] != spec["eval_episodes"]
                or row["pair_audit"]["same_label_pairs"] != spec["eval_episodes"]
                or row["source_reference_audit"] != {
                    "query_full_match": True, "query_quarter_match": True,
                }
                or set(row["models"]) != set(spec["models"])
            ):
                return invalid(f"evaluation {name} identity, pairing, source, or model roster changed")
            for model in row["models"].values():
                if set(model["conditions"]) != set(spec["conditions"]):
                    return invalid(f"evaluation {name} condition roster changed")
                if any(not _metric_shape(metric, spec["contexts"])
                       for metric in model["conditions"].values()):
                    return invalid(f"evaluation {name} metric shape changed")
                if set(model["paired_similarity"]) != {
                    "state_mean", "state_minimum", "address_mean", "address_minimum",
                }:
                    return invalid(f"evaluation {name} similarity audit changed")

            source_metrics = row["models"]["source"]["conditions"]
            storage_pass = _passes(source_metrics["storage_quarter_missing"], thresholds)
            query_pass = _passes(source_metrics["query_quarter_missing"], thresholds)
            profiles.append((storage_pass, query_pass))
            candidates = []
            for model_name in ("query_only", "combined"):
                metrics = row["models"][model_name]["conditions"]
                recovered = _passes(metrics["query_full"], thresholds) and _passes(
                    metrics["query_quarter_missing"], thresholds
                )
                regression = (
                    row["models"]["storage_only"]["conditions"]["storage_full"]["accuracy"]
                    - metrics["storage_full"]["accuracy"]
                ) > thresholds["maximum_storage_full_regression"]
                candidates.append((model_name, recovered, regression))
            recoveries.append(candidates)
            fake = row["models"]["fake_query"]["conditions"]
            if (
                fake["query_full"]["accuracy"] > thresholds["fake_category_max_accuracy"]
                or fake["query_quarter_missing"]["accuracy"] > thresholds["fake_category_max_accuracy"]
            ):
                return invalid(f"evaluation {name} fake-label control failed")
            judged[name] = {
                "source_storage_quarter_pass": storage_pass,
                "source_query_quarter_pass": query_pass,
                "candidate_recovery": {
                    model_name: {"query_recovered": recovered, "storage_regression": regression}
                    for model_name, recovered, regression in candidates
                },
            }

        if all(storage and query for storage, query in profiles):
            verdict = "CC5_NO_QUERY_TIME_LOSS"
            reason = "existing readout passes damaged storage-time and query-time context states"
        elif all(not storage and not query for storage, query in profiles):
            verdict = "CC2_SHARED_PARTIAL_CUE_LOSS"
            reason = "damaged storage-time and query-time states both fail the category gate"
        elif all(storage and not query for storage, query in profiles):
            if all(any(recovered and not regression for _, recovered, regression in rows)
                   for rows in recoveries):
                verdict = "CC1_QUERY_TIME_SHIFT_CAUSAL"
                reason = "query-time calibration recovers held-out query states without storage regression"
            elif all(any(recovered for _, recovered, _ in rows) for rows in recoveries):
                verdict = "CC4_QUERY_RECOVERY_WITH_STORAGE_REGRESSION"
                reason = "query-time calibration recovers query states but regresses storage states"
            else:
                verdict = "CC3_QUERY_REFIT_INSUFFICIENT"
                reason = "query-time state loss remains after registered query calibration"
        else:
            return invalid("evaluation combinations disagree on the registered time-shift profile")
        return {
            "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "evaluations": judged,
        }
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError):
        return invalid("payload is incomplete or inconsistent with the registered experiment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/cue_context_results.json")
    parser.add_argument("--output", default="measurement/cue_context_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    Path(args.output).write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
