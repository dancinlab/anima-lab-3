#!/usr/bin/env python3
"""Fail-closed adjudication for CUE-ALIGN-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

try:
    from measurement.cue_align_registry import CUE_ALIGN_SPEC, calibration_pairs, spec_sha256
    from measurement.cue_context_gate import adjudicate as adjudicate_context
    from measurement.cue_context_registry import CUE_CONTEXT_SPEC, spec_sha256 as context_spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.cue_align_registry import CUE_ALIGN_SPEC, calibration_pairs, spec_sha256
    from measurement.cue_context_gate import adjudicate as adjudicate_context
    from measurement.cue_context_registry import CUE_CONTEXT_SPEC, spec_sha256 as context_spec_sha256
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
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    except (KeyError, TypeError):
        return False


def _metric_valid(metric: dict, classes: int) -> bool:
    return (
        set(metric) == {
            "accuracy", "per_class_recall", "minimum_class_recall",
            "correct_similarity_mean", "closest_wrong_similarity_mean",
            "center_margin_mean", "center_margin_minimum",
            "positive_center_margin_fraction",
        }
        and len(metric["per_class_recall"]) == classes
        and 0 <= metric["accuracy"] <= 1
        and 0 <= metric["minimum_class_recall"] <= 1
    )


def _passes(metric: dict, thresholds: dict) -> bool:
    return metric["accuracy"] >= thresholds["category_accuracy"] and metric[
        "minimum_class_recall"
    ] >= thresholds["minimum_category_recall"]


def _fit_audit_valid(audit: dict, examples: int, spec: dict) -> bool:
    return (
        set(audit) == {
            "method", "examples", "input_dim", "target_dim", "ridge",
            "input_sha256", "target_sha256", "mse", "deterministic",
        }
        and audit["method"] == spec["fit_method"]
        and audit["examples"] == examples
        and audit["input_dim"] == spec["state_dim"]
        and audit["target_dim"] == spec["state_dim"]
        and audit["ridge"] == spec["ridge"]
        and audit["deterministic"] is True
    )


def adjudicate(payload: dict, spec: dict = CUE_ALIGN_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {"experiment": payload.get("experiment", spec["experiment"]),
                "verdict": "CA0_INVALID", "reason": reason,
                "spec_sha256": spec_sha256(spec)}

    try:
        if (payload["experiment"] != spec["experiment"] or payload["spec"] != spec
                or payload["spec_sha256"] != spec_sha256(spec) or not _finite(payload)):
            return invalid("experiment, registered spec, digest, or finite-value check failed")
        for name in ("results", "verdict", "context_checkpoint"):
            if not _receipt_valid(payload["source"][name]):
                return invalid(f"registered source {name} changed")
        upstream = payload["source"]["upstream"]
        for name in ("results", "verdict", "robust_checkpoint", "component_checkpoint",
                     "value_checkpoint"):
            if not _receipt_valid(upstream[name]):
                return invalid(f"registered upstream source {name} changed")
        if any(not _receipt_valid(row) for row in upstream["prototype_checkpoints"].values()):
            return invalid("registered prototype checkpoint changed")
        if not _receipt_valid(payload["checkpoint"]):
            return invalid("CUE-ALIGN-1 checkpoint changed")

        if source_results is None:
            source_results = json.loads(Path(payload["source"]["results"]["path"]).read_text())
        source_verdict = json.loads(Path(payload["source"]["verdict"]["path"]).read_text())
        source_sha = context_spec_sha256(CUE_CONTEXT_SPEC)
        if (source_results.get("spec") != CUE_CONTEXT_SPEC
                or source_results.get("spec_sha256") != source_sha
                or source_verdict.get("verdict") != spec["source_verdict"]
                or adjudicate_context(source_results) != source_verdict
                or payload["source"]["source_spec_sha256"] != source_sha
                or payload["source"]["context_checkpoint"] != source_results["checkpoint"]):
            return invalid("registered CUE-CONTEXT-1 identity changed")

        checkpoint = torch.load(payload["checkpoint"]["path"], map_location="cpu", weights_only=True)
        if (checkpoint.get("experiment") != spec["experiment"]
                or checkpoint.get("spec_sha256") != spec_sha256(spec)
                or checkpoint.get("deterministic") is not True
                or checkpoint.get("fit_audits") != payload["fit_audits"]
                or set(checkpoint.get("states", {})) != {"global_affine", "category_oracle", "wrong_pair"}
                or set(checkpoint["states"]["category_oracle"]) != {
                    str(label) for label in range(spec["contexts"])
                }):
            return invalid("checkpoint identity, model roster, or fit audit changed")

        pairs = calibration_pairs(spec)
        expected_counts = {str(label): pairs // spec["contexts"]
                           for label in range(spec["contexts"])}
        if (payload["calibration_evaluation_overlap"] != 0
                or payload["calibration_pair_audit"]["pairs"] != pairs
                or payload["calibration_pair_audit"]["same_label_pairs"] != pairs
                or payload["calibration_pair_audit"]["label_counts"] != expected_counts
                or payload["mask_overlap_audit"]["exact_overlap"] != 0
                or payload["label_use_audit"] != {
                    "global_affine": False, "category_oracle": True,
                    "wrong_pair_control": True,
                }
                or payload["wrong_pair_audit"] != {
                    "pairs": pairs * 2, "mismatched_label_fraction": 1.0,
                    "rule": spec["wrong_pair_rule"],
                }):
            return invalid("calibration pairing, balance, overlap, or label-use audit changed")
        if not _fit_audit_valid(payload["fit_audits"]["global_affine"], pairs * 2, spec):
            return invalid("global alignment fit audit changed")
        if not _fit_audit_valid(payload["fit_audits"]["wrong_pair"], pairs * 2, spec):
            return invalid("wrong-pair fit audit changed")
        for audit in payload["fit_audits"]["category_oracle"].values():
            if not _fit_audit_valid(audit, pairs * 2 // spec["contexts"], spec):
                return invalid("category alignment fit audit changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        profiles, judged = [], {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (row["prototype_seed"] != identity["prototype_seed"]
                    or row["engine_seed"] != identity["engine_seed"]
                    or row["pair_audit"]["pairs"] != spec["eval_episodes"]
                    or row["source_reference_audit"] != {
                        "query_full": True, "query_quarter_missing": True,
                    }
                    or set(row["models"]) != set(spec["models"])):
                return invalid(f"evaluation {name} identity or source reproduction changed")
            for model in row["models"].values():
                if set(model["conditions"]) != set(spec["conditions"]):
                    return invalid(f"evaluation {name} condition roster changed")
                if any(not _metric_valid(metric, spec["contexts"])
                       for metric in model["conditions"].values()):
                    return invalid(f"evaluation {name} metric shape changed")
                for diagnostic in model["alignment"].values():
                    if set(diagnostic) != {
                        "before_similarity", "after_similarity", "before_mse", "after_mse",
                    }:
                        return invalid(f"evaluation {name} alignment audit changed")

            wrong = row["models"]["wrong_pair"]["conditions"]
            if any(wrong[condition]["accuracy"] > thresholds["wrong_pair_max_accuracy"]
                   for condition in spec["conditions"]):
                return invalid(f"evaluation {name} wrong-pair control failed")
            source = row["models"]["source"]["conditions"]
            global_metrics = row["models"]["global_affine"]["conditions"]
            oracle = row["models"]["category_oracle"]["conditions"]
            source_full = _passes(source["query_full"], thresholds)
            source_damaged = _passes(source["query_quarter_missing"], thresholds)
            if not source_full or source_damaged:
                return invalid(f"evaluation {name} registered source failure profile changed")
            global_full = _passes(global_metrics["query_full"], thresholds)
            global_damaged = _passes(global_metrics["query_quarter_missing"], thresholds)
            oracle_full = _passes(oracle["query_full"], thresholds)
            oracle_damaged = _passes(oracle["query_quarter_missing"], thresholds)
            gain = (global_metrics["query_quarter_missing"]["accuracy"]
                    - source["query_quarter_missing"]["accuracy"])
            profiles.append((global_full, global_damaged, oracle_full, oracle_damaged,
                             gain >= thresholds["minimum_damaged_gain"]))
            judged[name] = {
                "global_full_pass": global_full, "global_damaged_pass": global_damaged,
                "oracle_full_pass": oracle_full, "oracle_damaged_pass": oracle_damaged,
                "damaged_accuracy_gain": gain,
            }

        if all(full and damaged and gain for full, damaged, _, _, gain in profiles):
            verdict, reason = "CA1_COMMON_ALIGNMENT_VALID_NOT_UNIQUE", (
                "one label-free affine alignment recovers held-out full and damaged query states"
            )
        elif all(full and damaged and not gain for full, damaged, _, _, gain in profiles):
            verdict, reason = "CA5_NO_ALIGNMENT_GAIN", (
                "common alignment passes but does not meet the registered damaged-state gain"
            )
        elif all(full and not damaged for full, damaged, _, _, _ in profiles):
            verdict, reason = "CA3_FULL_ONLY_ALIGNMENT", (
                "common alignment recovers full query states but not damaged query states"
            )
        elif all((not full or not damaged) and oracle_full and oracle_damaged
                 for full, damaged, oracle_full, oracle_damaged, _ in profiles):
            verdict, reason = "CA2_CATEGORY_DEPENDENT_WARP", (
                "only the true-label category oracle recovers both query conditions"
            )
        else:
            verdict, reason = "CA4_NONLINEAR_OR_EPISODE_SHIFT", (
                "neither common alignment nor the category oracle recovers all query conditions"
            )
        return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
                "spec_sha256": spec_sha256(spec), "evaluations": judged}
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError):
        return invalid("payload is incomplete or inconsistent with the registered experiment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/cue_align_results.json")
    parser.add_argument("--output", default="measurement/cue_align_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    Path(args.output).write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()

