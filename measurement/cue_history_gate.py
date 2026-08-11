#!/usr/bin/env python3
"""Fail-closed adjudication for CUE-HISTORY-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.cue_align_gate import adjudicate as adjudicate_align
    from measurement.cue_align_registry import CUE_ALIGN_SPEC, spec_sha256 as align_spec_sha256
    from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.cue_align_gate import adjudicate as adjudicate_align
    from measurement.cue_align_registry import CUE_ALIGN_SPEC, spec_sha256 as align_spec_sha256
    from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256
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


def _comparison_valid(row: dict) -> bool:
    return set(row) == {
        "prediction_agreement", "prediction_disagreement",
        "baseline_errors_corrected_fraction", "baseline_correct_regression_fraction",
        "accuracy_gain", "state_cosine_similarity", "state_mse",
    } and all(0 <= row[name] <= 1 for name in (
        "prediction_agreement", "prediction_disagreement",
        "baseline_errors_corrected_fraction", "baseline_correct_regression_fraction",
    )) and abs(row["prediction_agreement"] + row["prediction_disagreement"] - 1) < 1e-6


def _passes(metric: dict, thresholds: dict) -> bool:
    return metric["accuracy"] >= thresholds["category_accuracy"] and metric[
        "minimum_class_recall"
    ] >= thresholds["minimum_category_recall"]


def _recovered(history: dict, original: dict, thresholds: dict) -> bool:
    return (
        _passes(history["query_full"]["metric"], thresholds)
        and _passes(history["query_quarter_missing"]["metric"], thresholds)
        and history["query_quarter_missing"]["metric"]["accuracy"]
        - original["query_quarter_missing"]["metric"]["accuracy"]
        >= thresholds["minimum_damaged_gain"]
    )


def _sensitive(history: dict, thresholds: dict) -> bool:
    return history["query_quarter_missing"]["comparison_to_original"][
        "prediction_disagreement"
    ] >= thresholds["minimum_prediction_disagreement"]


def adjudicate(payload: dict, spec: dict = CUE_HISTORY_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {"experiment": payload.get("experiment", spec["experiment"]),
                "verdict": "CH0_INVALID", "reason": reason,
                "spec_sha256": spec_sha256(spec)}

    try:
        if (payload["experiment"] != spec["experiment"] or payload["spec"] != spec
                or payload["spec_sha256"] != spec_sha256(spec) or not _finite(payload)):
            return invalid("experiment, registered spec, digest, or finite-value check failed")
        for name in ("results", "verdict"):
            if not _receipt_valid(payload["source"][name]):
                return invalid(f"registered source {name} changed")
        if source_results is None:
            source_results = json.loads(Path(payload["source"]["results"]["path"]).read_text())
        source_verdict = json.loads(Path(payload["source"]["verdict"]["path"]).read_text())
        source_sha = align_spec_sha256(CUE_ALIGN_SPEC)
        if (source_results.get("spec") != CUE_ALIGN_SPEC
                or source_results.get("spec_sha256") != source_sha
                or source_verdict.get("verdict") != spec["source_verdict"]
                or adjudicate_align(source_results) != source_verdict
                or payload["source"]["source_spec_sha256"] != source_sha):
            return invalid("registered CUE-ALIGN-1 identity changed")

        count = spec["eval_episodes"]
        audit = payload["history_audit"]
        if set(audit) != set(spec["histories"]):
            return invalid("history roster changed")
        for history, row in audit.items():
            if (row["episodes"] != count or row["query_identity_preserved"] != count
                    or row["event_multiset_preserved"] != count
                    or row["distractor_roster_preserved_by_context"] is not True):
                return invalid(f"history {history} changed episode identity or content")
        if (audit["original"]["event_order_changed"] != 0
                or audit["original_repeat"]["event_order_changed"] != 0
                or audit["distractor_swapped"]["event_order_changed"] != 0
                or audit["event_reversed"]["event_order_changed"] != count
                or audit["both_changed"]["event_order_changed"] != count
                or audit["original"]["distractor_history_changed"] != 0
                or audit["original_repeat"]["distractor_history_changed"] != 0
                or audit["event_reversed"]["distractor_history_changed"] != 0):
            return invalid("registered event-order intervention changed")
        minimum_changed = count * spec["thresholds"]["minimum_distractor_changed_fraction"]
        if (audit["distractor_swapped"]["distractor_history_changed"] < minimum_changed
                or audit["both_changed"]["distractor_history_changed"] < minimum_changed):
            return invalid("distractor history intervention changed too few episodes")

        if (payload["dataset_audit"] != source_results["evaluation_dataset_audit"]
                or payload["mask_audit"]["states"] != count
                or payload["mask_audit"]["removed_per_state"]
                != round(spec["state_dim"] * spec["missing_fraction"])):
            return invalid("dataset or damage mask plan changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        profiles = []
        judged = {}
        thresholds = spec["thresholds"]
        for name, row in evaluations.items():
            identity = registered[name]
            if (row["prototype_seed"] != identity["prototype_seed"]
                    or row["engine_seed"] != identity["engine_seed"]
                    or row["source_reference_audit"] != {
                        "query_full": True, "query_quarter_missing": True,
                    }
                    or row["repeat_exact"] != {
                        "storage": True, "query": True, "labels": True,
                    }
                    or set(row["histories"]) != set(spec["histories"])):
                return invalid(f"evaluation {name} identity or source reproduction changed")
            base_audit = row["pair_audits"]["original"]
            expected_counts = {str(label): count // spec["contexts"]
                               for label in range(spec["contexts"])}
            for history in spec["histories"]:
                pair_audit = row["pair_audits"][history]
                if (pair_audit["pairs"] != count or pair_audit["label_counts"] != expected_counts
                        or pair_audit["context_step_calls"] != base_audit["context_step_calls"]
                        or pair_audit["key_step_calls"] != base_audit["key_step_calls"]
                        or pair_audit["value_step_calls"] != base_audit["value_step_calls"]
                        or pair_audit["distractor_step_calls"] != base_audit["distractor_step_calls"]):
                    return invalid(f"evaluation {name} history {history} call audit changed")
                conditions = row["histories"][history]["conditions"]
                if set(conditions) != set(spec["conditions"]):
                    return invalid(f"evaluation {name} condition roster changed")
                for condition in conditions.values():
                    if (not _metric_valid(condition["metric"], spec["contexts"])
                            or not _comparison_valid(condition["comparison_to_original"])):
                        return invalid(f"evaluation {name} metric shape changed")
            original = row["histories"]["original"]["conditions"]
            repeat = row["histories"]["original_repeat"]["conditions"]
            if repeat != original:
                return invalid(f"evaluation {name} repeat metrics changed")
            if (not _passes(original["query_full"]["metric"], thresholds)
                    or _passes(original["query_quarter_missing"]["metric"], thresholds)):
                return invalid(f"evaluation {name} source failure profile changed")
            recovered = {
                history: _recovered(row["histories"][history]["conditions"], original, thresholds)
                for history in spec["judged_histories"]
            }
            sensitive = {
                history: _sensitive(row["histories"][history]["conditions"], thresholds)
                for history in spec["judged_histories"]
            }
            profiles.append((recovered, sensitive))
            judged[name] = {"recovered": recovered, "sensitive": sensitive}

        all_recovered = {
            history: all(row[0][history] for row in profiles)
            for history in spec["judged_histories"]
        }
        all_sensitive = {
            history: all(row[1][history] for row in profiles)
            for history in spec["judged_histories"]
        }
        event = all_recovered["event_reversed"]
        distractor = all_recovered["distractor_swapped"]
        both = all_recovered["both_changed"]
        if event and not distractor:
            verdict, reason = "CH1_EVENT_ORDER_CAUSAL", (
                "reversing event order recovers held-out full and damaged query classification"
            )
        elif distractor and not event:
            verdict, reason = "CH2_DISTRACTOR_HISTORY_CAUSAL", (
                "swapping distractor history recovers held-out full and damaged query classification"
            )
        elif (event and distractor) or (both and not event and not distractor):
            verdict, reason = "CH3_MIXED_HISTORY_CAUSAL", (
                "both history components recover or their combined intervention is required"
            )
        elif any(all_sensitive.values()):
            verdict, reason = "CH4_HISTORY_SENSITIVE_NOT_SUFFICIENT", (
                "processing history changes decisions but no registered intervention recovers them"
            )
        else:
            verdict, reason = "CH5_HISTORY_NOT_PRIMARY", (
                "registered processing-history changes neither recover nor materially alter decisions"
            )
        return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
                "spec_sha256": spec_sha256(spec), "evaluations": judged,
                "all_recovered": all_recovered, "all_sensitive": all_sensitive}
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError):
        return invalid("payload is incomplete or malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/cue_history_results.json")
    parser.add_argument("--output", default="measurement/cue_history_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    Path(args.output).write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
