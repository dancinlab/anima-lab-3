#!/usr/bin/env python3
"""Fail-closed adjudication for QUERY-REFRESH-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.cue_history_gate import adjudicate as adjudicate_history
    from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256 as history_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.cue_history_gate import adjudicate as adjudicate_history
    from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256 as history_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256


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
    except (KeyError, TypeError, OSError):
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
    return (
        set(row) == {
            "prediction_agreement", "prediction_disagreement",
            "baseline_errors_corrected_fraction", "baseline_correct_regression_fraction",
            "accuracy_gain", "state_cosine_similarity", "state_mse",
        }
        and all(0 <= row[name] <= 1 for name in (
            "prediction_agreement", "prediction_disagreement",
            "baseline_errors_corrected_fraction", "baseline_correct_regression_fraction",
        ))
        and abs(row["prediction_agreement"] + row["prediction_disagreement"] - 1) < 1e-6
        and row["state_mse"] >= 0
    )


def _passes(metric: dict, thresholds: dict) -> bool:
    return (
        metric["accuracy"] >= thresholds["category_accuracy"]
        and metric["minimum_class_recall"] >= thresholds["minimum_category_recall"]
    )


def _first_sustained(profile: dict[int, bool]) -> int | None:
    ordered = sorted(profile)
    for offset, steps in enumerate(ordered):
        if all(profile[later] for later in ordered[offset:]):
            return steps
    return None


def _classify(recovery_step: int | None, convergence_step: int | None,
              processing_influence: bool) -> tuple[str, str]:
    if recovery_step is not None and convergence_step is not None:
        minimum = max(recovery_step, convergence_step)
        return "QR1_REFRESH_RECOVERS_AND_CONVERGES", (
            f"full and damaged queries recover and event histories converge from {minimum} steps"
        )
    if recovery_step is not None:
        return "QR2_REFRESH_RECOVERS_NOT_CONVERGED", (
            "full and damaged queries recover but event histories do not converge"
        )
    if convergence_step is not None:
        return "QR3_HISTORY_CONVERGES_NOT_RECOVERED", (
            "event histories converge but at least one query condition does not recover"
        )
    if processing_influence:
        return "QR4_REFRESH_CHANGES_NOT_SUFFICIENT", (
            "query refresh changes decisions but does not recover and converge"
        )
    return "QR5_REFRESH_NOT_PRIMARY", (
        "query refresh neither recovers nor materially changes registered decisions"
    )


def adjudicate(payload: dict, spec: dict = QUERY_REFRESH_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "QR0_INVALID",
            "reason": reason,
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
        if any(not _receipt_valid(payload["source"][name]) for name in ("results", "verdict")):
            return invalid("registered CUE-HISTORY-1 source file changed")
        if source_results is None:
            source_results = json.loads(Path(payload["source"]["results"]["path"]).read_text())
        source_verdict = json.loads(Path(payload["source"]["verdict"]["path"]).read_text())
        source_sha = history_spec_sha256(CUE_HISTORY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CUE_HISTORY_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_history(source_results) != source_verdict
            or payload["source"]["source_spec_sha256"] != source_sha
        ):
            return invalid("registered CUE-HISTORY-1 identity changed")

        count = spec["eval_episodes"]
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("evaluation dataset changed")
        source_mask = source_results["mask_audit"]
        if (
            payload["mask_audit"] != source_mask
            or source_mask["states"] != count
            or source_mask["removed_per_state"]
            != round(spec["state_dim"] * spec["missing_fraction"])
        ):
            return invalid("registered damage mask plan changed")
        history_audit = payload["history_audit"]
        if set(history_audit) != set(spec["histories"]):
            return invalid("history roster changed")
        for history, row in history_audit.items():
            if (
                row["episodes"] != count
                or row["query_identity_preserved"] != count
                or row["event_multiset_preserved"] != count
                or row["distractor_history_changed"] != 0
                or row["distractor_roster_preserved_by_context"] is not True
            ):
                return invalid(f"history {history} changed episode identity or content")
        if (
            history_audit["original"]["event_order_changed"] != 0
            or history_audit["event_reversed"]["event_order_changed"] != count
        ):
            return invalid("registered event-order intervention changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        expected_counts = {
            str(label): count // spec["contexts"] for label in range(spec["contexts"])
        }
        baseline_steps = spec["baseline_query_context_steps"]
        steps_roster = {str(steps) for steps in spec["query_context_steps"]}
        recovery_profiles = {steps: [] for steps in spec["query_context_steps"]}
        convergence_profiles = {steps: [] for steps in spec["query_context_steps"]}
        influence_profiles: dict[tuple[int, str], list[bool]] = {
            (steps, history): []
            for steps in spec["query_context_steps"] if steps != baseline_steps
            for history in spec["histories"]
        }
        public_evaluations = {}
        thresholds = spec["thresholds"]
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["source_reference_audit"]) != set(spec["histories"])
                or not all(
                    all(values.values())
                    and set(values) == {
                        "pair_digest_match", "query_full_metric_match",
                        "query_quarter_missing_metric_match",
                    }
                    for values in row["source_reference_audit"].values()
                )
                or set(row["pair_audits"]) != steps_roster
                or set(row["pair_digests"]) != steps_roster
                or set(row["candidates"]) != steps_roster
            ):
                return invalid(f"evaluation {name} identity or source reproduction changed")
            baseline_history_disagreement = row["candidates"][str(baseline_steps)][
                "history_comparison"
            ]["query_quarter_missing"]["prediction_disagreement"]
            public_steps = {}
            baseline_storage_digest = {}
            baseline_label_digest = None
            for steps in spec["query_context_steps"]:
                step_key = str(steps)
                candidate = row["candidates"][step_key]
                if candidate["query_context_steps"] != steps:
                    return invalid(f"evaluation {name} candidate step identity changed")
                if set(candidate["histories"]) != set(spec["histories"]):
                    return invalid(f"evaluation {name} history roster changed at {steps} steps")
                audits = row["pair_audits"][step_key]
                digests = row["pair_digests"][step_key]
                if set(audits) != set(spec["histories"]) or set(digests) != set(spec["histories"]):
                    return invalid(f"evaluation {name} pair roster changed at {steps} steps")
                history_pass = []
                history_public = {}
                for history in spec["histories"]:
                    audit = audits[history]
                    expected_storage_calls = (
                        count * spec["events_per_episode"] * spec["settled_context_steps"]
                    )
                    expected_query_calls = count * steps
                    if (
                        audit["pairs"] != count
                        or audit["same_label_pairs"] != count
                        or audit["unique_trial_seeds"] != count
                        or audit["label_counts"] != expected_counts
                        or not spec["minimum_cells"] <= audit["minimum_cells"]
                        or audit["minimum_cells"] > audit["maximum_cells"]
                        or audit["maximum_cells"] > spec["maximum_cells"]
                        or audit["storage_context_step_calls"] != expected_storage_calls
                        or audit["query_context_step_calls"] != expected_query_calls
                        or audit["query_context_sense_steps"] != steps
                        or audit["context_step_calls"] != expected_storage_calls + expected_query_calls
                        or audit["key_step_calls"]
                        != count * (spec["events_per_episode"] + 1) * spec["key_sense_steps"]
                        or audit["value_step_calls"]
                        != count * spec["events_per_episode"] * spec["value_sense_steps"]
                        or audit["distractor_step_calls"]
                        != count * spec["distractor_steps"] * spec["distractor_sense_steps"]
                    ):
                        return invalid(f"evaluation {name} {history} call audit changed at {steps} steps")
                    digest = digests[history]
                    if set(digest) != {"storage", "query", "labels"} or any(
                        not isinstance(value, str) or len(value) != 64 for value in digest.values()
                    ):
                        return invalid(f"evaluation {name} {history} pair digest changed")
                    if history not in baseline_storage_digest:
                        baseline_storage_digest[history] = digest["storage"]
                    if digest["storage"] != baseline_storage_digest[history]:
                        return invalid(f"evaluation {name} {history} storage changed with query steps")
                    if baseline_label_digest is None:
                        baseline_label_digest = digest["labels"]
                    if digest["labels"] != baseline_label_digest:
                        return invalid(f"evaluation {name} labels changed with history or query steps")
                    conditions = candidate["histories"][history]["conditions"]
                    if set(conditions) != set(spec["conditions"]):
                        return invalid(f"evaluation {name} condition roster changed")
                    condition_pass = []
                    for condition_name, condition in conditions.items():
                        if (
                            not _metric_valid(condition["metric"], spec["contexts"])
                            or not _comparison_valid(condition["comparison_to_baseline_steps"])
                        ):
                            return invalid(f"evaluation {name} metric shape changed")
                        condition_pass.append(_passes(condition["metric"], thresholds))
                    if steps == baseline_steps and any(
                        condition["comparison_to_baseline_steps"]["prediction_disagreement"] != 0
                        or condition["comparison_to_baseline_steps"]["state_mse"] != 0
                        for condition in conditions.values()
                    ):
                        return invalid(f"evaluation {name} baseline comparison is not exact")
                    passed = all(condition_pass)
                    history_pass.append(passed)
                    if steps != baseline_steps:
                        influence_profiles[(steps, history)].append(
                            conditions["query_quarter_missing"][
                                "comparison_to_baseline_steps"
                            ]["prediction_disagreement"]
                            >= thresholds["minimum_processing_disagreement"]
                        )
                    history_public[history] = {
                        condition_name: {
                            "accuracy": condition["metric"]["accuracy"],
                            "minimum_recall": condition["metric"]["minimum_class_recall"],
                            "passed": condition_pass[index],
                            "baseline_disagreement": condition[
                                "comparison_to_baseline_steps"
                            ]["prediction_disagreement"],
                        }
                        for index, (condition_name, condition) in enumerate(conditions.items())
                    }
                comparison = candidate["history_comparison"]
                if set(comparison) != set(spec["conditions"]) or any(
                    not _comparison_valid(value) for value in comparison.values()
                ):
                    return invalid(f"evaluation {name} history comparison changed")
                damaged_disagreement = comparison["query_quarter_missing"][
                    "prediction_disagreement"
                ]
                converged = (
                    damaged_disagreement <= thresholds["maximum_history_disagreement"]
                    and baseline_history_disagreement - damaged_disagreement
                    >= thresholds["minimum_history_disagreement_reduction"]
                )
                recovery_profiles[steps].append(all(history_pass))
                convergence_profiles[steps].append(converged)
                public_steps[step_key] = {
                    "histories": history_public,
                    "history_disagreement": damaged_disagreement,
                    "recovered": all(history_pass),
                    "converged": converged,
                }
            public_evaluations[name] = public_steps

        recovery = {steps: all(values) for steps, values in recovery_profiles.items()}
        convergence = {steps: all(values) for steps, values in convergence_profiles.items()}
        influence = {
            f"{steps}:{history}": all(values)
            for (steps, history), values in influence_profiles.items()
        }
        recovery_step = _first_sustained(recovery)
        convergence_step = _first_sustained(convergence)
        processing_influence = any(influence.values())
        verdict, reason = _classify(recovery_step, convergence_step, processing_influence)
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "minimum_sustained_recovery_steps": recovery_step,
            "minimum_sustained_convergence_steps": convergence_step,
            "processing_influence": processing_influence,
            "recovery": {str(key): value for key, value in recovery.items()},
            "convergence": {str(key): value for key, value in convergence.items()},
            "influence": influence,
            "evaluations": public_evaluations,
        }
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError):
        return invalid("payload is incomplete or malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/query_refresh_results.json")
    parser.add_argument("--output", default="measurement/query_refresh_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output)
    path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
