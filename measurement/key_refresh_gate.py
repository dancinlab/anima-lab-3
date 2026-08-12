#!/usr/bin/env python3
"""Fail-closed adjudication for KEY-REFRESH-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.key_refresh_registry import KEY_REFRESH_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_gate import (
        _classification_shape, _distance_shape, _finite, _metric_shape,
        _passes, _receipt_valid, adjudicate as adjudicate_source,
    )
    from measurement.query_refresh2_registry import (
        QUERY_REFRESH2_SPEC, spec_sha256 as source_spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.key_refresh_registry import KEY_REFRESH_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_gate import (
        _classification_shape, _distance_shape, _finite, _metric_shape,
        _passes, _receipt_valid, adjudicate as adjudicate_source,
    )
    from measurement.query_refresh2_registry import (
        QUERY_REFRESH2_SPEC, spec_sha256 as source_spec_sha256,
    )


def _first_sustained(profile: dict[int, bool], baseline: int) -> int | None:
    ordered = [steps for steps in sorted(profile) if steps > baseline]
    for offset, steps in enumerate(ordered):
        if all(profile[later] for later in ordered[offset:]):
            return steps
    return None


def _classify(sustained_step: int | None, any_recovery: bool,
              causal: bool, improved: bool) -> tuple[str, str]:
    if sustained_step is not None and causal:
        return "KRF1_KEY_PATH_RECOVERED_AND_SUSTAINED", (
            f"damaged key paths recover and remain recovered from {sustained_step} steps"
        )
    if any_recovery and sustained_step is None:
        return "KRF2_RECOVERED_NOT_SUSTAINED", (
            "a query-key candidate recovers but a later registered candidate regresses"
        )
    if improved:
        return "KRF3_IMPROVED_NOT_RECOVERED", (
            "query-key refresh materially improves both-cue damage but does not sustain recovery"
        )
    return "KRF4_KEY_REFRESH_NOT_CAUSAL", (
        "query-key refresh does not produce the registered causal recovery"
    )


def adjudicate(payload: dict, spec: dict = KEY_REFRESH_SPEC, *,
               source_results: dict | None = None,
               robust_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "KRF0_INVALID", "reason": reason,
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
        if any(not _receipt_valid(source[name]) for name in ("results", "verdict")):
            return invalid("registered QUERY-REFRESH-2 source file changed")
        if source_results is None:
            source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        expected_source_sha = source_spec_sha256(QUERY_REFRESH2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != QUERY_REFRESH2_SPEC
            or source_results.get("spec_sha256") != expected_source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_source(source_results, robust_results=robust_results)
            != source_verdict
            or source["source_spec_sha256"] != expected_source_sha
            or {
                key: value for key, value in source.items()
                if key not in {"results", "verdict", "source_spec_sha256"}
            } != source_results["source"]
        ):
            return invalid("registered QUERY-REFRESH-2 identity changed")

        audit = payload["dataset_audit"]
        if (
            audit != source_results["dataset_audit"]
            or audit["episodes"] != spec["eval_episodes"]
            or audit["unique_fingerprints"] != spec["eval_episodes"]
            or audit["latin_valid_episodes"] != spec["eval_episodes"]
        ):
            return invalid("registered evaluation dataset changed")
        if payload["execution_audit"] != {
            "unique_engine_step_runs": (
                len({row["engine_seed"] for row in spec["evaluation_combinations"]})
                * len(spec["query_key_steps"])
            ),
            "logical_candidate_evaluations": (
                len(spec["evaluation_combinations"]) * len(spec["query_key_steps"])
            ),
            "representative_prototype_seed": min(
                row["prototype_seed"] for row in spec["evaluation_combinations"]
            ),
            "stable_value_path_is_prototype_independent": True,
        }:
            return invalid("registered execution reuse audit changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        baselines = {row["name"]: row for row in source_results["evaluations"]}
        if (
            set(evaluations) != set(registered)
            or set(baselines) != set(registered)
            or len(evaluations) != len(payload["evaluations"])
        ):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        candidate_names = {str(steps) for steps in spec["query_key_steps"]}
        trace_names = {
            "storage_context", "storage_key", "storage_value",
            "query_context", "query_key", "labels",
        }
        expected_path = {
            "minimum_calls": spec["transform_calls_per_episode"],
            "maximum_calls": spec["transform_calls_per_episode"],
            "minimum_components": spec["components_per_key"],
            "maximum_components": spec["components_per_key"],
            "minimum_address_width": spec["composite_address_dim"],
            "maximum_address_width": spec["composite_address_dim"],
            "value_calls": spec["eval_episodes"] * spec["stores_per_episode"],
            "stores": spec["eval_episodes"] * spec["stores_per_episode"],
            "retrievals": spec["eval_episodes"] * spec["retrievals_per_episode"],
        }
        recovery_profile = {steps: [] for steps in spec["query_key_steps"]}
        gains = {steps: [] for steps in spec["query_key_steps"]}
        baseline_failures = []
        public = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["candidates"]) != candidate_names
            ):
                return invalid(f"evaluation {name} identity or candidate roster changed")
            source_baseline = baselines[name]["conditions"]["refreshed_8"]
            stable_trace = None
            for steps in spec["query_key_steps"]:
                result = row["candidates"][str(steps)]
                if (
                    result["prototype_seed"] != identity["prototype_seed"]
                    or result["engine_seed"] != identity["engine_seed"]
                    or set(result["arms"]) != set(spec["arms"])
                    or any(not _metric_shape(arm, spec["values"])
                           for arm in result["arms"].values())
                    or result["frozen_audit"] != {
                        "context": True, "key": True, "value": True,
                    }
                    or result["restoration_audit"] != {
                        "context_restored_to_full": True,
                        "key_restored_to_full": True,
                    }
                    or set(result["component_metrics"]) != set(spec["conditions"])
                    or set(result["distance_metrics"]) != set(spec["conditions"])
                    or set(result["memory_path_audit"]) != set(spec["conditions"])
                    or any(value != expected_path
                           for value in result["memory_path_audit"].values())
                    or set(result["trace_digests"]) != trace_names
                    or set(result["record_digests"]) != set(spec["arms"])
                    or any(len(value) != 64 for value in (
                        *result["trace_digests"].values(),
                        *result["record_digests"].values(),
                    ))
                ):
                    return invalid(f"evaluation {name} candidate {steps} path changed")
                for component_row in result["component_metrics"].values():
                    if (
                        set(component_row) != {"context", "key"}
                        or not _classification_shape(
                            component_row["context"], spec["contexts"]
                        )
                        or not _classification_shape(component_row["key"], spec["keys"])
                    ):
                        return invalid(f"evaluation {name} candidate {steps} component changed")
                if any(not _distance_shape(value)
                       for value in result["distance_metrics"].values()):
                    return invalid(f"evaluation {name} candidate {steps} distance changed")
                state = result["state_audit"]
                if (
                    state["episodes"] != spec["eval_episodes"]
                    or state["unique_episode_seeds"] != spec["eval_episodes"]
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    or state["minimum_cells"] > state["maximum_cells"]
                    or state["maximum_cells"] > spec["maximum_cells"]
                    or state["context_step_calls"] != spec["eval_episodes"] * (
                        spec["events_per_episode"] * spec["settled_context_steps"]
                        + spec["query_context_sense_steps"]
                    )
                    or state["key_step_calls"] != spec["eval_episodes"] * (
                        spec["events_per_episode"] * spec["key_sense_steps"] + steps
                    )
                    or state["value_step_calls"] != spec["eval_episodes"] * (
                        spec["events_per_episode"] * spec["value_sense_steps"]
                    )
                    or state["distractor_step_calls"] != spec["eval_episodes"] * (
                        spec["distractor_steps"] * spec["distractor_sense_steps"]
                    )
                ):
                    return invalid(f"evaluation {name} candidate {steps} sensory audit changed")
                invariant_trace = tuple(result["trace_digests"][key] for key in (
                    "storage_context", "storage_key", "storage_value",
                    "query_context", "labels",
                ))
                if stable_trace is None:
                    stable_trace = invariant_trace
                elif invariant_trace != stable_trace:
                    return invalid(f"evaluation {name} storage, context, or labels changed")
                arms = result["arms"]
                if any(arm["retrieval_api_match"] != 1.0 for arm in arms.values()):
                    return invalid(f"evaluation {name} candidate {steps} API mismatch")
                exact = arms["exact_context_key_control"]
                partner = arms["exact_context_key_partner_swap"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"])
                    < thresholds["exact_minimum_value_recall"]
                    or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
                ):
                    return invalid(f"evaluation {name} candidate {steps} control failed")
                full_ok = _passes(arms["full_cue"], thresholds, full=True)
                context_ok = _passes(arms["context_quarter_missing"], thresholds)
                key_ok = _passes(arms["key_quarter_missing"], thresholds)
                both_ok = _passes(arms["both_quarter_missing"], thresholds)
                recovery_profile[steps].append(
                    full_ok and context_ok and key_ok and both_ok
                )
                public.setdefault(name, {})[str(steps)] = {
                    arm: {
                        "selection_accuracy": arms[arm]["selection_accuracy"],
                        "final_accuracy": arms[arm]["accuracy"],
                    }
                    for arm in spec["conditions"]
                }
            baseline = row["candidates"][str(spec["baseline_query_key_steps"])]
            if baseline != source_baseline:
                return invalid(f"evaluation {name} three-step source reproduction changed")
            baseline_both = baseline["arms"]["both_quarter_missing"]["accuracy"]
            baseline_failures.append(
                baseline_both < thresholds["damaged_final_accuracy"]
            )
            for steps in spec["query_key_steps"]:
                gains[steps].append(
                    row["candidates"][str(steps)]["arms"][
                        "both_quarter_missing"
                    ]["accuracy"] - baseline_both
                )

        combined_profile = {
            steps: all(values) for steps, values in recovery_profile.items()
        }
        sustained_step = _first_sustained(
            combined_profile, spec["baseline_query_key_steps"]
        )
        any_recovery = any(
            passed for steps, passed in combined_profile.items()
            if steps > spec["baseline_query_key_steps"]
        )
        causal = (
            sustained_step is not None
            and all(baseline_failures)
            and all(
                gain >= thresholds["minimum_key_final_gain"]
                for gain in gains[sustained_step]
            )
        )
        improved = any(
            all(gain >= thresholds["minimum_key_final_gain"] for gain in values)
            for steps, values in gains.items()
            if steps > spec["baseline_query_key_steps"]
        )
        verdict, reason = _classify(sustained_step, any_recovery, causal, improved)
        return {
            "experiment": spec["experiment"], "verdict": verdict,
            "reason": reason, "spec_sha256": spec_sha256(spec),
            "minimum_sustained_recovery_steps": sustained_step,
            "recovery_profile": {
                str(steps): value for steps, value in combined_profile.items()
            },
            "causal": causal, "improved": improved,
            "evaluations": public,
        }
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError,
            ZeroDivisionError):
        return invalid("payload is incomplete or malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/key_refresh_results.json")
    parser.add_argument("--output", default="measurement/key_refresh_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
