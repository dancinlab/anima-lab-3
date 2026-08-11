#!/usr/bin/env python3
"""Fail-closed adjudication for QUERY-REFRESH-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _metric_shape
    from measurement.cue_mechanism_gate import (
        _classification_shape, _distance_shape, _finite, _receipt_valid,
    )
    from measurement.cue_robust_gate import adjudicate as adjudicate_robust
    from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_registry import QUERY_REFRESH2_SPEC, spec_sha256
    from measurement.query_refresh_gate import adjudicate as adjudicate_refresh
    from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256 as refresh_spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _metric_shape
    from measurement.cue_mechanism_gate import (
        _classification_shape, _distance_shape, _finite, _receipt_valid,
    )
    from measurement.cue_robust_gate import adjudicate as adjudicate_robust
    from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_registry import QUERY_REFRESH2_SPEC, spec_sha256
    from measurement.query_refresh_gate import adjudicate as adjudicate_refresh
    from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256 as refresh_spec_sha256


def _passes(arm: dict, thresholds: dict, *, full: bool = False) -> bool:
    return (
        arm["selection_accuracy"] >= thresholds[
            "full_selection_accuracy" if full else "damaged_selection_accuracy"
        ]
        and arm["accuracy"] >= thresholds[
            "full_final_accuracy" if full else "damaged_final_accuracy"
        ]
        and (not full or min(arm["per_value_recall"])
             >= thresholds["full_minimum_value_recall"])
    )


def _classify(full_pass: bool, context_pass: bool, key_pass: bool,
              both_pass: bool, context_gain: bool, causal: bool) -> tuple[str, str]:
    if not full_pass:
        return "QRI5_FULL_BEHAVIOR_REGRESSION", "eight-step refresh regresses full-cue behavior"
    if context_pass and key_pass and both_pass and causal:
        return "QRI1_FULL_PATH_RECOVERED", "all registered partial-cue behavior recovers causally"
    if context_pass and causal:
        return "QRI2_CONTEXT_PATH_RECOVERED", (
            "context-cue behavior recovers causally but key or joint damage remains"
        )
    if context_gain:
        return "QRI3_BEHAVIOR_IMPROVED_NOT_RECOVERED", (
            "context-cue behavior improves materially but remains below criterion"
        )
    return "QRI4_REFRESH_NOT_CAUSAL", "query refresh does not meet the causal behavior criterion"


def adjudicate(payload: dict, spec: dict = QUERY_REFRESH2_SPEC,
               *, robust_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "QRI0_INVALID", "reason": reason,
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
        for name in (
            "refresh_results", "refresh_verdict", "robust_results", "robust_verdict",
            "robust_checkpoint", "component_checkpoint", "value_checkpoint",
        ):
            if not _receipt_valid(source[name]):
                return invalid(f"registered source {name} changed")
        if any(not _receipt_valid(row) for row in source["prototype_checkpoints"].values()):
            return invalid("registered prototype checkpoint changed")

        refresh_results = json.loads(Path(source["refresh_results"]["path"]).read_text())
        refresh_verdict = json.loads(Path(source["refresh_verdict"]["path"]).read_text())
        refresh_sha = refresh_spec_sha256(QUERY_REFRESH_SPEC)
        if (
            refresh_results.get("experiment") != spec["source_refresh_experiment"]
            or refresh_results.get("spec") != QUERY_REFRESH_SPEC
            or refresh_results.get("spec_sha256") != refresh_sha
            or refresh_verdict.get("verdict") != spec["source_refresh_verdict"]
            or refresh_verdict.get("minimum_sustained_recovery_steps") != 8
            or adjudicate_refresh(refresh_results) != refresh_verdict
            or source["refresh_spec_sha256"] != refresh_sha
        ):
            return invalid("registered QUERY-REFRESH-1 identity changed")

        if robust_results is None:
            robust_results = json.loads(Path(source["robust_results"]["path"]).read_text())
        robust_verdict = json.loads(Path(source["robust_verdict"]["path"]).read_text())
        robust_sha = robust_spec_sha256(CUE_ROBUST_SPEC)
        if (
            robust_results.get("experiment") != spec["source_robust_experiment"]
            or robust_results.get("spec") != CUE_ROBUST_SPEC
            or robust_results.get("spec_sha256") != robust_sha
            or robust_verdict.get("verdict") != spec["source_robust_verdict"]
            or adjudicate_robust(robust_results) != robust_verdict
            or source["robust_spec_sha256"] != robust_sha
            or source["robust_checkpoint"] != robust_results["checkpoint"]
            or source["component_checkpoint"] != robust_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != robust_results["source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != robust_results["source"]["prototype_checkpoints"]
        ):
            return invalid("registered CUE-ROBUST-1 identity changed")

        audit = payload["dataset_audit"]
        if (
            audit != robust_results["dataset_audit"]
            or audit["episodes"] != spec["eval_episodes"]
            or audit["unique_fingerprints"] != spec["eval_episodes"]
            or audit["latin_valid_episodes"] != spec["eval_episodes"]
        ):
            return invalid("registered evaluation dataset changed")
        if payload["execution_audit"] != {
            "unique_engine_step_runs": (
                len({row["engine_seed"] for row in spec["evaluation_combinations"]})
                * len(set(spec["runtime_conditions"].values()))
            ),
            "logical_condition_evaluations": (
                len(spec["evaluation_combinations"]) * len(spec["runtime_conditions"])
            ),
            "representative_prototype_seed": min(
                row["prototype_seed"] for row in spec["evaluation_combinations"]
            ),
            "stable_value_path_is_prototype_independent": True,
            "condition_aliases": {
                "disabled_6": "baseline_6", "recovered_8": "refreshed_8",
            },
        }:
            return invalid("registered execution reuse audit changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        baselines = {row["name"]: row for row in robust_results["evaluations"]}
        if (
            set(evaluations) != set(registered)
            or set(baselines) != set(registered)
            or len(evaluations) != len(payload["evaluations"])
        ):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        condition_names = set(spec["runtime_conditions"])
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
        public = {}
        full_profile = []
        context_profile = []
        key_profile = []
        both_profile = []
        gain_profile = []
        causal_profile = []
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["conditions"]) != condition_names
            ):
                return invalid(f"evaluation {name} identity or condition roster changed")
            baseline_source = baselines[name]
            storage_reference = None
            labels_reference = None
            for condition, result in row["conditions"].items():
                steps = spec["runtime_conditions"][condition]
                if (
                    result["prototype_seed"] != identity["prototype_seed"]
                    or result["engine_seed"] != identity["engine_seed"]
                    or set(result["arms"]) != set(spec["arms"])
                    or any(not _metric_shape(arm, spec["values"])
                           for arm in result["arms"].values())
                    or result["frozen_audit"] != {"context": True, "key": True, "value": True}
                    or result["restoration_audit"] != {
                        "context_restored_to_full": True, "key_restored_to_full": True,
                    }
                    or set(result["component_metrics"]) != set(spec["conditions"])
                    or set(result["distance_metrics"]) != set(spec["conditions"])
                    or set(result["memory_path_audit"]) != set(spec["conditions"])
                    or any(value != expected_path for value in result["memory_path_audit"].values())
                    or set(result["trace_digests"]) != trace_names
                    or set(result["record_digests"]) != set(spec["arms"])
                    or any(len(value) != 64 for value in (
                        *result["trace_digests"].values(), *result["record_digests"].values()
                    ))
                ):
                    return invalid(f"evaluation {name} {condition} path or digest shape changed")
                for component_row in result["component_metrics"].values():
                    if (
                        set(component_row) != {"context", "key"}
                        or not _classification_shape(component_row["context"], spec["contexts"])
                        or not _classification_shape(component_row["key"], spec["keys"])
                    ):
                        return invalid(f"evaluation {name} {condition} component metric changed")
                if any(not _distance_shape(value) for value in result["distance_metrics"].values()):
                    return invalid(f"evaluation {name} {condition} distance metric changed")
                state = result["state_audit"]
                expected_context_calls = spec["eval_episodes"] * (
                    spec["events_per_episode"] * spec["settled_context_steps"] + steps
                )
                if (
                    state["episodes"] != spec["eval_episodes"]
                    or state["unique_episode_seeds"] != spec["eval_episodes"]
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    or state["minimum_cells"] > state["maximum_cells"]
                    or state["maximum_cells"] > spec["maximum_cells"]
                    or state["context_step_calls"] != expected_context_calls
                    or state["key_step_calls"] != spec["eval_episodes"] * (
                        spec["events_per_episode"] + 1
                    ) * spec["key_sense_steps"]
                    or state["value_step_calls"] != spec["eval_episodes"] * (
                        spec["events_per_episode"] * spec["value_sense_steps"]
                    )
                    or state["distractor_step_calls"] != spec["eval_episodes"] * (
                        spec["distractor_steps"] * spec["distractor_sense_steps"]
                    )
                ):
                    return invalid(f"evaluation {name} {condition} sensory call audit changed")
                storage = tuple(result["trace_digests"][key] for key in (
                    "storage_context", "storage_key", "storage_value",
                ))
                labels = result["trace_digests"]["labels"]
                if storage_reference is None:
                    storage_reference, labels_reference = storage, labels
                elif storage != storage_reference or labels != labels_reference:
                    return invalid(f"evaluation {name} storage or labels changed with query refresh")
                arms = result["arms"]
                if any(arm["retrieval_api_match"] != 1.0 for arm in arms.values()):
                    return invalid(f"evaluation {name} {condition} memory API mismatch")
                exact = arms["exact_context_key_control"]
                partner = arms["exact_context_key_partner_swap"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
                ):
                    return invalid(f"evaluation {name} {condition} exact or partner control failed")

            baseline = row["conditions"][spec["baseline_condition"]]
            refreshed = row["conditions"][spec["refreshed_condition"]]
            disabled = row["conditions"][spec["disabled_condition"]]
            recovered = row["conditions"][spec["recovered_condition"]]
            if (
                baseline["arms"] != baseline_source["arms"]
                or baseline["component_metrics"] != baseline_source["component_metrics"]
                or baseline["distance_metrics"] != baseline_source["distance_metrics"]
                or baseline["state_audit"] != baseline_source["state_audit"]
                or baseline["reference_audit"] != {
                    "full_metric_match": True, "both_quarter_metric_match": True,
                }
                or baseline["record_digests"] != disabled["record_digests"]
                or baseline["trace_digests"] != disabled["trace_digests"]
                or refreshed["record_digests"] != recovered["record_digests"]
                or refreshed["trace_digests"] != recovered["trace_digests"]
            ):
                return invalid(f"evaluation {name} baseline, disable, or recovery reproduction changed")

            base_context = baseline["arms"]["context_quarter_missing"]
            fresh_arms = refreshed["arms"]
            context = fresh_arms["context_quarter_missing"]
            full_ok = _passes(fresh_arms["full_cue"], thresholds, full=True)
            context_ok = _passes(context, thresholds)
            key_ok = _passes(fresh_arms["key_quarter_missing"], thresholds)
            both_ok = _passes(fresh_arms["both_quarter_missing"], thresholds)
            gain = context["accuracy"] - base_context["accuracy"]
            gain_ok = gain >= thresholds["minimum_context_final_gain"]
            causal_ok = gain_ok and base_context["accuracy"] < thresholds["damaged_final_accuracy"]
            full_profile.append(full_ok); context_profile.append(context_ok)
            key_profile.append(key_ok); both_profile.append(both_ok)
            gain_profile.append(gain_ok); causal_profile.append(causal_ok)
            public[name] = {
                "baseline": {
                    arm: {
                        "selection_accuracy": baseline["arms"][arm]["selection_accuracy"],
                        "final_accuracy": baseline["arms"][arm]["accuracy"],
                    } for arm in spec["conditions"]
                },
                "refreshed": {
                    arm: {
                        "selection_accuracy": fresh_arms[arm]["selection_accuracy"],
                        "final_accuracy": fresh_arms[arm]["accuracy"],
                    } for arm in spec["conditions"]
                },
                "context_final_gain": gain,
                "full_pass": full_ok, "context_pass": context_ok,
                "key_pass": key_ok, "both_pass": both_ok, "causal": causal_ok,
            }

        full_pass = all(full_profile)
        context_pass = all(context_profile)
        key_pass = all(key_profile)
        both_pass = all(both_profile)
        context_gain = all(gain_profile)
        causal = all(causal_profile)
        verdict, reason = _classify(
            full_pass, context_pass, key_pass, both_pass, context_gain, causal
        )
        return {
            "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "full_pass": full_pass,
            "context_pass": context_pass, "key_pass": key_pass,
            "both_pass": both_pass, "context_gain": context_gain,
            "causal": causal, "evaluations": public,
        }
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError, ZeroDivisionError):
        return invalid("payload is incomplete or malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/query_refresh2_results.json")
    parser.add_argument("--output", default="measurement/query_refresh2_verdict.json")
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
