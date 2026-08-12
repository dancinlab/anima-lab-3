#!/usr/bin/env python3
"""Fail-closed adjudication for KEY-REFRESH-2."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

try:
    from measurement.cue_mechanism_gate import _finite, _receipt_valid
    from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC, spec_sha256
    from measurement.key_refresh_gate import adjudicate as adjudicate_source
    from measurement.key_refresh_registry import KEY_REFRESH_SPEC
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_gate import _passes
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.cue_mechanism_gate import _finite, _receipt_valid
    from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC, spec_sha256
    from measurement.key_refresh_gate import adjudicate as adjudicate_source
    from measurement.key_refresh_registry import KEY_REFRESH_SPEC
    from measurement.projector_registry import evaluation_name
    from measurement.query_refresh2_gate import _passes


def _classify(full: bool, context: bool, key: bool, both: bool,
              causal: bool) -> tuple[str, str]:
    if not full:
        return "KR2I_FULL_BEHAVIOR_REGRESSION", "four-step query key regresses full-cue behavior"
    if full and context and key and both and causal:
        return "KR2I_FULL_PATH_RECOVERED", "all registered partial-cue paths recover causally"
    if not (context and key and both):
        return "KR2I_PARTIAL_PATH_RECOVERED", "one or more partial-cue paths remain below criterion"
    return "KR2I_NOT_CAUSAL", "behavior passes but the registered disable-recover cause is absent"


def adjudicate(payload: dict, spec: dict = KEY_REFRESH2_SPEC, *,
               source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "KR2I_INVALID", "reason": reason,
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
            return invalid("registered KEY-REFRESH-1 source file changed")
        if source_results is None:
            source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != KEY_REFRESH_SPEC
            or source_results.get("spec_sha256") != source["source_spec_sha256"]
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("minimum_sustained_recovery_steps")
            != spec["query_key_sense_steps"]
            or adjudicate_source(source_results) != source_verdict
            or source.get("upstream") != source_results["source"]
            or set(source) != {"results", "verdict", "source_spec_sha256", "upstream"}
        ):
            return invalid("registered KEY-REFRESH-1 identity changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("registered evaluation dataset changed")
        expected_execution = {
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
                "disabled_3": "baseline_3", "recovered_4": "integrated_4",
            },
        }
        if payload["execution_audit"] != expected_execution:
            return invalid("registered execution reuse audit changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        source_evaluations = {row["name"]: row for row in source_results["evaluations"]}
        if (
            set(evaluations) != set(registered)
            or set(source_evaluations) != set(registered)
            or len(evaluations) != len(payload["evaluations"])
        ):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        profiles = {name: [] for name in ("full", "context", "key", "both", "causal")}
        public = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["conditions"]) != set(spec["runtime_conditions"])
            ):
                return invalid(f"evaluation {name} identity or condition roster changed")
            source_row = source_evaluations[name]
            expected_three = source_row["candidates"][str(spec["baseline_query_key_sense_steps"])]
            expected_four = source_row["candidates"][str(spec["query_key_sense_steps"])]
            baseline = row["conditions"][spec["baseline_condition"]]
            integrated = row["conditions"][spec["integrated_condition"]]
            disabled = row["conditions"][spec["disabled_condition"]]
            recovered = row["conditions"][spec["recovered_condition"]]
            if baseline != expected_three or integrated != expected_four:
                return invalid(f"evaluation {name} source candidate reproduction changed")
            if disabled != baseline or recovered != integrated:
                return invalid(f"evaluation {name} disable or recovery reproduction changed")

            arms = integrated["arms"]
            full = _passes(arms["full_cue"], thresholds, full=True)
            context = _passes(arms["context_quarter_missing"], thresholds)
            key = _passes(arms["key_quarter_missing"], thresholds)
            both = _passes(arms["both_quarter_missing"], thresholds)
            baseline_both = baseline["arms"]["both_quarter_missing"]["accuracy"]
            gain = arms["both_quarter_missing"]["accuracy"] - baseline_both
            causal = (
                baseline_both < thresholds["damaged_final_accuracy"]
                and gain >= thresholds["minimum_key_final_gain"]
            )
            profiles["full"].append(full); profiles["context"].append(context)
            profiles["key"].append(key); profiles["both"].append(both)
            profiles["causal"].append(causal)
            public[name] = {
                "baseline_both_final_accuracy": baseline_both,
                "integrated_both_final_accuracy": arms["both_quarter_missing"]["accuracy"],
                "both_final_gain": gain,
                "integrated": {
                    arm: {
                        "selection_accuracy": arms[arm]["selection_accuracy"],
                        "final_accuracy": arms[arm]["accuracy"],
                    }
                    for arm in spec["conditions"]
                },
            }
        combined = {name: all(values) for name, values in profiles.items()}
        verdict, reason = _classify(
            combined["full"], combined["context"], combined["key"],
            combined["both"], combined["causal"],
        )
        return {
            "experiment": spec["experiment"], "verdict": verdict,
            "reason": reason, "spec_sha256": spec_sha256(spec),
            "query_key_sense_steps": spec["query_key_sense_steps"],
            "profiles": combined, "evaluations": public,
        }
    except (KeyError, TypeError, ValueError, IndexError, OSError, RuntimeError,
            ZeroDivisionError):
        return invalid("payload is incomplete or malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/key_refresh2_results.json")
    parser.add_argument("--output", default="measurement/key_refresh2_verdict.json")
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
