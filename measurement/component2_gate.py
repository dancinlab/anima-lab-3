#!/usr/bin/env python3
"""Fail-closed adjudication for COMPONENT-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256
    from measurement.component_gate import adjudicate as adjudicate_source
    from measurement.component_registry import COMPONENT_SPEC, spec_sha256 as source_spec_sha256
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256
    from measurement.component_gate import adjudicate as adjudicate_source
    from measurement.component_registry import COMPONENT_SPEC, spec_sha256 as source_spec_sha256
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC
    from measurement.projector_registry import evaluation_name


def adjudicate(payload, spec=COMPONENT2_SPEC):
    def invalid(reason): return {"experiment": payload.get("experiment", spec["experiment"]), "verdict": "CS0_INVALID", "reason": reason, "spec_sha256": spec_sha256(spec)}
    try:
        if payload["experiment"] != spec["experiment"] or payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec): return invalid("experiment or spec changed")
        if not _finite_tree(payload): return invalid("non-finite result")
        source = payload["source_component1"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]) or not _valid_receipt(payload["checkpoint"]): return invalid("source or checkpoint changed")
        sr = json.loads(Path(source["results"]["path"]).read_text()); sv = json.loads(Path(source["verdict"]["path"]).read_text()); sha = source_spec_sha256(COMPONENT_SPEC)
        if sr.get("experiment") != spec["source_experiment"] or sr.get("spec") != COMPONENT_SPEC or sr.get("spec_sha256") != sha or sv.get("verdict") != spec["source_verdict"] or adjudicate_source(sr) != sv or source["source_spec_sha256"] != sha: return invalid("source identity changed")
        checkpoint = torch.load(payload["checkpoint"]["path"], map_location="cpu", weights_only=True)
        if checkpoint.get("experiment") != spec["experiment"] or checkpoint.get("spec_sha256") != spec_sha256(spec) or checkpoint.get("deterministic") is not True: return invalid("component checkpoint identity changed")
        expected_states = spec["calibration_episodes"] * len(spec["calibration_engine_seeds"]) * spec["events_per_episode"]
        state = payload["calibration_state_audit"]
        expected_counts = {str(i): expected_states // 8 for i in range(8)}
        if state["states_per_component"] != expected_states or state["unique_engine_seeds"] != spec["calibration_episodes"] * 2 or state["context_counts"] != expected_counts or state["key_counts"] != expected_counts or not payload["deterministic"]: return invalid("calibration balance or determinism changed")
        diagnostics = {row["engine_seed"]: row for row in payload["diagnostics"]}
        if set(diagnostics) != set(spec["engine_seeds"]): return invalid("diagnostic engines changed")
        classification_pass = True
        for seed, row in diagnostics.items():
            if row["frozen_audit"] != {"context": True, "key": True}: return invalid(f"engine {seed} changed projectors")
            positions = {item["position"]: item for item in row["positions"]}
            if set(positions) != set(spec["positions"]): return invalid(f"engine {seed} positions changed")
            for item in positions.values():
                for component in ("context", "key"):
                    metric = item[component]
                    classification_pass &= metric["accuracy"] >= spec["thresholds"]["classification_accuracy"] and min(metric["per_key_recall"]) >= spec["thresholds"]["minimum_class_recall"]
        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered): return invalid("evaluation roster changed")
        integration_pass = True; causal_pass = True; judged = {}
        source_conj2 = json.loads(Path(sr["source_conjunction2"]["results"]["path"]).read_text())
        old = {row["name"]: row for row in source_conj2["evaluations"]}
        for name, row in evaluations.items():
            if row["frozen_audit"] != {"context": True, "key": True, "value": True}: return invalid(f"evaluation {name} changed projectors")
            arms = row["arms"]
            if set(arms) != set(CONJUNCTION2_SPEC["arms"]): return invalid(f"evaluation {name} arms changed")
            for arm in arms.values():
                if not _metric_shape(arm, spec["values"]): return invalid(f"evaluation {name} metrics changed")
            normal = arms["integrated_stable_conjunction_normal"]; exact = arms["exact_stable_context_key_control"]
            controls = (
                exact["accuracy"] >= .90 and min(exact["per_value_recall"]) >= .75
                and arms["integrated_stable_context_masked"]["accuracy"] <= .35
                and arms["integrated_stable_key_masked"]["accuracy"] <= .35
                and arms["exact_stable_context_only_control"]["accuracy"] <= .35
                and arms["exact_stable_key_only_control"]["accuracy"] <= .35
                and arms["exact_stable_partner_swap"]["accuracy"] <= .05
                and normal["reference_prediction_match"] == 1.0 and normal["reference_selection_match"] == 1.0
                and arms["integrated_stable_conjunction_recovered"]["prediction_match"] == 1.0
            )
            if not controls: return invalid(f"evaluation {name} control failed")
            passed = normal["selection_accuracy"] >= spec["thresholds"]["selection_accuracy"] and normal["accuracy"] >= spec["thresholds"]["final_accuracy"] and min(normal["per_value_recall"]) >= spec["thresholds"]["minimum_value_recall"]
            old_selection = old[name]["arms"]["integrated_stable_conjunction_normal"]["selection_accuracy"]
            causal = old_selection <= spec["thresholds"]["old_selection_max_accuracy"] and normal["selection_accuracy"] - old_selection >= spec["thresholds"]["minimum_causal_gain"]
            integration_pass &= passed; causal_pass &= causal
            judged[name] = {"selection_accuracy": normal["selection_accuracy"], "final_accuracy": normal["accuracy"], "minimum_value_recall": min(normal["per_value_recall"]), "old_selection_accuracy": old_selection, "passed": passed, "causal": causal}
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc: return invalid(str(exc))
    if not causal_pass: verdict, reason = "CS4_NOT_CAUSAL", "old component path did not reproduce the registered loss"
    elif not classification_pass: verdict, reason = "CS2_COMPONENT_FIT_INVALID", "stable components failed held-out serial classification"
    elif not integration_pass: verdict, reason = "CS3_COMPOSITION_LOSS", "components classified but the common composite path failed"
    else: verdict, reason = "CS1_STABLE_COMPOSITE_PATH_VALID_NOT_UNIQUE", "stable context, key, and value paths passed integrated conjunction"
    return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason, "spec_sha256": spec_sha256(spec), "evaluations": judged, "checkpoint": payload["checkpoint"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("results", nargs="?", default="measurement/component2_results.json"); parser.add_argument("--output", default="measurement/component2_verdict.json"); args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text()); verdict = adjudicate(payload); path = Path(args.output); temporary = path.with_name(path.name + ".tmp"); temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"); os.replace(temporary, path); print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__": main()
