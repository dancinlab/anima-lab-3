#!/usr/bin/env python3
"""Fail-closed adjudication for CONTEXT-SETTLE-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.component2_gate import adjudicate as adjudicate_component2
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component2_spec_sha256
    from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC, spec_sha256
    from measurement.context_settle_gate import adjudicate as adjudicate_context_settle
    from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256 as settle_spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.component2_gate import adjudicate as adjudicate_component2
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component2_spec_sha256
    from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC, spec_sha256
    from measurement.context_settle_gate import adjudicate as adjudicate_context_settle
    from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256 as settle_spec_sha256
    from measurement.projector_registry import evaluation_name


def _path_valid(path: dict, spec: dict) -> bool:
    expected = {
        "integrated_stable_conjunction_normal", "integrated_stable_context_masked",
        "integrated_stable_key_masked", "integrated_stable_conjunction_recovered",
        "integrated_raw_value_control",
    }
    if set(path) != expected:
        return False
    for name, calls in path.items():
        stable = name != "integrated_raw_value_control"
        if (
            calls["key_calls"] != spec["transform_calls_per_episode"]
            or calls["key_minimum_components"] != spec["components_per_key"]
            or calls["key_maximum_components"] != spec["components_per_key"]
            or calls["key_minimum_width"] != spec["composite_address_dim"]
            or calls["key_maximum_width"] != spec["composite_address_dim"]
            or calls["value_calls"] != (spec["value_transform_calls_per_episode"] if stable else 0)
            or calls["value_minimum_width"] != (spec["value_address_dim"] if stable else 0)
            or calls["value_maximum_width"] != (spec["value_address_dim"] if stable else 0)
            or calls["stores"] != spec["stores_per_episode"]
            or calls["retrievals"] != spec["retrievals_per_episode"]
        ):
            return False
    return True


def adjudicate(payload: dict, spec: dict = CONTEXT_SETTLE2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CT2I0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"] or payload["spec"] != spec:
            return invalid("experiment or registered spec changed")
        if payload["spec_sha256"] != spec_sha256(spec) or not _finite_tree(payload):
            return invalid("registered spec digest or finite-value check failed")
        source = payload["source"]
        for name in (
            "settle_results", "settle_verdict", "component_results",
            "component_verdict", "component_checkpoint",
        ):
            if not _valid_receipt(source[name]):
                return invalid(f"registered source {name} changed")
        settle_results = json.loads(Path(source["settle_results"]["path"]).read_text())
        settle_verdict = json.loads(Path(source["settle_verdict"]["path"]).read_text())
        settle_sha = settle_spec_sha256(CONTEXT_SETTLE_SPEC)
        component_results = json.loads(Path(source["component_results"]["path"]).read_text())
        component_verdict = json.loads(Path(source["component_verdict"]["path"]).read_text())
        component_sha = component2_spec_sha256(COMPONENT2_SPEC)
        if (
            settle_results.get("experiment") != spec["source_settle_experiment"]
            or settle_results.get("spec") != CONTEXT_SETTLE_SPEC
            or settle_results.get("spec_sha256") != settle_sha
            or settle_verdict.get("verdict") != spec["source_settle_verdict"]
            or settle_verdict.get("minimum_settling_steps") != spec["settled_context_steps"]
            or adjudicate_context_settle(settle_results) != settle_verdict
            or source["settle_spec_sha256"] != settle_sha
            or component_results.get("experiment") != spec["source_component_experiment"]
            or component_results.get("spec") != COMPONENT2_SPEC
            or component_results.get("spec_sha256") != component_sha
            or component_verdict.get("verdict") != spec["source_component_verdict"]
            or adjudicate_component2(component_results) != component_verdict
            or source["component_spec_sha256"] != component_sha
            or source["component_checkpoint"] != component_results.get("checkpoint")
        ):
            return invalid("registered source identity changed")
        checkpoint = torch.load(source["component_checkpoint"]["path"], map_location="cpu", weights_only=True)
        if (
            checkpoint.get("experiment") != COMPONENT2_SPEC["experiment"]
            or checkpoint.get("spec_sha256") != component_sha
            or checkpoint.get("deterministic") is not True
        ):
            return invalid("frozen component checkpoint identity changed")

        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        if (
            audit["episodes"] != total or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
            or not _balanced(audit["target_counts"], spec["values"], total)
        ):
            return invalid("registered conjunction dataset changed")
        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        old_evaluations = {row["name"]: row for row in component_results["evaluations"]}
        judged = {}
        baseline_loss = True
        settled_pass = True
        causal = True
        thresholds = spec["thresholds"]
        for name, row in evaluations.items():
            identity = registered[name]
            if row["prototype_seed"] != identity["prototype_seed"] or row["engine_seed"] != identity["engine_seed"]:
                return invalid(f"evaluation {name} identity changed")
            if set(row["conditions"]) != set(spec["conditions"]):
                return invalid(f"evaluation {name} condition roster changed")
            public = {}
            for condition, steps in (
                ("baseline_3", spec["baseline_context_steps"]),
                ("settled_6", spec["settled_context_steps"]),
            ):
                result = row["conditions"][condition]
                if (
                    result["prototype_seed"] != identity["prototype_seed"]
                    or result["engine_seed"] != identity["engine_seed"]
                    or result["frozen_audit"] != {"context": True, "key": True, "value": True}
                    or not _path_valid(result["path_audit"], spec)
                ):
                    return invalid(f"evaluation {name} {condition} identity or call path changed")
                state = result["state_audit"]
                event_queries = total * (spec["events_per_episode"] + 1)
                if (
                    state["episodes"] != total or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
                    or state["context_sense_steps_minimum"] != steps
                    or state["context_sense_steps_maximum"] != steps
                    or state["key_sense_steps_minimum"] != spec["key_sense_steps"]
                    or state["key_sense_steps_maximum"] != spec["key_sense_steps"]
                    or state["value_sense_steps_minimum"] != spec["value_sense_steps"]
                    or state["value_sense_steps_maximum"] != spec["value_sense_steps"]
                    or state["context_step_calls"] != event_queries * steps
                    or state["key_step_calls"] != event_queries * spec["key_sense_steps"]
                    or state["value_step_calls"] != total * spec["events_per_episode"] * spec["value_sense_steps"]
                    or state["distractor_step_calls"] != total * spec["distractor_steps"] * spec["distractor_sense_steps"]
                ):
                    return invalid(f"evaluation {name} {condition} sensing audit changed")
                arms = result["arms"]
                if set(arms) != set(spec["arms"]):
                    return invalid(f"evaluation {name} {condition} arm roster changed")
                if any(not _metric_shape(arm, spec["values"]) or arm["retrieval_api_match"] != 1.0 for arm in arms.values()):
                    return invalid(f"evaluation {name} {condition} metrics changed")
                normal = arms["integrated_stable_conjunction_normal"]
                exact = arms["exact_stable_context_key_control"]
                controls = (
                    exact["selection_accuracy"] >= thresholds["exact_selection_accuracy"]
                    and exact["accuracy"] >= thresholds["exact_final_accuracy"]
                    and min(exact["per_value_recall"]) >= thresholds["exact_minimum_value_recall"]
                    and arms["integrated_stable_context_masked"]["accuracy"] <= thresholds["component_masked_max_accuracy"]
                    and arms["integrated_stable_key_masked"]["accuracy"] <= thresholds["component_masked_max_accuracy"]
                    and arms["exact_stable_context_only_control"]["accuracy"] <= thresholds["component_masked_max_accuracy"]
                    and arms["exact_stable_key_only_control"]["accuracy"] <= thresholds["component_masked_max_accuracy"]
                    and arms["exact_stable_partner_swap"]["accuracy"] <= thresholds["partner_swap_max_accuracy"]
                    and normal["reference_prediction_match"] == 1.0
                    and normal["reference_selection_match"] == 1.0
                    and arms["integrated_stable_conjunction_recovered"]["prediction_match"] == 1.0
                )
                if not controls:
                    return invalid(f"evaluation {name} {condition} registered control failed")
                public[condition] = {
                    "selection_accuracy": normal["selection_accuracy"],
                    "final_accuracy": normal["accuracy"],
                    "minimum_value_recall": min(normal["per_value_recall"]),
                }
            if row["conditions"]["baseline_3"]["arms"] != old_evaluations[name]["arms"]:
                return invalid(f"evaluation {name} three-step source replay changed")
            baseline = public["baseline_3"]
            settled = public["settled_6"]
            gain = settled["selection_accuracy"] - baseline["selection_accuracy"]
            baseline_ok = baseline["selection_accuracy"] <= thresholds["baseline_selection_max_accuracy"]
            settled_ok = (
                settled["selection_accuracy"] >= thresholds["selection_accuracy"]
                and settled["final_accuracy"] >= thresholds["final_accuracy"]
                and settled["minimum_value_recall"] >= thresholds["minimum_value_recall"]
            )
            causal_ok = gain >= thresholds["minimum_causal_gain"]
            baseline_loss &= baseline_ok
            settled_pass &= settled_ok
            causal &= causal_ok
            judged[name] = {
                **public, "selection_gain": gain,
                "baseline_reproduced": baseline_ok,
                "settled_passed": settled_ok, "causal": causal_ok,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not baseline_loss or not causal:
        verdict = "CT2I_CONTEXT_SETTLING_NOT_CAUSAL"
        reason = "the registered three-step loss or minimum six-step gain did not reproduce"
    elif not settled_pass:
        verdict = "CT2I_COMPOSITION_LOSS"
        reason = "six-step context states passed upstream, but integrated behavior did not recover"
    else:
        verdict = "CT2I_PATH_RECOVERED_NOT_UNIQUE"
        reason = "six-step context sensing recovered the frozen common conjunction path"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "source_checkpoint": source["component_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/context_settle2_results.json")
    parser.add_argument("--output", default="measurement/context_settle2_verdict.json")
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
