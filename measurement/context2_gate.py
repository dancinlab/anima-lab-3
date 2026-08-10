#!/usr/bin/env python3
"""Fail-closed adjudication for CONTEXT-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256
    from measurement.context_gate import adjudicate as adjudicate_context
    from measurement.context_registry import CONTEXT_SPEC, spec_sha256 as context_spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256
    from measurement.context_gate import adjudicate as adjudicate_context
    from measurement.context_registry import CONTEXT_SPEC, spec_sha256 as context_spec_sha256
    from measurement.projector_registry import evaluation_name


def _normal_pass(metrics: dict, spec: dict) -> bool:
    thresholds = spec["thresholds"]
    return (
        metrics["selection_accuracy"] >= thresholds["normal_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["normal_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["normal_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = CONTEXT2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CX2I0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_context1"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CONTEXT-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = context_spec_sha256(CONTEXT_SPEC)
        source_context = source_results.get("source_separation2", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CONTEXT_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_context(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("context_checkpoint") != source_results.get("context_checkpoint")
            or source.get("canonical_checkpoint") != source_context.get("canonical_checkpoint")
            or source.get("prototype_checkpoints") != source_context.get("prototype_checkpoints")
            or not _valid_receipt(source["context_checkpoint"])
            or not _valid_receipt(source["canonical_checkpoint"])
            or any(not _valid_receipt(item) for item in source["prototype_checkpoints"].values())
        ):
            return invalid("registered CONTEXT-1 source identity changed")

        audit = payload["evaluation_dataset_audit"]
        total = spec["eval_episodes"]
        if (
            audit != source_results["evaluation_dataset_audit"]
            or audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or len(audit["fingerprint_set_sha256"]) != 64
            or not _balanced(audit["target_counts"], spec["values"], total)
            or not _balanced(
                audit["query_position_counts"], spec["events_per_episode"], total
            )
            or not _balanced(audit["shared_key_counts"], spec["keys"], total)
            or not _balanced(audit["query_context_counts"], spec["contexts"], total)
        ):
            return invalid("evaluation dataset changed from CONTEXT-1")

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
        judged = {}
        for name, row in evaluations.items():
            expected = registered[name]
            source_row = source_evaluations[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
                or row["state_audit"] != source_row["state_audit"]
                or row["update_audit"] != source_row["update_audit"]
            ):
                return invalid(f"evaluation {name} source stream changed")
            integration = row["integration_audit"]
            if (
                integration["component_weight"] != spec["component_weight"]
                or integration["component_address_dim"] != spec["component_address_dim"]
                or integration["composite_address_dim"] != spec["composite_address_dim"]
                or integration["context_projector_frozen"] is not True
                or integration["context_projector_unchanged"] is not True
                or integration["key_projector_frozen"] is not True
                or integration["key_projector_unchanged"] is not True
                or integration["source_normal_metrics_match"] is not True
                or integration["source_state_audit_match"] is not True
                or integration["source_update_audit_match"] is not True
            ):
                return invalid(f"evaluation {name} frozen integration changed")
            path_audit = row["memory_path_audit"]
            expected_integrated = {
                "integrated_composite_normal", "integrated_context_masked",
                "integrated_composite_recovered",
            }
            if set(path_audit) != expected_integrated:
                return invalid(f"evaluation {name} integrated arm audit changed")
            for arm_name, calls in path_audit.items():
                if (
                    calls["minimum_calls"] != spec["transform_calls_per_episode"]
                    or calls["maximum_calls"] != spec["transform_calls_per_episode"]
                    or calls["minimum_components"] != spec["components_per_key"]
                    or calls["maximum_components"] != spec["components_per_key"]
                    or calls["minimum_address_width"] != spec["composite_address_dim"]
                    or calls["maximum_address_width"] != spec["composite_address_dim"]
                    or calls["minimum_stores"] != spec["stores_per_episode"]
                    or calls["maximum_stores"] != spec["stores_per_episode"]
                    or calls["minimum_retrievals"] != spec["retrievals_per_episode"]
                    or calls["maximum_retrievals"] != spec["retrievals_per_episode"]
                ):
                    return invalid(f"evaluation {name} {arm_name} call path changed")

            arms = row["arms"]
            if set(arms) != set(spec["arms"]):
                return invalid(f"evaluation {name} arm roster changed")
            for arm_name in spec["arms"]:
                if (
                    not _metric_shape(arms[arm_name], spec["values"])
                    or arms[arm_name]["retrieval_api_match"]
                    != thresholds["retrieval_api_match"]
                ):
                    return invalid(f"evaluation {name} {arm_name} metrics changed")
            normal = arms["integrated_composite_normal"]
            reference = arms["external_composite_reference"]
            exact = arms["exact_context_key_control"]
            recovered = arms["integrated_composite_recovered"]
            if (
                reference != source_row["arms"]["composite_context_key_normal"]
                or normal["reference_prediction_match"]
                != thresholds["reference_prediction_match"]
                or normal["reference_selection_match"]
                != thresholds["reference_selection_match"]
                or arms["integrated_context_masked"]["accuracy"]
                > thresholds["context_masked_max_accuracy"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"])
                < thresholds["exact_minimum_value_recall"]
                or arms["exact_context_key_partner_swap"]["accuracy"]
                > thresholds["partner_swap_max_accuracy"]
                or recovered["prediction_match"]
                != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"evaluation {name} reference, control, or recovery failed")
            judged[name] = {
                "prototype_seed": row["prototype_seed"],
                "engine_seed": row["engine_seed"],
                "normal_selection_accuracy": normal["selection_accuracy"],
                "normal_final_accuracy": normal["accuracy"],
                "normal_minimum_value_recall": min(normal["per_value_recall"]),
                "normal_content_accuracy": normal["correct_content_accuracy"],
                "context_masked_accuracy": arms["integrated_context_masked"]["accuracy"],
                "exact_final_accuracy": exact["accuracy"],
                "partner_swap_accuracy": arms["exact_context_key_partner_swap"]["accuracy"],
                "selection_passed": (
                    normal["selection_accuracy"] >= thresholds["normal_selection_accuracy"]
                ),
                "normal_passed": _normal_pass(normal, spec),
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    rows = list(judged.values())
    selection_pass = all(row["selection_passed"] for row in rows)
    normal_pass = all(row["normal_passed"] for row in rows)
    if not selection_pass:
        verdict = "CX2I_MEMORY_PATH_LOSS"
        reason = "the common memory path did not reproduce the registered composite selection"
    elif not normal_pass:
        verdict = "CX2I_VALUE_READOUT_LOSS"
        reason = "the common path selected episodes, but value readout did not support balanced behavior"
    else:
        verdict = "CX2I_PATH_RECOVERED_NOT_UNIQUE"
        reason = "the optional common memory transform reproduced the frozen context+key address in every evaluation"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "evaluations": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/context2_results.json")
    parser.add_argument("--output", default="measurement/context2_verdict.json")
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
