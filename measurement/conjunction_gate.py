#!/usr/bin/env python3
"""Fail-closed adjudication for CONJUNCTION-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256
    from measurement.context2_gate import adjudicate as adjudicate_context2
    from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256 as context2_spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256
    from measurement.context2_gate import adjudicate as adjudicate_context2
    from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256 as context2_spec_sha256
    from measurement.projector_registry import evaluation_name


def _normal_pass(metrics: dict, spec: dict) -> bool:
    thresholds = spec["thresholds"]
    return (
        metrics["selection_accuracy"] >= thresholds["normal_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["normal_final_accuracy"]
        and min(metrics["per_value_recall"])
        >= thresholds["normal_minimum_value_recall"]
        and metrics["correct_content_accuracy"]
        >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = CONJUNCTION_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CJ0_INVALID",
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

        source = payload["source_context2"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CONTEXT-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = context2_spec_sha256(CONTEXT2_SPEC)
        source_context1 = source_results.get("source_context1", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CONTEXT2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_context2(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("context_checkpoint") != source_context1.get("context_checkpoint")
            or source.get("canonical_checkpoint") != source_context1.get("canonical_checkpoint")
            or source.get("prototype_checkpoints")
            != source_context1.get("prototype_checkpoints")
            or not _valid_receipt(source["context_checkpoint"])
            or not _valid_receipt(source["canonical_checkpoint"])
            or any(
                not _valid_receipt(item)
                for item in source["prototype_checkpoints"].values()
            )
        ):
            return invalid("registered CONTEXT-2 source identity changed")

        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        triple_total = spec["contexts"] * spec["keys"] * spec["values"]
        expected_triples = {
            f"{context}:{key}:{value}": total // triple_total
            for context in range(spec["contexts"])
            for key in range(spec["keys"])
            for value in range(spec["values"])
        }
        if (
            audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
            or len(audit["fingerprint_set_sha256"]) != 64
            or not _balanced(audit["target_counts"], spec["values"], total)
            or not _balanced(audit["query_context_counts"], spec["contexts"], total)
            or not _balanced(audit["query_key_counts"], spec["keys"], total)
            or audit["query_triple_counts"] != expected_triples
        ):
            return invalid("registered balanced conjunction dataset changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        judged = {}
        integrated_names = {
            "integrated_conjunction_normal", "integrated_context_masked",
            "integrated_key_masked", "integrated_conjunction_recovered",
        }
        for name, row in evaluations.items():
            expected = registered[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
            ):
                return invalid(f"evaluation {name} identity changed")
            state = row["state_audit"]
            updates = row["update_audit"]
            if (
                state["episodes"] != total
                or state["unique_episode_seeds"] != total
                or len(state["episode_seed_sha256"]) != 64
                or not spec["minimum_cells"] <= state["minimum_cells"]
                <= state["maximum_cells"] <= spec["maximum_cells"]
                or updates["requested_updates"] != spec["pre_query_updates"]
                or updates["performed_updates_minimum"] != spec["pre_query_updates"]
                or updates["performed_updates_maximum"] != spec["pre_query_updates"]
                or updates["disabled"] != list(spec["pre_query_dynamics_ablation"])
                or any(len(updates[key]) != 64 for key in (
                    "state_before_sha256", "state_after_sha256", "query_rng_sha256"
                ))
            ):
                return invalid(f"evaluation {name} state stream changed")
            integration = row["integration_audit"]
            if (
                integration["component_weight"] != spec["component_weight"]
                or integration["component_address_dim"] != spec["component_address_dim"]
                or integration["composite_address_dim"] != spec["composite_address_dim"]
                or integration["context_projector_frozen"] is not True
                or integration["context_projector_unchanged"] is not True
                or integration["key_projector_frozen"] is not True
                or integration["key_projector_unchanged"] is not True
            ):
                return invalid(f"evaluation {name} frozen integration changed")
            path_audit = row["memory_path_audit"]
            if set(path_audit) != integrated_names:
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
            normal = arms["integrated_conjunction_normal"]
            reference = arms["external_conjunction_reference"]
            exact = arms["exact_context_key_control"]
            recovered = arms["integrated_conjunction_recovered"]
            if (
                normal["reference_prediction_match"]
                != thresholds["reference_prediction_match"]
                or normal["reference_selection_match"]
                != thresholds["reference_selection_match"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"])
                < thresholds["exact_minimum_value_recall"]
                or arms["exact_context_key_partner_swap"]["accuracy"]
                > thresholds["partner_swap_max_accuracy"]
                or recovered["prediction_match"]
                != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"evaluation {name} reference, positive, or fake control failed")
            component_scores = {
                "integrated_context_masked": arms["integrated_context_masked"]["accuracy"],
                "integrated_key_masked": arms["integrated_key_masked"]["accuracy"],
                "exact_context_only": arms["exact_context_only_control"]["accuracy"],
                "exact_key_only": arms["exact_key_only_control"]["accuracy"],
            }
            judged[name] = {
                "prototype_seed": row["prototype_seed"],
                "engine_seed": row["engine_seed"],
                "normal_selection_accuracy": normal["selection_accuracy"],
                "normal_final_accuracy": normal["accuracy"],
                "normal_minimum_value_recall": min(normal["per_value_recall"]),
                "normal_content_accuracy": normal["correct_content_accuracy"],
                **component_scores,
                "partner_swap_accuracy": arms["exact_context_key_partner_swap"]["accuracy"],
                "selection_passed": (
                    normal["selection_accuracy"] >= thresholds["normal_selection_accuracy"]
                ),
                "normal_passed": _normal_pass(normal, spec),
                "component_controls_passed": all(
                    score <= thresholds["component_masked_max_accuracy"]
                    for score in component_scores.values()
                ),
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    rows = list(judged.values())
    if not all(row["component_controls_passed"] for row in rows):
        verdict = "CJ4_NOT_CONJUNCTIVE"
        reason = "at least one single-component control exceeded the registered chance ceiling"
    elif not all(row["selection_passed"] for row in rows):
        verdict = "CJ2_COMPONENT_COLLISION"
        reason = "exact pair addresses worked, but a frozen state component collided"
    elif not all(row["normal_passed"] for row in rows):
        verdict = "CJ3_VALUE_READOUT_LOSS"
        reason = "pair selection passed, but balanced value readout did not"
    else:
        verdict = "CJ1_CONJUNCTION_VALID_NOT_UNIQUE"
        reason = "the common memory path required and used both frozen address components"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/conjunction_results.json")
    parser.add_argument("--output", default="measurement/conjunction_verdict.json")
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
