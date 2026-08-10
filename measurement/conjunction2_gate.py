#!/usr/bin/env python3
"""Fail-closed adjudication for CONJUNCTION-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256
    from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.value2_gate import adjudicate as adjudicate_value2
    from measurement.value2_registry import VALUE2_SPEC, spec_sha256 as value2_spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256
    from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.value2_gate import adjudicate as adjudicate_value2
    from measurement.value2_registry import VALUE2_SPEC, spec_sha256 as value2_spec_sha256


def adjudicate(payload: dict, spec: dict = CONJUNCTION2_SPEC) -> dict:
    def invalid(reason: str):
        return {"experiment": payload.get("experiment", spec["experiment"]),
                "verdict": "CJ2_0_INVALID", "reason": reason,
                "spec_sha256": spec_sha256(spec)}
    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")
        source = payload["source"]
        required = (
            "value_results", "value_verdict", "value_checkpoint",
            "conjunction_results", "conjunction_verdict", "context_checkpoint",
            "canonical_checkpoint",
        )
        if any(not _valid_receipt(source[name]) for name in required):
            return invalid("a registered source file changed")
        value_results = json.loads(Path(source["value_results"]["path"]).read_text())
        value_verdict = json.loads(Path(source["value_verdict"]["path"]).read_text())
        value_sha = value2_spec_sha256(VALUE2_SPEC)
        conjunction_results = json.loads(Path(source["conjunction_results"]["path"]).read_text())
        conjunction_verdict = json.loads(Path(source["conjunction_verdict"]["path"]).read_text())
        conjunction_sha = conjunction_spec_sha256(CONJUNCTION_SPEC)
        inherited = conjunction_results.get("source_context2", {})
        if (
            value_results.get("experiment") != spec["source_value_experiment"]
            or value_results.get("spec") != VALUE2_SPEC
            or value_results.get("spec_sha256") != value_sha
            or value_verdict.get("verdict") != spec["source_value_verdict"]
            or value_verdict.get("spec_sha256") != value_sha
            or adjudicate_value2(value_results) != value_verdict
            or source["value_spec_sha256"] != value_sha
            or source["value_checkpoint"] != value_results.get("checkpoint")
            or conjunction_results.get("experiment") != spec["source_conjunction_experiment"]
            or conjunction_results.get("spec") != CONJUNCTION_SPEC
            or conjunction_results.get("spec_sha256") != conjunction_sha
            or conjunction_verdict.get("verdict") != spec["source_conjunction_verdict"]
            or adjudicate_conjunction(conjunction_results) != conjunction_verdict
            or source["conjunction_spec_sha256"] != conjunction_sha
            or source["context_checkpoint"] != inherited.get("context_checkpoint")
            or source["canonical_checkpoint"] != inherited.get("canonical_checkpoint")
            or source["prototype_checkpoints"] != inherited.get("prototype_checkpoints")
            or any(not _valid_receipt(row) for row in source["prototype_checkpoints"].values())
        ):
            return invalid("registered source identity changed")
        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        expected_triples = {
            f"{context}:{key}:{value}": total // (spec["contexts"] * spec["keys"] * spec["values"])
            for context in range(spec["contexts"])
            for key in range(spec["keys"])
            for value in range(spec["values"])
        }
        if (
            audit["episodes"] != total or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
            or not _balanced(audit["target_counts"], spec["values"], total)
            or audit["query_triple_counts"] != expected_triples
        ):
            return invalid("registered balanced conjunction dataset changed")
        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        judged = {}
        for name, row in evaluations.items():
            expected = registered[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"] != source["prototype_checkpoints"][str(expected["prototype_seed"])]
                or row["frozen_audit"] != {"context": True, "key": True, "value": True}
            ):
                return invalid(f"evaluation {name} identity or frozen model changed")
            state = row["state_audit"]
            if (
                state["episodes"] != total or state["unique_episode_seeds"] != total
                or len(state["episode_seed_sha256"]) != 64
                or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
            ):
                return invalid(f"evaluation {name} state stream changed")
            path = row["path_audit"]
            expected_paths = {
                "integrated_stable_conjunction_normal", "integrated_stable_context_masked",
                "integrated_stable_key_masked", "integrated_stable_conjunction_recovered",
                "integrated_raw_value_control",
            }
            if set(path) != expected_paths:
                return invalid(f"evaluation {name} path roster changed")
            for arm_name, calls in path.items():
                stable = arm_name != "integrated_raw_value_control"
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
                    return invalid(f"evaluation {name} {arm_name} call path changed")
            arms = row["arms"]
            if set(arms) != set(spec["arms"]):
                return invalid(f"evaluation {name} arm roster changed")
            for arm_name in spec["arms"]:
                if (
                    not _metric_shape(arms[arm_name], spec["values"])
                    or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                ):
                    return invalid(f"evaluation {name} {arm_name} metrics changed")
            normal = arms["integrated_stable_conjunction_normal"]
            exact = arms["exact_stable_context_key_control"]
            recovered = arms["integrated_stable_conjunction_recovered"]
            if (
                exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or arms["exact_stable_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                or normal["reference_prediction_match"] != thresholds["reference_prediction_match"]
                or normal["reference_selection_match"] != thresholds["reference_selection_match"]
                or recovered["prediction_match"] != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"evaluation {name} positive, reference, fake, or recovery control failed")
            component_scores = {
                "context_masked": arms["integrated_stable_context_masked"]["accuracy"],
                "key_masked": arms["integrated_stable_key_masked"]["accuracy"],
                "exact_context_only": arms["exact_stable_context_only_control"]["accuracy"],
                "exact_key_only": arms["exact_stable_key_only_control"]["accuracy"],
            }
            judged[name] = {
                "prototype_seed": row["prototype_seed"], "engine_seed": row["engine_seed"],
                "selection_accuracy": normal["selection_accuracy"],
                "final_accuracy": normal["accuracy"],
                "minimum_value_recall": min(normal["per_value_recall"]),
                "exact_final_accuracy": exact["accuracy"],
                "component_scores": component_scores,
                "components_passed": all(value <= thresholds["component_masked_max_accuracy"] for value in component_scores.values()),
                "selection_passed": normal["selection_accuracy"] >= thresholds["normal_selection_accuracy"],
                "final_passed": normal["accuracy"] >= thresholds["normal_final_accuracy"] and min(normal["per_value_recall"]) >= thresholds["normal_minimum_value_recall"],
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))
    rows = list(judged.values())
    if not all(row["components_passed"] for row in rows):
        verdict, reason = "CJ2_4_NOT_CONJUNCTIVE", "a single-component control exceeded the registered chance ceiling"
    elif not all(row["selection_passed"] for row in rows):
        verdict, reason = "CJ2_2_COMPONENT_COLLISION", "stable exact values worked, but a frozen address component collided"
    elif not all(row["final_passed"] for row in rows):
        verdict, reason = "CJ2_3_STABLE_VALUE_INTEGRATION_LOSS", "pair selection passed, but stable value readout did not"
    else:
        verdict, reason = "CJ2_1_CONJUNCTION_VALID_NOT_UNIQUE", "the common stable memory path required and used both address components"
    return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "evaluations": judged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/conjunction2_results.json")
    parser.add_argument("--output", default="measurement/conjunction2_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output); temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
