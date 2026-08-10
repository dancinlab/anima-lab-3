#!/usr/bin/env python3
"""Fail-closed adjudication for VALUE-MECHANISM-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.projector_registry import evaluation_name
    from measurement.value_gate import adjudicate as adjudicate_value
    from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256
    from measurement.value_registry import VALUE_SPEC, spec_sha256 as value_spec_sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.projector_registry import evaluation_name
    from measurement.value_gate import adjudicate as adjudicate_value
    from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256
    from measurement.value_registry import VALUE_SPEC, spec_sha256 as value_spec_sha256


def position_verdict(passed: list[bool], accuracies: list[float], minimum_effect: float) -> tuple[str, str]:
    if all(passed):
        return "VP1_POSITION_INVARIANT", "value readout passed at every registered serial position"
    if passed[0] and not any(passed[index] and not all(passed[:index]) for index in range(1, len(passed))):
        return "VP2_LATE_POSITION_LOSS", "value readout failed monotonically at later serial positions"
    if max(accuracies) - min(accuracies) >= minimum_effect:
        return "VP3_POSITION_SPECIFIC", "serial position changed readout but not with a monotonic boundary"
    return "VP4_POSITION_NOT_CAUSAL", "serial position did not explain the source readout failure"


def adjudicate(payload: dict, spec: dict = VALUE_MECHANISM_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "VP0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")
        source = payload["source_value1"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("VALUE-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = value_spec_sha256(VALUE_SPEC)
        inherited = source_results.get("source_conjunction1", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != VALUE_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_value(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("prototype_checkpoints") != inherited.get("prototype_checkpoints")
            or any(not _valid_receipt(item) for item in source["prototype_checkpoints"].values())
        ):
            return invalid("registered VALUE-1 source identity changed")

        audit = payload["dataset_audit"]
        base = audit["base"]
        total = spec["eval_episodes"]
        expected_triples = {
            f"{context}:{key}:{value}": 1
            for context in range(spec["contexts"])
            for key in range(spec["keys"])
            for value in range(spec["values"])
        }
        if (
            base["episodes"] != total or base["unique_fingerprints"] != total
            or base["latin_valid_episodes"] != total
            or not _balanced(base["target_counts"], spec["values"], total)
            or base["query_triple_counts"] != expected_triples
        ):
            return invalid("registered base dataset changed")
        positions_audit = audit["positions"]
        if set(positions_audit) != {str(value) for value in spec["query_positions"]}:
            return invalid("query-position dataset roster changed")
        if not isinstance(audit["shared_event_set_sha256"], str) or len(audit["shared_event_set_sha256"]) != 64:
            return invalid("query positions did not share one event set")
        for position in spec["query_positions"]:
            row = positions_audit[str(position)]
            if (
                row["episodes"] != total
                or row["query_position_matches"] != total
                or row["minimum_unique_pairs"] != spec["events_per_episode"]
                or row["maximum_unique_pairs"] != spec["events_per_episode"]
                or row["event_set_sha256"] != audit["shared_event_set_sha256"]
                or len(row["ordered_fingerprint_sha256"]) != 64
            ):
                return invalid(f"query position {position + 1} dataset changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        judged = {str(position): {} for position in spec["query_positions"]}
        engine_hashes: dict[tuple[str, int], str] = {}
        for name, row in evaluations.items():
            expected = registered[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
            ):
                return invalid(f"evaluation {name} identity changed")
            positions = {item["query_position"]: item for item in row["positions"]}
            if set(positions) != set(spec["query_positions"]) or len(positions) != len(row["positions"]):
                return invalid(f"evaluation {name} query-position roster changed")
            expected_seed_hash = None
            for position in spec["query_positions"]:
                item = positions[position]
                state, path = item["state_audit"], item["path_audit"]
                if (
                    item["query_position_label"] != position + 1
                    or state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    <= state["maximum_cells"] <= spec["maximum_cells"]
                    or path["stores_per_episode"] != spec["stores_per_episode"]
                    or path["retrievals_per_episode"] != spec["retrievals_per_episode"]
                    or path["address_width"] != spec["contexts"] + spec["keys"]
                ):
                    return invalid(f"evaluation {name} position {position + 1} path changed")
                if expected_seed_hash is None:
                    expected_seed_hash = state["episode_seed_sha256"]
                elif state["episode_seed_sha256"] != expected_seed_hash:
                    return invalid(f"evaluation {name} positions used different engine starts")
                engine_hashes[(str(row["engine_seed"]), position)] = state["episode_seed_sha256"]
                arms = item["arms"]
                if set(arms) != set(spec["arms"]):
                    return invalid(f"evaluation {name} position {position + 1} arms changed")
                for arm_name in spec["arms"]:
                    if (
                        not _metric_shape(arms[arm_name], spec["values"])
                        or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    ):
                        return invalid(f"evaluation {name} position {position + 1} metrics changed")
                normal = arms["exact_value_normal"]
                if (
                    normal["selection_accuracy"] < thresholds["selection_accuracy"]
                    or arms["exact_value_partner_swap"]["accuracy"]
                    > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_value_recovered"]["prediction_match"]
                    != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"evaluation {name} position {position + 1} control failed")
                passed = (
                    normal["accuracy"] >= thresholds["final_accuracy"]
                    and min(normal["per_value_recall"]) >= thresholds["minimum_value_recall"]
                    and normal["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
                )
                judged[str(position)][name] = {
                    "passed": passed,
                    "final_accuracy": normal["accuracy"],
                    "minimum_value_recall": min(normal["per_value_recall"]),
                    "content_readout_accuracy": normal["correct_content_accuracy"],
                    "selection_accuracy": normal["selection_accuracy"],
                    "partner_swap_accuracy": arms["exact_value_partner_swap"]["accuracy"],
                    "prototype_seed": row["prototype_seed"],
                    "engine_seed": row["engine_seed"],
                }
        passed = [
            all(row["passed"] for row in judged[str(position)].values())
            for position in spec["query_positions"]
        ]
        accuracies = [
            min(row["final_accuracy"] for row in judged[str(position)].values())
            for position in spec["query_positions"]
        ]
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    verdict, reason = position_verdict(
        passed, accuracies, spec["thresholds"]["minimum_position_effect"]
    )
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "positions": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/value_mechanism_results.json")
    parser.add_argument("--output", default="measurement/value_mechanism_verdict.json")
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
