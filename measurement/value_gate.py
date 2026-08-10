#!/usr/bin/env python3
"""Fail-closed adjudication for VALUE-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.value_registry import VALUE_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.conjunction_gate import adjudicate as adjudicate_conjunction
    from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256 as conjunction_spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.value_registry import VALUE_SPEC, spec_sha256


def boundary_verdict(passed: list[bool]) -> tuple[str, str]:
    if any(passed[index] and not all(passed[:index]) for index in range(1, len(passed))):
        return "VB6_NON_MONOTONIC", "a longer event prefix passed after a shorter prefix failed"
    labels = {
        4: ("VB1_READOUT_VALID_THROUGH_16", "value readout passed through all 16 events"),
        3: ("VB2_BOUNDARY_12", "value readout passed through 12 events and failed at 16"),
        2: ("VB3_BOUNDARY_8", "value readout passed through 8 events and failed at 12"),
        1: ("VB4_BOUNDARY_4", "value readout passed through 4 events and failed at 8"),
        0: ("VB5_BELOW_4", "value readout failed from the first four-event prefix"),
    }
    return labels[sum(passed)]


def adjudicate(payload: dict, spec: dict = VALUE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "VB0_INVALID",
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

        source = payload["source_conjunction1"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CONJUNCTION-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = conjunction_spec_sha256(CONJUNCTION_SPEC)
        inherited = source_results.get("source_context2", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CONJUNCTION_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_conjunction(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("prototype_checkpoints") != inherited.get("prototype_checkpoints")
            or any(not _valid_receipt(item) for item in source["prototype_checkpoints"].values())
        ):
            return invalid("registered CONJUNCTION-1 source identity changed")

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
            base["episodes"] != total
            or base["unique_fingerprints"] != total
            or base["latin_valid_episodes"] != total
            or base["minimum_unique_pairs"] != spec["events_per_episode"]
            or base["maximum_unique_pairs"] != spec["events_per_episode"]
            or not _balanced(base["target_counts"], spec["values"], total)
            or not _balanced(base["query_context_counts"], spec["contexts"], total)
            or not _balanced(base["query_key_counts"], spec["keys"], total)
            or base["query_triple_counts"] != expected_triples
            or len(base["fingerprint_set_sha256"]) != 64
        ):
            return invalid("registered balanced base dataset changed")
        prefixes = audit["prefixes"]
        if set(prefixes) != {str(count) for count in spec["event_counts"]}:
            return invalid("event-count prefix roster changed")
        for count in spec["event_counts"]:
            row = prefixes[str(count)]
            if (
                row["episodes"] != total
                or row["query_included"] != total
                or row["value_balanced"] != total
                or row["minimum_unique_pairs"] != count
                or row["maximum_unique_pairs"] != count
                or len(row["fingerprint_set_sha256"]) != 64
            ):
                return invalid(f"event count {count} prefix balance changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        judged = {str(count): {} for count in spec["event_counts"]}
        seed_hashes = {str(count): set() for count in spec["event_counts"]}
        for name, row in evaluations.items():
            expected = registered[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
            ):
                return invalid(f"evaluation {name} identity changed")
            counts = {item["event_count"]: item for item in row["counts"]}
            if set(counts) != set(spec["event_counts"]) or len(counts) != len(row["counts"]):
                return invalid(f"evaluation {name} event-count roster changed")
            for count in spec["event_counts"]:
                item = counts[count]
                state = item["state_audit"]
                path = item["path_audit"]
                if (
                    state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    <= state["maximum_cells"] <= spec["maximum_cells"]
                    or path["stores_per_episode"] != count
                    or path["retrievals_per_episode"] != spec["retrievals_per_episode"]
                    or path["address_width"] != spec["contexts"] + spec["keys"]
                    or path["query_position"] != 0
                ):
                    return invalid(f"evaluation {name} count {count} path changed")
                seed_hashes[str(count)].add(state["episode_seed_sha256"])
                arms = item["arms"]
                if set(arms) != set(spec["arms"]):
                    return invalid(f"evaluation {name} count {count} arm roster changed")
                for arm_name in spec["arms"]:
                    if (
                        not _metric_shape(arms[arm_name], spec["values"])
                        or arms[arm_name]["retrieval_api_match"]
                        != thresholds["retrieval_api_match"]
                    ):
                        return invalid(f"evaluation {name} count {count} metrics changed")
                normal = arms["exact_value_normal"]
                if (
                    normal["selection_accuracy"] < thresholds["selection_accuracy"]
                    or arms["exact_value_partner_swap"]["accuracy"]
                    > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_value_recovered"]["prediction_match"]
                    != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"evaluation {name} count {count} control failed")
                passed = (
                    normal["accuracy"] >= thresholds["final_accuracy"]
                    and min(normal["per_value_recall"]) >= thresholds["minimum_value_recall"]
                    and normal["correct_content_accuracy"]
                    >= thresholds["content_readout_accuracy"]
                )
                judged[str(count)][name] = {
                    "passed": passed,
                    "final_accuracy": normal["accuracy"],
                    "minimum_value_recall": min(normal["per_value_recall"]),
                    "content_readout_accuracy": normal["correct_content_accuracy"],
                    "selection_accuracy": normal["selection_accuracy"],
                    "partner_swap_accuracy": arms["exact_value_partner_swap"]["accuracy"],
                    "prototype_seed": row["prototype_seed"],
                    "engine_seed": row["engine_seed"],
                }
        for count in spec["event_counts"]:
            if len(seed_hashes[str(count)]) != len({row["engine_seed"] for row in spec["evaluation_combinations"]}):
                return invalid(f"event count {count} did not reuse paired engine starts")
        passed = [
            all(row["passed"] for row in judged[str(count)].values())
            for count in spec["event_counts"]
        ]
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    verdict, reason = boundary_verdict(passed)
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "counts": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/value_results.json")
    parser.add_argument("--output", default="measurement/value_verdict.json")
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
