#!/usr/bin/env python3
"""Fail-closed adjudication for VALUE-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.projector_registry import evaluation_name
    from measurement.value2_registry import VALUE2_SPEC, spec_sha256
    from measurement.value_mechanism_gate import adjudicate as adjudicate_mechanism
    from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256 as mechanism_spec_sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _metric_shape, _valid_receipt
    from measurement.projector_registry import evaluation_name
    from measurement.value2_registry import VALUE2_SPEC, spec_sha256
    from measurement.value_mechanism_gate import adjudicate as adjudicate_mechanism
    from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256 as mechanism_spec_sha256


def _classification_shape(value: dict, classes: int) -> bool:
    return (
        len(value["per_key_recall"]) == classes
        and len(value["confusion_matrix"]) == classes
        and all(len(row) == classes for row in value["confusion_matrix"])
    )


def _checkpoint_valid(receipt: dict, spec: dict) -> bool:
    if not _valid_receipt(receipt):
        return False
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    state = checkpoint.get("projector", {})
    expected = {
        "projection.weight": (spec["address_dim"], spec["input_dim"]),
        "projection.bias": (spec["address_dim"],),
        "prototypes": (spec["values"], spec["address_dim"]),
    }
    return (
        checkpoint.get("experiment") == spec["experiment"]
        and checkpoint.get("spec_sha256") == spec_sha256(spec)
        and checkpoint.get("model_class") == spec["model_class"]
        and set(state) == set(expected)
        and all(
            tuple(state[name].shape) == shape and torch.isfinite(state[name]).all()
            for name, shape in expected.items()
        )
        and checkpoint.get("deterministic")
        == {"repeat_equal": True, "reverse_order_equal": True}
    )


def adjudicate(payload: dict, spec: dict = VALUE2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "VT0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_value_mechanism1"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("VALUE-MECHANISM-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = mechanism_spec_sha256(VALUE_MECHANISM_SPEC)
        inherited = source_results.get("source_value1", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != VALUE_MECHANISM_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_mechanism(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("prototype_checkpoints") != inherited.get("prototype_checkpoints")
            or any(not _valid_receipt(item) for item in source["prototype_checkpoints"].values())
        ):
            return invalid("registered VALUE-MECHANISM-1 source identity changed")

        calibration = payload["calibration_dataset_audit"]["base"]
        evaluation = payload["eval_dataset_audit"]["base"]
        total = spec["calibration_episodes"]
        if (
            calibration["episodes"] != total
            or calibration["unique_fingerprints"] != total
            or calibration["latin_valid_episodes"] != total
            or not _balanced(calibration["target_counts"], spec["values"], total)
            or evaluation["episodes"] != spec["eval_episodes"]
            or evaluation["unique_fingerprints"] != spec["eval_episodes"]
            or calibration["fingerprint_set_sha256"] == evaluation["fingerprint_set_sha256"]
        ):
            return invalid("calibration and evaluation datasets changed or overlap")
        state = payload["calibration_state_audit"]
        expected_states = (
            spec["calibration_episodes"] * len(spec["calibration_engine_seeds"])
            * spec["events_per_episode"]
        )
        if (
            state["episodes"] != spec["calibration_episodes"] * len(spec["calibration_engine_seeds"])
            or state["states"] != expected_states
            or state["unique_engine_seeds"] != state["episodes"]
            or state["label_counts"]
            != {str(value): expected_states // spec["values"] for value in range(spec["values"])}
            or any(len(state[key]) != 64 for key in (
                "engine_seed_sha256", "state_sha256", "label_sha256"
            ))
            or not spec["minimum_cells"] <= state["minimum_cells"]
            <= state["maximum_cells"] <= spec["maximum_cells"]
        ):
            return invalid("calibration state stream changed")
        fit = payload["fit_audit"]
        if (
            fit["method"] != "ridge_fixed_orthogonal_targets"
            or fit["examples"] != expected_states
            or fit["input_dim"] != spec["input_dim"]
            or fit["address_dim"] != spec["address_dim"]
            or fit["keys"] != spec["values"]
            or fit["weight_regularization"] != spec["weight_decay"]
            or fit["label_sha256"] != state["label_sha256"]
            or payload["deterministic_audit"]
            != {"repeat_equal": True, "reverse_order_equal": True}
            or not _checkpoint_valid(payload["checkpoint"], spec)
        ):
            return invalid("deterministic value fit changed")

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
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
            ):
                return invalid(f"evaluation {name} identity changed")
            positions = {item["query_position"]: item for item in row["positions"]}
            if set(positions) != set(spec["query_positions"]) or len(positions) != len(row["positions"]):
                return invalid(f"evaluation {name} position roster changed")
            position_rows = {}
            seed_hash = None
            for position in spec["query_positions"]:
                item = positions[position]
                state_row, path = item["state_audit"], item["path_audit"]
                if (
                    item["query_position_label"] != position + 1
                    or state_row["episodes"] != spec["eval_episodes"]
                    or state_row["unique_episode_seeds"] != spec["eval_episodes"]
                    or len(state_row["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state_row["minimum_cells"]
                    <= state_row["maximum_cells"] <= spec["maximum_cells"]
                    or path["minimum_calls"] != spec["value_transform_calls_per_episode"]
                    or path["maximum_calls"] != spec["value_transform_calls_per_episode"]
                    or path["minimum_input_width"] != spec["input_dim"]
                    or path["maximum_input_width"] != spec["input_dim"]
                    or path["minimum_output_width"] != spec["address_dim"]
                    or path["maximum_output_width"] != spec["address_dim"]
                    or path["minimum_stores"] != spec["stores_per_episode"]
                    or path["maximum_stores"] != spec["stores_per_episode"]
                    or path["minimum_retrievals"] != spec["retrievals_per_episode"]
                    or path["maximum_retrievals"] != spec["retrievals_per_episode"]
                ):
                    return invalid(f"evaluation {name} position {position + 1} path changed")
                if seed_hash is None:
                    seed_hash = state_row["episode_seed_sha256"]
                elif state_row["episode_seed_sha256"] != seed_hash:
                    return invalid(f"evaluation {name} positions used different engine starts")
                classification = item["value_classification"]
                if not _classification_shape(classification, spec["values"]):
                    return invalid(f"evaluation {name} position {position + 1} classification changed")
                arms = item["arms"]
                if set(arms) != set(spec["arms"]):
                    return invalid(f"evaluation {name} position {position + 1} arms changed")
                for arm_name in spec["arms"]:
                    if (
                        not _metric_shape(arms[arm_name], spec["values"])
                        or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    ):
                        return invalid(f"evaluation {name} position {position + 1} metrics changed")
                normal = arms["integrated_stable_value_normal"]
                if (
                    normal["selection_accuracy"] < thresholds["selection_accuracy"]
                    or normal["reference_prediction_match"]
                    != thresholds["reference_prediction_match"]
                    or arms["integrated_stable_value_partner_swap"]["accuracy"]
                    > thresholds["partner_swap_max_accuracy"]
                    or arms["integrated_stable_value_recovered"]["prediction_match"]
                    != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"evaluation {name} position {position + 1} control failed")
                classification_passed = (
                    classification["accuracy"] >= thresholds["value_classification_accuracy"]
                    and min(classification["per_key_recall"])
                    >= thresholds["value_classification_minimum_recall"]
                )
                integration_passed = (
                    normal["accuracy"] >= thresholds["final_accuracy"]
                    and min(normal["per_value_recall"]) >= thresholds["minimum_value_recall"]
                )
                position_rows[str(position)] = {
                    "classification_passed": classification_passed,
                    "integration_passed": integration_passed,
                    "classification_accuracy": classification["accuracy"],
                    "classification_minimum_recall": min(classification["per_key_recall"]),
                    "final_accuracy": normal["accuracy"],
                    "minimum_value_recall": min(normal["per_value_recall"]),
                    "raw_accuracy": arms["raw_value_control"]["accuracy"],
                    "partner_swap_accuracy": arms["integrated_stable_value_partner_swap"]["accuracy"],
                }
            late = [position_rows[str(position)] for position in spec["query_positions"][1:]]
            causal = any(
                item["raw_accuracy"] <= thresholds["raw_late_max_accuracy"]
                and item["final_accuracy"] - item["raw_accuracy"]
                >= thresholds["minimum_causal_drop"]
                for item in late
            )
            judged[name] = {
                "prototype_seed": row["prototype_seed"],
                "engine_seed": row["engine_seed"],
                "positions": position_rows,
                "causal_control_passed": causal,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    rows = list(judged.values())
    if not all(row["causal_control_passed"] for row in rows):
        verdict = "VT4_NOT_CAUSAL"
        reason = "disabling the value transform did not reproduce the registered late-position loss"
    elif not all(
        item["classification_passed"]
        for row in rows for item in row["positions"].values()
    ):
        verdict = "VT2_VALUE_TRANSFORM_INVALID"
        reason = "the deterministic value transform did not classify held-out serial states"
    elif not all(
        item["integration_passed"]
        for row in rows for item in row["positions"].values()
    ):
        verdict = "VT3_MEMORY_INTEGRATION_LOSS"
        reason = "value classification passed, but the common memory path did not"
    else:
        verdict = "VT1_STABLE_VALUE_PATH_VALID_NOT_UNIQUE"
        reason = "the common memory path read a deterministic value representation at every position"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "checkpoint": payload["checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/value2_results.json")
    parser.add_argument("--output", default="measurement/value2_verdict.json")
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
