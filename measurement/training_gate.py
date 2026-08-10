#!/usr/bin/env python3
"""Fail-closed adjudication for TRAINING-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from measurement.capacity2_gate import _passes
from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
from measurement.projector_gate import adjudicate as adjudicate_projector
from measurement.projector_registry import PROJECTOR_SPEC, evaluation_name, projector_name, spec_sha256 as projector_spec_sha256
from measurement.seedmap_registry import SEEDMAP_SPEC
from measurement.training_registry import TRAINING_SPEC, spec_sha256, training_name


def _classify(grid: dict[str, bool], low: int, high: int) -> tuple[str, str]:
    a = grid[training_name({"initialization_seed": low, "batch_seed": low})]
    b = grid[training_name({"initialization_seed": low, "batch_seed": high})]
    c = grid[training_name({"initialization_seed": high, "batch_seed": low})]
    d = grid[training_name({"initialization_seed": high, "batch_seed": high})]
    if not a and not b and c and d:
        return "TR1_INITIALIZATION_CAUSAL", "robustness followed initialization across minibatch orders"
    if not a and b and not c and d:
        return "TR2_BATCH_ORDER_CAUSAL", "robustness followed minibatch order across initializations"
    if not a and b and c and d:
        return "TR3_EITHER_FACTOR_SUFFICIENT", "changing either training-randomness factor recovered robustness"
    if not a and not b and not c and d:
        return "TR4_BOTH_FACTORS_REQUIRED", "both training-randomness factors had to change together"
    return "TR5_FACTOR_INTERACTION_OR_MIXED", "the robust grid did not match a registered simple factor pattern"


def _checkpoint_valid(receipt: dict, row: dict, spec: dict) -> bool:
    path = Path(receipt["path"])
    if not path.is_file() or sha256_file(path) != receipt["sha256"]:
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "projection.weight": (spec["address_dim"], spec["input_dim"]),
        "projection.bias": (spec["address_dim"],),
        "prototypes": (spec["keys"], spec["address_dim"]),
    }
    state = checkpoint.get("projector", {})
    return (
        checkpoint.get("experiment") == spec["experiment"]
        and checkpoint.get("spec_sha256") == spec_sha256(spec)
        and checkpoint.get("initialization_seed") == row["initialization_seed"]
        and checkpoint.get("batch_seed") == row["batch_seed"]
        and checkpoint.get("calibration_seed") == spec["calibration_seed"]
        and checkpoint.get("model_class") == spec["model_class"]
        and checkpoint.get("calibration_audit") == row["calibration_audit"]
        and checkpoint.get("training_audit") == row["training_audit"]
        and checkpoint.get("diagonal_checkpoint_match") == row["diagonal_checkpoint_match"]
        and set(state) == set(expected)
        and all(tuple(state[name].shape) == shape and torch.isfinite(state[name]).all() for name, shape in expected.items())
    )


def adjudicate(payload: dict, spec: dict = TRAINING_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "TR0_INVALID",
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
        source = payload["source_projector"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("PROJECTOR-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        expected_sha = projector_spec_sha256(PROJECTOR_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != PROJECTOR_SPEC
            or source_results.get("spec_sha256") != expected_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_sha
            or adjudicate_projector(source_results) != source_verdict
            or source["source_spec_sha256"] != expected_sha
        ):
            return invalid("registered PROJECTOR-1 source identity changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("calibration dataset changed")
        if payload["capacity_dataset_audit"] != source_results["capacity_dataset_audit"]:
            return invalid("event-four evaluation dataset changed")
        source_rows = {row["name"]: row for row in source_results["projectors"]}
        if source["projector_checkpoints"] != {name: row["checkpoint"] for name, row in source_rows.items()}:
            return invalid("source projector checkpoint roster changed")
        if source["prototype_checkpoints"] != source_results["source"]["prototype_checkpoints"]:
            return invalid("source prototype roster changed")
        for seed in spec["factor_seeds"]:
            if not _verify_prototypes(source["prototype_checkpoints"][str(seed)], SEEDMAP_SPEC):
                return invalid(f"prototype seed {seed} changed")

        expected_rows = {training_name(row): row for row in spec["training_combinations"]}
        rows = {row["name"]: row for row in payload["training_combinations"]}
        if set(rows) != set(expected_rows) or len(rows) != len(payload["training_combinations"]):
            return invalid("training combination roster changed")
        expected_evals = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        thresholds = spec["thresholds"]
        robust, summaries = {}, {}
        for name, row in rows.items():
            registered = expected_rows[name]
            if (
                row["initialization_seed"] != registered["initialization_seed"]
                or row["batch_seed"] != registered["batch_seed"]
                or row["calibration_seed"] != spec["calibration_seed"]
            ):
                return invalid(f"training combination {name} identity changed")
            calibration, training = row["calibration_audit"], row["training_audit"]
            if (
                calibration["episodes"] != spec["calibration_episodes"]
                or calibration["states"] != spec["calibration_episodes"] * 3
                or calibration["unique_engine_seeds"] != spec["calibration_episodes"]
                or len(calibration["engine_seed_sha256"]) != 64
                or training["examples"] != spec["calibration_episodes"] * 3
                or training["steps"] != spec["train_steps"]
                or training["shuffled"]
                or len(training["training_label_sha256"]) != 64
                or not row["projector_frozen"]
                or not row["projector_unchanged"]
                or not _checkpoint_valid(row["checkpoint"], row, spec)
            ):
                return invalid(f"training combination {name} audit or checkpoint changed")
            diagonal = row["initialization_seed"] == row["batch_seed"]
            if diagonal and row["diagonal_checkpoint_match"] is not True:
                return invalid(f"training combination {name} did not reproduce PROJECTOR-1")
            if not diagonal and row["diagonal_checkpoint_match"] is not None:
                return invalid(f"training combination {name} mixed identity changed")
            evals = {item["name"]: item for item in row["evaluations"]}
            if set(evals) != set(expected_evals) or len(evals) != len(row["evaluations"]):
                return invalid(f"training combination {name} evaluation roster changed")
            passed_rows, eval_summary, signatures = [], {}, []
            for eval_name, item in evals.items():
                registered_eval = expected_evals[eval_name]
                if (
                    item["prototype_seed"] != registered_eval["prototype_seed"]
                    or item["engine_seed"] != registered_eval["engine_seed"]
                    or item["source_reused"] is not diagonal
                ):
                    return invalid(f"training combination {name} evaluation {eval_name} identity changed")
                result = item["result"]
                total = spec["eval_episodes"]
                state, update, integration = result["state_audit"], result["update_audit"], result["integration_audit"]
                calls = spec["event_count"] + 1
                if (
                    result["event_count"] != spec["event_count"]
                    or set(result["arms"]) != set(spec["arms"])
                    or state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
                    or update["requested_updates"] != spec["settling_updates"]
                    or update["performed_updates_minimum"] != spec["settling_updates"]
                    or update["performed_updates_maximum"] != spec["settling_updates"]
                    or update["disabled"] != []
                    or integration["stable_transform_calls"] != {
                        "episodes": total, "total": total * calls, "minimum": calls, "maximum": calls,
                    }
                    or integration["address_width_minimum"] != spec["address_dim"]
                    or integration["address_width_maximum"] != spec["address_dim"]
                ):
                    return invalid(f"training combination {name} evaluation {eval_name} execution changed")
                signatures.append((item["engine_seed"], state["episode_seed_sha256"], update["state_before_sha256"], update["query_rng_sha256"]))
                arms = result["arms"]
                if any(
                    not _metric_shape(arms[arm], spec["values"])
                    or arms[arm]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    for arm in spec["arms"]
                ):
                    return invalid(f"training combination {name} evaluation {eval_name} metrics changed")
                exact = arms["exact_key_control"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"training combination {name} evaluation {eval_name} control failed")
                if diagonal:
                    source_name = projector_name({
                        "calibration_seed": spec["calibration_seed"],
                        "training_seed": row["initialization_seed"],
                    })
                    source_eval = next(value for value in source_rows[source_name]["evaluations"] if value["name"] == eval_name)
                    if result != source_eval["result"]:
                        return invalid(f"training combination {name} evaluation {eval_name} did not replay source")
                metrics = arms["stable_distinct_normal"]
                passed = _passes(metrics, thresholds)
                passed_rows.append(passed)
                eval_summary[eval_name] = {
                    "passed": passed,
                    "selection_accuracy": metrics["selection_accuracy"],
                    "content_accuracy": metrics["correct_content_accuracy"],
                    "final_accuracy": metrics["accuracy"],
                    "minimum_value_recall": min(metrics["per_value_recall"]),
                }
            for engine_seed in spec["factor_seeds"]:
                if len({value[1:] for value in signatures if value[0] == engine_seed}) != 1:
                    return invalid(f"training combination {name} changed paired engine state")
            robust[name] = all(passed_rows)
            summaries[name] = {
                "robust": robust[name],
                "diagonal_checkpoint_match": row["diagonal_checkpoint_match"],
                "evaluations": eval_summary,
                "checkpoint": row["checkpoint"],
            }
        low, high = spec["factor_seeds"]
        verdict, reason = _classify(robust, low, high)
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "robust_grid": robust,
            "training_combinations": summaries,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/training_results.json")
    parser.add_argument("--output", default="measurement/training_verdict.json")
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
