#!/usr/bin/env python3
"""Fail-closed adjudication for PROJECTOR-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from measurement.capacity2_gate import _passes
from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from measurement.projector_registry import (
    PROJECTOR_SPEC,
    evaluation_name,
    projector_name,
    spec_sha256,
)
from measurement.seedmap_gate import adjudicate as adjudicate_seedmap
from measurement.seedmap_registry import SEEDMAP_SPEC, combination_name, spec_sha256 as seedmap_spec_sha256


def _classify(grid: dict[str, bool], low: int, high: int) -> tuple[str, str]:
    a = grid[projector_name({"calibration_seed": low, "training_seed": low})]
    b = grid[projector_name({"calibration_seed": low, "training_seed": high})]
    c = grid[projector_name({"calibration_seed": high, "training_seed": low})]
    d = grid[projector_name({"calibration_seed": high, "training_seed": high})]
    if not a and not b and c and d:
        return "PD1_CALIBRATION_STREAM_CAUSAL", "robustness followed the calibration-state seed across training seeds"
    if not a and b and not c and d:
        return "PD2_TRAINING_RANDOMNESS_CAUSAL", "robustness followed the training-randomness seed across calibration streams"
    if not a and b and c and d:
        return "PD3_EITHER_FACTOR_SUFFICIENT", "changing either bundled training factor recovered robustness"
    if not a and not b and not c and d:
        return "PD4_BOTH_FACTORS_REQUIRED", "both bundled training factors had to change together"
    return "PD5_FACTOR_INTERACTION_OR_MIXED", "the robust-projector grid did not match a registered simple factor pattern"


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
        and checkpoint.get("calibration_seed") == row["calibration_seed"]
        and checkpoint.get("training_seed") == row["training_seed"]
        and checkpoint.get("model_class") == spec["model_class"]
        and checkpoint.get("calibration_audit") == row["calibration_audit"]
        and checkpoint.get("training_audit") == row["training_audit"]
        and checkpoint.get("native_checkpoint_match") == row["native_checkpoint_match"]
        and set(state) == set(expected)
        and all(tuple(state[name].shape) == shape and torch.isfinite(state[name]).all() for name, shape in expected.items())
    )


def adjudicate(payload: dict, spec: dict = PROJECTOR_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "PD0_INVALID",
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
        source = payload["source"]
        for name in ("seedmap_results", "seedmap_verdict", "key_results", "key_verdict"):
            if not _valid_receipt(source[name]):
                return invalid(f"source receipt {name} changed")
        seedmap_results = json.loads(Path(source["seedmap_results"]["path"]).read_text())
        seedmap_verdict = json.loads(Path(source["seedmap_verdict"]["path"]).read_text())
        key_results = json.loads(Path(source["key_results"]["path"]).read_text())
        key_verdict = json.loads(Path(source["key_verdict"]["path"]).read_text())
        if (
            seedmap_results.get("experiment") != spec["source_experiment"]
            or seedmap_results.get("spec") != SEEDMAP_SPEC
            or seedmap_results.get("spec_sha256") != seedmap_spec_sha256(SEEDMAP_SPEC)
            or seedmap_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_seedmap(seedmap_results) != seedmap_verdict
            or key_results.get("experiment") != spec["source_key_experiment"]
            or key_results.get("spec") != KEY_SPEC
            or key_results.get("spec_sha256") != key_spec_sha256(KEY_SPEC)
            or key_verdict.get("verdict") != spec["source_key_verdict"]
            or source["seedmap_spec_sha256"] != seedmap_spec_sha256(SEEDMAP_SPEC)
            or source["key_spec_sha256"] != key_spec_sha256(KEY_SPEC)
        ):
            return invalid("registered SEEDMAP-1 or KEY-1 source identity changed")
        if payload["dataset_audit"] != key_results["dataset_audit"]:
            return invalid("KEY-1 calibration dataset changed")
        if payload["capacity_dataset_audit"] != seedmap_results["dataset_audit"]:
            return invalid("SEEDMAP-1 evaluation dataset changed")
        if (
            source["projector_checkpoints"] != seedmap_results["projector_checkpoints"]
            or source["prototype_checkpoints"] != seedmap_results["prototype_checkpoints"]
        ):
            return invalid("inherited checkpoint roster changed")
        for seed in spec["factor_seeds"]:
            if not _verify_prototypes(source["prototype_checkpoints"][str(seed)], spec):
                return invalid(f"prototype seed {seed} changed")

        expected_projectors = {projector_name(row): row for row in spec["training_combinations"]}
        rows = {row["name"]: row for row in payload["projectors"]}
        if set(rows) != set(expected_projectors) or len(rows) != len(payload["projectors"]):
            return invalid("training combination roster changed")
        expected_evaluations = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        seedmap_combinations = {row["name"]: row["result"] for row in seedmap_results["combinations"]}
        thresholds = spec["thresholds"]
        robust = {}
        summaries = {}
        for name, row in rows.items():
            registered = expected_projectors[name]
            if any(row[key] != registered[key] for key in ("calibration_seed", "training_seed")):
                return invalid(f"projector {name} factor identity changed")
            calibration = row["calibration_audit"]
            training = row["training_audit"]
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
                return invalid(f"projector {name} training or checkpoint audit changed")
            native = row["calibration_seed"] == row["training_seed"]
            if native and row["native_checkpoint_match"] is not True:
                return invalid(f"projector {name} did not reproduce KEY-1 weights")
            if not native and row["native_checkpoint_match"] is not None:
                return invalid(f"projector {name} mixed identity changed")
            evaluations = {item["name"]: item for item in row["evaluations"]}
            if set(evaluations) != set(expected_evaluations) or len(evaluations) != len(row["evaluations"]):
                return invalid(f"projector {name} evaluation roster changed")
            projector_passes = []
            evaluation_summary = {}
            audit_signatures = []
            for eval_name, item in evaluations.items():
                registered_eval = expected_evaluations[eval_name]
                if any(item[key] != registered_eval[key] for key in ("prototype_seed", "engine_seed")):
                    return invalid(f"projector {name} evaluation {eval_name} identity changed")
                if item["source_reused"] is not native:
                    return invalid(f"projector {name} evaluation reuse identity changed")
                result = item["result"]
                if result["event_count"] != spec["event_count"] or set(result["arms"]) != set(spec["arms"]):
                    return invalid(f"projector {name} evaluation {eval_name} arm roster changed")
                total = spec["eval_episodes"]
                state, update, integration = result["state_audit"], result["update_audit"], result["integration_audit"]
                calls = spec["event_count"] + 1
                if (
                    state["episodes"] != total
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
                    return invalid(f"projector {name} evaluation {eval_name} execution audit changed")
                audit_signatures.append((
                    item["engine_seed"], state["episode_seed_sha256"],
                    update["state_before_sha256"], update["query_rng_sha256"],
                ))
                arms = result["arms"]
                if any(
                    not _metric_shape(arms[arm], spec["values"])
                    or arms[arm]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    for arm in spec["arms"]
                ):
                    return invalid(f"projector {name} evaluation {eval_name} metrics changed")
                exact = arms["exact_key_control"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"projector {name} evaluation {eval_name} control failed")
                if native:
                    source_name = combination_name({"projector_seed": row["training_seed"], **registered_eval})
                    if result != seedmap_combinations[source_name]:
                        return invalid(f"projector {name} evaluation {eval_name} did not replay SEEDMAP-1")
                metrics = arms["stable_distinct_normal"]
                passed = _passes(metrics, thresholds)
                projector_passes.append(passed)
                evaluation_summary[eval_name] = {
                    "passed": passed,
                    "selection_accuracy": metrics["selection_accuracy"],
                    "content_accuracy": metrics["correct_content_accuracy"],
                    "final_accuracy": metrics["accuracy"],
                    "minimum_value_recall": min(metrics["per_value_recall"]),
                }
            for engine_seed in spec["factor_seeds"]:
                signatures = {value[1:] for value in audit_signatures if value[0] == engine_seed}
                if len(signatures) != 1:
                    return invalid(f"projector {name} crossing changed paired engine state")
            robust[name] = all(projector_passes)
            summaries[name] = {
                "robust": robust[name],
                "native_checkpoint_match": row["native_checkpoint_match"],
                "evaluations": evaluation_summary,
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
            "projectors": summaries,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/projector_results.json")
    parser.add_argument("--output", default="measurement/projector_verdict.json")
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
