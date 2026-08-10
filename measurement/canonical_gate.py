#!/usr/bin/env python3
"""Fail-closed adjudication for CANONICAL-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from measurement.canonical_registry import CANONICAL_SPEC, spec_sha256
from measurement.capacity2_gate import _passes
from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
from measurement.projector_registry import PROJECTOR_SPEC, evaluation_name
from measurement.seedmap_registry import SEEDMAP_SPEC
from measurement.training_gate import adjudicate as adjudicate_training
from measurement.training_registry import TRAINING_SPEC, spec_sha256 as training_spec_sha256


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
        and checkpoint.get("name") == row["name"]
        and checkpoint.get("calibration_seeds") == row["calibration_seeds"]
        and checkpoint.get("model_class") == spec["model_class"]
        and checkpoint.get("source_audits") == row["source_audits"]
        and checkpoint.get("fit_audit") == row["fit_audit"]
        and checkpoint.get("repeat_equal") == row["repeat_equal"]
        and checkpoint.get("reverse_order_max_abs_delta") == row["reverse_order_max_abs_delta"]
        and set(state) == set(expected)
        and all(tuple(state[name].shape) == shape and torch.isfinite(state[name]).all() for name, shape in expected.items())
    )


def adjudicate(payload: dict, spec: dict = CANONICAL_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CN0_INVALID",
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
        source = payload["source_training"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("TRAINING-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        expected_sha = training_spec_sha256(TRAINING_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != TRAINING_SPEC
            or source_results.get("spec_sha256") != expected_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_sha
            or adjudicate_training(source_results) != source_verdict
            or source["source_spec_sha256"] != expected_sha
        ):
            return invalid("registered TRAINING-1 source identity changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("calibration dataset changed")
        if payload["capacity_dataset_audit"] != source_results["capacity_dataset_audit"]:
            return invalid("event-four dataset changed")
        if source["prototype_checkpoints"] != source_results["source_projector"]["prototype_checkpoints"]:
            return invalid("prototype checkpoint roster changed")
        for seed in spec["factor_seeds"]:
            if not _verify_prototypes(source["prototype_checkpoints"][str(seed)], SEEDMAP_SPEC):
                return invalid(f"prototype seed {seed} changed")

        projector_source = json.loads(Path(source_results["source_projector"]["results"]["path"]).read_text())
        source_audits = {}
        for row in projector_source["projectors"]:
            source_audits.setdefault(row["calibration_seed"], row["calibration_audit"])
        expected_arms = {row["name"]: row for row in spec["calibration_arms"]}
        rows = {row["name"]: row for row in payload["canonical_projectors"]}
        if set(rows) != set(expected_arms) or len(rows) != len(payload["canonical_projectors"]):
            return invalid("canonical projector roster changed")
        expected_evals = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        thresholds = spec["thresholds"]
        robust, summaries = {}, {}
        order_sensitive = False
        for name, row in rows.items():
            registered = expected_arms[name]
            if row["calibration_seeds"] != registered["calibration_seeds"]:
                return invalid(f"canonical projector {name} source identity changed")
            expected_source_audits = {str(seed): source_audits[seed] for seed in row["calibration_seeds"]}
            fit = row["fit_audit"]
            if (
                row["source_audits"] != expected_source_audits
                or fit["method"] != spec["method"]
                or fit["examples"] != spec["calibration_episodes"] * len(row["calibration_seeds"]) * 3
                or fit["input_dim"] != spec["input_dim"]
                or fit["address_dim"] != spec["address_dim"]
                or fit["keys"] != spec["keys"]
                or fit["weight_regularization"] != spec["weight_decay"]
                or fit["bias_regularized"] is not False
                or fit["design_rank"] <= 0
                or len(fit["label_sha256"]) != 64
                or row["repeat_equal"] is not True
                or not row["projector_frozen"]
                or not row["projector_unchanged"]
                or not _checkpoint_valid(row["checkpoint"], row, spec)
            ):
                return invalid(f"canonical projector {name} fit or checkpoint changed")
            delta = row["reverse_order_max_abs_delta"]
            if len(row["calibration_seeds"]) > 1:
                if delta is None:
                    return invalid(f"canonical projector {name} order audit missing")
                order_sensitive = order_sensitive or delta > spec["order_tolerance"]
            elif delta is not None:
                return invalid(f"canonical projector {name} unexpected order audit")
            evaluations = {item["name"]: item for item in row["evaluations"]}
            if set(evaluations) != set(expected_evals) or len(evaluations) != len(row["evaluations"]):
                return invalid(f"canonical projector {name} evaluation roster changed")
            passed_rows, eval_summary, signatures = [], {}, []
            for eval_name, item in evaluations.items():
                registered_eval = expected_evals[eval_name]
                if item["prototype_seed"] != registered_eval["prototype_seed"] or item["engine_seed"] != registered_eval["engine_seed"]:
                    return invalid(f"canonical projector {name} evaluation {eval_name} identity changed")
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
                    return invalid(f"canonical projector {name} evaluation {eval_name} execution changed")
                signatures.append((item["engine_seed"], state["episode_seed_sha256"], update["state_before_sha256"], update["query_rng_sha256"]))
                arms = result["arms"]
                if any(
                    not _metric_shape(arms[arm], spec["values"])
                    or arms[arm]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    for arm in spec["arms"]
                ):
                    return invalid(f"canonical projector {name} evaluation {eval_name} metrics changed")
                exact = arms["exact_key_control"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"canonical projector {name} evaluation {eval_name} control failed")
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
                    return invalid(f"canonical projector {name} changed paired engine state")
            robust[name] = all(passed_rows)
            summaries[name] = {
                "robust": robust[name],
                "repeat_equal": row["repeat_equal"],
                "reverse_order_max_abs_delta": delta,
                "evaluations": eval_summary,
                "checkpoint": row["checkpoint"],
            }
        if order_sensitive:
            verdict, reason = "CN4_ORDER_SENSITIVE", "the pooled closed-form address exceeded the registered order tolerance"
        elif all(robust.values()):
            verdict, reason = "CN1_CANONICAL_ADDRESS_VALID", "all deterministic calibration arms were robust"
        elif robust.get("pooled"):
            verdict, reason = "CN2_POOLED_ONLY_VALID", "only the pooled deterministic address was robust"
        else:
            verdict, reason = "CN3_CANONICAL_NOT_ROBUST", "the pooled deterministic address was not robust"
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "robust": robust,
            "canonical_projectors": summaries,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/canonical_results.json")
    parser.add_argument("--output", default="measurement/canonical_verdict.json")
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
