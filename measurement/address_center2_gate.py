#!/usr/bin/env python3
"""Fail-closed adjudication for ADDRESS-CENTER-2."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from graft_behavior import sha256_file
    from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC, spec_sha256
    from measurement.address_margin_gate import adjudicate as adjudicate_address_margin
    from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC
    from measurement.capacity_gate import _metric_shape
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graft_behavior import sha256_file
    from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC, spec_sha256
    from measurement.address_margin_gate import adjudicate as adjudicate_address_margin
    from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC
    from measurement.capacity_gate import _metric_shape
    from measurement.projector_registry import evaluation_name


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _receipt_valid(receipt: dict) -> bool:
    try:
        path = Path(receipt["path"])
        return path.is_file() and sha256_file(path) == receipt["sha256"]
    except (KeyError, TypeError, OSError):
        return False


def adjudicate(payload: dict, spec: dict = ADDRESS_CENTER2_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "AC0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if (
            payload["experiment"] != spec["experiment"]
            or payload["spec"] != spec
            or payload["spec_sha256"] != spec_sha256(spec)
            or not _finite(payload)
        ):
            return invalid("experiment, registered spec, digest, or finite-value check failed")
        source = payload["source"]
        for name in ("results", "verdict", "component_checkpoint", "value_checkpoint"):
            if not _receipt_valid(source[name]):
                return invalid(f"registered source {name} changed")
        if any(not _receipt_valid(row) for row in source["prototype_checkpoints"].values()):
            return invalid("registered prototype checkpoint changed")
        if source_results is None:
            source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != ADDRESS_MARGIN_SPEC
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_address_margin(source_results) != source_verdict
            or source["component_checkpoint"] != source_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != source_results["source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != source_results["source"]["prototype_checkpoints"]
        ):
            return invalid("registered ADDRESS-MARGIN-1 identity changed")
        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        if (
            audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
        ):
            return invalid("registered dataset changed")
        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        center_pass = True
        disabled_loss = True
        causal = True
        judged = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["arms"]) != set(spec["arms"])
                or any(not _metric_shape(arm, spec["values"]) for arm in row["arms"].values())
            ):
                return invalid(f"evaluation {name} identity or arm shape changed")
            if row["frozen_audit"] != {"context": True, "key": True, "value": True}:
                return invalid(f"evaluation {name} changed a frozen transform")
            reference = row["reference_audit"]
            if reference != {
                "center_prediction_selection_match": 1.0,
                "disabled_prediction_selection_match": 1.0,
                "center_metric_match": True,
                "disabled_metric_match": True,
                "recovery_metric_match": True,
            }:
                return invalid(f"evaluation {name} disagrees with its registered references")
            event_queries = total * (spec["events_per_episode"] + 1)
            states = row["state_audit"]
            if (
                states["episodes"] != total
                or states["unique_episode_seeds"] != total
                or len(states["episode_seed_sha256"]) != 64
                or states["expected_context_states"] != event_queries
                or states["expected_key_states"] != event_queries
                or not spec["minimum_cells"] <= states["minimum_cells"] <= states["maximum_cells"] <= spec["maximum_cells"]
                or states["context_step_calls"] != event_queries * spec["settled_context_steps"]
                or states["key_step_calls"] != event_queries * spec["key_sense_steps"]
                or states["value_step_calls"] != total * spec["events_per_episode"] * spec["value_sense_steps"]
                or states["distractor_step_calls"] != total * spec["distractor_steps"] * spec["distractor_sense_steps"]
            ):
                return invalid(f"evaluation {name} state audit changed")
            if set(row["memory_path_audit"]) != set(spec["arms"][:4]):
                return invalid(f"evaluation {name} memory path roster changed")
            expected_path = {
                "minimum_calls": spec["transform_calls_per_episode"],
                "maximum_calls": spec["transform_calls_per_episode"],
                "minimum_components": spec["components_per_key"],
                "maximum_components": spec["components_per_key"],
                "minimum_address_width": spec["composite_address_dim"],
                "maximum_address_width": spec["composite_address_dim"],
                "value_calls": total * spec["stores_per_episode"],
                "stores": total * spec["stores_per_episode"],
                "retrievals": total * spec["retrievals_per_episode"],
            }
            if any(value != expected_path for value in row["memory_path_audit"].values()):
                return invalid(f"evaluation {name} common memory path changed")
            normal = row["arms"]["integrated_context_center"]
            disabled = row["arms"]["integrated_center_disabled"]
            recovered = row["arms"]["integrated_context_center_recovered"]
            masked = row["arms"]["integrated_context_masked"]
            exact = row["arms"]["exact_context_key_control"]
            partner = row["arms"]["exact_context_key_partner_swap"]
            if (
                normal != recovered
                or any(arm["retrieval_api_match"] != 1.0 for arm in row["arms"].values())
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or masked["selection_accuracy"] > thresholds["context_masked_max_accuracy"]
                or masked["accuracy"] > thresholds["context_masked_max_accuracy"]
                or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
            ):
                return invalid(f"evaluation {name} positive, masked, recovery, or swap control failed")
            center_ok = (
                normal["selection_accuracy"] >= thresholds["center_selection_accuracy"]
                and normal["accuracy"] >= thresholds["center_final_accuracy"]
                and min(normal["per_value_recall"]) >= thresholds["minimum_value_recall"]
            )
            disabled_ok = disabled["selection_accuracy"] <= thresholds["disabled_selection_max_accuracy"]
            gain = normal["selection_accuracy"] - disabled["selection_accuracy"]
            causal_ok = gain >= thresholds["minimum_selection_gain"]
            center_pass &= center_ok
            disabled_loss &= disabled_ok
            causal &= causal_ok
            judged[name] = {
                "center_selection": normal["selection_accuracy"],
                "center_final_accuracy": normal["accuracy"],
                "minimum_value_recall": min(normal["per_value_recall"]),
                "disabled_selection": disabled["selection_accuracy"],
                "selection_gain": gain,
                "masked_selection": masked["selection_accuracy"],
                "exact_selection": exact["selection_accuracy"],
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not center_pass:
        verdict = "AC2_INTEGRATED_CENTER_PATH_LOSS"
        reason = "the shared context-center memory path did not clear every registered threshold"
    elif not disabled_loss or not causal:
        verdict = "AC3_CENTER_NOT_CAUSAL"
        reason = "the disabled loss or minimum causal gain did not reproduce"
    else:
        verdict = "AC1_CONTEXT_CENTER_INTEGRATED_NOT_UNIQUE"
        reason = "the optional shared context-center path recovered retrieval and behavior"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "source_checkpoint": source["component_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/address_center2_results.json")
    parser.add_argument("--output", default="measurement/address_center2_verdict.json")
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
