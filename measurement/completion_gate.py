#!/usr/bin/env python3
"""Fail-closed adjudication for COMPLETION-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from graft_behavior import sha256_file
    from measurement.address_center2_gate import adjudicate as adjudicate_source
    from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC
    from measurement.capacity_gate import _metric_shape
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit, spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graft_behavior import sha256_file
    from measurement.address_center2_gate import adjudicate as adjudicate_source
    from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC
    from measurement.capacity_gate import _metric_shape
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit, spec_sha256
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


def adjudicate(payload: dict, spec: dict = COMPLETION_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CP0_INVALID", "reason": reason,
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
            or source_results.get("spec") != ADDRESS_CENTER2_SPEC
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_source(source_results) != source_verdict
            or source["component_checkpoint"] != source_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != source_results["source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != source_results["source"]["prototype_checkpoints"]
        ):
            return invalid("registered ADDRESS-CENTER-2 identity changed")
        if payload["mask_plan_audit"] != mask_plan_audit(spec):
            return invalid("registered partial-cue mask plan changed")
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
        status = {
            "full": True, "quarter": True, "context_half": True,
            "key_half": True, "both_half_selection": True,
            "both_half_readout": True,
        }
        judged = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["arms"]) != set(spec["arms"])
                or any(not _metric_shape(arm, spec["values"]) for arm in row["arms"].values())
                or row["frozen_audit"] != {"context": True, "key": True, "value": True}
                or row["reference_audit"] != {"full_metric_match": True}
            ):
                return invalid(f"evaluation {name} identity, arm, frozen, or source match changed")
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
            if set(row["memory_path_audit"]) != set(spec["arms"][:6]):
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

            arms = row["arms"]
            full = arms["full_cue"]
            quarter = arms["both_quarter_missing"]
            context_half = arms["context_half_cue"]
            key_half = arms["key_half_cue"]
            both_half = arms["both_half_missing"]
            exact = arms["exact_context_key_control"]
            partner = arms["exact_context_key_partner_swap"]
            if (
                any(arm["retrieval_api_match"] != 1.0 for arm in arms.values())
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
            ):
                return invalid(f"evaluation {name} positive, API, or swap control failed")
            status["full"] &= (
                full["selection_accuracy"] >= thresholds["full_selection_accuracy"]
                and full["accuracy"] >= thresholds["full_final_accuracy"]
                and min(full["per_value_recall"]) >= thresholds["full_minimum_value_recall"]
            )
            status["quarter"] &= (
                quarter["selection_accuracy"] >= thresholds["quarter_selection_accuracy"]
                and quarter["accuracy"] >= thresholds["quarter_final_accuracy"]
            )
            status["context_half"] &= (
                context_half["selection_accuracy"] >= thresholds["single_half_selection_accuracy"]
                and context_half["accuracy"] >= thresholds["single_half_final_accuracy"]
            )
            status["key_half"] &= (
                key_half["selection_accuracy"] >= thresholds["single_half_selection_accuracy"]
                and key_half["accuracy"] >= thresholds["single_half_final_accuracy"]
            )
            status["both_half_selection"] &= (
                both_half["selection_accuracy"] >= thresholds["both_half_selection_accuracy"]
            )
            status["both_half_readout"] &= (
                both_half["accuracy"] >= thresholds["both_half_final_accuracy"]
                and min(both_half["per_value_recall"]) >= thresholds["both_half_minimum_value_recall"]
            )
            judged[name] = {
                arm_name: {
                    "selection_accuracy": arms[arm_name]["selection_accuracy"],
                    "accuracy": arms[arm_name]["accuracy"],
                    "minimum_value_recall": min(arms[arm_name]["per_value_recall"]),
                }
                for arm_name in spec["arms"][:6]
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not status["full"]:
        return invalid("the registered full-cue source path did not clear its thresholds")
    if not status["quarter"]:
        verdict = "CP2_FRAGILE_CUE_PATH"
        reason = "removing one quarter of both cue states broke the registered path"
    elif not status["context_half"]:
        verdict = "CP3_CONTEXT_PARTIAL_LOSS"
        reason = "the context state did not tolerate half-cue removal"
    elif not status["key_half"]:
        verdict = "CP4_KEY_PARTIAL_LOSS"
        reason = "the key state did not tolerate half-cue removal"
    elif not status["both_half_selection"]:
        verdict = "CP5_COMBINED_PARTIAL_LOSS"
        reason = "each half cue survived alone but their combined memory margin did not"
    elif not status["both_half_readout"]:
        verdict = "CP6_VALUE_READOUT_LOSS"
        reason = "the partial address selected a memory but its value readout did not clear thresholds"
    else:
        verdict = "CP1_PARTIAL_CUE_COMPLETION_VALID_NOT_UNIQUE"
        reason = "the unchanged shared memory path retrieved events from every registered partial cue"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "source_checkpoint": source["component_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/completion_results.json")
    parser.add_argument("--output", default="measurement/completion_verdict.json")
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
