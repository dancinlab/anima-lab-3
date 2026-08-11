#!/usr/bin/env python3
"""Fail-closed adjudication for CUE-MECHANISM-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from graft_behavior import sha256_file
    from measurement.capacity_gate import _metric_shape
    from measurement.completion_gate import adjudicate as adjudicate_source
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit
    from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graft_behavior import sha256_file
    from measurement.capacity_gate import _metric_shape
    from measurement.completion_gate import adjudicate as adjudicate_source
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit
    from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256
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


def _classification_shape(metric: dict, classes: int) -> bool:
    try:
        return (
            0.0 <= metric["accuracy"] <= 1.0
            and len(metric["per_class_recall"]) == classes
            and metric["minimum_class_recall"] == min(metric["per_class_recall"])
            and all(0.0 <= value <= 1.0 for value in metric["per_class_recall"])
            and len(metric["confusion_matrix"]) == classes
            and all(len(row) == classes for row in metric["confusion_matrix"])
            and sum(map(sum, metric["confusion_matrix"])) == CUE_MECHANISM_SPEC["eval_episodes"]
            and 0.0 <= metric["positive_center_margin_fraction"] <= 1.0
            and -1.000001 <= metric["full_address_similarity_minimum"] <= 1.000001
            and -1.000001 <= metric["full_address_similarity_mean"] <= 1.000001
        )
    except (KeyError, TypeError, ValueError):
        return False


def _distance_shape(metric: dict) -> bool:
    try:
        return 0.0 <= metric["positive_margin_fraction"] <= 1.0
    except (KeyError, TypeError):
        return False


def adjudicate(payload: dict, spec: dict = CUE_MECHANISM_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CM0_INVALID", "reason": reason,
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
            or source_results.get("spec") != COMPLETION_SPEC
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_source(source_results) != source_verdict
            or source["component_checkpoint"] != source_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != source_results["source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != source_results["source"]["prototype_checkpoints"]
        ):
            return invalid("registered COMPLETION-1 identity changed")
        if payload["mask_plan_audit"] != mask_plan_audit(COMPLETION_SPEC):
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
            "context_category": True,
            "key_category": True,
            "context_retrieval": True,
            "key_retrieval": True,
            "both_retrieval": True,
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
                or row["reference_audit"] != {
                    "full_metric_match": True, "both_quarter_metric_match": True,
                }
                or row["restoration_audit"] != {
                    "context_restored_to_full": True, "key_restored_to_full": True,
                }
            ):
                return invalid(f"evaluation {name} identity, arm, frozen, source, or restore check changed")
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
            if set(row["memory_path_audit"]) != set(spec["conditions"]):
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
            if (
                set(row["component_metrics"]) != set(spec["conditions"])
                or set(row["distance_metrics"]) != set(spec["conditions"])
            ):
                return invalid(f"evaluation {name} diagnostic condition roster changed")
            for condition in spec["conditions"]:
                metrics = row["component_metrics"][condition]
                if (
                    set(metrics) != {"context", "key"}
                    or not _classification_shape(metrics["context"], spec["contexts"])
                    or not _classification_shape(metrics["key"], spec["keys"])
                    or not _distance_shape(row["distance_metrics"][condition])
                ):
                    return invalid(f"evaluation {name} diagnostic shape changed")

            arms = row["arms"]
            full = arms["full_cue"]
            context = arms["context_quarter_missing"]
            key = arms["key_quarter_missing"]
            both = arms["both_quarter_missing"]
            exact = arms["exact_context_key_control"]
            partner = arms["exact_context_key_partner_swap"]
            if (
                any(arm["retrieval_api_match"] != 1.0 for arm in arms.values())
                or full["selection_accuracy"] < thresholds["full_selection_accuracy"]
                or full["accuracy"] < thresholds["full_final_accuracy"]
                or min(full["per_value_recall"]) < thresholds["full_minimum_value_recall"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
            ):
                return invalid(f"evaluation {name} full, positive, API, or swap control failed")

            context_category = row["component_metrics"]["context_quarter_missing"]["context"]
            key_category = row["component_metrics"]["key_quarter_missing"]["key"]
            status["context_category"] &= (
                context_category["accuracy"] >= thresholds["component_category_accuracy"]
                and context_category["minimum_class_recall"]
                >= thresholds["component_minimum_category_recall"]
            )
            status["key_category"] &= (
                key_category["accuracy"] >= thresholds["component_category_accuracy"]
                and key_category["minimum_class_recall"]
                >= thresholds["component_minimum_category_recall"]
            )
            status["context_retrieval"] &= (
                context["selection_accuracy"] >= thresholds["single_quarter_selection_accuracy"]
                and context["accuracy"] >= thresholds["single_quarter_final_accuracy"]
            )
            status["key_retrieval"] &= (
                key["selection_accuracy"] >= thresholds["single_quarter_selection_accuracy"]
                and key["accuracy"] >= thresholds["single_quarter_final_accuracy"]
            )
            status["both_retrieval"] &= (
                both["selection_accuracy"] >= thresholds["both_quarter_selection_accuracy"]
                and both["accuracy"] >= thresholds["both_quarter_final_accuracy"]
            )
            judged[name] = {
                "context_category_accuracy": context_category["accuracy"],
                "context_minimum_category_recall": context_category["minimum_class_recall"],
                "key_category_accuracy": key_category["accuracy"],
                "key_minimum_category_recall": key_category["minimum_class_recall"],
                "context_selection_accuracy": context["selection_accuracy"],
                "context_final_accuracy": context["accuracy"],
                "key_selection_accuracy": key["selection_accuracy"],
                "key_final_accuracy": key["accuracy"],
                "both_selection_accuracy": both["selection_accuracy"],
                "both_final_accuracy": both["accuracy"],
                "retrieval_margin_mean": {
                    condition: row["distance_metrics"][condition]["retrieval_margin_mean"]
                    for condition in spec["conditions"]
                },
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not status["context_category"] and not status["key_category"]:
        verdict = "CM3_DUAL_CATEGORY_LOSS"
        reason = "both partial component states lost registered category readout"
    elif not status["context_category"]:
        verdict = "CM1_CONTEXT_CATEGORY_LOSS"
        reason = "the partial context state lost registered category readout first"
    elif not status["key_category"]:
        verdict = "CM2_KEY_CATEGORY_LOSS"
        reason = "the partial key state lost registered category readout first"
    elif not status["context_retrieval"] and not status["key_retrieval"]:
        verdict = "CM6_DUAL_RETRIEVAL_MARGIN_LOSS"
        reason = "both categories survived but each partial component broke retrieval margin"
    elif not status["context_retrieval"]:
        verdict = "CM4_CONTEXT_RETRIEVAL_MARGIN_LOSS"
        reason = "context category survived but its partial address broke retrieval margin"
    elif not status["key_retrieval"]:
        verdict = "CM5_KEY_RETRIEVAL_MARGIN_LOSS"
        reason = "key category survived but its partial address broke retrieval margin"
    elif not status["both_retrieval"]:
        verdict = "CM7_COMPOSITE_ONLY_MARGIN_LOSS"
        reason = "each partial component survived alone but their combined address margin failed"
    else:
        verdict = "CM8_QUARTER_CUES_ROBUST"
        reason = "the unchanged shared memory path tolerated every registered quarter-cue removal"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "status": status,
        "evaluations": judged, "source_checkpoint": source["component_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/cue_mechanism_results.json")
    parser.add_argument("--output", default="measurement/cue_mechanism_verdict.json")
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
