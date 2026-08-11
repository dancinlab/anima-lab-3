#!/usr/bin/env python3
"""Fail-closed adjudication for ADDRESS-MARGIN-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from graft_behavior import sha256_file
from measurement.address_margin_registry import ADDRESS_MARGIN_SPEC, spec_sha256
from measurement.context_settle2_gate import adjudicate as adjudicate_context_settle2
from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC
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


def _class_valid(metric: dict, classes: int) -> bool:
    return (
        set(metric) == {"accuracy", "per_class_recall", "confusion_matrix"}
        and len(metric["per_class_recall"]) == classes
        and len(metric["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metric["confusion_matrix"])
    )


def adjudicate(payload: dict, spec: dict = ADDRESS_MARGIN_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "AM0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"] or payload["spec"] != spec:
            return invalid("experiment or registered spec changed")
        if payload["spec_sha256"] != spec_sha256(spec) or not _finite(payload):
            return invalid("registered spec digest or finite-value check failed")
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
            or source_results.get("spec") != CONTEXT_SETTLE2_SPEC
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_context_settle2(source_results) != source_verdict
            or source["component_checkpoint"] != source_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != source_results["source"]["conjunction_source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != source_results["source"]["conjunction_source"]["prototype_checkpoints"]
        ):
            return invalid("registered CONTEXT-SETTLE-2 identity changed")
        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        if (
            audit["episodes"] != total or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
        ):
            return invalid("registered dataset changed")
        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        source_evaluations = {row["name"]: row for row in source_results["evaluations"]}
        thresholds = spec["thresholds"]
        category_pass = True
        center_pass = True
        causal = True
        source_loss = True
        judged = {}
        for name, row in evaluations.items():
            identity = registered[name]
            if row["prototype_seed"] != identity["prototype_seed"] or row["engine_seed"] != identity["engine_seed"]:
                return invalid(f"evaluation {name} identity changed")
            if set(row["arms"]) != set(spec["arms"]):
                return invalid(f"evaluation {name} arm roster changed")
            if row["frozen_audit"] != {"context": True, "key": True, "value": True}:
                return invalid(f"evaluation {name} changed a frozen transform")
            states = row["state_audit"]
            event_queries = total * (spec["events_per_episode"] + 1)
            if (
                states["episodes"] != total or states["unique_episode_seeds"] != total
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
            path = row["path_audit"]
            arm_count = len(spec["arms"])
            if path != {
                "arms_per_episode": arm_count,
                "value_calls": total * arm_count * spec["events_per_episode"],
                "stores": total * arm_count * spec["events_per_episode"],
                "retrievals": total * arm_count,
                "address_width": spec["composite_address_dim"],
            }:
                return invalid(f"evaluation {name} memory path changed")
            continuous = row["arms"]["continuous_frozen"]
            source_arm = source_evaluations[name]["conditions"]["settled_6"]["arms"][
                "integrated_stable_conjunction_normal"
            ]
            if continuous != {
                key: value for key, value in source_arm.items()
                if key not in {"reference_prediction_match", "reference_selection_match"}
            }:
                return invalid(f"evaluation {name} continuous source replay changed")
            oracle = row["arms"]["oracle_centers"]
            shifted = row["arms"]["shifted_center_control"]
            if (
                oracle["selection_accuracy"] != thresholds["oracle_selection_accuracy"]
                or oracle["accuracy"] < thresholds["oracle_final_accuracy"]
                or shifted["selection_accuracy"] > thresholds["shifted_selection_max_accuracy"]
                or any(arm["retrieval_api_match"] != 1.0 for arm in row["arms"].values())
            ):
                return invalid(f"evaluation {name} positive or shifted control failed")
            if set(row["category_metrics"]) != {"context", "key"}:
                return invalid(f"evaluation {name} category roster changed")
            category_ok = True
            for component, classes in (("context", spec["contexts"]), ("key", spec["keys"])):
                metric = row["category_metrics"][component]
                if not _class_valid(metric, classes):
                    return invalid(f"evaluation {name} {component} metric shape changed")
                category_ok &= (
                    metric["accuracy"] >= thresholds["category_accuracy"]
                    and min(metric["per_class_recall"]) >= thresholds["minimum_category_recall"]
                )
            centered = row["arms"]["predicted_centers"]
            center_ok = (
                centered["selection_accuracy"] >= thresholds["center_selection_accuracy"]
                and centered["accuracy"] >= thresholds["center_final_accuracy"]
                and min(centered["per_value_recall"]) >= thresholds["minimum_value_recall"]
            )
            gain = centered["selection_accuracy"] - continuous["selection_accuracy"]
            source_ok = continuous["selection_accuracy"] <= thresholds["source_selection_max_accuracy"]
            causal_ok = gain >= thresholds["minimum_selection_gain"]
            source_loss &= source_ok
            category_pass &= category_ok
            center_pass &= center_ok
            causal &= causal_ok
            judged[name] = {
                "continuous_selection": continuous["selection_accuracy"],
                "centered_selection": centered["selection_accuracy"],
                "selection_gain": gain,
                "centered_final_accuracy": centered["accuracy"],
                "minimum_value_recall": min(centered["per_value_recall"]),
                "context_category_accuracy": row["category_metrics"]["context"]["accuracy"],
                "key_category_accuracy": row["category_metrics"]["key"]["accuracy"],
                "continuous_margin": row["distance_metrics"]["continuous_frozen"],
                "centered_margin": row["distance_metrics"]["predicted_centers"],
                "deviation": row["deviation_metrics"],
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not source_loss or not causal:
        verdict = "AM4_MARGIN_NOT_CAUSAL"
        reason = "the registered continuous loss or minimum centered gain did not reproduce"
    elif not category_pass:
        verdict = "AM2_CATEGORY_READOUT_LOSS"
        reason = "a frozen component did not classify its registered category reliably"
    elif not center_pass:
        verdict = "AM3_COMPOSITE_DISTANCE_LOSS"
        reason = "categories were readable, but fixed category centers did not recover retrieval"
    else:
        verdict = "AM1_WITHIN_CLASS_MARGIN_LOSS"
        reason = "fixed predicted category centers recovered retrieval from continuous within-category drift"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "source_checkpoint": source["component_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/address_margin_results.json")
    parser.add_argument("--output", default="measurement/address_margin_verdict.json")
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
