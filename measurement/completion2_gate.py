#!/usr/bin/env python3
"""Fail-closed adjudication for COMPLETION-2."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from graft_behavior import sha256_file
    from measurement.capacity_gate import _metric_shape
    from measurement.completion2_registry import (
        COMPLETION2_SPEC, mask_plan_audit, spec_sha256,
    )
    from measurement.key_refresh2_gate import adjudicate as adjudicate_source
    from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graft_behavior import sha256_file
    from measurement.capacity_gate import _metric_shape
    from measurement.completion2_registry import (
        COMPLETION2_SPEC, mask_plan_audit, spec_sha256,
    )
    from measurement.key_refresh2_gate import adjudicate as adjudicate_source
    from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC
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


def _all_receipts_valid(value) -> bool:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return _receipt_valid(value)
        return all(_all_receipts_valid(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_receipts_valid(item) for item in value)
    return True


def _passes(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["selection_accuracy"] >= thresholds["damage_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["damage_final_accuracy"]
        and min(metrics["per_value_recall"])
        >= thresholds["damage_minimum_value_recall"]
    )


def _classify(common_boundary: int) -> tuple[str, str]:
    if common_boundary == 75:
        return "C2_BOUNDARY_75", "the unchanged repaired path survived every registered 75% cue loss"
    if common_boundary == 50:
        return "C2_BOUNDARY_50", "the common repaired-path boundary was 50% cue loss"
    if common_boundary == 25:
        return "C2_BOUNDARY_25", "the common repaired-path boundary remained at 25% cue loss"
    raise ValueError("common boundary is outside the registered levels")


def adjudicate(payload: dict, spec: dict = COMPLETION2_SPEC,
               *, source_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "C20_INVALID", "reason": reason,
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
        if not _all_receipts_valid(source):
            return invalid("a registered source receipt changed")
        if source_results is None:
            source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != KEY_REFRESH2_SPEC
            or source_results.get("spec_sha256") != source["source_spec_sha256"]
            or source_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_source(source_results) != source_verdict
            or source["upstream"] != source_results["source"]["upstream"]
        ):
            return invalid("registered KEY-REFRESH-2 identity changed")
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
        level_names = {
            25: (
                "context_quarter_missing", "key_quarter_missing",
                "both_quarter_missing",
            ),
            50: (
                "context_half_missing", "key_half_missing", "both_half_missing",
            ),
            75: (
                "context_three_quarters_missing", "key_three_quarters_missing",
                "both_three_quarters_missing",
            ),
        }
        status = {
            level: {component: True for component in ("context", "key", "both")}
            for level in level_names
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
                or not all(
                    check == {"metric_match": True, "record_match": True}
                    for check in row["reference_audit"].values()
                )
                or set(row["reference_audit"]) != {
                    "full_cue", "context_quarter_missing", "key_quarter_missing",
                    "both_quarter_missing", "exact_context_key_control",
                    "exact_context_key_partner_swap",
                }
                or set(row["record_digests"]) != set(spec["arms"])
                or any(len(digest) != 64 for digest in row["record_digests"].values())
            ):
                return invalid(f"evaluation {name} identity, arm, frozen, or source match changed")

            event_queries = total * (spec["events_per_episode"] + 1)
            states = row["state_audit"]
            expected_context_steps = total * (
                spec["events_per_episode"] * spec["settled_context_steps"]
                + spec["query_context_sense_steps"]
            )
            expected_key_steps = total * (
                spec["events_per_episode"] * spec["key_sense_steps"]
                + spec["query_key_sense_steps"]
            )
            if (
                states["episodes"] != total
                or states["unique_episode_seeds"] != total
                or len(states["episode_seed_sha256"]) != 64
                or states["expected_context_states"] != event_queries
                or states["expected_key_states"] != event_queries
                or not spec["minimum_cells"] <= states["minimum_cells"] <= states["maximum_cells"] <= spec["maximum_cells"]
                or states["context_step_calls"] != expected_context_steps
                or states["key_step_calls"] != expected_key_steps
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

            arms = row["arms"]
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
            for absent in ("context_absent", "key_absent", "both_absent"):
                if (
                    arms[absent]["selection_accuracy"]
                    > thresholds["absent_max_selection_accuracy"]
                    or arms[absent]["accuracy"] > thresholds["absent_max_final_accuracy"]
                ):
                    return invalid(f"evaluation {name} missing-category information control leaked")
            for level, arm_names in level_names.items():
                for component, arm_name in zip(("context", "key", "both"), arm_names):
                    status[level][component] &= _passes(arms[arm_name], thresholds)
            judged[name] = {
                arm_name: {
                    "selection_accuracy": arms[arm_name]["selection_accuracy"],
                    "accuracy": arms[arm_name]["accuracy"],
                    "minimum_value_recall": min(arms[arm_name]["per_value_recall"]),
                }
                for arm_name in spec["arms"][:-2]
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError,
            ZeroDivisionError) as exc:
        return invalid(str(exc))

    if not all(status[25].values()):
        return invalid("the registered KEY-REFRESH-2 25% boundary did not reproduce")
    boundaries = {}
    for component in ("context", "key", "both"):
        boundaries[component] = 75 if status[75][component] else (
            50 if status[50][component] else 25
        )
    common_boundary = min(boundaries.values())
    verdict, reason = _classify(common_boundary)
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "common_boundary_percent": common_boundary,
        "component_boundaries_percent": boundaries,
        "profiles": {str(level): profile for level, profile in status.items()},
        "evaluations": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/completion2_results.json")
    parser.add_argument("--output", default="measurement/completion2_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
