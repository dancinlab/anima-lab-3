#!/usr/bin/env python3
"""Fail-closed adjudication for CONTEXT-SETTLE-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _valid_receipt
    from measurement.component2_gate import adjudicate as adjudicate_source
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as source_spec_sha256
    from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _valid_receipt
    from measurement.component2_gate import adjudicate as adjudicate_source
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as source_spec_sha256
    from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256


def _metric_shape(metric: dict, classes: int) -> bool:
    return (
        len(metric["per_key_recall"]) == classes
        and len(metric["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metric["confusion_matrix"])
    )


def _classify(baseline_reproduced: bool, candidate_pass: dict[int, bool]) -> tuple[str, str, int | None]:
    if not baseline_reproduced:
        return "CT3_BASELINE_ALREADY_STABLE", "three context steps did not reproduce the registered transition loss", None
    for steps in sorted(candidate_pass):
        if candidate_pass[steps]:
            return "CT1_MINIMUM_SETTLING_FOUND", f"all context positions recovered at {steps} settling steps", steps
    return "CT2_NO_REGISTERED_SETTLING_RECOVERY", "at least one context position still failed through nine settling steps", None


def adjudicate(payload: dict, spec: dict = CONTEXT_SETTLE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CT0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"] or payload["spec"] != spec:
            return invalid("experiment or registered spec changed")
        if payload["spec_sha256"] != spec_sha256(spec) or not _finite_tree(payload):
            return invalid("registered spec digest or finite-value check failed")
        source = payload["source_component2"]
        if any(not _valid_receipt(source[name]) for name in ("results", "verdict", "checkpoint")):
            return invalid("COMPONENT-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = source_spec_sha256(COMPONENT2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != COMPONENT2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_source(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("checkpoint") != source_results.get("checkpoint")
        ):
            return invalid("COMPONENT-2 source identity changed")
        checkpoint = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=True)
        if (
            checkpoint.get("experiment") != COMPONENT2_SPEC["experiment"]
            or checkpoint.get("spec_sha256") != source_sha
            or checkpoint.get("deterministic") is not True
        ):
            return invalid("frozen context projector identity changed")

        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        total_events = total * spec["events_per_episode"]
        if (
            audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total
            or not _balanced(audit["target_counts"], spec["values"], total)
            or not _balanced(audit["query_context_counts"], spec["contexts"], total)
            or not _balanced(audit["query_key_counts"], spec["keys"], total)
            or not _balanced(audit["event_context_counts"], spec["contexts"], total_events)
            or not _balanced(audit["event_key_counts"], spec["keys"], total_events)
            or set(audit["source_overlap"]) != set(spec["source_overlap_sets"])
            or any(audit["source_overlap"].values())
        ):
            return invalid("evaluation dataset balance, uniqueness, or isolation changed")

        engines = {row["engine_seed"]: row for row in payload["engines"]}
        if set(engines) != set(spec["engine_seeds"]) or len(engines) != len(payload["engines"]):
            return invalid("engine roster changed")
        judged: dict[str, dict[str, dict]] = {}
        candidate_pass = {steps: True for steps in spec["context_steps"] if steps != spec["baseline_steps"]}
        baseline_by_seed = {}
        expected_seed_digest = None
        for seed, engine in engines.items():
            if not engine["projector_frozen"] or not engine["projector_unchanged"]:
                return invalid(f"engine {seed} changed the frozen context projector")
            candidates = {row["context_steps"]: row for row in engine["candidates"]}
            if set(candidates) != set(spec["context_steps"]) or len(candidates) != len(engine["candidates"]):
                return invalid(f"engine {seed} context-step roster changed")
            judged[str(seed)] = {}
            for steps, row in candidates.items():
                state = row["state_audit"]
                if expected_seed_digest is None:
                    expected_seed_digest = {}
                previous_digest = expected_seed_digest.get(seed)
                if previous_digest is None:
                    expected_seed_digest[seed] = state["episode_seed_sha256"]
                if (
                    state["episodes"] != total
                    or state["states"] != total_events
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or state["episode_seed_sha256"] != expected_seed_digest[seed]
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    or state["minimum_cells"] > state["maximum_cells"]
                    or state["maximum_cells"] > spec["maximum_cells"]
                    or state["context_step_calls"] != total_events * steps
                    or state["key_step_calls"] != total_events * spec["key_steps"]
                    or state["value_step_calls"] != total_events * spec["value_steps"]
                ):
                    return invalid(f"engine {seed} step {steps} state or call audit changed")
                positions = {item["position"]: item for item in row["positions"]}
                if set(positions) != set(spec["positions"]) or len(positions) != len(row["positions"]):
                    return invalid(f"engine {seed} step {steps} position roster changed")
                public_positions = {}
                all_passed = True
                for position in spec["positions"]:
                    item = positions[position]
                    metric = item["context"]
                    if item["position_label"] != position + 1 or not _metric_shape(metric, spec["contexts"]):
                        return invalid(f"engine {seed} step {steps} position metrics changed")
                    minimum_recall = min(metric["per_key_recall"])
                    passed = (
                        metric["accuracy"] >= spec["thresholds"]["classification_accuracy"]
                        and minimum_recall >= spec["thresholds"]["minimum_class_recall"]
                    )
                    all_passed &= passed
                    public_positions[str(position)] = {
                        "accuracy": metric["accuracy"],
                        "minimum_recall": minimum_recall,
                        "passed": passed,
                    }
                judged[str(seed)][str(steps)] = {
                    "all_positions_passed": all_passed,
                    "positions": public_positions,
                }
                if steps != spec["baseline_steps"]:
                    candidate_pass[steps] &= all_passed
            baseline = judged[str(seed)][str(spec["baseline_steps"])]["positions"]
            baseline_by_seed[seed] = any(
                not baseline[str(position)]["passed"] for position in spec["transition_positions"]
            )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    verdict, reason, minimum = _classify(all(baseline_by_seed.values()), candidate_pass)
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "minimum_settling_steps": minimum,
        "baseline_reproduced": {str(seed): value for seed, value in baseline_by_seed.items()},
        "candidate_pass": {str(steps): value for steps, value in candidate_pass.items()},
        "engines": judged,
        "source_checkpoint": source["checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/context_settle_results.json")
    parser.add_argument("--output", default="measurement/context_settle_verdict.json")
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
