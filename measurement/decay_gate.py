#!/usr/bin/env python3
"""Fail-closed adjudication for DECAY-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
    from measurement.decay_registry import DECAY_SPEC, spec_sha256
    from measurement.episode2_registry import EPISODE2_SPEC
    from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
except ModuleNotFoundError:
    from capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
    from decay_registry import DECAY_SPEC, spec_sha256
    from episode2_registry import EPISODE2_SPEC
    from key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _valid_receipt(receipt: dict) -> bool:
    path = Path(receipt["path"])
    return path.is_file() and _sha256_file(path) == receipt["sha256"]


def _balanced(counts: dict, categories: int, total: int) -> bool:
    expected = total // categories
    return counts == {str(index): expected for index in range(categories)}


def _metric_shape(metrics: dict, classes: int) -> bool:
    return (
        len(metrics["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metrics["confusion_matrix"])
        and len(metrics["per_value_recall"]) == classes
        and len(metrics["selection_counts"]) == classes
        and all(name in metrics for name in (
            "selection_accuracy", "correct_content_accuracy", "retrieval_api_match",
            "key_margin_mean", "key_margin_min",
        ))
    )


def _verify_projector(receipt: dict, seed: int, spec: dict) -> bool:
    if not _valid_receipt(receipt):
        return False
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    state = checkpoint.get("projector", {})
    expected = {
        "projection.weight": (spec["address_dim"], spec["state_dim"]),
        "projection.bias": (spec["address_dim"],),
        "prototypes": (spec["keys"], spec["address_dim"]),
    }
    return (
        checkpoint.get("experiment") == KEY_SPEC["experiment"]
        and checkpoint.get("spec_sha256") == key_spec_sha256(KEY_SPEC)
        and checkpoint.get("seed") == seed
        and checkpoint.get("model_class") == spec["model_class"]
        and set(state) == set(expected)
        and all(
            tuple(state[name].shape) == shape and torch.isfinite(state[name]).all()
            for name, shape in expected.items()
        )
    )


def _verify_prototypes(receipt: dict, spec: dict) -> bool:
    if not _valid_receipt(receipt):
        return False
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint.get("prototypes", {})
    return (
        set(prototypes) == {"quantum", "sensory"}
        and all(
            tuple(value.shape) == (spec["values"], spec["state_dim"])
            and torch.isfinite(value).all()
            for value in prototypes.values()
        )
    )


def _passes(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["selection_accuracy"] >= thresholds["stable_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["stable_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["stable_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = DECAY_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "D0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_capacity"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CAPACITY-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = capacity_spec_sha256(CAPACITY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CAPACITY_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("CAPACITY-1 source identity or verdict changed")

        total = spec["eval_episodes_per_delay"]
        audit = payload["dataset_audit"]
        if (
            audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or len(audit["fingerprint_set_sha256"]) != 64
            or not _balanced(audit["target_counts"], spec["values"], total)
            or not _balanced(audit["query_position_counts"], spec["queryable_events"], total)
            or not _balanced(audit["query_key_counts"], spec["keys"], total)
            or not _balanced(audit["query_context_counts"], spec["contexts"], total)
        ):
            return invalid("dataset balance or uniqueness changed")

        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if (
            set(source_rows) != set(spec["seeds"])
            or set(rows) != set(spec["seeds"])
            or len(rows) != len(payload["seeds"])
        ):
            return invalid("registered seeds are missing or duplicated")

        thresholds = spec["thresholds"]
        judged = {
            str(delay): {name: {} for name in spec["stable_arms"]}
            for delay in spec["distractor_steps"]
        }
        for seed in spec["seeds"]:
            row = rows[seed]
            source_row = source_rows[seed]
            projector_receipt = row["source_checkpoint"]
            prototype_receipt = row["prototype_checkpoint"]
            if (
                projector_receipt != source["checkpoints"].get(str(seed))
                or projector_receipt != source_row["source_checkpoint"]
                or prototype_receipt != source["prototype_checkpoints"].get(str(seed))
                or prototype_receipt != source_row["prototype_checkpoint"]
                or not _verify_projector(projector_receipt, seed, spec)
                or not _verify_prototypes(prototype_receipt, spec)
            ):
                return invalid(f"seed {seed} source checkpoint changed")
            if not row["projector_frozen"] or not row["projector_unchanged"]:
                return invalid(f"seed {seed} stable projector changed during evaluation")
            delay_rows = {item["distractor_steps"]: item for item in row["delays"]}
            if set(delay_rows) != set(spec["distractor_steps"]) or len(delay_rows) != len(row["delays"]):
                return invalid(f"seed {seed} delay roster changed")
            for delay in spec["distractor_steps"]:
                item = delay_rows[delay]
                if set(item["arms"]) != set(spec["arms"]):
                    return invalid(f"seed {seed} delay {delay} arm roster changed")
                state = item["state_audit"]
                if (
                    state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or state["prefix_state_matches"] != total
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    or state["minimum_cells"] > state["maximum_cells"]
                    or state["maximum_cells"] > spec["maximum_cells"]
                ):
                    return invalid(f"seed {seed} delay {delay} nested state stream changed")
                integration = item["integration_audit"]
                calls = integration["stable_transform_calls"]
                if set(calls) != set(spec["stable_arms"]):
                    return invalid(f"seed {seed} delay {delay} transform roster changed")
                for name, expected_calls in spec["expected_transform_calls"].items():
                    value = calls[name]
                    if (
                        value["episodes"] != total
                        or value["total"] != total * expected_calls
                        or value["minimum"] != expected_calls
                        or value["maximum"] != expected_calls
                    ):
                        return invalid(f"seed {seed} delay {delay} {name} transform path changed")
                if (
                    integration["address_width_minimum"] != spec["address_dim"]
                    or integration["address_width_maximum"] != spec["address_dim"]
                ):
                    return invalid(f"seed {seed} delay {delay} address width changed")
                arms = item["arms"]
                for name in spec["arms"]:
                    if not _metric_shape(arms[name], spec["values"]):
                        return invalid(f"seed {seed} delay {delay} {name} metrics are incomplete")
                    if arms[name]["retrieval_api_match"] != thresholds["retrieval_api_match"]:
                        return invalid(f"seed {seed} delay {delay} memory API mismatch")
                exact = arms["exact_three_candidates"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_three_partner_swap"]["accuracy"]
                    > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_three_recovered"]["prediction_match"]
                    != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"seed {seed} delay {delay} control failed")
                for name in spec["stable_arms"]:
                    metrics = arms[name]
                    judged[str(delay)][name][str(seed)] = {
                        "passed": _passes(metrics, thresholds),
                        "selection_accuracy": metrics["selection_accuracy"],
                        "final_accuracy": metrics["accuracy"],
                        "minimum_value_recall": min(metrics["per_value_recall"]),
                        "content_accuracy": metrics["correct_content_accuracy"],
                        "key_margin_mean": metrics["key_margin_mean"],
                    }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    aggregate = {
        str(delay): {
            name: all(row["passed"] for row in judged[str(delay)][name].values())
            for name in spec["stable_arms"]
        }
        for delay in spec["distractor_steps"]
    }
    baseline = str(spec["baseline_distractor_steps"])
    two_name, history_name, competition_name = spec["stable_arms"]
    if not aggregate[baseline][two_name]:
        return invalid("registered two-event positive path did not reproduce at delay two")

    mixed = any(
        len({row["passed"] for row in judged[str(delay)][name].values()}) > 1
        for delay in spec["distractor_steps"] for name in spec["stable_arms"]
    )
    non_monotonic = False
    for name in spec["stable_arms"]:
        seen_failure = False
        for delay in spec["distractor_steps"]:
            passed = aggregate[str(delay)][name]
            non_monotonic |= seen_failure and passed
            seen_failure |= not passed
    h0 = aggregate["0"][history_name]
    c0 = aggregate["0"][competition_name]
    h2 = aggregate[baseline][history_name]
    c2 = aggregate[baseline][competition_name]
    if mixed or non_monotonic:
        verdict = "D5_NON_MONOTONIC_OR_MIXED"
        reason = "seed boundaries differed or a failed condition passed again at a longer delay"
    elif (h0 and not h2) or (c0 and not c2):
        verdict = "D3_DELAY_INTERACTION"
        reason = "the longer stream or third candidate passed without distractors and first failed after delay"
    elif not h2 and not h0:
        verdict = "D2_STREAM_HISTORY_LOSS"
        reason = "processing the third event caused the first registered loss before storing it"
    elif h2 and not c2 and not c0:
        verdict = "D1_RETRIEVAL_COMPETITION"
        reason = "adding the third stored candidate caused the first registered loss on the same state stream"
    elif h2 and c2:
        verdict = "D4_BOUNDARY_NOT_REPRODUCED"
        reason = "all nested stable-address paths passed at the registered two-distractor baseline"
    else:
        verdict = "D5_NON_MONOTONIC_OR_MIXED"
        reason = "the registered factors did not produce a single ordered boundary"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "condition_pass": aggregate, "delays": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/decay_results.json")
    parser.add_argument("--output", default="measurement/decay_verdict.json")
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
