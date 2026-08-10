#!/usr/bin/env python3
"""Fail-closed adjudication for CAPACITY-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256
    from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
    from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
except ModuleNotFoundError:
    from capacity_registry import CAPACITY_SPEC, spec_sha256
    from episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
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
    if isinstance(value, float):
        return math.isfinite(value)
    return True


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


def adjudicate(payload: dict, spec: dict = CAPACITY_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "C0_INVALID",
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

        source = payload["source_episode2"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("EPISODE-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = episode2_spec_sha256(EPISODE2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != EPISODE2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("EPISODE-2 source identity or verdict changed")

        audits = payload["dataset_audit"]
        if set(audits) != {str(value) for value in spec["event_counts"]}:
            return invalid("event-count dataset roster changed")
        for count in spec["event_counts"]:
            audit = audits[str(count)]
            total = spec["eval_episodes_per_count"]
            if (
                audit["episodes"] != total
                or audit["unique_fingerprints"] != total
                or len(audit["fingerprint_set_sha256"]) != 64
                or not _balanced(audit["target_counts"], spec["values"], total)
                or not _balanced(audit["query_position_counts"], count, total)
                or not _balanced(audit["shared_key_counts"], spec["keys"], total)
                or not _balanced(audit["query_context_counts"], spec["contexts"], total)
            ):
                return invalid(f"event count {count} balance or uniqueness changed")

        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if (
            set(source_rows) != set(spec["seeds"])
            or set(rows) != set(spec["seeds"])
            or len(rows) != len(payload["seeds"])
        ):
            return invalid("registered seeds are missing or duplicated")

        thresholds = spec["thresholds"]
        judged = {str(count): {} for count in spec["event_counts"]}
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
            count_rows = {item["event_count"]: item for item in row["counts"]}
            if set(count_rows) != set(spec["event_counts"]) or len(count_rows) != len(row["counts"]):
                return invalid(f"seed {seed} event-count roster changed")
            for count in spec["event_counts"]:
                item = count_rows[count]
                if set(item["arms"]) != set(spec["arms"]):
                    return invalid(f"seed {seed} count {count} arm roster changed")
                state = item["state_audit"]
                total = spec["eval_episodes_per_count"]
                if (
                    state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"]
                    or state["minimum_cells"] > state["maximum_cells"]
                    or state["maximum_cells"] > spec["maximum_cells"]
                ):
                    return invalid(f"seed {seed} count {count} state stream changed")
                integration = item["integration_audit"]
                calls = integration["stable_transform_calls"]
                expected_calls = count + 1
                if (
                    calls["episodes"] != total
                    or calls["total"] != total * expected_calls
                    or calls["minimum"] != expected_calls
                    or calls["maximum"] != expected_calls
                    or integration["address_width_minimum"] != spec["address_dim"]
                    or integration["address_width_maximum"] != spec["address_dim"]
                ):
                    return invalid(f"seed {seed} count {count} stable transform path changed")
                arms = item["arms"]
                for name in spec["arms"]:
                    if not _metric_shape(arms[name], spec["values"]):
                        return invalid(f"seed {seed} count {count} {name} metrics are incomplete")
                    if arms[name]["retrieval_api_match"] != thresholds["retrieval_api_match"]:
                        return invalid(f"seed {seed} count {count} memory API mismatch")
                exact = arms["exact_key_control"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_key_partner_swap"]["accuracy"]
                    > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_key_recovered"]["prediction_match"]
                    != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"seed {seed} count {count} positive, negative, or recovery control failed")
                stable = arms["stable_distinct_normal"]
                passed = (
                    stable["selection_accuracy"] >= thresholds["stable_selection_accuracy"]
                    and stable["accuracy"] >= thresholds["stable_final_accuracy"]
                    and min(stable["per_value_recall"])
                    >= thresholds["stable_minimum_value_recall"]
                    and stable["correct_content_accuracy"]
                    >= thresholds["content_readout_accuracy"]
                )
                judged[str(count)][str(seed)] = {
                    "passed": passed,
                    "stable_selection_accuracy": stable["selection_accuracy"],
                    "stable_final_accuracy": stable["accuracy"],
                    "stable_minimum_value_recall": min(stable["per_value_recall"]),
                    "stable_content_accuracy": stable["correct_content_accuracy"],
                    "raw_selection_accuracy": arms["raw_distinct_control"]["selection_accuracy"],
                    "raw_final_accuracy": arms["raw_distinct_control"]["accuracy"],
                    "exact_final_accuracy": exact["accuracy"],
                    "partner_swap_accuracy": arms["exact_key_partner_swap"]["accuracy"],
                    "source_checkpoint": projector_receipt,
                    "prototype_checkpoint": prototype_receipt,
                }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    passed = [
        all(row["passed"] for row in judged[str(count)].values())
        for count in spec["event_counts"]
    ]
    if any(passed[index] and not all(passed[:index]) for index in range(1, len(passed))):
        verdict = "C5_NON_MONOTONIC"
        reason = "a larger event count passed after a smaller event count failed"
    elif all(passed):
        verdict = "C1_CAPACITY_AT_LEAST_4"
        reason = "the existing stable address path preserved two, three, and four distinct events"
    elif passed[:2] == [True, True] and not passed[2]:
        verdict = "C2_CAPACITY_BOUNDARY_3"
        reason = "the existing stable address path passed through three events and failed at four"
    elif passed[0] and not passed[1] and not passed[2]:
        verdict = "C3_CAPACITY_BOUNDARY_2"
        reason = "the existing stable address path passed two events and failed from three"
    else:
        verdict = "C4_CAPACITY_BELOW_2"
        reason = "the existing stable address path failed at two, three, and four events"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "capacity_pass": {str(count): value for count, value in zip(spec["event_counts"], passed)},
        "counts": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/capacity_results.json")
    parser.add_argument("--output", default="measurement/capacity_verdict.json")
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
