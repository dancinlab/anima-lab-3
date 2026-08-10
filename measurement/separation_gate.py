#!/usr/bin/env python3
"""Fail-closed adjudication for SEPARATION-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
    from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
    from measurement.separation_registry import SEPARATION_SPEC, spec_sha256
except ModuleNotFoundError:
    from episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
    from key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
    from separation_registry import SEPARATION_SPEC, spec_sha256


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


def _metric_shape(metrics: dict, classes: int, diagnostics: bool = True) -> bool:
    valid = (
        len(metrics["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metrics["confusion_matrix"])
        and len(metrics["per_value_recall"]) == classes
        and len(metrics["selection_counts"]) == classes
    )
    if diagnostics:
        valid = valid and all(name in metrics for name in (
            "selection_accuracy", "correct_content_accuracy", "retrieval_api_match",
            "key_margin_mean", "key_margin_min",
        ))
    return valid


def _balanced(counts: dict, categories: int, total: int) -> bool:
    expected = total // categories
    return counts == {str(index): expected for index in range(categories)}


def adjudicate(payload: dict, spec: dict = SEPARATION_SPEC) -> dict:
    invalid = lambda reason: {
        "experiment": payload.get("experiment", spec["experiment"]),
        "verdict": "S0_INVALID",
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
        expected_source_sha = episode2_spec_sha256(EPISODE2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != EPISODE2_SPEC
            or source_results.get("spec_sha256") != expected_source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != expected_source_sha
        ):
            return invalid("EPISODE-2 source identity or verdict changed")

        audit = payload["dataset_audit"]
        if (
            audit["episodes"] != spec["eval_episodes"]
            or audit["unique_fingerprints"] != spec["eval_episodes"]
            or len(audit["fingerprint_set_sha256"]) != 64
            or not _balanced(audit["target_counts"], spec["values"], spec["eval_episodes"])
            or not _balanced(
                audit["query_position_counts"], spec["events_per_episode"], spec["eval_episodes"]
            )
            or not _balanced(audit["shared_key_counts"], spec["keys"], spec["eval_episodes"])
            or not _balanced(
                audit["query_context_counts"], spec["contexts"], spec["eval_episodes"]
            )
        ):
            return invalid("registered episode balance or uniqueness changed")

        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if (
            set(source_rows) != set(spec["seeds"])
            or set(rows) != set(spec["seeds"])
            or len(rows) != len(payload["seeds"])
        ):
            return invalid("registered seeds are missing or duplicated")

        thresholds = spec["thresholds"]
        judged = {}
        for seed in spec["seeds"]:
            row = rows[seed]
            source_row = source_rows[seed]
            if set(row["arms"]) != set(spec["arms"]):
                return invalid(f"seed {seed} arm roster changed")
            projector_receipt = row["source_checkpoint"]
            prototype_receipt = row["prototype_checkpoint"]
            if (
                projector_receipt != source["checkpoints"].get(str(seed))
                or projector_receipt != source_row["source_checkpoint"]
                or prototype_receipt != source["prototype_checkpoints"].get(str(seed))
                or prototype_receipt != source_row["prototype_checkpoint"]
                or not _valid_receipt(projector_receipt)
                or not _valid_receipt(prototype_receipt)
            ):
                return invalid(f"seed {seed} source checkpoint changed")
            projector_checkpoint = torch.load(
                projector_receipt["path"], map_location="cpu", weights_only=True
            )
            projector_state = projector_checkpoint.get("projector", {})
            expected_tensors = {
                "projection.weight": (spec["address_dim"], spec["state_dim"]),
                "projection.bias": (spec["address_dim"],),
                "prototypes": (spec["keys"], spec["address_dim"]),
            }
            if (
                projector_checkpoint.get("experiment") != KEY_SPEC["experiment"]
                or projector_checkpoint.get("spec_sha256") != key_spec_sha256(KEY_SPEC)
                or projector_checkpoint.get("seed") != seed
                or projector_checkpoint.get("model_class") != spec["model_class"]
                or set(projector_state) != set(expected_tensors)
                or any(
                    tuple(projector_state[name].shape) != shape
                    or not torch.isfinite(projector_state[name]).all()
                    for name, shape in expected_tensors.items()
                )
            ):
                return invalid(f"seed {seed} stable projector identity changed")
            prototype_checkpoint = torch.load(
                prototype_receipt["path"], map_location="cpu", weights_only=True
            )
            prototypes = prototype_checkpoint.get("prototypes", {})
            if (
                set(prototypes) != {"quantum", "sensory"}
                or any(
                    tuple(value.shape) != (spec["values"], spec["state_dim"])
                    or not torch.isfinite(value).all()
                    for value in prototypes.values()
                )
            ):
                return invalid(f"seed {seed} value prototypes changed")

            state = row["state_audit"]
            if (
                state["episodes"] != spec["eval_episodes"]
                or state["unique_episode_seeds"] != spec["eval_episodes"]
                or len(state["episode_seed_sha256"]) != 64
            ):
                return invalid(f"seed {seed} state stream changed")
            integration = row["integration_audit"]
            calls = integration["stable_transform_calls"]
            expected_calls = spec["expected_stable_transform_calls_per_episode"]
            if (
                calls["episodes"] != spec["eval_episodes"]
                or calls["total"] != spec["eval_episodes"] * expected_calls
                or calls["minimum"] != expected_calls
                or calls["maximum"] != expected_calls
                or integration["address_width_minimum"] != spec["address_dim"]
                or integration["address_width_maximum"] != spec["address_dim"]
                or not integration["projector_frozen"]
                or not integration["projector_unchanged"]
            ):
                return invalid(f"seed {seed} stable transform was not frozen or canonical")

            arms = row["arms"]
            for name in spec["arms"]:
                if not _metric_shape(arms[name], spec["values"]):
                    return invalid(f"seed {seed} {name} metrics are incomplete")
                if arms[name]["retrieval_api_match"] != thresholds["retrieval_api_match"]:
                    return invalid(f"seed {seed} {name} memory API disagrees with cosine selection")
            distinct = arms["stable_distinct_key_control"]
            exact = arms["exact_context_key_control"]
            if (
                distinct["selection_accuracy"] < thresholds["distinct_selection_accuracy"]
                or distinct["accuracy"] < thresholds["distinct_final_accuracy"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or arms["context_removed_control"]["accuracy"]
                > thresholds["context_removed_max_accuracy"]
                or arms["exact_context_key_partner_swap"]["accuracy"]
                > thresholds["partner_swap_max_accuracy"]
                or arms["exact_context_key_recovered"]["prediction_match"]
                != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"seed {seed} registered positive, negative, or recovery control failed")

            stable = arms["stable_similar_normal"]
            raw = arms["raw_similar_normal"]
            judged[str(seed)] = {
                "stable_selection_accuracy": stable["selection_accuracy"],
                "stable_final_accuracy": stable["accuracy"],
                "stable_minimum_value_recall": min(stable["per_value_recall"]),
                "stable_content_accuracy": stable["correct_content_accuracy"],
                "raw_selection_accuracy": raw["selection_accuracy"],
                "raw_final_accuracy": raw["accuracy"],
                "raw_minimum_value_recall": min(raw["per_value_recall"]),
                "distinct_final_accuracy": distinct["accuracy"],
                "exact_final_accuracy": exact["accuracy"],
                "context_removed_accuracy": arms["context_removed_control"]["accuracy"],
                "partner_swap_accuracy": arms["exact_context_key_partner_swap"]["accuracy"],
                "source_checkpoint": projector_receipt,
                "prototype_checkpoint": prototype_receipt,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    values = list(judged.values())
    selection_pass = all(
        row["stable_selection_accuracy"] >= thresholds["similar_selection_accuracy"]
        for row in values
    )
    stable_behavior_pass = all(
        row["stable_final_accuracy"] >= thresholds["similar_final_accuracy"]
        and row["stable_minimum_value_recall"] >= thresholds["similar_minimum_value_recall"]
        and row["stable_content_accuracy"] >= thresholds["content_readout_accuracy"]
        for row in values
    )
    raw_pass = all(
        row["raw_selection_accuracy"] >= thresholds["similar_selection_accuracy"]
        and row["raw_final_accuracy"] >= thresholds["similar_final_accuracy"]
        and row["raw_minimum_value_recall"] >= thresholds["similar_minimum_value_recall"]
        for row in values
    )
    if selection_pass and not stable_behavior_pass:
        verdict = "S4_VALUE_READOUT_LOSS"
        reason = "similar-event retrieval selected the right slot, but stored values did not support balanced behavior"
    elif selection_pass and stable_behavior_pass:
        verdict = "S1_SIMILAR_EPISODES_SEPARATED_NOT_UNIQUE"
        reason = "the existing stable memory path separated four similar episodes for both seeds"
    elif raw_pass:
        verdict = "S2_STABLE_ADDRESS_COLLISION"
        reason = "raw states separated similar episodes, but the stable address transform collapsed their context"
    else:
        verdict = "S3_CONTEXT_ADDRESS_LOSS"
        reason = "exact composite addresses worked, but neither current raw nor stable key-time states preserved context"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/separation_results.json")
    parser.add_argument("--output", default="measurement/separation_verdict.json")
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
