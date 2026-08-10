#!/usr/bin/env python3
"""Fail-closed adjudication for EPISODE-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256
    from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
except ModuleNotFoundError:
    from episode2_registry import EPISODE2_SPEC, spec_sha256
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


def _metric_shape(metrics: dict, classes: int) -> bool:
    return (
        len(metrics["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metrics["confusion_matrix"])
        and len(metrics["per_value_recall"]) == classes
        and len(metrics["selection_counts"]) == classes
    )


def _checkpoint_is_valid(path: Path, receipt: dict) -> bool:
    return path.is_file() and _sha256_file(path) == receipt["sha256"]


def adjudicate(payload: dict, spec: dict = EPISODE2_SPEC) -> dict:
    invalid = lambda reason: {
        "experiment": payload.get("experiment", spec["experiment"]),
        "verdict": "E2I0_INVALID",
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

        source = payload["source_key"]
        results_path = Path(source["results"]["path"])
        verdict_path = Path(source["verdict"]["path"])
        for receipt, path in ((source["results"], results_path), (source["verdict"], verdict_path)):
            if not _checkpoint_is_valid(path, receipt):
                return invalid("KEY-1 source file changed")
        source_results = json.loads(results_path.read_text())
        source_verdict = json.loads(verdict_path.read_text())
        expected_source_sha = key_spec_sha256(KEY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != KEY_SPEC
            or source_results.get("spec_sha256") != expected_source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != expected_source_sha
        ):
            return invalid("KEY-1 source identity or verdict changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("evaluation dataset no longer matches KEY-1")

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
            state = row["state_audit"]
            source_state = source_row["eval_state_audit"]
            if (
                state["episodes"] != spec["eval_episodes"]
                or state["unique_episode_seeds"] != spec["eval_episodes"]
                or state["episode_seed_sha256"] != source_state["engine_seed_sha256"]
            ):
                return invalid(f"seed {seed} evaluation state stream changed")

            projector_receipt = row["source_checkpoint"]
            prototype_receipt = row["prototype_checkpoint"]
            if (
                projector_receipt != source["checkpoints"].get(str(seed))
                or projector_receipt != source_row["checkpoint"]
                or prototype_receipt != source["prototype_checkpoints"].get(str(seed))
                or prototype_receipt != source_row["source_checkpoint"]
            ):
                return invalid(f"seed {seed} source checkpoint roster changed")
            projector_path = Path(projector_receipt["path"])
            prototype_path = Path(prototype_receipt["path"])
            if not _checkpoint_is_valid(projector_path, projector_receipt):
                return invalid(f"seed {seed} KEY-1 projector changed")
            if not _checkpoint_is_valid(prototype_path, prototype_receipt):
                return invalid(f"seed {seed} EPISODE-1 value prototype changed")
            projector_checkpoint = torch.load(projector_path, map_location="cpu", weights_only=True)
            projector_state = projector_checkpoint.get("projector", {})
            expected_tensors = {
                "projection.weight": (spec["address_dim"], spec["state_dim"]),
                "projection.bias": (spec["address_dim"],),
                "prototypes": (spec["keys"], spec["address_dim"]),
            }
            if (
                projector_checkpoint.get("experiment") != spec["source_experiment"]
                or projector_checkpoint.get("spec_sha256") != expected_source_sha
                or projector_checkpoint.get("seed") != seed
                or projector_checkpoint.get("model_class") != spec["model_class"]
                or set(projector_state) != set(expected_tensors)
                or any(
                    tuple(projector_state[name].shape) != shape
                    or not torch.isfinite(projector_state[name]).all()
                    for name, shape in expected_tensors.items()
                )
            ):
                return invalid(f"seed {seed} KEY-1 projector identity changed")
            prototype_checkpoint = torch.load(prototype_path, map_location="cpu", weights_only=True)
            prototypes = prototype_checkpoint.get("prototypes", {})
            if (
                set(prototypes) != {"quantum", "sensory"}
                or any(
                    tuple(value.shape) != (spec["values"], spec["state_dim"])
                    or not torch.isfinite(value).all()
                    for value in prototypes.values()
                )
            ):
                return invalid(f"seed {seed} value prototype tensor changed")

            for arm in spec["arms"]:
                if not _metric_shape(row["arms"][arm], spec["values"]):
                    return invalid(f"seed {seed} {arm} metrics are incomplete")
            arms = row["arms"]
            if arms["manual_stable_reference"] != source_row["arms"]["stabilized_memory_normal"]:
                return invalid(f"seed {seed} manual KEY-1 reference replay changed")
            if arms["transform_disabled"] != source_row["arms"]["raw_quantum_memory"]:
                return invalid(f"seed {seed} transform-disabled EPISODE-1 replay changed")
            if arms["sensory_memory"] != source_row["arms"]["sensory_memory"]:
                return invalid(f"seed {seed} sensory-memory control changed")
            if arms["keyed_attention"] != source_row["arms"]["keyed_attention"]:
                return invalid(f"seed {seed} keyed-attention control changed")
            if arms["no_memory"] != source_row["arms"]["no_memory"]:
                return invalid(f"seed {seed} no-memory control changed")

            audit = row["integration_audit"]
            expected_calls = spec["expected_transform_calls_per_episode"]
            for name in (
                "normal_transform_calls", "partner_swap_transform_calls",
                "recovery_transform_calls",
            ):
                calls = audit[name]
                if (
                    calls["episodes"] != spec["eval_episodes"]
                    or calls["total"] != spec["eval_episodes"] * expected_calls
                    or calls["minimum"] != expected_calls
                    or calls["maximum"] != expected_calls
                ):
                    return invalid(f"seed {seed} {name} changed")
            if (
                audit["address_width_minimum"] != spec["address_dim"]
                or audit["address_width_maximum"] != spec["address_dim"]
                or not audit["projector_frozen"]
                or not audit["projector_unchanged"]
            ):
                return invalid(f"seed {seed} address transform was not frozen or canonical")

            integrated = arms["integrated_stable_normal"]
            if integrated["retrieval_api_match"] != 1.0:
                return invalid(f"seed {seed} VectorMemory retrieval disagrees with cosine selection")
            if (
                arms["integrated_stable_partner_swap"]["accuracy"]
                > thresholds["negative_control_max_accuracy"]
                or arms["integrated_stable_recovered"]["prediction_match"]
                != thresholds["recovery_prediction_match"]
                or arms["sensory_memory"]["accuracy"] < thresholds["positive_control_accuracy"]
                or arms["keyed_attention"]["accuracy"] < thresholds["positive_control_accuracy"]
                or arms["no_memory"]["accuracy"] > thresholds["negative_control_max_accuracy"]
            ):
                return invalid(f"seed {seed} registered control or intervention failed")

            judged[str(seed)] = {
                "selection_accuracy": integrated["selection_accuracy"],
                "final_accuracy": integrated["accuracy"],
                "minimum_value_recall": min(integrated["per_value_recall"]),
                "content_accuracy": integrated["correct_content_accuracy"],
                "partner_swap_accuracy": arms["integrated_stable_partner_swap"]["accuracy"],
                "manual_prediction_match": audit["manual_prediction_match"],
                "manual_selection_match": audit["manual_selection_match"],
                "source_checkpoint": projector_receipt,
                "prototype_checkpoint": prototype_receipt,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    values = list(judged.values())
    memory_pass = all(
        row["manual_prediction_match"] == thresholds["reference_prediction_match"]
        and row["manual_selection_match"] == thresholds["reference_prediction_match"]
        and row["selection_accuracy"] >= thresholds["selection_accuracy"]
        for row in values
    )
    behavior_pass = all(
        row["final_accuracy"] >= thresholds["final_accuracy"]
        and row["minimum_value_recall"] >= thresholds["minimum_value_recall"]
        and row["content_accuracy"] >= thresholds["content_readout_accuracy"]
        for row in values
    )
    if not memory_pass:
        verdict = "E2I_MEMORY_INTEGRATION_LOSS"
        reason = "the frozen KEY-1 reference passed, but the shared memory path did not preserve its retrieval"
    elif behavior_pass:
        verdict = "E2I_PATH_RECOVERED_NOT_UNIQUE"
        reason = "the optional shared-memory key transform preserved one-shot retrieval and behavior for both seeds"
    else:
        verdict = "E2I_BEHAVIOR_LOSS"
        reason = "shared-memory retrieval passed, but the stored state did not support balanced final behavior"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/episode2_results.json")
    parser.add_argument("--output", default="measurement/episode2_verdict.json")
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
