#!/usr/bin/env python3
"""Fail-closed adjudication for EPISODE-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC, spec_sha256 as source_spec_sha256
    from measurement.episode_registry import EPISODE_SPEC, spec_sha256
except ModuleNotFoundError:
    from episode_control_registry import ATTENTION_CONTROL_SPEC, spec_sha256 as source_spec_sha256
    from episode_registry import EPISODE_SPEC, spec_sha256


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


def adjudicate(payload: dict, spec: dict = EPISODE_SPEC) -> dict:
    invalid = lambda reason: {
        "experiment": payload.get("experiment", spec["experiment"]),
        "verdict": "E0_INVALID",
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
        source = payload["source_control"]
        source_results_path = Path(source["results"]["path"])
        source_verdict_path = Path(source["verdict"]["path"])
        for receipt, path in (
            (source["results"], source_results_path),
            (source["verdict"], source_verdict_path),
        ):
            if not path.is_file() or _sha256_file(path) != receipt["sha256"]:
                return invalid("CONTROL-3 source file changed")
        source_results = json.loads(source_results_path.read_text())
        source_verdict = json.loads(source_verdict_path.read_text())
        expected_source_sha = source_spec_sha256(ATTENTION_CONTROL_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != ATTENTION_CONTROL_SPEC
            or source_results.get("spec_sha256") != expected_source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != expected_source_sha
        ):
            return invalid("CONTROL-3 source identity or verdict changed")
        source_rows = {str(row["seed"]): row for row in source_results["seeds"]}
        if set(source_rows) != {str(seed) for seed in spec["seeds"]}:
            return invalid("CONTROL-3 source seeds changed")
        if any(
            source_rows[str(seed)]["checkpoint"] != source["checkpoints"].get(str(seed))
            for seed in spec["seeds"]
        ):
            return invalid("CONTROL-3 source checkpoint roster changed")
        audit = payload["dataset_audit"]
        source_audit = source_results["dataset_audit"]
        if audit != source_audit:
            return invalid("evaluation dataset no longer matches CONTROL-3")
        eval_audit = audit["splits"][spec["eval_split"]]
        expected_eval = spec["splits"][spec["eval_split"]]
        if eval_audit["episodes"] != expected_eval or eval_audit["unique_fingerprints"] != expected_eval:
            return invalid("evaluation episodes are missing or repeated")
        maximum_delta = spec["thresholds"]["maximum_balance_delta"]
        for field, categories in (
            ("target_counts", spec["values"]),
            ("query_key_counts", spec["keys"]),
            ("query_position_counts", spec["relations_per_episode"]),
        ):
            counts = eval_audit[field]
            if set(counts) != {str(index) for index in range(categories)}:
                return invalid(f"evaluation {field} categories changed")
            values = list(counts.values())
            if sum(values) != expected_eval or max(values) - min(values) > maximum_delta:
                return invalid(f"evaluation {field} is not balanced")
        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")
        judged = {}
        for seed in spec["seeds"]:
            row = rows[seed]
            state = row["state_audit"]
            expected_prototypes = spec["values"] * spec["prototype_repeats_per_value"]
            if (
                state["episodes"] != expected_eval
                or state["unique_episode_seeds"] != expected_eval
                or state["prototype"]["states"] != expected_prototypes
                or state["prototype"]["unique_seeds"] != expected_prototypes
                or state["prototype_episode_seed_overlap"] != 0
            ):
                return invalid(f"seed {seed} state stream is incomplete or overlapping")
            if len(state["episode_seed_sha256"]) != 64 or len(state["prototype"]["seed_sha256"]) != 64:
                return invalid(f"seed {seed} state seed receipt is malformed")
            if set(row["arms"]) != set(spec["arms"]):
                return invalid(f"seed {seed} arm roster changed")
            receipt = row["checkpoint"]
            checkpoint_path = Path(receipt["path"])
            if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != receipt["sha256"]:
                return invalid(f"seed {seed} prototype checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("experiment") != spec["experiment"]
                or checkpoint.get("spec_sha256") != spec_sha256(spec)
                or checkpoint.get("seed") != seed
                or checkpoint.get("prototype_audit") != state["prototype"]
            ):
                return invalid(f"seed {seed} prototype checkpoint identity changed")
            prototypes = checkpoint.get("prototypes", {})
            expected_shape = (spec["values"], spec["state_dim"])
            if (
                set(prototypes) != {"quantum", "sensory"}
                or any(tuple(value.shape) != expected_shape for value in prototypes.values())
                or any(not torch.isfinite(value).all() for value in prototypes.values())
            ):
                return invalid(f"seed {seed} prototype tensor changed")
            source_receipt = row["source_checkpoint"]
            if source_receipt != source["checkpoints"].get(str(seed)):
                return invalid(f"seed {seed} source checkpoint receipt changed")
            source_checkpoint_path = Path(source_receipt["path"])
            if (
                not source_checkpoint_path.is_file()
                or _sha256_file(source_checkpoint_path) != source_receipt["sha256"]
            ):
                return invalid(f"seed {seed} source attention checkpoint changed")
            source_checkpoint = torch.load(
                source_checkpoint_path, map_location="cpu", weights_only=True
            )
            if (
                source_checkpoint.get("experiment") != spec["source_experiment"]
                or source_checkpoint.get("spec_sha256") != source_spec_sha256(ATTENTION_CONTROL_SPEC)
                or source_checkpoint.get("seed") != seed
                or source_checkpoint.get("model_class") != ATTENTION_CONTROL_SPEC["model_class"]
            ):
                return invalid(f"seed {seed} source attention checkpoint identity changed")
            for arm in spec["arms"]:
                if not _metric_shape(row["arms"][arm], spec["values"]):
                    return invalid(f"seed {seed} {arm} metrics are incomplete")
            arms = row["arms"]
            thresholds = spec["thresholds"]
            if arms["keyed_attention"]["accuracy"] < thresholds["positive_control_accuracy"]:
                return invalid(f"seed {seed} keyed-attention positive control failed")
            sensory = arms["sensory_memory_normal"]
            if (
                sensory["accuracy"] < thresholds["positive_control_accuracy"]
                or sensory["selection_accuracy"] < thresholds["positive_control_accuracy"]
                or sensory["correct_content_accuracy"] < thresholds["content_readout_accuracy"]
                or sensory["retrieval_api_match"] != 1.0
            ):
                return invalid(f"seed {seed} direct sensory memory control failed")
            ceiling = thresholds["negative_control_max_accuracy"]
            if arms["sensory_memory_partner_swap"]["accuracy"] > ceiling:
                return invalid(f"seed {seed} direct sensory partner-swap control failed")
            if arms["no_memory"]["accuracy"] > ceiling:
                return invalid(f"seed {seed} no-memory control exceeded its ceiling")
            quantum = arms["quantum_memory_normal"]
            if quantum["retrieval_api_match"] != 1.0:
                return invalid(f"seed {seed} VectorMemory retrieval disagrees with cosine selection")
            if arms["quantum_memory_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]:
                return invalid(f"seed {seed} recovery prediction changed")
            judged[str(seed)] = {
                "quantum_accuracy": quantum["accuracy"],
                "selection_accuracy": quantum["selection_accuracy"],
                "content_accuracy": quantum["correct_content_accuracy"],
                "minimum_value_recall": min(quantum["per_value_recall"]),
                "partner_swap_accuracy": arms["quantum_memory_partner_swap"]["accuracy"],
                "arms": arms,
                "checkpoint": receipt,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))
    threshold = spec["thresholds"]
    rows = list(judged.values())
    selection_pass = all(row["selection_accuracy"] >= threshold["quantum_selection_accuracy"] for row in rows)
    content_pass = all(row["content_accuracy"] >= threshold["content_readout_accuracy"] for row in rows)
    final_pass = all(
        row["quantum_accuracy"] >= threshold["quantum_accuracy"]
        and row["minimum_value_recall"] >= threshold["quantum_min_value_recall"]
        and row["partner_swap_accuracy"] <= threshold["negative_control_max_accuracy"]
        for row in rows
    )
    if not selection_pass and content_pass:
        verdict = "E2_KEY_RETRIEVAL_LOSS"
        reason = "value states remained readable, but QuantumC keys did not select the matching episode entry"
    elif selection_pass and not content_pass:
        verdict = "E3_CONTENT_READOUT_LOSS"
        reason = "episode positions were selected, but stored QuantumC value states were not readable"
    elif selection_pass and content_pass and final_pass:
        verdict = "E1_STATE_MEMORY_VALID_NOT_UNIQUE"
        reason = "QuantumC states supported causal one-shot recall, alongside the standard memory controls"
    else:
        verdict = "E4_EPISODIC_PATH_LOSS"
        reason = "component checks did not establish a causal, value-balanced one-shot memory path"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    args = parser.parse_args()
    verdict = adjudicate(json.loads(Path(args.results).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
