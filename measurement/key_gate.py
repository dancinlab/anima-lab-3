#!/usr/bin/env python3
"""Fail-closed adjudication for KEY-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

try:
    from measurement.episode_registry import EPISODE_SPEC, spec_sha256 as episode_spec_sha256
    from measurement.key_registry import KEY_SPEC, spec_sha256
except ModuleNotFoundError:
    from episode_registry import EPISODE_SPEC, spec_sha256 as episode_spec_sha256
    from key_registry import KEY_SPEC, spec_sha256


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


def _key_metric_shape(metrics: dict, classes: int) -> bool:
    return (
        len(metrics["confusion_matrix"]) == classes
        and all(len(row) == classes for row in metrics["confusion_matrix"])
        and len(metrics["per_key_recall"]) == classes
    )


def adjudicate(payload: dict, spec: dict = KEY_SPEC) -> dict:
    invalid = lambda reason: {
        "experiment": payload.get("experiment", spec["experiment"]),
        "verdict": "K0_INVALID",
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
        source = payload["source_episode"]
        results_path = Path(source["results"]["path"])
        verdict_path = Path(source["verdict"]["path"])
        for receipt, path in ((source["results"], results_path), (source["verdict"], verdict_path)):
            if not path.is_file() or _sha256_file(path) != receipt["sha256"]:
                return invalid("EPISODE-1 source file changed")
        source_results = json.loads(results_path.read_text())
        source_verdict = json.loads(verdict_path.read_text())
        expected_source_sha = episode_spec_sha256(EPISODE_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != EPISODE_SPEC
            or source_results.get("spec_sha256") != expected_source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != expected_source_sha
        ):
            return invalid("EPISODE-1 source identity or verdict changed")
        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        if set(source_rows) != set(spec["seeds"]):
            return invalid("EPISODE-1 source seeds changed")
        if payload["dataset_audit"] != source_results["dataset_audit"]:
            return invalid("calibration/evaluation dataset no longer matches EPISODE-1")
        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")
        thresholds = spec["thresholds"]
        judged = {}
        for seed in spec["seeds"]:
            row = rows[seed]
            source_row = source_rows[seed]
            if set(row["arms"]) != set(spec["arms"]):
                return invalid(f"seed {seed} arm roster changed")
            calibration = row["calibration_audit"]
            evaluation = row["eval_state_audit"]
            if (
                calibration["episodes"] != spec["calibration_episodes"]
                or calibration["states"] != spec["calibration_episodes"] * 3
                or calibration["unique_engine_seeds"] != spec["calibration_episodes"]
                or evaluation["episodes"] != spec["eval_episodes"]
                or evaluation["states"] != spec["eval_episodes"] * 3
                or evaluation["unique_engine_seeds"] != spec["eval_episodes"]
                or evaluation["calibration_engine_seed_overlap"] != 0
            ):
                return invalid(f"seed {seed} state streams are incomplete or overlapping")
            if len(calibration["engine_seed_sha256"]) != 64 or len(evaluation["engine_seed_sha256"]) != 64:
                return invalid(f"seed {seed} state seed receipt is malformed")
            for audit in (row["training_audit"], row["shuffled_training_audit"]):
                if (
                    audit["examples"] != spec["calibration_episodes"] * 3
                    or audit["steps"] != spec["train_steps"]
                    or len(audit["training_label_sha256"]) != 64
                ):
                    return invalid(f"seed {seed} training stream changed")
            if row["training_audit"]["shuffled"] or not row["shuffled_training_audit"]["shuffled"]:
                return invalid(f"seed {seed} shuffled-label identity changed")
            checkpoint_receipt = row["checkpoint"]
            checkpoint_path = Path(checkpoint_receipt["path"])
            if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != checkpoint_receipt["sha256"]:
                return invalid(f"seed {seed} projector checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("experiment") != spec["experiment"]
                or checkpoint.get("spec_sha256") != spec_sha256(spec)
                or checkpoint.get("seed") != seed
                or checkpoint.get("model_class") != spec["model_class"]
                or checkpoint.get("training_audit") != row["training_audit"]
                or checkpoint.get("shuffled_training_audit") != row["shuffled_training_audit"]
            ):
                return invalid(f"seed {seed} projector checkpoint identity changed")
            for state_name in ("projector", "shuffled_label_projector"):
                state = checkpoint.get(state_name, {})
                expected = {
                    "projection.weight": (spec["address_dim"], spec["input_dim"]),
                    "projection.bias": (spec["address_dim"],),
                    "prototypes": (spec["keys"], spec["address_dim"]),
                }
                if set(state) != set(expected) or any(
                    tuple(state[name].shape) != shape or not torch.isfinite(state[name]).all()
                    for name, shape in expected.items()
                ):
                    return invalid(f"seed {seed} {state_name} tensor changed")
            source_checkpoint = row["source_checkpoint"]
            if source_checkpoint != source["checkpoints"].get(str(seed)):
                return invalid(f"seed {seed} source checkpoint receipt changed")
            source_path = Path(source_checkpoint["path"])
            if not source_path.is_file() or _sha256_file(source_path) != source_checkpoint["sha256"]:
                return invalid(f"seed {seed} source prototype checkpoint changed")
            for arm in (
                "stabilized_memory_normal", "stabilized_memory_partner_swap",
                "stabilized_memory_recovered", "raw_quantum_memory", "sensory_memory",
                "keyed_attention", "no_memory",
            ):
                if not _metric_shape(row["arms"][arm], spec["values"]):
                    return invalid(f"seed {seed} {arm} metrics are incomplete")
            if not _key_metric_shape(row["key_classification"], spec["keys"]):
                return invalid(f"seed {seed} key classification metrics are incomplete")
            fake = row["arms"]["shuffled_label_projector"]
            if not _key_metric_shape(fake, spec["keys"]):
                return invalid(f"seed {seed} shuffled-label metrics are incomplete")
            if row["arms"]["raw_quantum_memory"] != source_row["arms"]["quantum_memory_normal"]:
                return invalid(f"seed {seed} raw QuantumC replay changed")
            if row["arms"]["sensory_memory"] != source_row["arms"]["sensory_memory_normal"]:
                return invalid(f"seed {seed} direct sensory replay changed")
            if row["arms"]["keyed_attention"] != source_row["arms"]["keyed_attention"]:
                return invalid(f"seed {seed} keyed-attention replay changed")
            if row["arms"]["no_memory"] != source_row["arms"]["no_memory"]:
                return invalid(f"seed {seed} no-memory replay changed")
            stable = row["arms"]["stabilized_memory_normal"]
            if (
                stable["retrieval_api_match"] != 1.0
                or stable["correct_content_accuracy"] < thresholds["content_readout_accuracy"]
                or row["arms"]["stabilized_memory_partner_swap"]["accuracy"]
                > thresholds["negative_control_max_accuracy"]
                or row["arms"]["stabilized_memory_recovered"]["prediction_match"]
                != thresholds["recovery_prediction_match"]
                or row["arms"]["sensory_memory"]["accuracy"]
                < thresholds["positive_control_accuracy"]
                or row["arms"]["keyed_attention"]["accuracy"]
                < thresholds["positive_control_accuracy"]
                or row["arms"]["no_memory"]["accuracy"]
                > thresholds["negative_control_max_accuracy"]
                or fake["accuracy"] > thresholds["negative_control_max_accuracy"]
            ):
                return invalid(f"seed {seed} registered control or intervention failed")
            judged[str(seed)] = {
                "key_classification_accuracy": row["key_classification"]["accuracy"],
                "selection_accuracy": stable["selection_accuracy"],
                "final_accuracy": stable["accuracy"],
                "minimum_value_recall": min(stable["per_value_recall"]),
                "content_accuracy": stable["correct_content_accuracy"],
                "partner_swap_accuracy": row["arms"]["stabilized_memory_partner_swap"]["accuracy"],
                "shuffled_key_accuracy": fake["accuracy"],
                "checkpoint": checkpoint_receipt,
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))
    values = list(judged.values())
    alignment_pass = all(
        row["key_classification_accuracy"] >= thresholds["key_classification_accuracy"]
        and row["selection_accuracy"] >= thresholds["selection_accuracy"]
        for row in values
    )
    final_pass = all(
        row["final_accuracy"] >= thresholds["final_accuracy"]
        and row["minimum_value_recall"] >= thresholds["minimum_value_recall"]
        for row in values
    )
    if not alignment_pass:
        verdict = "K2_KEY_ALIGNMENT_LOSS"
        reason = "the registered linear transform did not make held-out QuantumC keys stable enough"
    elif final_pass:
        verdict = "K1_STABLE_KEY_VALID_NOT_UNIQUE"
        reason = "the frozen linear address recovered one-shot retrieval, alongside standard memory controls"
    else:
        verdict = "K3_EPISODIC_PATH_NOT_RECOVERED"
        reason = "key alignment passed, but the registered one-shot memory path did not recover"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/key_results.json")
    parser.add_argument("--output", default="measurement/key_verdict.json")
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
