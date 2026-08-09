#!/usr/bin/env python3
"""Fail-closed adjudicator for the registered META-1 experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.metacognition_registry import experiment, spec_sha256
except ModuleNotFoundError:
    from metacognition_registry import experiment, spec_sha256


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _reader_pass(metrics: dict, bars: dict) -> bool:
    return (
        metrics["auroc"] >= bars["auroc"]
        and metrics["brier"] <= bars["brier"]
        and metrics["ece"] <= bars["ece"]
        and metrics["selective_accuracy_gap"] >= bars["selective_accuracy_gap"]
    )


def _judge_arm(row: dict, bars: dict, levels: list[float]) -> dict:
    normal = row["confidence"]["normal"]
    shuffled = row["confidence"]["shuffle"]
    output_only = row["output_only"]
    action = row["action"]
    expected_levels = {str(level) for level in levels}
    difficulty_valid = (
        set(action["by_noise_level"]) == expected_levels
        and action["by_noise_level"][str(levels[0])]["accuracy"] >= bars["clean_action_accuracy"]
        and action["by_noise_level"][str(levels[-1])]["accuracy"] <= bars["hard_action_accuracy_max"]
        and action["correct_examples"] >= bars["minimum_correct_examples"]
        and action["incorrect_examples"] >= bars["minimum_incorrect_examples"]
    )
    reader_pass = _reader_pass(normal, bars)
    causal = (
        normal["auroc"] - shuffled["auroc"] >= bars["shuffle_auroc_drop"]
        and shuffled["brier"] - normal["brier"] >= bars["shuffle_brier_increase"]
        and row["intervention_actions_identical"] is True
        and row["recovery_confidence_identical"] is True
    )
    output_advantage = (
        normal["auroc"] - output_only["auroc"] >= bars["output_auroc_advantage"]
        and output_only["brier"] - normal["brier"] >= bars["output_brier_advantage"]
    )
    return {
        "difficulty_valid": difficulty_valid,
        "reader_pass": reader_pass,
        "causal": causal,
        "output_advantage": output_advantage,
        "normal": normal,
        "shuffle": shuffled,
        "output_only": output_only,
        "action": action,
    }


def adjudicate(payload: dict) -> dict:
    try:
        spec = experiment(payload.get("experiment"))
    except ValueError:
        return {"experiment": payload.get("experiment"), "verdict": "M0_INVALID",
                "reason": "result names an unregistered experiment"}
    if payload.get("spec_sha256") != spec_sha256(spec) or payload.get("spec") != spec:
        return {"experiment": spec["experiment"], "verdict": "M0_INVALID",
                "reason": "result spec does not match the registered SSOT"}
    if not _finite_tree(payload):
        return {"experiment": spec["experiment"], "verdict": "M0_INVALID",
                "reason": "result contains a non-finite measurement"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"experiment": spec["experiment"], "verdict": "M0_INVALID",
                "reason": "registered seed pair is incomplete"}
    judged = {}
    expected_hashes = spec["archive"]["checkpoint_sha256"]
    try:
        for seed in spec["seeds"]:
            arms = rows[seed]["arms"]
            if set(arms) != set(spec["arms"]):
                raise KeyError(f"seed {seed} arm set is incomplete")
            seed_key = str(seed)
            judged[seed_key] = {}
            for arm in spec["arms"]:
                if arms[arm]["source_checkpoint_sha256"] != expected_hashes[str(seed)][arm]:
                    raise KeyError(f"seed {seed} {arm} source checkpoint mismatch")
                judged[seed_key][arm] = _judge_arm(
                    arms[arm], spec["thresholds"], spec["readout_noise_levels"]
                )
    except (KeyError, TypeError) as exc:
        return {"experiment": spec["experiment"], "verdict": "M0_INVALID", "reason": str(exc)}

    all_rows = [judged[str(seed)][arm] for seed in spec["seeds"] for arm in spec["arms"]]
    consciousness = [judged[str(seed)]["consciousness"] for seed in spec["seeds"]]
    if not all(row["difficulty_valid"] for row in all_rows):
        verdict, reason = "M0_INVALID", "noise range did not produce a valid easy-to-hard action task"
    elif not all(row["reader_pass"] and row["causal"] for row in judged_seed_memory(judged, spec)):
        verdict, reason = "M0_INVALID", "direct-memory monitor did not validate the calibration task"
    elif not all(row["reader_pass"] and row["causal"] for row in consciousness):
        verdict, reason = "M3_NOT_CALIBRATED", "QuantumC readout did not causally calibrate correctness in both seeds"
    elif all(row["output_advantage"] for row in consciousness):
        verdict, reason = "M1_STATE_MONITORING_ADVANTAGE", "QuantumC readout calibrated correctness beyond output logits"
    else:
        verdict, reason = "M2_CALIBRATED_NOT_UNIQUE", "QuantumC readout calibrated correctness but output logits were equivalent"
    return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "seeds": judged}


def judged_seed_memory(judged: dict, spec: dict) -> list[dict]:
    return [judged[str(seed)]["memory"] for seed in spec["seeds"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    Path(args.output).write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
