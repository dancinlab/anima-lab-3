#!/usr/bin/env python3
"""Fail-closed adjudicator for the registered SYNERGY-1 experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.synergy_registry import experiment, spec_sha256
except ModuleNotFoundError:
    from synergy_registry import experiment, spec_sha256


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _expected_audit(spec: dict, split: str) -> dict:
    n_a = len(spec["module_a_cues"])
    n_b = len(spec["module_b_cues"])
    repeats = spec[f"{split}_repeats_per_pair"]
    target_count = n_a * n_b * repeats // len(spec["actions"])
    per_module_target_count = n_b * repeats // len(spec["actions"])
    return {
        "examples": n_a * n_b * repeats,
        "pair_count": n_a * n_b,
        "examples_per_pair": repeats,
        "target_counts": {str(index): target_count for index in range(len(spec["actions"]))},
        "module_a_target_counts": {
            str(cue): {str(index): per_module_target_count for index in range(len(spec["actions"]))}
            for cue in range(n_a)
        },
        "module_b_target_counts": {
            str(cue): {str(index): per_module_target_count for index in range(len(spec["actions"]))}
            for cue in range(n_b)
        },
    }


def _judge_arm(metrics: dict, spec: dict, positive_control: bool) -> dict:
    bars = spec["thresholds"]
    conditions = metrics["conditions"]
    if set(conditions) != set(spec["interventions"]):
        raise ValueError("intervention set is incomplete")
    values = {name: float(row["accuracy"]) for name, row in conditions.items()}
    if not all(0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("accuracy is not a probability")
    normal_bar = bars["positive_control_accuracy"] if positive_control else bars["joint_accuracy"]
    single_best = max(values["module_a_only"], values["module_b_only"])
    integrated = (
        values["normal"] >= normal_bar
        and single_best <= bars["single_module_max_accuracy"]
        and values["partner_shuffle"] <= bars["partner_shuffle_max_accuracy"]
        and values["normal"] - single_best >= bars["minimum_joint_gain"]
        and abs(values["normal"] - values["recovered"]) <= bars["recovery_accuracy_tolerance"]
        and conditions["recovered"].get("logits_identical") is True
    )
    return {
        "integrated": integrated,
        "language_ok": float(metrics["neutral_kl_nats"]) <= bars["neutral_kl_nats"],
        "normal": values["normal"],
        "module_a_only": values["module_a_only"],
        "module_b_only": values["module_b_only"],
        "partner_shuffle": values["partner_shuffle"],
        "recovered": values["recovered"],
        "neutral_kl_nats": float(metrics["neutral_kl_nats"]),
    }


def adjudicate(payload: dict) -> dict:
    try:
        spec = experiment(payload.get("experiment"))
    except ValueError:
        return {"experiment": payload.get("experiment"), "verdict": "Y0_INVALID",
                "reason": "result names an unregistered experiment"}
    if payload.get("spec_sha256") != spec_sha256(spec) or payload.get("spec") != spec:
        return {"experiment": spec["experiment"], "verdict": "Y0_INVALID",
                "reason": "result spec does not match the registered SSOT"}
    if not _finite_tree(payload):
        return {"experiment": spec["experiment"], "verdict": "Y0_INVALID",
                "reason": "result contains a non-finite measurement"}
    expected_audit = {split: _expected_audit(spec, split) for split in ("train", "eval")}
    if payload.get("dataset_audit") != expected_audit:
        return {"experiment": spec["experiment"], "verdict": "Y0_INVALID",
                "reason": "split-cue dataset is not exactly balanced"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"experiment": spec["experiment"], "verdict": "Y0_INVALID",
                "reason": "registered seed pair is incomplete"}
    judged = {}
    try:
        for seed in spec["seeds"]:
            arms = rows[seed]["arms"]
            if set(arms) != set(spec["arms"]):
                raise ValueError(f"seed {seed} arm set is incomplete")
            judged[str(seed)] = {
                arm: _judge_arm(arms[arm], spec, positive_control=arm != "quantum_pair")
                for arm in spec["arms"]
            }
            checkpoints = rows[seed]["checkpoints"]
            if set(checkpoints) != set(spec["arms"]):
                raise ValueError(f"seed {seed} checkpoint set is incomplete")
            for arm, receipt in checkpoints.items():
                digest = receipt.get("sha256", "")
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError(f"seed {seed} {arm} checkpoint hash is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        return {"experiment": spec["experiment"], "verdict": "Y0_INVALID", "reason": str(exc)}

    validation_arms = spec.get("validation_arms", ["direct_memory", "gru"])
    controls = [judged[str(seed)][arm] for seed in spec["seeds"] for arm in validation_arms]
    quantum = [judged[str(seed)]["quantum_pair"] for seed in spec["seeds"]]
    if not all(row["integrated"] and row["language_ok"] for row in controls):
        verdict, reason = "Y0_INVALID", "a registered task-validation arm did not validate the task"
    elif not all(row["integrated"] for row in quantum):
        verdict, reason = "Y3_NOT_INTEGRATED", "QuantumC pair did not integrate both clues in both seeds"
    elif not all(row["language_ok"] for row in quantum):
        verdict, reason = "Y4_CONFOUNDED", "integration was accompanied by excessive neutral-language drift"
    else:
        margin = spec["thresholds"]["quantum_advantage_margin"]
        comparison_arms = spec.get("comparison_arms", ["direct_memory", "gru"])
        advantage = all(
            judged[str(seed)]["quantum_pair"]["normal"]
            > max(judged[str(seed)][arm]["normal"] for arm in comparison_arms) + margin
            for seed in spec["seeds"]
        )
        if advantage:
            verdict, reason = "Y2_QUANTUM_ADVANTAGE", "QuantumC integration exceeded both general controls"
        else:
            verdict, reason = "Y1_INTEGRATED_NOT_UNIQUE", "QuantumC integrated both clues but general controls were equivalent"
    return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "seeds": judged}


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
