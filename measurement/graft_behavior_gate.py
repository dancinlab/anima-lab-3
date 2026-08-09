#!/usr/bin/env python3
"""Adjudicate the pre-registered GRAFT hidden-situation behavior experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from measurement.graft_behavior_registry import BEHAVIOR_SPEC, spec_sha256
except ModuleNotFoundError:
    from graft_behavior_registry import BEHAVIOR_SPEC, spec_sha256


def judge_arm(metrics: dict, positive_control: bool = False) -> dict:
    bars = BEHAVIOR_SPEC["thresholds"]
    normal_bar = bars["positive_control_accuracy"] if positive_control else bars["normal_accuracy"]
    normal = metrics["normal"]["accuracy"]
    off = metrics["off"]["accuracy"]
    shuffled = metrics["shuffle"]["accuracy"]
    noise = metrics["noise"]["accuracy"]
    recovered = metrics["recovered"]["accuracy"]
    content_ok = max(shuffled, noise) <= bars["content_control_max_accuracy"]
    causal = (
        normal >= normal_bar
        and normal - off >= bars["minimum_off_drop"]
        and content_ok
        and abs(normal - recovered) <= bars["recovery_accuracy_tolerance"]
        and metrics["recovered"].get("logits_identical") is True
    )
    language_ok = metrics["neutral_kl_nats"] <= bars["neutral_kl_nats"]
    return {"causal": causal, "language_ok": language_ok, "normal": normal, "off": off,
            "shuffle": shuffled, "noise": noise, "recovered": recovered,
            "neutral_kl_nats": metrics["neutral_kl_nats"]}


def adjudicate(payload: dict) -> dict:
    if payload.get("spec_sha256") != spec_sha256():
        return {"experiment": BEHAVIOR_SPEC["experiment"], "verdict": "B0_INVALID",
                "reason": "result spec does not match the registered SSOT"}
    expected = set(BEHAVIOR_SPEC["seeds"])
    rows = {row["seed"]: row for row in payload.get("seeds", [])}
    if set(rows) != expected:
        return {"experiment": BEHAVIOR_SPEC["experiment"], "verdict": "B0_INVALID",
                "reason": "registered seed pair is incomplete"}
    judged = {}
    for seed in sorted(rows):
        arms = rows[seed].get("arms", {})
        if set(arms) != {"consciousness", "memory"}:
            return {"experiment": BEHAVIOR_SPEC["experiment"], "verdict": "B0_INVALID",
                    "reason": f"seed {seed} arm set is incomplete"}
        judged[seed] = {
            "consciousness": judge_arm(arms["consciousness"]),
            "memory": judge_arm(arms["memory"], positive_control=True),
        }
    memory_valid = all(row["memory"]["causal"] and row["memory"]["language_ok"] for row in judged.values())
    consciousness_causal = all(row["consciousness"]["causal"] for row in judged.values())
    consciousness_language = all(row["consciousness"]["language_ok"] for row in judged.values())
    if not memory_valid:
        verdict, reason = "B0_INVALID", "direct-memory positive control did not validate the task"
    elif not consciousness_language:
        verdict, reason = "B4_CONFOUNDED", "action effect was accompanied by excessive neutral-language drift"
    elif not consciousness_causal:
        verdict, reason = "B3_NOT_CAUSAL", "QuantumC state did not causally control held-out actions in both seeds"
    else:
        margin = BEHAVIOR_SPEC["thresholds"]["memory_equivalence_margin"]
        advantage = all(
            row["consciousness"]["normal"] > row["memory"]["normal"] + margin
            for row in judged.values()
        )
        if advantage:
            verdict, reason = "B1_CAUSAL_ADVANTAGE", "QuantumC state causally controlled action and beat direct memory"
        else:
            verdict, reason = "B2_CAUSAL_NOT_UNIQUE", "QuantumC state causally controlled action but direct memory was equivalent"
    return {"experiment": BEHAVIOR_SPEC["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(), "seeds": judged}


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
