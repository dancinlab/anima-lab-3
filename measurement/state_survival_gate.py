#!/usr/bin/env python3
"""Fail-closed adjudication for the pre-registered STATE-1 map."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.state_survival_registry import STATE_SURVIVAL_SPEC, spec_sha256
except ModuleNotFoundError:
    from state_survival_registry import STATE_SURVIVAL_SPEC, spec_sha256


def _status(metrics: dict, positive_control: bool = False) -> dict:
    bars = STATE_SURVIVAL_SPEC["thresholds"]
    normal_bar = bars["positive_control_accuracy"] if positive_control else bars["signal_accuracy"]
    accuracy = float(metrics["accuracy"])
    shuffled = float(metrics["shuffled_label_accuracy"])
    if not (math.isfinite(accuracy) and math.isfinite(shuffled)):
        raise ValueError("probe metrics must be finite")
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= shuffled <= 1.0):
        raise ValueError("probe metrics must be probabilities")
    passed = (
        accuracy >= normal_bar
        and shuffled <= bars["shuffled_label_max_accuracy"]
    )
    return {"pass": passed, **metrics}


def _all_pass(judged: dict, channel: str, delays: list[int]) -> bool:
    return all(judged[seed][str(delay)][channel]["pass"] for seed in judged for delay in delays)


def _at_delay(judged: dict, channel: str, delay: int) -> bool:
    return all(judged[seed][str(delay)][channel]["pass"] for seed in judged)


def adjudicate(payload: dict) -> dict:
    spec = STATE_SURVIVAL_SPEC
    if payload.get("experiment") != spec["experiment"]:
        return {"verdict": "S0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "S0_INVALID", "reason": "result spec does not match the registered SSOT"}
    expected_seeds = set(spec["seeds"])
    seed_rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(seed_rows) != expected_seeds:
        return {"verdict": "S0_INVALID", "reason": "registered seed pair is incomplete"}

    delays = spec["delay_steps"]
    channels = set(spec["channels"])
    judged = {}
    for seed in sorted(seed_rows):
        rows = seed_rows[seed].get("delays", {})
        if set(rows) != {str(delay) for delay in delays}:
            return {"verdict": "S0_INVALID", "reason": f"seed {seed} delay set is incomplete"}
        seed_key = str(seed)
        judged[seed_key] = {}
        for delay in delays:
            metrics = rows[str(delay)]
            if set(metrics) != channels:
                return {"verdict": "S0_INVALID", "reason": f"seed {seed} delay {delay} channel set is incomplete"}
            try:
                judged[seed_key][str(delay)] = {
                    channel: _status(value, positive_control=channel == "sense_input")
                    for channel, value in metrics.items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                return {"verdict": "S0_INVALID", "reason": f"invalid probe metrics: {exc}"}

    behavior = payload.get("downstream_behavior", {})
    expected_behavior = spec["downstream_behavior"]["required_experiment"]
    if behavior.get("experiment") != expected_behavior or behavior.get("verdict") != "B3_NOT_CAUSAL":
        return {"verdict": "S0_INVALID", "reason": "registered downstream GRAFT result is missing or changed"}

    all_delays = delays
    delayed = [delay for delay in delays if delay > 0]
    if not _all_pass(judged, "sense_input", all_delays):
        verdict, reason = "S0_INVALID", "direct sense-input positive control did not validate the probe"
    elif not _at_delay(judged, "phase", 0):
        verdict, reason = "S1_SENSORY_ENCODING_LOSS", "situation information was absent immediately after sensing"
    elif not _all_pass(judged, "phase", delayed):
        verdict, reason = "S2_TEMPORAL_RETENTION_LOSS", "phase information survived sensing but failed during delay"
    elif not _all_pass(judged, "full_state", all_delays):
        verdict, reason = "S3_STATE_CHANNEL_LOSS", "phase survived but the combined runtime state was not stable"
    elif not _all_pass(judged, "bridge_cells", all_delays):
        verdict, reason = "S4_BRIDGE_TRANSFORM_LOSS", "information was lost before cell pooling in ThalamicBridge"
    elif not _all_pass(judged, "bridge_pooled", all_delays):
        verdict, reason = "S5_CELL_POOLING_LOSS", "information survived per-cell bridge codes but not cell averaging"
    elif not _all_pass(judged, "bridge_gate", all_delays):
        verdict, reason = "S6_GATE_TRANSFORM_LOSS", "information survived pooling but not the final bridge gate"
    else:
        verdict, reason = "S7_BEHAVIOR_GROUNDING_LOSS", (
            "information reached the language gate, while the registered GRAFT action test remained non-causal"
        )
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
        "downstream_behavior": behavior,
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
