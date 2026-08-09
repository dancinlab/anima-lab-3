#!/usr/bin/env python3
"""Fail-closed adjudication for the registered STATE-2 bridge width sweep."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.bridge_capacity_registry import BRIDGE_CAPACITY_SPEC, spec_sha256
except ModuleNotFoundError:
    from bridge_capacity_registry import BRIDGE_CAPACITY_SPEC, spec_sha256


def _judge(metrics: dict, threshold: float, shuffled_max: float) -> dict:
    accuracy = float(metrics["accuracy"])
    shuffled = float(metrics["shuffled_label_accuracy"])
    if not (math.isfinite(accuracy) and math.isfinite(shuffled)):
        raise ValueError("probe metrics must be finite")
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= shuffled <= 1.0):
        raise ValueError("probe metrics must be probabilities")
    return {"pass": accuracy >= threshold and shuffled <= shuffled_max, **metrics}


def adjudicate(payload: dict) -> dict:
    spec = BRIDGE_CAPACITY_SPEC
    if payload.get("experiment") != spec["experiment"]:
        return {"verdict": "C0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "C0_INVALID", "reason": "result spec does not match the registered SSOT"}

    expected_widths = spec["bridge"]["hub_dims"]
    width_rows = {row.get("hub_dim"): row for row in payload.get("widths", [])}
    if set(width_rows) != set(expected_widths):
        return {"verdict": "C0_INVALID", "reason": "registered width set is incomplete"}

    expected_seeds = set(spec["seeds"])
    expected_delays = {str(delay) for delay in spec["delay_steps"]}
    expected_channels = set(spec["channels"])
    thresholds = spec["thresholds"]
    judged = {}
    try:
        for width in expected_widths:
            width_row = width_rows[width]
            if width_row.get("pooling") != spec["bridge"]["pooling"]:
                raise ValueError(f"width {width} used an unregistered pooling method")
            seed_rows = {row.get("seed"): row for row in width_row.get("seeds", [])}
            if set(seed_rows) != expected_seeds:
                raise ValueError(f"width {width} seed pair is incomplete")
            judged[str(width)] = {}
            for seed in sorted(seed_rows):
                delays = seed_rows[seed].get("delays", {})
                if set(delays) != expected_delays:
                    raise ValueError(f"width {width} seed {seed} delay set is incomplete")
                judged[str(width)][str(seed)] = {}
                for delay in spec["delay_steps"]:
                    row = delays[str(delay)]
                    if set(row) != expected_channels:
                        raise ValueError(
                            f"width {width} seed {seed} delay {delay} channel set is incomplete"
                        )
                    judged[str(width)][str(seed)][str(delay)] = {}
                    for channel, metrics in row.items():
                        threshold = (
                            thresholds["positive_control_accuracy"]
                            if channel == "sense_input"
                            else thresholds["signal_accuracy"]
                        )
                        judged[str(width)][str(seed)][str(delay)][channel] = _judge(
                            metrics, threshold, thresholds["shuffled_label_max_accuracy"]
                        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "C0_INVALID", "reason": f"invalid sweep result: {exc}"}

    def all_pass(width: int, channel: str) -> bool:
        return all(
            judged[str(width)][str(seed)][str(delay)][channel]["pass"]
            for seed in spec["seeds"]
            for delay in spec["delay_steps"]
        )

    controls = spec["selection"]["positive_controls"]
    if not all(all_pass(width, channel) for width in expected_widths for channel in controls):
        return {
            "experiment": spec["experiment"],
            "verdict": "C0_INVALID",
            "reason": "a registered positive control failed",
            "spec_sha256": spec_sha256(spec),
            "widths": judged,
        }

    stage_pass = {
        str(width): {
            stage: all_pass(width, stage) for stage in spec["selection"]["ordered_stages"]
        }
        for width in expected_widths
    }
    cells = [width for width in expected_widths if stage_pass[str(width)]["bridge_cells"]]
    pooled = [width for width in cells if stage_pass[str(width)]["bridge_pooled"]]
    full = [width for width in pooled if stage_pass[str(width)]["bridge_gate"]]

    if not cells:
        verdict = "C1_NO_WIDTH_RECOVERY"
        reason = "no registered width preserved the per-cell bridge code"
        selected = None
        next_action = "do not change runtime; test the bridge transform objective"
    elif not pooled:
        verdict = "C2_TRANSFORM_RECOVERED_POOLING_LOSS"
        reason = "widening recovered per-cell codes but mean pooling still lost information"
        selected = min(cells)
        next_action = "keep runtime unchanged; test registered learned pooling"
    elif not full:
        verdict = "C3_GATE_TRANSFORM_LOSS"
        reason = "information survived mean pooling but not the final gate"
        selected = min(pooled)
        next_action = "keep runtime unchanged; isolate the gate transform"
    else:
        verdict = "C4_FULL_PATH_RECOVERY"
        reason = "a registered width preserved information through the complete bridge path"
        selected = min(full)
        next_action = "upgrade the canonical bridge width and rerun the causal behavior test"

    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "selected_hub_dim": selected,
        "next_action": next_action,
        "spec_sha256": spec_sha256(spec),
        "stage_pass": stage_pass,
        "widths": judged,
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
