#!/usr/bin/env python3
"""Fail-closed adjudicator for the registered WORKSPACE-1 information map."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.workspace_registry import WORKSPACE_INFORMATION_SPEC, spec_sha256
except ModuleNotFoundError:
    from workspace_registry import WORKSPACE_INFORMATION_SPEC, spec_sha256


def _judge(metrics: dict, positive: bool) -> dict:
    bars = WORKSPACE_INFORMATION_SPEC["thresholds"]
    accuracy = float(metrics["accuracy"])
    shuffled = float(metrics["shuffled_label_accuracy"])
    if not (math.isfinite(accuracy) and math.isfinite(shuffled)):
        raise ValueError("probe metrics must be finite")
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= shuffled <= 1.0):
        raise ValueError("probe metrics must be probabilities")
    threshold = bars["positive_control_accuracy"] if positive else bars["signal_accuracy"]
    return {
        "pass": accuracy >= threshold and shuffled <= bars["shuffled_label_max_accuracy"],
        **metrics,
    }


def _stage_passes(judged: dict, stage: str) -> bool:
    return all(
        judged[str(seed)][stage][label]["pass"]
        for seed in WORKSPACE_INFORMATION_SPEC["seeds"]
        for label in WORKSPACE_INFORMATION_SPEC["labels"]
    )


def adjudicate(payload: dict) -> dict:
    spec = WORKSPACE_INFORMATION_SPEC
    if payload.get("experiment") != spec["experiment"]:
        return {"verdict": "I0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec") != spec or payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "I0_INVALID", "reason": "result spec does not match the registered SSOT"}
    source = payload.get("source", {})
    if (source.get("experiment") != spec["source_experiment"]
            or source.get("verdict") != spec["source_verdict"]):
        return {"verdict": "I0_INVALID", "reason": "registered SYNERGY-1 source result is missing or changed"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"verdict": "I0_INVALID", "reason": "registered seed pair is incomplete"}
    judged = {}
    try:
        for seed in spec["seeds"]:
            row = rows[seed]
            if set(row["channels"]) != set(spec["channels"]):
                raise ValueError(f"seed {seed} channel set is incomplete")
            receipt = row["checkpoint"]
            digest = receipt["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"seed {seed} checkpoint hash is invalid")
            judged[str(seed)] = {}
            for channel in spec["channels"]:
                metrics = row["channels"][channel]
                if set(metrics) != set(spec["labels"]):
                    raise ValueError(f"seed {seed} {channel} label set is incomplete")
                judged[str(seed)][channel] = {
                    label: _judge(value, positive=channel == "raw_pair")
                    for label, value in metrics.items()
                }
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "I0_INVALID", "reason": str(exc)}

    if not _stage_passes(judged, "raw_pair"):
        verdict, reason = "I0_INVALID", "raw split-cue states did not validate both cue probes"
    elif not _stage_passes(judged, "bridge_cells"):
        verdict, reason = "I1_LOCAL_TRANSFORM_LOSS", "a cue was lost in the per-cell bridge transform"
    elif not _stage_passes(judged, "bridge_pooled"):
        verdict, reason = "I2_POOLING_LOSS", "both cues reached bridge cells but one was lost by cell pooling"
    elif not _stage_passes(judged, "bridge_gate") or not _stage_passes(judged, "normalized_code"):
        verdict, reason = "I3_GATE_LOSS", "both cues survived pooling but one was lost in the language gate code"
    else:
        verdict, reason = "I4_RELATION_COMPUTATION_LOSS", (
            "both cues survived the language gate while the registered joint action remained at chance"
        )
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
        "source": source,
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
