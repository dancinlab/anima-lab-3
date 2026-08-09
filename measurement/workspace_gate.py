#!/usr/bin/env python3
"""Fail-closed adjudicator for WORKSPACE-1 recurrent split-cue integration."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.synergy_gate import _expected_audit, _judge_arm
    from measurement.workspace_information_gate import adjudicate as adjudicate_information
    from measurement.workspace_registry import WORKSPACE_SPEC, spec_sha256
except ModuleNotFoundError:
    from synergy_gate import _expected_audit, _judge_arm
    from workspace_information_gate import adjudicate as adjudicate_information
    from workspace_registry import WORKSPACE_SPEC, spec_sha256


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def adjudicate(payload: dict) -> dict:
    spec = WORKSPACE_SPEC
    if payload.get("experiment") != spec["experiment"]:
        return {"verdict": "W0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec") != spec or payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "W0_INVALID", "reason": "result spec does not match the registered SSOT"}
    if not _finite_tree(payload):
        return {"verdict": "W0_INVALID", "reason": "result contains a non-finite measurement"}
    source = payload.get("source_map", {})
    source_results = source.get("results")
    source_verdict = source.get("verdict")
    if not isinstance(source_results, dict) or not isinstance(source_verdict, dict):
        return {"verdict": "W0_INVALID", "reason": "registered information map is missing"}
    reproduced = adjudicate_information(source_results)
    if reproduced != source_verdict:
        return {"verdict": "W0_INVALID", "reason": "information-map verdict does not reproduce"}
    if (source_verdict.get("experiment") != spec["source_map_experiment"]
            or source_verdict.get("verdict") not in spec["source_map_verdicts"]):
        return {"verdict": "W0_INVALID", "reason": "information map does not authorize the registered repair"}
    if payload.get("dataset_audit") != {
        split: _expected_audit(spec, split) for split in ("train", "eval")
    }:
        return {"verdict": "W0_INVALID", "reason": "split-cue dataset is not exactly balanced"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"verdict": "W0_INVALID", "reason": "registered seed pair is incomplete"}

    judged = {}
    try:
        for seed in spec["seeds"]:
            arms = rows[seed]["arms"]
            if set(arms) != set(spec["arms"]):
                raise ValueError(f"seed {seed} arm set is incomplete")
            judged[str(seed)] = {
                arm: _judge_arm(arms[arm], spec, positive_control=arm in spec["validation_arms"])
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
        return {"verdict": "W0_INVALID", "reason": str(exc)}

    controls = [judged[str(seed)]["gru"] for seed in spec["seeds"]]
    baseline = [judged[str(seed)]["quantum_single_pass"] for seed in spec["seeds"]]
    if not all(row["integrated"] and row["language_ok"] for row in controls):
        verdict, reason = "W0_INVALID", "GRU task-validation arm did not validate the task"
        selected = None
    elif spec["selection"]["require_single_pass_failure"] and all(
        row["integrated"] for row in baseline
    ):
        verdict, reason = "W0_INVALID", "single-pass baseline no longer reproduces the registered failure"
        selected = None
    else:
        selected = next((
            rounds for rounds in spec["workspace_rounds"]
            if all(judged[str(seed)][f"quantum_workspace_{rounds}"]["integrated"]
                   for seed in spec["seeds"])
        ), None)
        if selected is None:
            verdict, reason = "W3_NOT_INTEGRATED", (
                "no registered recurrence count integrated both clues in both seeds"
            )
        else:
            quantum = [judged[str(seed)][f"quantum_workspace_{selected}"] for seed in spec["seeds"]]
            if not all(row["language_ok"] for row in quantum):
                verdict, reason = "W4_CONFOUNDED", (
                    "integration was accompanied by excessive neutral-language drift"
                )
            else:
                margin = spec["thresholds"]["quantum_advantage_margin"]
                advantage = all(
                    judged[str(seed)][f"quantum_workspace_{selected}"]["normal"]
                    > max(
                        judged[str(seed)][f"memory_workspace_{selected}"]["normal"],
                        judged[str(seed)]["gru"]["normal"],
                    ) + margin
                    for seed in spec["seeds"]
                )
                if advantage:
                    verdict, reason = "W2_QUANTUM_ADVANTAGE", (
                        "QuantumC recurrent workspace exceeded both general controls"
                    )
                else:
                    verdict, reason = "W1_INTEGRATED_NOT_UNIQUE", (
                        "QuantumC recurrent workspace integrated both clues without a general-control advantage"
                    )
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "selected_workspace_rounds": selected,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
        "source_map_verdict": source_verdict.get("verdict"),
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
