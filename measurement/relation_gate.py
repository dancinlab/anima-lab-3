#!/usr/bin/env python3
"""Fail-closed adjudicator for RELATION-1 role-content binding."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.relation_registry import RELATION_SPEC, spec_sha256
    from measurement.synergy_gate import _expected_audit, _judge_arm
    from measurement.workspace_gate import adjudicate as adjudicate_workspace
except ModuleNotFoundError:
    from relation_registry import RELATION_SPEC, spec_sha256
    from synergy_gate import _expected_audit, _judge_arm
    from workspace_gate import adjudicate as adjudicate_workspace


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def adjudicate(payload: dict) -> dict:
    spec = RELATION_SPEC
    if payload.get("experiment") != spec["experiment"]:
        return {"verdict": "R0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec") != spec or payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "R0_INVALID", "reason": "result spec does not match the registered SSOT"}
    if not _finite_tree(payload):
        return {"verdict": "R0_INVALID", "reason": "result contains a non-finite measurement"}
    source = payload.get("source", {})
    source_results = source.get("results")
    source_verdict = source.get("verdict")
    if not isinstance(source_results, dict) or not isinstance(source_verdict, dict):
        return {"verdict": "R0_INVALID", "reason": "registered WORKSPACE-1 source is missing"}
    if adjudicate_workspace(source_results) != source_verdict:
        return {"verdict": "R0_INVALID", "reason": "WORKSPACE-1 source verdict does not reproduce"}
    if (source_verdict.get("experiment") != spec["source_experiment"]
            or source_verdict.get("verdict") != spec["source_verdict"]):
        return {"verdict": "R0_INVALID", "reason": "registered WORKSPACE-1 failure changed"}
    expected = {split: _expected_audit(spec, split) for split in ("train", "eval")}
    if payload.get("dataset_audit") != expected:
        return {"verdict": "R0_INVALID", "reason": "role-cue dataset is not exactly balanced"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"verdict": "R0_INVALID", "reason": "registered seed pair is incomplete"}

    judged = {}
    try:
        for seed in spec["seeds"]:
            arms = rows[seed]["arms"]
            if set(arms) != set(spec["arms"]):
                raise ValueError(f"seed {seed} arm set is incomplete")
            judged[str(seed)] = {
                arm: _judge_arm(arms[arm], spec, arm in spec["validation_arms"])
                for arm in spec["arms"]
            }
            checkpoints = rows[seed]["checkpoints"]
            if set(checkpoints) != set(spec["arms"]):
                raise ValueError(f"seed {seed} checkpoint set is incomplete")
            for arm, receipt in checkpoints.items():
                digest = receipt.get("sha256", "")
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"seed {seed} {arm} checkpoint hash is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "R0_INVALID", "reason": str(exc)}

    controls = [judged[str(seed)]["gru"] for seed in spec["seeds"]]
    baseline = [judged[str(seed)][spec["baseline_arm"]] for seed in spec["seeds"]]
    relation = [judged[str(seed)][spec["relation_arm"]] for seed in spec["seeds"]]
    if not all(row["integrated"] and row["language_ok"] for row in controls):
        verdict, reason = "R0_INVALID", "GRU task-validation arm did not validate the task"
    elif all(row["integrated"] and row["language_ok"] for row in baseline):
        verdict, reason = "R5_EXISTING_WORKSPACE_SUFFICIENT", (
            "the registered existing workspace integrated the role-sensitive task without binding"
        )
    elif not all(row["integrated"] for row in relation):
        verdict, reason = "R3_NOT_BOUND", (
            "QuantumC role-content binding did not pass every intervention in both seeds"
        )
    elif not all(row["language_ok"] for row in relation):
        verdict, reason = "R4_CONFOUNDED", (
            "role binding passed behavior with excessive neutral-language drift"
        )
    else:
        margin = spec["thresholds"]["quantum_advantage_margin"]
        advantage = all(
            judged[str(seed)][spec["relation_arm"]]["normal"]
            > max(judged[str(seed)][arm]["normal"] for arm in spec["comparison_arms"]) + margin
            for seed in spec["seeds"]
        )
        if advantage:
            verdict, reason = "R2_QUANTUM_ADVANTAGE", (
                "QuantumC role binding exceeded both registered general controls"
            )
        else:
            verdict, reason = "R1_BOUND_NOT_UNIQUE", (
                "QuantumC role binding integrated both roles without a general-control advantage"
            )
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
