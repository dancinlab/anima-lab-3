#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-CONTROL-2."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from measurement.semantic_memory_write_gate import adjudicate as adjudicate_control1
    from measurement.semantic_memory_write_matched_registry import (
        MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.semantic_memory_write_gate import adjudicate as adjudicate_control1
    from measurement.semantic_memory_write_matched_registry import (
        MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
        spec_sha256,
    )


def adjudicate(
    payload: dict,
    spec: dict = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "GCM0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
    ):
        return invalid("experiment, registered spec, or digest changed")

    control1_spec = spec["control1_spec"]
    data_spec = control1_spec["gate1_spec"]
    rows = payload.get("seeds", [])
    if [row.get("seed") for row in rows] != data_spec["seeds"]:
        return invalid("registered seed roster changed")
    for row in rows:
        seed = row.get("seed")
        audit = row.get("matching_audit", {})
        if set(audit) != {
            "method", "semantic_counts", "matched_shuffled_counts",
            "matched_random_counts", "semantic_selection_sha256",
            "matched_shuffled_selection_sha256", "matched_random_selection_sha256",
            "fake_scores_sha256",
        } or audit.get("method") != spec["matching"]:
            return invalid(f"seed {seed} matching audit changed")
        counts = [
            audit.get("semantic_counts"),
            audit.get("matched_shuffled_counts"),
            audit.get("matched_random_counts"),
        ]
        if any(
            not isinstance(values, list)
            or len(values) != data_spec["evaluation_episodes"]
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= data_spec["candidates_per_episode"]
                for value in values
            )
            for values in counts
        ):
            return invalid(f"seed {seed} per-episode counts changed")
        if counts[0] != counts[1] or counts[0] != counts[2]:
            return invalid(f"seed {seed} per-episode storage counts do not match")
        if any(
            not isinstance(audit.get(name), str) or len(audit[name]) != 64
            for name in (
                "semantic_selection_sha256", "matched_shuffled_selection_sha256",
                "matched_random_selection_sha256", "fake_scores_sha256",
            )
        ):
            return invalid(f"seed {seed} matching digest changed")
        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        expected_total = sum(counts[0])
        if (
            arms["semantic_gate"].get("stored") != expected_total
            or arms["matched_shuffled_gate"].get("stored") != expected_total
            or arms["matched_random"].get("stored") != expected_total
        ):
            return invalid(f"seed {seed} stored totals do not match the episode audit")

    compatibility_payload = copy.deepcopy(payload)
    compatibility_payload["experiment"] = control1_spec["experiment"]
    compatibility_payload["spec"] = control1_spec
    compatibility_payload["spec_sha256"] = spec["control1_spec_sha256"]
    for row in compatibility_payload["seeds"]:
        row["arms"]["shuffled_gate"] = row["arms"].pop("matched_shuffled_gate")
    base_verdict = adjudicate_control1(compatibility_payload, control1_spec)
    if base_verdict.get("verdict") == "GC0_INVALID":
        return invalid("shared semantic positive-control validation failed: " + base_verdict["reason"])

    passed = base_verdict.get("verdict") == "GC1_SEMANTIC_CONTROL_VALID"
    return {
        "experiment": spec["experiment"],
        "verdict": (
            "GCM1_MATCHED_SEMANTIC_CONTROL_VALID"
            if passed else "GCM2_MATCHED_SEMANTIC_CONTROL_LOSS"
        ),
        "reason": (
            "the frozen semantic selector beat storage-matched fake controls"
            if passed else "the matched semantic selector missed a registered threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": base_verdict.get("seeds", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_memory_write_matched_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
