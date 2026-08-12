#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-RUNTIME-3 collection readiness."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from measurement.runtime_memory_collection_registry import (
        RUNTIME_MEMORY_COLLECTION_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.runtime_memory_collection_registry import (
        RUNTIME_MEMORY_COLLECTION_SPEC,
        spec_sha256,
    )


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _invalid(reason: str, spec: dict) -> dict:
    return {
        "experiment": spec["experiment"],
        "verdict": "GR30_INVALID",
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }


def adjudicate(payload: dict, spec: dict = RUNTIME_MEMORY_COLLECTION_SPEC) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") == "__PREREGISTRATION_COMMIT__"
        or not _finite(payload)
    ):
        return _invalid("registration or finite-value check failed", spec)

    audit = payload.get("audit", {})
    counts = audit.get("counts", {})
    required_counts = (
        "total_rows", "eligible_user_turns", "unique_user_turns", "active_days", "sessions"
    )
    if (
        audit.get("format") != spec["audit"]["format"]
        or audit.get("source_read_only") is not True
        or audit.get("source_append_only") is not True
        or audit.get("baseline_prefix_sha256") != spec["baseline"]["source_manifest_sha256"]
        or audit.get("raw_text_leaks") != 0
        or any(type(counts.get(key)) is not int or counts[key] < 0 for key in required_counts)
        or counts.get("total_rows", -1) < spec["baseline"]["total_rows"]
        or counts.get("eligible_user_turns", -1) < spec["baseline"]["eligible_user_turns"]
    ):
        return _invalid("append-only, privacy, or count receipt failed", spec)

    thresholds = spec["thresholds"]
    ready = (
        counts["eligible_user_turns"] >= thresholds["minimum_user_turns"]
        and counts["unique_user_turns"] >= thresholds["minimum_unique_user_turns"]
        and counts["active_days"] >= thresholds["minimum_active_days"]
        and counts["sessions"] >= thresholds["minimum_sessions"]
    )
    expected_status = "ready_for_review" if ready else "collecting"
    if audit.get("collection_status") != expected_status:
        return _invalid("collection readiness status disagrees with registered thresholds", spec)
    return {
        "experiment": spec["experiment"],
        "verdict": "GR32_READY_FOR_REVIEW" if ready else "GR31_COLLECTING",
        "reason": (
            "the append-only real-dialogue sample reached every preregistered collection threshold"
            if ready else
            "the live shadow is safe but the real-dialogue sample has not reached every threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".local/gate-runtime3/verdict.json"))
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
