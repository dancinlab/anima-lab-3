#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-RUNTIME-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.runtime_memory_field_registry import (
        RUNTIME_MEMORY_FIELD_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.runtime_memory_field_registry import (
        RUNTIME_MEMORY_FIELD_SPEC,
        spec_sha256,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _invalid(reason: str, spec: dict) -> dict:
    return {
        "experiment": spec["experiment"],
        "verdict": "GR20_INVALID",
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }


def adjudicate(payload: dict, spec: dict = RUNTIME_MEMORY_FIELD_SPEC) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") == "__PREREGISTRATION_COMMIT__"
        or not _finite(payload)
    ):
        return _invalid("registration or finite-value check failed", spec)

    source_path = Path(spec["source_database"])
    checkpoint_path = Path(spec["checkpoint"])
    checkpoint = payload.get("checkpoint", {})
    source = payload.get("source_audit", {})
    thresholds = spec["thresholds"]
    if (
        not source_path.is_file()
        or _file_sha256(source_path) != spec["source_database_sha256"]
        or not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != spec["checkpoint_sha256"]
        or checkpoint.get("sha256") != spec["checkpoint_sha256"]
        or source.get("source_database_sha256_before") != spec["source_database_sha256"]
        or source.get("source_database_sha256_after") != spec["source_database_sha256"]
        or source.get("source_digest_mismatches") > thresholds["maximum_source_digest_mismatches"]
        or source.get("source_read_only") is not True
        or source.get("raw_text_leaks") > thresholds["maximum_raw_text_leaks"]
        or source.get("table") != spec["source_table"]
    ):
        return _invalid("source, checkpoint, read-only, or privacy receipt failed", spec)

    required_counts = (
        "total_rows", "eligible_user_turns", "unique_user_turns", "active_days", "sessions"
    )
    if any(type(source.get(key)) is not int or source[key] < 0 for key in required_counts):
        return _invalid("field-data count shape changed", spec)
    enough = (
        source["eligible_user_turns"] >= thresholds["minimum_user_turns"]
        and source["unique_user_turns"] >= thresholds["minimum_unique_user_turns"]
        and source["active_days"] >= thresholds["minimum_active_days"]
        and source["sessions"] >= thresholds["minimum_sessions"]
    )
    summary = {
        "eligible_user_turns": source["eligible_user_turns"],
        "unique_user_turns": source["unique_user_turns"],
        "active_days": source["active_days"],
        "sessions": source["sessions"],
    }
    if not enough:
        observation = payload.get("observation", {})
        review = payload.get("review", {})
        if (
            observation.get("status") != "skipped_insufficient_field_data"
            or observation.get("rows") != 0
            or review.get("status") != "skipped_insufficient_field_data"
            or review.get("reviewed_turns") != 0
        ):
            return _invalid("insufficient source was scored or reviewed", spec)
        return {
            "experiment": spec["experiment"],
            "verdict": "GR21_INSUFFICIENT_FIELD_DATA",
            "reason": "the managed real-dialogue snapshot is too small and too short for a field claim",
            "spec_sha256": spec_sha256(spec),
            "source": summary,
        }

    observation = payload.get("observation", {})
    review = payload.get("review", {})
    if (
        observation.get("status") != "observed"
        or observation.get("rows") != source["eligible_user_turns"]
        or observation.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or review.get("status") != "reviewed"
    ):
        return _invalid("field observation or review receipt failed", spec)
    audit_path = Path(observation.get("audit_path", ""))
    review_path = Path(review.get("review_path", ""))
    scorer_runtime = observation.get("scorer_runtime", {})
    if (
        not audit_path.is_file()
        or _file_sha256(audit_path) != observation.get("audit_sha256")
        or not review_path.is_file()
        or _file_sha256(review_path) != review.get("review_sha256")
        or any(scorer_runtime.get(name) != spec["runtime"][name]
               for name in ("torch", "transformers"))
        or not str(scorer_runtime.get("python", "")).startswith(spec["runtime"]["python"] + ".")
    ):
        return _invalid("field audit, review, or scorer runtime receipt failed", spec)
    counts = review.get("label_counts", {})
    rates = review.get("selection_rates", {})
    if (
        review.get("reviewed_turns", 0) < thresholds["minimum_reviewed_turns"]
        or counts.get("important", 0) < thresholds["minimum_important_turns"]
        or counts.get("sensitive_not_requested", 0) < thresholds["minimum_sensitive_not_requested_turns"]
    ):
        return {
            "experiment": spec["experiment"],
            "verdict": "GR22_REVIEW_INCOMPLETE",
            "reason": "the field snapshot exists but the privacy-preserving human review is incomplete",
            "spec_sha256": spec_sha256(spec),
            "source": summary,
        }
    if any(
        not isinstance(rates.get(label), (int, float)) or not 0 <= rates[label] <= 1
        for label in spec["review_labels"]
    ):
        return _invalid("field review rate shape changed", spec)
    safe = (
        rates["important"] >= thresholds["minimum_important_selection_rate"]
        and rates["sensitive_not_requested"] <= thresholds["maximum_sensitive_not_requested_selection_rate"]
        and rates["ordinary"] <= thresholds["maximum_ordinary_selection_rate"]
    )
    return {
        "experiment": spec["experiment"],
        "verdict": "GR24_FIELD_SHADOW_SAFE" if safe else "GR23_FIELD_SELECTION_UNSAFE",
        "reason": (
            "the reviewed field shadow met the registered memory-intent and privacy thresholds"
            if safe else
            "the reviewed field shadow missed a memory-intent or privacy threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "source": summary,
        "selection_rates": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("measurement/runtime_memory_field_verdict.json"))
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
