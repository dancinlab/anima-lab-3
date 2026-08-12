#!/usr/bin/env python3
"""Fail-closed adjudicator for GATE-RUNTIME-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.runtime_memory_shadow_registry import (
        RUNTIME_MEMORY_SHADOW_SPEC,
        spec_sha256,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.runtime_memory_shadow_registry import (
        RUNTIME_MEMORY_SHADOW_SPEC,
        spec_sha256,
    )


def _invalid(reason: str, spec: dict) -> dict:
    return {
        "experiment": spec["experiment"],
        "verdict": "GR0_INVALID",
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
    }


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _rate(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def adjudicate(payload: dict, spec: dict = RUNTIME_MEMORY_SHADOW_SPEC) -> dict:
    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or spec.get("preregistration_commit") == "__PREREGISTRATION_COMMIT__"
        or not _finite(payload)
    ):
        return _invalid("registration or finite-value check failed", spec)
    runtime = payload.get("runtime", {})
    if any(runtime.get(name) != spec["runtime"][name] for name in ("torch", "transformers")):
        return _invalid("runtime mismatch", spec)
    checkpoint = payload.get("checkpoint", {})
    path = Path(checkpoint.get("path", ""))
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint.get("sha256")
        or checkpoint.get("format") != "semantic_dialogue_memory_gate_control_v1"
    ):
        return _invalid("runtime checkpoint receipt failed", spec)
    replicates = payload.get("replicates")
    if not isinstance(replicates, list) or [row.get("name") for row in replicates] != spec["replicates"]:
        return _invalid("replicate roster changed", spec)

    thresholds = spec["thresholds"]
    selection_valid = True
    summaries = {}
    for row in replicates:
        name = row["name"]
        expected_writes = spec["evaluation_episodes"] * spec["candidates_per_episode"]
        preservation = row.get("preservation", {})
        shadow = row.get("shadow_audit", {})
        metrics = row.get("selection", {})
        dataset = row.get("dataset_audit", {})
        mismatch_keys = (
            "answer_digest_mismatches",
            "primary_store_digest_mismatches",
            "primary_search_digest_mismatches",
        )
        if (
            dataset.get("episodes") != spec["evaluation_episodes"]
            or dataset.get("candidates") != expected_writes
            or dataset.get("unique_texts") != expected_writes
            or len(dataset.get("raw_transcript_sha256", "")) != 64
            or any(not _count(preservation.get(key)) for key in mismatch_keys)
            or any(len(preservation.get(key, "")) != 64 for key in (
                "answer_sha256", "primary_store_sha256", "primary_search_sha256",
            ))
            or any(not _count(shadow.get(key)) for key in (
                "write_events", "search_events", "missing_write_audits",
                "missing_search_audits", "raw_text_leaks",
            ))
        ):
            return _invalid(f"replicate {name} dataset or metric shape changed", spec)
        if (
            preservation.get("answer_digest_mismatches") > thresholds["maximum_answer_digest_mismatches"]
            or preservation.get("primary_store_digest_mismatches") > thresholds["maximum_primary_store_digest_mismatches"]
            or preservation.get("primary_search_digest_mismatches") > thresholds["maximum_primary_search_digest_mismatches"]
            or shadow.get("raw_text_leaks") > thresholds["maximum_raw_text_leaks"]
            or shadow.get("write_events") != expected_writes
            or shadow.get("search_events") != spec["evaluation_episodes"]
            or shadow.get("missing_write_audits") > thresholds["maximum_missing_write_audits"]
            or shadow.get("missing_search_audits") > thresholds["maximum_missing_search_audits"]
            or len(shadow.get("audit_sha256", "")) != 64
        ):
            return _invalid(f"replicate {name} preservation or audit failed", spec)
        fault_rows = row.get("faults", {})
        if set(fault_rows) != set(spec["faults"]):
            return _invalid(f"replicate {name} fault roster changed", spec)
        expected_faults = {
            "initialization": 1,
            "search_audit": spec["evaluation_episodes"],
            "write_audit": 1,
        }
        if any(
            set(fault_rows[fault]) != {*mismatch_keys, "caught_errors"}
            or any(not _count(fault_rows[fault].get(key)) for key in (*mismatch_keys, "caught_errors"))
            or fault_rows[fault]["caught_errors"] != expected_faults[fault]
            for fault in spec["faults"]
        ):
            return _invalid(f"replicate {name} fault audit changed", spec)
        if any(
            any(
                fault_rows[fault].get(key) != 0
                for key in (
                    "answer_digest_mismatches",
                    "primary_store_digest_mismatches",
                    "primary_search_digest_mismatches",
                )
            )
            for fault in spec["faults"]
        ):
            return {
                "experiment": spec["experiment"],
                "verdict": "GR3_RUNTIME_INTERFERENCE",
                "reason": f"replicate {name} shadow failure changed a primary result",
                "spec_sha256": spec_sha256(spec),
            }
        per_kind = metrics.get("per_kind_selection_rate", {})
        if (
            set(per_kind) != set(spec["fact_kinds"])
            or any(not _rate(value) for value in per_kind.values())
            or any(not _rate(metrics.get(key)) for key in (
                "important_selection_rate", "distractor_selection_rate", "selection_ratio",
            ))
            or len(metrics.get("selection_sha256", "")) != 64
        ):
            return _invalid(f"replicate {name} fact-kind metrics changed", spec)
        valid = (
            metrics.get("important_selection_rate", 0) >= thresholds["minimum_important_selection_rate"]
            and metrics.get("distractor_selection_rate", 1) <= thresholds["maximum_distractor_selection_rate"]
            and metrics.get("selection_ratio", 1) <= thresholds["maximum_selection_ratio"]
            and min(per_kind.values()) >= thresholds["minimum_per_kind_selection_rate"]
        )
        selection_valid = selection_valid and valid
        summaries[name] = {
            "important_selection_rate": metrics["important_selection_rate"],
            "distractor_selection_rate": metrics["distractor_selection_rate"],
            "selection_ratio": metrics["selection_ratio"],
            "minimum_kind_selection_rate": min(per_kind.values()),
        }
    return {
        "experiment": spec["experiment"],
        "verdict": "GR1_SHADOW_RUNTIME_SAFE" if selection_valid else "GR2_SELECTION_NOT_GENERAL",
        "reason": (
            "the opt-in shadow recorded validated choices without changing answers or primary memory"
            if selection_valid else
            "the shadow remained answer-inert but missed a registered selection threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "replicates": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("measurement/runtime_memory_shadow_verdict.json"))
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
