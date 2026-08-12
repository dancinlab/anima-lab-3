#!/usr/bin/env python3
"""GATE-RUNTIME-2: privacy-preserving field preflight and shadow review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from measurement.runtime_memory_field_registry import (
    RUNTIME_MEMORY_FIELD_SPEC,
    spec_sha256,
)


FORBIDDEN_RAW_KEYS = {"text", "query", "response", "content", "raw_text"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dialogue_sha256(role: str, text: str) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError("unsupported dialogue role")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty dialogue row")
    return hashlib.sha256(f"{role}\0{text}".encode()).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing memory timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _contains_forbidden_raw_key(value) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_RAW_KEYS & set(value)) or any(
            _contains_forbidden_raw_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_raw_key(item) for item in value)
    return False


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def scan_source(path: Path, spec: dict = RUNTIME_MEMORY_FIELD_SPEC) -> tuple[dict, list[dict]]:
    """Read the registered source without copying raw dialogue into the audit."""
    before = _file_sha256(path)
    with _read_only_connection(path) as connection:
        columns = {
            row[1] for row in connection.execute(
                f"PRAGMA table_info({spec['source_table']})"
            )
        }
        required = {"id", "role", "text", "timestamp"}
        if not required <= columns:
            raise ValueError("registered memory table shape changed")
        database_rows = connection.execute(
            f"SELECT id, role, text, timestamp FROM {spec['source_table']} ORDER BY id"
        ).fetchall()
    after = _file_sha256(path)

    role_counts = Counter()
    eligible: list[dict] = []
    timestamps: list[datetime] = []
    source_manifest = []
    for source_id, role, text, timestamp in database_rows:
        digest = _dialogue_sha256(role, text)
        role_counts[role] += 1
        source_manifest.append(f"{source_id}\0{role}\0{timestamp}\0{digest}")
        if role == spec["eligible_role"]:
            parsed = _parse_timestamp(timestamp)
            timestamps.append(parsed)
            eligible.append({
                "source_id": str(source_id),
                "role": role,
                "text": text,
                "text_sha256": digest,
                "timestamp": parsed,
            })

    timestamps.sort()
    gap = timedelta(minutes=spec["session_gap_minutes"])
    sessions = 0
    previous = None
    for current in timestamps:
        if previous is None or current - previous > gap:
            sessions += 1
        previous = current
    active_days = sorted({value.date().isoformat() for value in timestamps})
    audit = {
        "source_database_sha256_before": before,
        "source_database_sha256_after": after,
        "source_digest_mismatches": int(before != after),
        "source_read_only": True,
        "table": spec["source_table"],
        "columns_sha256": hashlib.sha256("\n".join(sorted(columns)).encode()).hexdigest(),
        "total_rows": len(database_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "eligible_user_turns": len(eligible),
        "unique_user_turns": len({row["text_sha256"] for row in eligible}),
        "active_days": len(active_days),
        "sessions": sessions,
        "first_observed_at": timestamps[0].isoformat() if timestamps else None,
        "last_observed_at": timestamps[-1].isoformat() if timestamps else None,
        "source_manifest_sha256": hashlib.sha256("\n".join(source_manifest).encode()).hexdigest(),
        "raw_text_leaks": 0,
    }
    if _contains_forbidden_raw_key(audit):
        raise ValueError("raw dialogue key entered source audit")
    return audit, eligible


def _field_data_sufficient(audit: dict, spec: dict) -> bool:
    thresholds = spec["thresholds"]
    return (
        audit["eligible_user_turns"] >= thresholds["minimum_user_turns"]
        and audit["unique_user_turns"] >= thresholds["minimum_unique_user_turns"]
        and audit["active_days"] >= thresholds["minimum_active_days"]
        and audit["sessions"] >= thresholds["minimum_sessions"]
    )


def _score_rows(rows: list[dict], checkpoint: Path, audit_path: Path,
                spec: dict) -> tuple[dict, dict[str, bool]]:
    import torch
    import transformers
    from memory_shadow import SemanticMemoryWriteScorer

    scorer = SemanticMemoryWriteScorer(
        checkpoint, expected_sha256=spec["checkpoint_sha256"]
    )
    decisions: dict[str, bool] = {}
    events = []
    batch_size = spec["encoder"]["batch_size"]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        scores = scorer.score_rows([
            {"role": row["role"], "text": row["text"]} for row in chunk
        ])
        for row, score in zip(chunk, scores):
            selected = score >= scorer.threshold
            decisions[row["text_sha256"]] = selected
            events.append({
                "format": spec["audit"]["format"],
                "event": "field_write",
                "source_id": row["source_id"],
                "role": row["role"],
                "text_sha256": row["text_sha256"],
                "score": score,
                "threshold": scorer.threshold,
                "selected": selected,
            })
    if _contains_forbidden_raw_key(events):
        raise ValueError("raw dialogue key entered field audit")
    _atomic_jsonl(audit_path, events)
    return {
        "status": "observed",
        "rows": len(events),
        "selected": sum(event["selected"] for event in events),
        "selection_rate": sum(event["selected"] for event in events) / len(events),
        "audit_path": str(audit_path),
        "audit_sha256": _file_sha256(audit_path),
        "checkpoint_sha256": scorer.checkpoint_sha256,
        "scorer_runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }, decisions


def _review(review_path: Path | None, decisions: dict[str, bool], spec: dict) -> dict:
    if review_path is None:
        return {"status": "not_provided", "reviewed_turns": 0, "label_counts": {}}
    payload = json.loads(review_path.read_text())
    if set(payload) != {"format", "labels"} or payload["format"] != "runtime_memory_field_review_v1":
        raise ValueError("field-review format changed")
    labels = payload["labels"]
    if not isinstance(labels, list):
        raise ValueError("field-review labels must be a list")
    allowed = set(spec["review_labels"])
    seen = set()
    counts = Counter()
    selected = Counter()
    for row in labels:
        if set(row) != {"text_sha256", "label"}:
            raise ValueError("field-review row shape changed")
        digest = row["text_sha256"]
        label = row["label"]
        if digest in seen or digest not in decisions or label not in allowed:
            raise ValueError("field-review row is duplicate or outside the observed source")
        seen.add(digest)
        counts[label] += 1
        selected[label] += int(decisions[digest])
    return {
        "status": "reviewed",
        "reviewed_turns": len(labels),
        "label_counts": dict(sorted(counts.items())),
        "selection_rates": {
            label: selected[label] / counts[label] if counts[label] else None
            for label in spec["review_labels"]
        },
        "review_sha256": _file_sha256(review_path),
        "review_path": str(review_path),
    }


def run(source: Path, checkpoint: Path, audit_path: Path,
        review_path: Path | None = None,
        spec: dict = RUNTIME_MEMORY_FIELD_SPEC) -> dict:
    source_audit, rows = scan_source(source, spec)
    checkpoint_receipt = {
        "path": str(checkpoint),
        "sha256": _file_sha256(checkpoint),
    }
    if _field_data_sufficient(source_audit, spec):
        observation, decisions = _score_rows(rows, checkpoint, audit_path, spec)
        review = _review(review_path, decisions, spec)
    else:
        observation = {
            "status": "skipped_insufficient_field_data",
            "rows": 0,
            "selected": 0,
        }
        review = {
            "status": "skipped_insufficient_field_data",
            "reviewed_turns": 0,
            "label_counts": {},
        }
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "runtime": {"python": platform.python_version()},
        "checkpoint": checkpoint_receipt,
        "source_audit": source_audit,
        "observation": observation,
        "review": review,
    }
    if _contains_forbidden_raw_key(payload):
        raise ValueError("raw dialogue key entered field result")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(RUNTIME_MEMORY_FIELD_SPEC["source_database"]))
    parser.add_argument("--checkpoint", type=Path, default=Path(RUNTIME_MEMORY_FIELD_SPEC["checkpoint"]))
    parser.add_argument("--audit", type=Path, default=Path("measurement/runtime-memory-field/field_audit.jsonl"))
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("measurement/runtime_memory_field_results.json"))
    args = parser.parse_args()
    _atomic_json(args.output, run(args.source, args.checkpoint, args.audit, args.review))


if __name__ == "__main__":
    main()
