#!/usr/bin/env python3
"""GATE-RUNTIME-3: append-only, raw-text-free live dialogue collection audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from measurement.runtime_memory_collection_registry import (
    RUNTIME_MEMORY_COLLECTION_SPEC,
    spec_sha256,
)


FORBIDDEN_RAW_KEYS = {"text", "query", "response", "content", "raw_text"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


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


def _manifest_sha256(rows: list[tuple]) -> str:
    manifest = [
        f"{source_id}\0{role}\0{timestamp}\0{_dialogue_sha256(role, text)}"
        for source_id, role, text, timestamp in rows
    ]
    return hashlib.sha256("\n".join(manifest).encode()).hexdigest()


def _contains_forbidden_raw_key(value) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_RAW_KEYS & set(value)) or any(
            _contains_forbidden_raw_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_raw_key(item) for item in value)
    return False


def _read_snapshot(path: Path, table: str) -> tuple[list[str], list[tuple]]:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        required = {"id", "role", "text", "timestamp"}
        if not required <= set(columns):
            raise ValueError("registered memory table shape changed")
        rows = connection.execute(
            f"SELECT id, role, text, timestamp FROM {table} ORDER BY id"
        ).fetchall()
    return columns, rows


def scan_collection_source(
    path: Path,
    spec: dict = RUNTIME_MEMORY_COLLECTION_SPEC,
) -> dict:
    columns, rows = _read_snapshot(path, spec["source_table"])
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("memory IDs are not unique and ordered")

    baseline = spec["baseline"]
    baseline_rows = rows[:baseline["total_rows"]]
    prefix_sha256 = _manifest_sha256(baseline_rows)
    append_only = (
        len(rows) >= baseline["total_rows"]
        and prefix_sha256 == baseline["source_manifest_sha256"]
    )

    role_counts = Counter()
    user_digests = []
    timestamps = []
    for _, role, text, timestamp in rows:
        digest = _dialogue_sha256(role, text)
        role_counts[role] += 1
        if role == spec["eligible_role"]:
            user_digests.append(digest)
            timestamps.append(_parse_timestamp(timestamp))

    timestamps.sort()
    gap = timedelta(minutes=spec["session_gap_minutes"])
    sessions = 0
    previous = None
    for current in timestamps:
        if previous is None or current - previous > gap:
            sessions += 1
        previous = current
    active_days = sorted({value.date().isoformat() for value in timestamps})
    thresholds = spec["thresholds"]
    counts = {
        "total_rows": len(rows),
        "eligible_user_turns": len(user_digests),
        "unique_user_turns": len(set(user_digests)),
        "active_days": len(active_days),
        "sessions": sessions,
    }
    ready = (
        counts["eligible_user_turns"] >= thresholds["minimum_user_turns"]
        and counts["unique_user_turns"] >= thresholds["minimum_unique_user_turns"]
        and counts["active_days"] >= thresholds["minimum_active_days"]
        and counts["sessions"] >= thresholds["minimum_sessions"]
    )
    audit = {
        "format": spec["audit"]["format"],
        "source_read_only": True,
        "source_append_only": append_only,
        "baseline_prefix_sha256": prefix_sha256,
        "current_manifest_sha256": _manifest_sha256(rows),
        "columns_sha256": hashlib.sha256("\n".join(sorted(columns)).encode()).hexdigest(),
        "counts": counts,
        "new_user_turns": counts["eligible_user_turns"] - baseline["eligible_user_turns"],
        "first_observed_at": timestamps[0].isoformat() if timestamps else None,
        "last_observed_at": timestamps[-1].isoformat() if timestamps else None,
        "collection_status": "ready_for_review" if ready else "collecting",
        "raw_text_leaks": 0,
    }
    if _contains_forbidden_raw_key(audit):
        raise ValueError("raw dialogue key entered collection audit")
    return audit


def run(source: Path, spec: dict = RUNTIME_MEMORY_COLLECTION_SPEC) -> dict:
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": str(source),
        "audit": scan_collection_source(source, spec),
    }
    if _contains_forbidden_raw_key(payload):
        raise ValueError("raw dialogue key entered collection result")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(RUNTIME_MEMORY_COLLECTION_SPEC["source_database"]))
    parser.add_argument("--output", type=Path, default=Path(".local/gate-runtime3/status.json"))
    args = parser.parse_args()
    _atomic_json(args.output, run(args.source))


if __name__ == "__main__":
    main()
