#!/usr/bin/env python3
"""GATE-RUNTIME-1: replay the opt-in memory shadow around the primary path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_write_control1 import (
    build_balanced_calibration,
    build_balanced_evaluation,
)
from memory_gate import fit_canonical_ridge
from memory_shadow import (
    RuntimeMemoryShadow,
    SemanticMemoryWriteScorer,
    create_memory_shadow,
    observe_shadow_search,
    observe_shadow_writes,
)
from measurement.runtime_memory_shadow_registry import (
    RUNTIME_MEMORY_SHADOW_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


class _FixedClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"2026-08-12T00:00:{self.value:06d}Z"


class _FaultingShadow:
    threshold = 0.5

    def __init__(self, fault: str) -> None:
        self.fault = fault

    def record_writes(self, rows):
        if self.fault == "write_audit":
            raise OSError("registered write-audit fault")
        return []

    def record_search(self, query, candidates):
        if self.fault == "search_audit":
            raise OSError("registered search-audit fault")
        return {}


def _checkpoint(encoder: FrozenSentenceEncoder, checkpoint_path: Path, spec: dict) -> dict:
    calibration = sorted(
        build_balanced_calibration(spec["calibration_seed"]),
        key=lambda row: (row["role"], row["text"], row["kind"], row["template_index"]),
    )
    features, feature_audit = encoder.encode_rows(calibration)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        features, labels, ridge=spec["ridge"],
    )
    payload = _checkpoint_payload(weight, bias, threshold, spec["encoder"])
    _atomic_json(checkpoint_path, payload)
    return {
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "format": payload["format"],
        "calibration_sha256": _digest([
            {"role": row["role"], "text": row["text"], "important": row["important"]}
            for row in calibration
        ]),
        "feature_audit": feature_audit,
        "fit_audit": fit_audit,
    }


def _primary_trace(episodes: list[dict]) -> dict:
    stored = []
    searches = []
    answers = []
    next_id = 1
    episode_rows = []
    for episode in episodes:
        candidates = []
        for position, row in enumerate(episode["candidates"]):
            primary = {
                "id": next_id,
                "role": row["role"],
                "text": row["text"],
                "timestamp": f"turn-{next_id}",
                "similarity": 1.0 - position / 10.0,
            }
            next_id += 1
            stored.append({key: primary[key] for key in ("id", "role", "text", "timestamp")})
            candidates.append(primary)
            episode_rows.append({
                "role": row["role"],
                "text": row["text"],
                "memory_id": primary["id"],
                "important": bool(row["important"]),
                "kind": episode["kind"] if row["important"] else row["kind"],
            })
        search_view = [
            {key: candidate[key] for key in ("id", "role", "text", "timestamp", "similarity")}
            for candidate in candidates
        ]
        searches.append(search_view)
        context = [candidate["text"] for candidate in search_view[:3] if candidate["similarity"] > 0.3]
        answers.append(f"{episode['query']}\n" + "\n".join(context))
    return {
        "stored": stored,
        "searches": searches,
        "answers": answers,
        "rows": episode_rows,
        "answer_sha256": _digest(answers),
        "store_sha256": _digest(stored),
        "search_sha256": _digest(searches),
    }


def _selection_metrics(rows: list[dict], events: list[dict], spec: dict) -> dict:
    if len(rows) != len(events):
        raise ValueError("runtime shadow write-event count changed")
    important = [event["selected"] for row, event in zip(rows, events) if row["important"]]
    distractors = [event["selected"] for row, event in zip(rows, events) if not row["important"]]
    per_kind = {
        kind: sum(
            event["selected"] for row, event in zip(rows, events)
            if row["important"] and row["kind"] == kind
        ) / sum(row["important"] and row["kind"] == kind for row in rows)
        for kind in spec["fact_kinds"]
    }
    return {
        "important_selection_rate": sum(important) / len(important),
        "distractor_selection_rate": sum(distractors) / len(distractors),
        "selection_ratio": sum(event["selected"] for event in events) / len(events),
        "per_kind_selection_rate": per_kind,
        "selection_sha256": _digest([event["selected"] for event in events]),
    }


def _fault_audit(primary: dict, fault: str) -> dict:
    before = {
        "answer": _digest(primary["answers"]),
        "store": _digest(primary["stored"]),
        "search": _digest(primary["searches"]),
    }
    errors = []
    if fault == "initialization":
        shadow = create_memory_shadow(
            lambda: (_ for _ in ()).throw(RuntimeError("registered initialization fault")),
            on_error=lambda error: errors.append(type(error).__name__),
        )
    else:
        shadow = _FaultingShadow(fault)
    observe_shadow_writes(
        shadow,
        primary["rows"],
        on_error=lambda error: errors.append(type(error).__name__),
    )
    for query_index, candidates in enumerate(primary["searches"]):
        observe_shadow_search(
            shadow,
            f"registered-query-{query_index}",
            candidates,
            on_error=lambda error: errors.append(type(error).__name__),
        )
    after = {
        "answer": _digest(primary["answers"]),
        "store": _digest(primary["stored"]),
        "search": _digest(primary["searches"]),
    }
    return {
        "answer_digest_mismatches": int(before["answer"] != after["answer"]),
        "primary_store_digest_mismatches": int(before["store"] != after["store"]),
        "primary_search_digest_mismatches": int(before["search"] != after["search"]),
        "caught_errors": len(errors),
    }


def _run_replicate(name: str, scorer: SemanticMemoryWriteScorer,
                   audit_dir: Path, spec: dict) -> dict:
    episodes = build_balanced_evaluation(spec["evaluation_seed"], name)
    primary = _primary_trace(episodes)
    baseline = {
        "answer": _digest(primary["answers"]),
        "store": _digest(primary["stored"]),
        "search": _digest(primary["searches"]),
    }
    audit_path = audit_dir / f"{name}.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    shadow = RuntimeMemoryShadow(scorer, audit_path, clock=_FixedClock())
    events = []
    chunk = 256
    for start in range(0, len(primary["rows"]), chunk):
        events.extend(shadow.record_writes(primary["rows"][start:start + chunk]))
    for episode, candidates in zip(episodes, primary["searches"]):
        shadow.record_search(episode["query"], candidates)

    parsed = [json.loads(line) for line in audit_path.read_text().splitlines()]
    writes = [event for event in parsed if event.get("event") == "write"]
    searches = [event for event in parsed if event.get("event") == "search"]
    source_texts = {row["text"] for row in primary["rows"]}
    audit_strings = set(_strings(parsed))
    missing_search = sum(
        candidate["selected"] is None
        for event in searches
        for candidate in event["candidates"]
    )
    observed = {
        "answer": _digest(primary["answers"]),
        "store": _digest(primary["stored"]),
        "search": _digest(primary["searches"]),
    }
    return {
        "name": name,
        "dataset_audit": {
            "episodes": len(episodes),
            "candidates": len(primary["rows"]),
            "unique_texts": len(source_texts),
            "raw_transcript_sha256": _digest(primary["stored"]),
        },
        "preservation": {
            "answer_digest_mismatches": int(baseline["answer"] != observed["answer"]),
            "primary_store_digest_mismatches": int(baseline["store"] != observed["store"]),
            "primary_search_digest_mismatches": int(baseline["search"] != observed["search"]),
            "answer_sha256": primary["answer_sha256"],
            "primary_store_sha256": primary["store_sha256"],
            "primary_search_sha256": primary["search_sha256"],
        },
        "shadow_audit": {
            "path": str(audit_path),
            "write_events": len(writes),
            "search_events": len(searches),
            "missing_write_audits": len(primary["rows"]) - len(writes),
            "missing_search_audits": missing_search,
            "raw_text_leaks": len(source_texts & audit_strings),
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        },
        "selection": _selection_metrics(primary["rows"], events, spec),
        "faults": {
            fault: _fault_audit(primary, fault) for fault in spec["faults"]
        },
    }


def run(output_dir: Path, checkpoint_path: Path,
        spec: dict = RUNTIME_MEMORY_SHADOW_SPEC) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    encoder = FrozenSentenceEncoder(spec["encoder"])
    checkpoint = _checkpoint(encoder, checkpoint_path, spec)
    scorer = SemanticMemoryWriteScorer(
        checkpoint_path,
        encoder=encoder,
        expected_sha256=checkpoint["sha256"],
    )
    return {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": spec["encoder"]["device"],
        },
        "checkpoint": checkpoint,
        "encoder_audit": encoder.audit(),
        "replicates": [
            _run_replicate(name, scorer, output_dir / "audit", spec)
            for name in spec["replicates"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("measurement/runtime_memory_shadow_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/runtime-memory-shadow"))
    parser.add_argument("--checkpoint", type=Path, default=Path(RUNTIME_MEMORY_SHADOW_SPEC["checkpoint"]))
    args = parser.parse_args()
    _atomic_json(args.output, run(args.output_dir, args.checkpoint))


if __name__ == "__main__":
    main()
