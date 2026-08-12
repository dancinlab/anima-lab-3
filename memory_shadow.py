#!/usr/bin/env python3
"""Answer-inert audit path for dialogue-memory write decisions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import torch


AUDIT_FORMAT = "runtime_memory_shadow_v1"


def _sha256_text(role: str, text: str) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError(f"unsupported dialogue role: {role}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("shadow memory text must be a non-empty string")
    return hashlib.sha256(f"{role}\0{text}".encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SemanticMemoryWriteScorer:
    """Frozen GATE-4-compatible sentence scorer for the live shadow path."""

    def __init__(self, checkpoint: str | Path, *, encoder=None,
                 expected_sha256: str | None = None) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"memory-shadow checkpoint not found: {checkpoint}")
        if expected_sha256 and _file_sha256(checkpoint) != expected_sha256:
            raise ValueError("memory-shadow checkpoint digest mismatch")
        payload = json.loads(checkpoint.read_text())
        required = {
            "format", "method", "model_id", "revision", "feature_dim",
            "weight", "bias", "threshold",
        }
        if set(payload) != required:
            raise ValueError("memory-shadow checkpoint shape changed")
        if (
            payload["format"] != "semantic_dialogue_memory_gate_control_v1"
            or payload["method"] != "canonical_ridge"
        ):
            raise ValueError("unsupported memory-shadow checkpoint")

        from measurement.runtime_memory_shadow_registry import RUNTIME_MEMORY_SHADOW_SPEC

        encoder_spec = RUNTIME_MEMORY_SHADOW_SPEC["encoder"]
        if (
            payload["model_id"] != encoder_spec["model_id"]
            or payload["revision"] != encoder_spec["revision"]
            or payload["feature_dim"] != encoder_spec["feature_dim"]
        ):
            raise ValueError("memory-shadow encoder registration mismatch")
        weight = torch.tensor(payload["weight"], dtype=torch.float64)
        if weight.dim() != 1 or weight.numel() != payload["feature_dim"]:
            raise ValueError("memory-shadow checkpoint weight width changed")
        if not torch.isfinite(weight).all():
            raise ValueError("memory-shadow checkpoint contains non-finite weights")
        self.weight = weight
        self.bias = float(payload["bias"])
        self.threshold = float(payload["threshold"])
        if not math.isfinite(self.bias) or not math.isfinite(self.threshold):
            raise ValueError("memory-shadow checkpoint contains non-finite scalars")
        if encoder is None:
            from gate_control1 import FrozenSentenceEncoder
            encoder = FrozenSentenceEncoder(encoder_spec)
        self.encoder = encoder
        self.checkpoint_sha256 = _file_sha256(checkpoint)

    def score_rows(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        features, _ = self.encoder.encode_rows(rows)
        if features.shape != (len(rows), self.weight.numel()):
            raise ValueError("memory-shadow feature shape changed")
        scores = features @ self.weight + self.bias
        if not torch.isfinite(scores).all():
            raise ValueError("memory-shadow produced a non-finite score")
        return [float(value) for value in scores]


class RuntimeMemoryShadow:
    """Append-only, raw-text-free observer that never filters primary memory."""

    def __init__(self, scorer, audit_path: str | Path, *,
                 clock: Callable[[], str] | None = None) -> None:
        self.scorer = scorer
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._lock = threading.Lock()
        self._decisions: dict[str, bool] = {}
        self._load_decisions()

    def _load_decisions(self) -> None:
        if not self.audit_path.is_file():
            return
        for line in self.audit_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("format") == AUDIT_FORMAT
                and event.get("event") == "write"
                and isinstance(event.get("text_sha256"), str)
                and type(event.get("selected")) is bool
            ):
                self._decisions[event["text_sha256"]] = event["selected"]

    @staticmethod
    def _validate_event(event: dict) -> None:
        forbidden = {"text", "query", "response", "content", "raw_text"}

        def validate_keys(value) -> None:
            if isinstance(value, dict):
                if forbidden & set(value):
                    raise ValueError("raw dialogue text is forbidden in shadow audit events")
                for item in value.values():
                    validate_keys(item)
            elif isinstance(value, list):
                for item in value:
                    validate_keys(item)

        validate_keys(event)
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
        if "NaN" in encoded or "Infinity" in encoded:
            raise ValueError("non-finite shadow audit value")

    def _append_many(self, events: list[dict], *, decisions: dict[str, bool] | None = None) -> None:
        if not events:
            return
        for event in events:
            self._validate_event(event)
        payload = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        )
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if decisions:
                self._decisions.update(decisions)

    def record_writes(self, rows: list[dict]) -> list[dict]:
        scores = self.scorer.score_rows(rows)
        if len(scores) != len(rows):
            raise ValueError("memory-shadow scorer result count changed")
        events = []
        decisions = {}
        for row, score in zip(rows, scores):
            role = row.get("role")
            text = row.get("text")
            digest = _sha256_text(role, text)
            if not math.isfinite(score):
                raise ValueError("memory-shadow score must be finite")
            selected = score >= self.scorer.threshold
            event = {
                "format": AUDIT_FORMAT,
                "event": "write",
                "occurred_at": self.clock(),
                "role": role,
                "text_sha256": digest,
                "score": score,
                "threshold": float(self.scorer.threshold),
                "selected": selected,
                "source_id": str(row["memory_id"]) if row.get("memory_id") is not None else None,
            }
            events.append(event)
            decisions[digest] = selected
        self._append_many(events, decisions=decisions)
        return events

    def record_search(self, query_text: str, candidates: Iterable[dict]) -> dict:
        query_sha256 = hashlib.sha256(query_text.encode()).hexdigest()
        with self._lock:
            decisions = dict(self._decisions)
        candidate_rows = []
        for rank, candidate in enumerate(candidates):
            role = candidate.get("role")
            text = candidate.get("text")
            digest = _sha256_text(role, text)
            similarity = candidate.get("similarity")
            if similarity is not None and (
                not isinstance(similarity, (int, float)) or not math.isfinite(similarity)
            ):
                raise ValueError("memory-shadow candidate similarity must be finite")
            candidate_rows.append({
                "rank": rank,
                "source_id": str(candidate["id"]) if candidate.get("id") is not None else None,
                "role": role,
                "text_sha256": digest,
                "selected": decisions.get(digest),
                "similarity": float(similarity) if similarity is not None else None,
            })
        event = {
            "format": AUDIT_FORMAT,
            "event": "search",
            "occurred_at": self.clock(),
            "query_sha256": query_sha256,
            "candidates": candidate_rows,
        }
        self._append_many([event])
        return event


def create_memory_shadow(factory: Callable[[], RuntimeMemoryShadow], *, on_error=None):
    """Fail open during optional shadow initialization."""
    try:
        return factory()
    except Exception as error:
        if on_error:
            on_error(error)
        return None


def observe_shadow_writes(shadow, rows: list[dict], *, on_error=None) -> None:
    """Observe completed primary writes without allowing audit failure to escape."""
    if shadow is None:
        return
    try:
        shadow.record_writes(rows)
    except Exception as error:
        if on_error:
            on_error(error)


def observe_shadow_search(shadow, query: str, candidates: Iterable[dict], *, on_error=None) -> None:
    """Observe an already-built primary search context without changing it."""
    if shadow is None:
        return
    try:
        shadow.record_search(query, list(candidates))
    except Exception as error:
        if on_error:
            on_error(error)
