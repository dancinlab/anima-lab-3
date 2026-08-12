#!/usr/bin/env python3
"""Deterministic long-term-memory write gate used by GATE-1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from creativity_classifier import text_to_vector


ROLE_ORDER = ("user", "assistant")


def fit_canonical_ridge(features: torch.Tensor, labels: torch.Tensor, *,
                        ridge: float = 1e-3) -> tuple[torch.Tensor, float, float, dict]:
    """Fit the shared deterministic binary ridge readout to precomputed features."""
    features = features.detach().to(dtype=torch.float64)
    labels = labels.detach().to(dtype=torch.float64)
    if features.dim() != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("canonical-ridge features must be a non-empty matrix")
    if labels.dim() != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("canonical-ridge labels must match the feature rows")
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("canonical-ridge labels must contain both binary classes")
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError("canonical-ridge inputs must be finite")
    if not float(ridge) > 0:
        raise ValueError("canonical-ridge penalty must be positive")

    design = torch.cat([features, torch.ones(len(features), 1, dtype=torch.float64)], dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[-1, -1] = 0.0
    solution = torch.linalg.solve(
        design.T @ design + float(ridge) * penalty,
        design.T @ labels,
    )
    scores = design @ solution
    positive_mean = float(scores[labels == 1].mean())
    negative_mean = float(scores[labels == 0].mean())
    threshold = (positive_mean + negative_mean) / 2.0
    return solution[:-1].detach().clone(), float(solution[-1]), threshold, {
        "method": "canonical_ridge",
        "examples": len(features),
        "positives": int(labels.sum()),
        "negatives": int((1 - labels).sum()),
        "feature_dim": features.shape[1],
        "design_rank": int(torch.linalg.matrix_rank(design)),
        "ridge": float(ridge),
        "positive_score_mean": positive_mean,
        "negative_score_mean": negative_mean,
        "threshold": threshold,
    }


def memory_gate_features(role: str, text: str, dim: int = 128) -> torch.Tensor:
    """Reuse the native trigram vector and append a fixed role indicator."""
    if role not in ROLE_ORDER:
        raise ValueError(f"unsupported dialogue role: {role}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("memory-gate text must be a non-empty string")
    base = text_to_vector(text, dim=dim).detach().to(dtype=torch.float64)
    role_features = torch.zeros(len(ROLE_ORDER), dtype=torch.float64)
    role_features[ROLE_ORDER.index(role)] = 1.0
    return torch.cat([base, role_features])


@dataclass(frozen=True)
class DialogueMemoryGate:
    """Frozen canonical-ridge binary decision for long-term search indexing."""

    weight: torch.Tensor
    bias: float
    threshold: float
    vector_dim: int = 128

    def __post_init__(self) -> None:
        expected = self.vector_dim + len(ROLE_ORDER)
        if self.weight.dim() != 1 or self.weight.numel() != expected:
            raise ValueError("memory-gate weight width is invalid")
        if not torch.isfinite(self.weight).all():
            raise ValueError("memory-gate weight must be finite")
        if not all(torch.isfinite(torch.tensor(value)) for value in (self.bias, self.threshold)):
            raise ValueError("memory-gate bias and threshold must be finite")

    def score(self, role: str, text: str) -> float:
        features = memory_gate_features(role, text, self.vector_dim)
        return float(features @ self.weight.to(dtype=torch.float64) + self.bias)

    def should_index(self, role: str, text: str) -> bool:
        return self.score(role, text) >= self.threshold

    def to_payload(self) -> dict:
        return {
            "format": "dialogue_memory_gate_v1",
            "method": "canonical_ridge",
            "vector_dim": self.vector_dim,
            "roles": list(ROLE_ORDER),
            "weight": self.weight.detach().to(dtype=torch.float64).tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DialogueMemoryGate":
        if (
            payload.get("format") != "dialogue_memory_gate_v1"
            or payload.get("method") != "canonical_ridge"
            or payload.get("roles") != list(ROLE_ORDER)
        ):
            raise ValueError("unsupported memory-gate checkpoint")
        return cls(
            weight=torch.tensor(payload["weight"], dtype=torch.float64),
            bias=float(payload["bias"]),
            threshold=float(payload["threshold"]),
            vector_dim=int(payload["vector_dim"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DialogueMemoryGate":
        return cls.from_payload(json.loads(Path(path).read_text()))


def fit_canonical_memory_gate(rows: list[dict], *, vector_dim: int = 128,
                              ridge: float = 1e-3) -> tuple[DialogueMemoryGate, dict]:
    """Fit the unique ridge solution; no optimizer seed or evaluation data is used."""
    if not rows:
        raise ValueError("memory-gate calibration rows must not be empty")
    labels = torch.tensor([int(row["important"]) for row in rows], dtype=torch.float64)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("memory-gate calibration must contain both labels")
    features = torch.stack([
        memory_gate_features(row["role"], row["text"], vector_dim) for row in rows
    ])
    weight, bias, threshold, audit = fit_canonical_ridge(features, labels, ridge=ridge)
    gate = DialogueMemoryGate(
        weight=weight,
        bias=bias,
        threshold=threshold,
        vector_dim=vector_dim,
    )
    return gate, audit
