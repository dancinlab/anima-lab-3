#!/usr/bin/env python3
"""GATE-CONTROL-1: frozen semantic sentence-encoder positive control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

from gate1 import _matched_random, _metrics, build_calibration, build_evaluation, dataset_audit
from memory_gate import ROLE_ORDER, fit_canonical_ridge
from measurement.semantic_memory_write_registry import SEMANTIC_MEMORY_WRITE_SPEC, spec_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def attention_mask_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the pooling declared by the pinned sentence-transformer model."""
    if hidden.dim() != 3 or mask.dim() != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("semantic pooling shapes are invalid")
    expanded = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denominator = expanded.sum(dim=1).clamp_min(1e-9)
    return (hidden * expanded).sum(dim=1) / denominator


class FrozenSentenceEncoder:
    """Pinned standard encoder used only by the registered positive control."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        runtime = SEMANTIC_MEMORY_WRITE_SPEC["runtime"]
        if torch.__version__.split("+")[0] != runtime["torch"]:
            raise RuntimeError(f"registered torch {runtime['torch']} is required")
        if transformers.__version__ != runtime["transformers"]:
            raise RuntimeError(f"registered transformers {runtime['transformers']} is required")
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"], use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            spec["model_id"], revision=spec["revision"],
        ).to(spec["device"]).eval()
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision != spec["revision"]:
            raise RuntimeError("loaded encoder revision does not match the registration")
        if int(self.model.config.hidden_size) != spec["embedding_dim"]:
            raise RuntimeError("loaded encoder width does not match the registration")

    def encode_rows(self, rows: list[dict]) -> tuple[torch.Tensor, dict]:
        texts = []
        role_indices = []
        for row in rows:
            role = row.get("role")
            text = row.get("text")
            if role not in ROLE_ORDER:
                raise ValueError(f"unsupported dialogue role: {role}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("semantic memory-gate text must be non-empty")
            texts.append(text)
            role_indices.append(ROLE_ORDER.index(role))

        pooled_rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.spec["batch_size"]):
                encoded = self.tokenizer(
                    texts[start:start + self.spec["batch_size"]],
                    padding=True,
                    truncation=True,
                    max_length=self.spec["max_length"],
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.spec["device"]) for name, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                pooled = attention_mask_mean(hidden, encoded["attention_mask"])
                pooled_rows.append(F.normalize(pooled, p=2, dim=1).cpu())
        sentence_features = torch.cat(pooled_rows).to(dtype=torch.float64)
        role_features = torch.zeros(len(rows), len(ROLE_ORDER), dtype=torch.float64)
        role_features[torch.arange(len(rows)), torch.tensor(role_indices)] = 1.0
        features = torch.cat((sentence_features, role_features), dim=1)
        if features.shape[1] != self.spec["feature_dim"] or not torch.isfinite(features).all():
            raise RuntimeError("semantic feature validation failed")
        norms = torch.linalg.vector_norm(sentence_features, dim=1)
        return features, {
            "rows": len(rows),
            "feature_dim": features.shape[1],
            "sentence_norm_min": float(norms.min()),
            "sentence_norm_max": float(norms.max()),
            "features_sha256": hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest(),
        }

    def audit(self) -> dict:
        return {
            "model_id": self.spec["model_id"],
            "requested_revision": self.spec["revision"],
            "loaded_revision": getattr(self.model.config, "_commit_hash", None),
            "embedding_dim": int(self.model.config.hidden_size),
            "pooling": self.spec["pooling"],
            "normalize": self.spec["normalize"],
            "max_length": self.spec["max_length"],
        }


def _checkpoint_payload(weight: torch.Tensor, bias: float, threshold: float, spec: dict) -> dict:
    return {
        "format": "semantic_dialogue_memory_gate_control_v1",
        "method": SEMANTIC_MEMORY_WRITE_SPEC["fit_method"],
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "feature_dim": spec["feature_dim"],
        "weight": weight.tolist(),
        "bias": bias,
        "threshold": threshold,
    }


def _predict(features: torch.Tensor, weight: torch.Tensor, bias: float,
             threshold: float, width: int) -> list[list[bool]]:
    selected = (features @ weight + bias >= threshold).tolist()
    if len(selected) % width:
        raise ValueError("evaluation feature rows do not match the episode width")
    return [selected[index:index + width] for index in range(0, len(selected), width)]


def run_seed(seed: int, encoder: FrozenSentenceEncoder, checkpoint_dir: Path,
             spec: dict = SEMANTIC_MEMORY_WRITE_SPEC) -> dict:
    data_spec = spec["gate1_spec"]
    calibration = build_calibration(seed, data_spec)
    episodes = build_evaluation(seed, data_spec)
    evaluation_rows = [row for episode in episodes for row in episode["candidates"]]
    calibration_features, calibration_embedding_audit = encoder.encode_rows(calibration)
    evaluation_features, evaluation_embedding_audit = encoder.encode_rows(evaluation_rows)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"],
    )

    generator = torch.Generator().manual_seed(seed + data_spec["shuffle_seed_offset"])
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, shuffled_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=spec["ridge"],
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_semantic_memory_gate.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_shuffled_semantic_memory_gate.json"
    _atomic_json(checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"]))
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(fake_weight, fake_bias, fake_threshold, spec["encoder"]),
    )

    semantic = _predict(
        evaluation_features, weight, bias, threshold, data_spec["candidates_per_episode"],
    )
    shuffled = _predict(
        evaluation_features, fake_weight, fake_bias, fake_threshold,
        data_spec["candidates_per_episode"],
    )
    arms = {
        "semantic_gate": semantic,
        "store_all": [[True] * data_spec["candidates_per_episode"] for _ in episodes],
        "oracle_gate": [
            [bool(row["important"]) for row in episode["candidates"]] for episode in episodes
        ],
        "matched_random": [
            _matched_random(selection, seed + data_spec["random_seed_offset"] + index)
            for index, selection in enumerate(semantic)
        ],
        "shuffled_gate": shuffled,
        "no_memory": [[False] * data_spec["candidates_per_episode"] for _ in episodes],
    }
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, episodes, data_spec),
        "encoder_audit": encoder.audit(),
        "embedding_audit": {
            "calibration": calibration_embedding_audit,
            "evaluation": evaluation_embedding_audit,
        },
        "fit_audit": fit_audit,
        "shuffled_fit_audit": shuffled_fit_audit,
        "checkpoints": {
            "semantic": {
                "path": str(checkpoint_path),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            },
            "shuffled": {
                "path": str(fake_checkpoint_path),
                "sha256": hashlib.sha256(fake_checkpoint_path.read_bytes()).hexdigest(),
            },
        },
        "arms": {name: _metrics(episodes, selections, data_spec["top_k"])
                 for name, selections in arms.items()},
    }


def run(checkpoint_dir: Path, spec: dict = SEMANTIC_MEMORY_WRITE_SPEC) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    encoder = FrozenSentenceEncoder(spec["encoder"])
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
        "seeds": [run_seed(seed, encoder, checkpoint_dir, spec)
                  for seed in spec["gate1_spec"]["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_memory_write_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/gate-control1"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
