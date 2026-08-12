#!/usr/bin/env python3
"""GATE-CONTROL-2: storage-matched semantic write positive control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate1 import _matched_random, _metrics, build_calibration, build_evaluation, dataset_audit
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from memory_gate import fit_canonical_ridge
from measurement.semantic_memory_write_matched_registry import (
    MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _scores(features: torch.Tensor, weight: torch.Tensor, bias: float) -> torch.Tensor:
    scores = features @ weight + bias
    if scores.dim() != 1 or not torch.isfinite(scores).all():
        raise ValueError("semantic selector scores must be a finite vector")
    return scores


def _threshold_rows(scores: torch.Tensor, threshold: float, width: int) -> list[list[bool]]:
    if width <= 0 or scores.numel() % width:
        raise ValueError("selector scores do not match the episode width")
    selected = (scores >= threshold).tolist()
    return [selected[index:index + width] for index in range(0, len(selected), width)]


def match_ranked_counts(
    scores: torch.Tensor,
    reference: list[list[bool]],
    width: int,
) -> list[list[bool]]:
    """Select the top fake scores while matching every episode's reference count."""
    if width <= 0 or scores.dim() != 1 or scores.numel() != len(reference) * width:
        raise ValueError("ranked matching shapes are invalid")
    if not torch.isfinite(scores).all() or any(len(row) != width for row in reference):
        raise ValueError("ranked matching inputs are invalid")
    matched = []
    for episode_index, reference_row in enumerate(reference):
        count = sum(bool(value) for value in reference_row)
        start = episode_index * width
        episode_scores = scores[start:start + width].tolist()
        ranked = sorted(range(width), key=lambda index: (-episode_scores[index], index))
        chosen = set(ranked[:count])
        matched.append([index in chosen for index in range(width)])
    return matched


def _selection_digest(rows: list[list[bool]]) -> str:
    encoded = "\n".join("".join("1" if value else "0" for value in row) for row in rows)
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_seed(
    seed: int,
    encoder: FrozenSentenceEncoder,
    checkpoint_dir: Path,
    spec: dict = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
) -> dict:
    control1_spec = spec["control1_spec"]
    data_spec = control1_spec["gate1_spec"]
    calibration = build_calibration(seed, data_spec)
    episodes = build_evaluation(seed, data_spec)
    evaluation_rows = [row for episode in episodes for row in episode["candidates"]]
    calibration_features, calibration_embedding_audit = encoder.encode_rows(calibration)
    evaluation_features, evaluation_embedding_audit = encoder.encode_rows(evaluation_rows)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=control1_spec["ridge"],
    )

    generator = torch.Generator().manual_seed(seed + data_spec["shuffle_seed_offset"])
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, shuffled_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=control1_spec["ridge"],
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_semantic_memory_gate.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_shuffled_semantic_memory_gate.json"
    _atomic_json(
        checkpoint_path,
        _checkpoint_payload(weight, bias, threshold, control1_spec["encoder"]),
    )
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(
            fake_weight, fake_bias, fake_threshold, control1_spec["encoder"],
        ),
    )

    semantic_scores = _scores(evaluation_features, weight, bias)
    fake_scores = _scores(evaluation_features, fake_weight, fake_bias)
    semantic = _threshold_rows(
        semantic_scores, threshold, data_spec["candidates_per_episode"],
    )
    matched_shuffled = match_ranked_counts(
        fake_scores, semantic, data_spec["candidates_per_episode"],
    )
    matched_random = [
        _matched_random(selection, seed + data_spec["random_seed_offset"] + index)
        for index, selection in enumerate(semantic)
    ]
    arms = {
        "semantic_gate": semantic,
        "store_all": [[True] * data_spec["candidates_per_episode"] for _ in episodes],
        "oracle_gate": [
            [bool(row["important"]) for row in episode["candidates"]] for episode in episodes
        ],
        "matched_random": matched_random,
        "matched_shuffled_gate": matched_shuffled,
        "no_memory": [[False] * data_spec["candidates_per_episode"] for _ in episodes],
    }
    semantic_counts = [sum(row) for row in semantic]
    matched_counts = [sum(row) for row in matched_shuffled]
    random_counts = [sum(row) for row in matched_random]
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
        "matching_audit": {
            "method": spec["matching"],
            "semantic_counts": semantic_counts,
            "matched_shuffled_counts": matched_counts,
            "matched_random_counts": random_counts,
            "semantic_selection_sha256": _selection_digest(semantic),
            "matched_shuffled_selection_sha256": _selection_digest(matched_shuffled),
            "matched_random_selection_sha256": _selection_digest(matched_random),
            "fake_scores_sha256": hashlib.sha256(
                fake_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
        },
        "arms": {
            name: _metrics(episodes, selections, data_spec["top_k"])
            for name, selections in arms.items()
        },
    }


def run(
    checkpoint_dir: Path,
    spec: dict = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    control1_spec = spec["control1_spec"]
    encoder = FrozenSentenceEncoder(control1_spec["encoder"])
    return {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": control1_spec["encoder"]["device"],
        },
        "seeds": [
            run_seed(seed, encoder, checkpoint_dir, spec)
            for seed in control1_spec["gate1_spec"]["seeds"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_memory_write_matched_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/gate-control2"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
