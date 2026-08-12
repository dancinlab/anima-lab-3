#!/usr/bin/env python3
"""GATE-WRITE-MECHANISM-1: decompose seed-coupled write selection factors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate2 import _scores, build_calibration, dataset_audit
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_control2 import _selection_digest, _threshold_rows
from gate_retrieval_control3 import build_balanced_evaluation
from memory_gate import fit_canonical_ridge
from measurement.integrated_dialogue_memory_registry import INTEGRATED_DIALOGUE_MEMORY_SPEC
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC
from measurement.write_mechanism_registry import WRITE_MECHANISM_SPEC, spec_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def factor_sources(seed: int, peer: int, swapped: list[str]) -> dict[str, int]:
    unknown = set(swapped) - set(WRITE_MECHANISM_SPEC["factors"])
    if unknown or seed == peer:
        raise ValueError("invalid write-mechanism factor assignment")
    return {
        factor: peer if factor in swapped else seed
        for factor in WRITE_MECHANISM_SPEC["factors"]
    }


def _factor_audit(sources: dict[str, int]) -> dict:
    return {
        "sources": sources,
        "source_sha256": {
            factor: hashlib.sha256(f"{factor}:{source}".encode()).hexdigest()
            for factor, source in sources.items()
        },
    }


def _score_summary(values: torch.Tensor) -> dict:
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("score summary requires finite values")
    return {
        "count": values.numel(),
        "minimum": float(values.min()),
        "q25": float(torch.quantile(values, 0.25)),
        "median": float(torch.quantile(values, 0.50)),
        "q75": float(torch.quantile(values, 0.75)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def selection_metrics(
    episodes: list[dict], selections: list[list[bool]], scores: torch.Tensor, spec: dict
) -> dict:
    width = spec["candidates_per_episode"]
    if (
        len(episodes) != len(selections)
        or scores.shape != (len(episodes), width)
        or any(len(row) != width for row in selections)
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("write-mechanism selections have incompatible shapes")
    kind_hits = {kind: [0, 0] for kind in spec["fact_kinds"]}
    position_hits = {str(position): [0, 0] for position in spec["fact_positions"]}
    distractor_hits = {kind: [0, 0] for kind in spec["distractor_kinds"]}
    fact_scores = {kind: [] for kind in spec["fact_kinds"]}
    distractor_scores = []
    important_kept = distractors_kept = total_kept = 0
    for episode_index, (episode, selected) in enumerate(zip(episodes, selections)):
        fact_position = episode["fact_position"]
        kept = bool(selected[fact_position])
        important_kept += kept
        total_kept += sum(bool(value) for value in selected)
        kind_hits[episode["kind"]][0] += kept
        kind_hits[episode["kind"]][1] += 1
        position_hits[str(fact_position)][0] += kept
        position_hits[str(fact_position)][1] += 1
        fact_scores[episode["kind"]].append(scores[episode_index, fact_position])
        for position, (row, keep) in enumerate(zip(episode["candidates"], selected)):
            if position == fact_position:
                continue
            distractors_kept += bool(keep)
            distractor_hits[row["kind"]][0] += bool(keep)
            distractor_hits[row["kind"]][1] += 1
            distractor_scores.append(scores[episode_index, position])
    total = len(episodes)
    distractor_total = total * (width - 1)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / distractor_total,
        "search_size_ratio": total_kept / (total * width),
        "stored": total_kept,
        "per_kind_storage_rate": {
            kind: hits / count for kind, (hits, count) in kind_hits.items()
        },
        "per_position_storage_rate": {
            position: hits / count for position, (hits, count) in position_hits.items()
        },
        "per_distractor_storage_rate": {
            kind: hits / count for kind, (hits, count) in distractor_hits.items()
        },
        "per_kind_score_summary": {
            kind: _score_summary(torch.stack(values)) for kind, values in fact_scores.items()
        },
        "distractor_score_summary": _score_summary(torch.stack(distractor_scores)),
        "selection_sha256": _selection_digest(selections),
        "scores_sha256": hashlib.sha256(scores.contiguous().numpy().tobytes()).hexdigest(),
    }


def _source_spec(spec: dict) -> dict:
    source = dict(REALISTIC_MEMORY_WRITE_SPEC)
    source["evaluation_episodes"] = spec["evaluation_episodes"]
    source["fact_positions"] = list(spec["fact_positions"])
    return source


def run_arm(
    seed: int,
    peer: int,
    arm_name: str,
    encoder: FrozenSentenceEncoder,
    checkpoint_dir: Path,
    spec: dict,
) -> dict:
    sources = factor_sources(seed, peer, spec["arms"][arm_name])
    calibration = build_calibration(
        seed,
        REALISTIC_MEMORY_WRITE_SPEC,
        template_seed=sources["template"],
        identifier_seed=sources["identifier"],
        layout_seed=sources["layout"],
    )
    episodes = build_balanced_evaluation(
        seed,
        INTEGRATED_DIALOGUE_MEMORY_SPEC,
        template_seed=sources["template"],
        identifier_seed=sources["identifier"],
        layout_seed=sources["layout"],
    )
    candidates = [row for episode in episodes for row in episode["candidates"]]
    calibration_features, calibration_embedding = encoder.encode_rows(calibration)
    candidate_features, candidate_embedding = encoder.encode_rows(candidates)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"]
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_{arm_name}_write_gate.json"
    _atomic_json(
        checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"])
    )
    width = spec["candidates_per_episode"]
    scores = _scores(candidate_features, weight, bias).reshape(-1, width)
    selections = _threshold_rows(scores.reshape(-1), threshold, width)
    audit = dataset_audit(calibration, episodes, _source_spec(spec))
    audit["calibration_labels_sha256"] = hashlib.sha256(
        labels.contiguous().numpy().tobytes()
    ).hexdigest()
    return {
        "factor_audit": _factor_audit(sources),
        "dataset_audit": audit,
        "embedding_audit": {
            "calibration": calibration_embedding,
            "candidates": candidate_embedding,
        },
        "fit_audit": fit_audit,
        "selection_threshold": threshold,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "metrics": selection_metrics(episodes, selections, scores, spec),
    }


def run_seed(
    seed: int, encoder: FrozenSentenceEncoder, checkpoint_dir: Path, spec: dict
) -> dict:
    peers = [value for value in spec["seeds"] if value != seed]
    if len(peers) != 1:
        raise ValueError("write-mechanism experiment requires exactly two distinct seeds")
    peer = peers[0]
    return {
        "seed": seed,
        "peer_seed": peer,
        "arms": {
            name: run_arm(seed, peer, name, encoder, checkpoint_dir, spec)
            for name in spec["arms"]
        },
    }


def run(checkpoint_dir: Path, spec: dict = WRITE_MECHANISM_SPEC) -> dict:
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
        "encoder_audit": encoder.audit(),
        "seeds": [run_seed(seed, encoder, checkpoint_dir, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("measurement/write_mechanism_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/gate-write-mechanism1")
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
