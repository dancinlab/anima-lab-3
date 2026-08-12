#!/usr/bin/env python3
"""GATE-RETRIEVAL-CONTROL-1: frozen semantic retrieval positive control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from creativity_classifier import text_to_vector
from gate2 import build_evaluation
from gate_control1 import FrozenSentenceEncoder
from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    template_sha256,
)
from measurement.semantic_retrieval_control_registry import (
    SEMANTIC_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def dataset_audit(episodes: list[dict], spec: dict) -> dict:
    rows = [row for episode in episodes for row in episode["candidates"]]
    texts = [row["text"] for row in rows]
    queries = [episode["query"] for episode in episodes]
    return {
        "evaluation_episodes": len(episodes),
        "evaluation_candidates": len(rows),
        "evaluation_unique": len(set(texts)),
        "query_unique": len(set(queries)),
        "query_candidate_overlap": len(set(queries) & set(texts)),
        "fact_counts": {
            kind: sum(episode["kind"] == kind for episode in episodes)
            for kind in spec["fact_kinds"]
        },
        "fact_position_counts": {
            str(position): sum(episode["fact_position"] == position for episode in episodes)
            for position in spec["fact_positions"]
        },
        "topic_switch_counts": {
            str(count): sum(episode["topic_switches"] == count for episode in episodes)
            for count in {episode["topic_switches"] for episode in episodes}
        },
        "template_sha256": template_sha256(REALISTIC_MEMORY_WRITE_SPEC),
        "evaluation_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "query_sha256": hashlib.sha256("\n".join(queries).encode()).hexdigest(),
    }


def _rank(scores: torch.Tensor) -> list[int]:
    values = scores.tolist()
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("retrieval scores must be a non-empty finite vector")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def _semantic_rankings(query_features: torch.Tensor, candidate_features: torch.Tensor,
                       width: int) -> tuple[list[list[int]], list[list[float]]]:
    if candidate_features.shape[0] != query_features.shape[0] * width:
        raise ValueError("semantic candidates do not match the registered episode width")
    grouped = candidate_features.reshape(query_features.shape[0], width, -1)
    scores = torch.einsum("ed,ewd->ew", query_features, grouped)
    return [_rank(row) for row in scores], scores.tolist()


def _character_rankings(episodes: list[dict]) -> tuple[list[list[int]], list[list[float]]]:
    rankings = []
    all_scores = []
    for episode in episodes:
        query = text_to_vector(episode["query"]).float()
        candidates = torch.stack([
            text_to_vector(row["text"]).float() for row in episode["candidates"]
        ])
        scores = F.cosine_similarity(query.unsqueeze(0), candidates, dim=1)
        rankings.append(_rank(scores))
        all_scores.append(scores.tolist())
    return rankings, all_scores


def _metrics(episodes: list[dict], rankings: list[list[int]], scores: list[list[float]],
             top_k: int) -> dict:
    if len(rankings) != len(episodes) or len(scores) != len(episodes):
        raise ValueError("retrieval outputs must cover every episode")
    hits = 0
    kind_hits = {kind: [0, 0] for kind in REALISTIC_MEMORY_WRITE_SPEC["fact_kinds"]}
    position_hits = {
        str(position): [0, 0] for position in REALISTIC_MEMORY_WRITE_SPEC["fact_positions"]
    }
    fact_ranks = []
    margins = []
    digest_rows = []
    for episode, ranking, row_scores in zip(episodes, rankings, scores):
        fact = episode["fact_position"]
        if ranking:
            if (
                len(set(ranking)) != len(ranking)
                or any(index < 0 or index >= len(episode["candidates"]) for index in ranking)
            ):
                raise ValueError("retrieval ranking must contain unique candidate indices")
            if len(row_scores) != len(ranking):
                raise ValueError("retrieval scores must match ranking width")
            hit = fact in ranking[:top_k]
            if fact in ranking:
                rank = ranking.index(fact) + 1
                fact_ranks.append(rank)
                if len(ranking) == 1:
                    margins.append(1.0)
                else:
                    score_by_candidate = (
                        dict(enumerate(row_scores))
                        if len(row_scores) == len(episode["candidates"])
                        else dict(zip(ranking, row_scores))
                    )
                    margins.append(score_by_candidate[fact] - max(
                        score for index, score in score_by_candidate.items() if index != fact
                    ))
            digest_rows.append(",".join(map(str, ranking)))
        else:
            if row_scores:
                raise ValueError("no-memory rankings cannot carry scores")
            hit = False
            digest_rows.append("")
        hits += int(hit)
        kind_hits[episode["kind"]][0] += int(hit)
        kind_hits[episode["kind"]][1] += 1
        position = str(fact)
        position_hits[position][0] += int(hit)
        position_hits[position][1] += 1
    count = len(episodes)
    return {
        "recall_at_3": hits / count,
        "per_kind_recall": {
            kind: matched / total for kind, (matched, total) in kind_hits.items()
        },
        "per_position_recall": {
            position: matched / total
            for position, (matched, total) in position_hits.items()
        },
        "mean_fact_rank": sum(fact_ranks) / len(fact_ranks) if fact_ranks else 0.0,
        "mean_fact_margin": sum(margins) / len(margins) if margins else 0.0,
        "rankings_sha256": hashlib.sha256("\n".join(digest_rows).encode()).hexdigest(),
    }


def run_seed(seed: int, encoder: FrozenSentenceEncoder,
             spec: dict = SEMANTIC_RETRIEVAL_CONTROL_SPEC) -> dict:
    episodes = build_evaluation(seed, REALISTIC_MEMORY_WRITE_SPEC)
    query_rows = [{"role": "user", "text": episode["query"]} for episode in episodes]
    candidate_rows = [row for episode in episodes for row in episode["candidates"]]
    query_all, query_audit = encoder.encode_rows(query_rows)
    candidate_all, candidate_audit = encoder.encode_rows(candidate_rows)
    width = spec["retrieval"]["feature_dim"]
    query_features = query_all[:, :width]
    candidate_features = candidate_all[:, :width]
    semantic_rankings, semantic_scores = _semantic_rankings(
        query_features, candidate_features, spec["candidates_per_episode"],
    )
    shifted = torch.roll(query_features, shifts=-1, dims=0)
    shuffled_rankings, shuffled_scores = _semantic_rankings(
        shifted, candidate_features, spec["candidates_per_episode"],
    )
    character_rankings, character_scores = _character_rankings(episodes)
    oracle_rankings = [[episode["fact_position"]] for episode in episodes]
    oracle_scores = [[1.0] for _ in episodes]
    no_memory_rankings = [[] for _ in episodes]
    no_memory_scores = [[] for _ in episodes]
    arms = {
        "semantic_retrieval": (semantic_rankings, semantic_scores),
        "character_retrieval": (character_rankings, character_scores),
        "oracle_memory": (oracle_rankings, oracle_scores),
        "shuffled_query": (shuffled_rankings, shuffled_scores),
        "no_memory": (no_memory_rankings, no_memory_scores),
    }
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(episodes, REALISTIC_MEMORY_WRITE_SPEC),
        "encoder_audit": encoder.audit(),
        "embedding_audit": {"queries": query_audit, "candidates": candidate_audit},
        "feature_audit": {
            "feature_dim": width,
            "query_features_sha256": hashlib.sha256(
                query_features.contiguous().numpy().tobytes()
            ).hexdigest(),
            "candidate_features_sha256": hashlib.sha256(
                candidate_features.contiguous().numpy().tobytes()
            ).hexdigest(),
        },
        "arms": {
            name: _metrics(episodes, rankings, scores, spec["top_k"])
            for name, (rankings, scores) in arms.items()
        },
    }


def run(spec: dict = SEMANTIC_RETRIEVAL_CONTROL_SPEC) -> dict:
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
        "seeds": [run_seed(seed, encoder, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_retrieval_control_results.json"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run())


if __name__ == "__main__":
    main()
