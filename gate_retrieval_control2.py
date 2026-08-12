#!/usr/bin/env python3
"""GATE-RETRIEVAL-CONTROL-2: split topic-address/content retrieval control."""
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
from gate2 import _scores, _token, build_calibration, build_evaluation
from gate_control1 import FrozenSentenceEncoder
from gate_retrieval_control1 import dataset_audit
from memory_gate import fit_canonical_ridge
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC
from measurement.split_retrieval_control_registry import (
    SPLIT_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _rank(indices: list[int], scores: torch.Tensor) -> list[int]:
    values = scores.tolist()
    if not indices or any(index < 0 or index >= len(values) for index in indices):
        raise ValueError("retrieval indices must select a non-empty score vector")
    if any(not math.isfinite(values[index]) for index in indices):
        raise ValueError("retrieval scores must be finite")
    return sorted(indices, key=lambda index: (-values[index], index))


def _candidate_topic(episode: dict, seed: int, episode_index: int, row: dict) -> str:
    segment = row["topic_segment"]
    if segment == episode["fact_position"] // 2:
        return episode["subject"]
    return _token("evaluation", f"보조주제{segment}", seed, episode_index)


def _topic_features(episodes: list[dict], seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    query_rows = []
    candidate_rows = []
    topics = []
    for episode_index, episode in enumerate(episodes):
        query_rows.append(text_to_vector(episode["subject"]).float())
        row_topics = [
            _candidate_topic(episode, seed, episode_index, row)
            for row in episode["candidates"]
        ]
        topics.extend(row_topics)
        candidate_rows.extend(text_to_vector(topic).float() for topic in row_topics)
    queries = F.normalize(torch.stack(query_rows), p=2, dim=1)
    candidates = F.normalize(torch.stack(candidate_rows), p=2, dim=1)
    return queries, candidates, {
        "query_topics": len(query_rows),
        "candidate_topics": len(topics),
        "candidate_topic_unique": len(set(topics)),
        "query_address_sha256": hashlib.sha256(queries.contiguous().numpy().tobytes()).hexdigest(),
        "candidate_address_sha256": hashlib.sha256(
            candidates.contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def _address_pools(query_addresses: torch.Tensor, candidate_addresses: torch.Tensor,
                   width: int, pool_size: int) -> tuple[list[list[int]], torch.Tensor]:
    if candidate_addresses.shape[0] != query_addresses.shape[0] * width:
        raise ValueError("topic address rows do not match the registered episode width")
    grouped = candidate_addresses.reshape(query_addresses.shape[0], width, -1)
    scores = torch.einsum("ed,ewd->ew", query_addresses, grouped)
    pools = [_rank(list(range(width)), row)[:pool_size] for row in scores]
    return pools, scores


def _content_rankings(pools: list[list[int]], content_scores: torch.Tensor) -> list[list[int]]:
    if content_scores.dim() != 2 or len(pools) != content_scores.shape[0]:
        raise ValueError("content scores must match every address pool")
    return [_rank(pool, scores) for pool, scores in zip(pools, content_scores)]


def _metrics(
    episodes: list[dict],
    rankings: list[list[int]],
    scores: torch.Tensor,
    *,
    fact_kinds: list[str] | None = None,
    fact_positions: list[int] | None = None,
) -> dict:
    if len(rankings) != len(episodes) or scores.shape[0] != len(episodes):
        raise ValueError("retrieval outputs must cover every episode")
    fact_kinds = fact_kinds or REALISTIC_MEMORY_WRITE_SPEC["fact_kinds"]
    fact_positions = fact_positions or REALISTIC_MEMORY_WRITE_SPEC["fact_positions"]
    top1 = top3 = 0
    kinds = {kind: [0, 0] for kind in fact_kinds}
    positions = {
        str(position): [0, 0] for position in fact_positions
    }
    ranks = []
    margins = []
    digest = []
    for episode, ranking, row_scores in zip(episodes, rankings, scores):
        if len(set(ranking)) != len(ranking) or any(
            index < 0 or index >= len(episode["candidates"]) for index in ranking
        ):
            raise ValueError("retrieval ranking contains an invalid candidate index")
        fact = episode["fact_position"]
        hit1 = bool(ranking and ranking[0] == fact)
        hit3 = fact in ranking[:3]
        top1 += hit1
        top3 += hit3
        kinds[episode["kind"]][0] += hit1
        kinds[episode["kind"]][1] += 1
        position = str(fact)
        positions[position][0] += hit1
        positions[position][1] += 1
        if fact in ranking:
            ranks.append(ranking.index(fact) + 1)
            competitors = [row_scores[index].item() for index in ranking if index != fact]
            margins.append(
                1.0 if not competitors else row_scores[fact].item() - max(competitors)
            )
        digest.append(",".join(map(str, ranking)))
    total = len(episodes)
    return {
        "recall_at_1": top1 / total,
        "recall_at_3": top3 / total,
        "per_kind_recall_at_1": {
            name: hits / count for name, (hits, count) in kinds.items()
        },
        "per_position_recall_at_1": {
            name: hits / count for name, (hits, count) in positions.items()
        },
        "mean_fact_rank": sum(ranks) / len(ranks) if ranks else 0.0,
        "mean_fact_margin": sum(margins) / len(margins) if margins else 0.0,
        "rankings_sha256": hashlib.sha256("\n".join(digest).encode()).hexdigest(),
    }


def run_seed(seed: int, encoder: FrozenSentenceEncoder,
             spec: dict = SPLIT_RETRIEVAL_CONTROL_SPEC) -> dict:
    episodes = build_evaluation(seed, REALISTIC_MEMORY_WRITE_SPEC)
    calibration = build_calibration(seed, REALISTIC_MEMORY_WRITE_SPEC)
    candidate_rows = [row for episode in episodes for row in episode["candidates"]]
    query_rows = [{"role": "user", "text": episode["query"]} for episode in episodes]
    calibration_features, calibration_audit = encoder.encode_rows(calibration)
    candidate_all, candidate_audit = encoder.encode_rows(candidate_rows)
    query_all, query_audit = encoder.encode_rows(query_rows)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, _threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"],
    )
    width = spec["candidates_per_episode"]
    content_scores = _scores(candidate_all, weight, bias).reshape(-1, width)
    semantic_query = query_all[:, :spec["encoder"]["embedding_dim"]]
    semantic_candidates = candidate_all[:, :spec["encoder"]["embedding_dim"]].reshape(
        -1, width, spec["encoder"]["embedding_dim"]
    )
    content_only_scores = torch.einsum("ed,ewd->ew", semantic_query, semantic_candidates)
    content_only_rankings = [
        _rank(list(range(width)), row) for row in content_only_scores
    ]

    query_addresses, candidate_addresses, address_audit = _topic_features(episodes, seed)
    pool_size = spec["retrieval"]["address_pool"]
    normal_pools, address_scores = _address_pools(
        query_addresses, candidate_addresses, width, pool_size,
    )
    shuffled_pools, shuffled_address_scores = _address_pools(
        torch.roll(query_addresses, shifts=-1, dims=0), candidate_addresses, width, pool_size,
    )
    split_rankings = _content_rankings(normal_pools, content_scores)
    topic_only_rankings = normal_pools
    shuffled_topic_rankings = _content_rankings(shuffled_pools, content_scores)
    shuffled_content_scores = torch.roll(content_scores, shifts=-1, dims=0)
    shuffled_content_rankings = _content_rankings(normal_pools, shuffled_content_scores)
    oracle_rankings = [[episode["fact_position"]] for episode in episodes]
    no_memory_rankings = [[] for _ in episodes]
    zero_scores = torch.zeros_like(content_scores)
    arms = {
        "split_topic_content": (split_rankings, content_scores),
        "topic_only": (topic_only_rankings, address_scores),
        "content_only": (content_only_rankings, content_only_scores),
        "shuffled_topic": (shuffled_topic_rankings, shuffled_address_scores),
        "shuffled_content": (shuffled_content_rankings, shuffled_content_scores),
        "oracle_memory": (oracle_rankings, zero_scores),
        "no_memory": (no_memory_rankings, zero_scores),
    }
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(episodes, REALISTIC_MEMORY_WRITE_SPEC),
        "encoder_audit": encoder.audit(),
        "embedding_audit": {
            "calibration": calibration_audit,
            "queries": query_audit,
            "candidates": candidate_audit,
        },
        "fit_audit": fit_audit,
        "address_audit": address_audit,
        "score_audit": {
            "content_scores_sha256": hashlib.sha256(
                content_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
            "address_scores_sha256": hashlib.sha256(
                address_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
            "normal_pool_fact_coverage": sum(
                episode["fact_position"] in pool for episode, pool in zip(episodes, normal_pools)
            ) / len(episodes),
        },
        "arms": {
            name: _metrics(episodes, rankings, scores)
            for name, (rankings, scores) in arms.items()
        },
    }


def run(spec: dict = SPLIT_RETRIEVAL_CONTROL_SPEC) -> dict:
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
        default=Path("measurement/split_retrieval_control_results.json"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run())


if __name__ == "__main__":
    main()
