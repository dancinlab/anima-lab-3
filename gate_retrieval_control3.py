#!/usr/bin/env python3
"""GATE-RETRIEVAL-CONTROL-3: balanced order and episode-address control."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from gate2 import _scores, build_calibration, build_evaluation
from gate_control1 import FrozenSentenceEncoder
from gate_retrieval_control1 import dataset_audit
from gate_retrieval_control2 import _address_pools, _content_rankings, _metrics, _rank
from memory_gate import fit_canonical_ridge
from measurement.balanced_retrieval_control_registry import (
    BALANCED_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_balanced_evaluation(seed: int, spec: dict = BALANCED_RETRIEVAL_CONTROL_SPEC) -> list[dict]:
    source_spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    source_spec["evaluation_episodes"] = spec["evaluation_episodes"]
    episodes = build_evaluation(seed, source_spec)
    width = spec["candidates_per_episode"]
    positions = spec["fact_positions"]
    if width != len(positions) or len(episodes) % len(positions):
        raise ValueError("registered episodes must balance every candidate position")
    for episode_index, episode in enumerate(episodes):
        target = positions[(episode_index // len(spec["fact_kinds"])) % len(positions)]
        current = episode["fact_position"]
        if target != current:
            episode["candidates"][current], episode["candidates"][target] = (
                episode["candidates"][target], episode["candidates"][current]
            )
        for position, row in enumerate(episode["candidates"]):
            row["position"] = position
            row["topic_segment"] = position // 2
        episode["fact_position"] = target
    return episodes


def _episode_addresses(seed: int, episode_count: int, segments: int,
                       dim: int, offset: int) -> torch.Tensor:
    if episode_count <= 0 or segments <= 0 or dim <= 0:
        raise ValueError("episode address dimensions must be positive")
    generator = torch.Generator().manual_seed(seed + offset)
    addresses = torch.randn(episode_count, segments, dim, generator=generator)
    return F.normalize(addresses, p=2, dim=2)


def _topic_features(episodes: list[dict], seed: int, spec: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    width = spec["candidates_per_episode"]
    segments = width // 2
    addresses = _episode_addresses(
        seed, len(episodes), segments, spec["retrieval"]["address_dim"],
        spec["retrieval"]["address_seed_offset"],
    )
    queries = torch.stack([
        addresses[index, episode["fact_position"] // 2]
        for index, episode in enumerate(episodes)
    ])
    candidates = torch.stack([
        addresses[index, row["topic_segment"]]
        for index, episode in enumerate(episodes)
        for row in episode["candidates"]
    ])
    flat = addresses.reshape(-1, addresses.shape[-1])
    unique = len({row.contiguous().numpy().tobytes() for row in flat})
    return queries, candidates, {
        "query_topics": len(queries),
        "candidate_topics": len(candidates),
        "episode_segment_addresses": flat.shape[0],
        "unique_episode_segment_addresses": unique,
        "query_address_sha256": hashlib.sha256(queries.contiguous().numpy().tobytes()).hexdigest(),
        "candidate_address_sha256": hashlib.sha256(candidates.contiguous().numpy().tobytes()).hexdigest(),
    }


def _seed_measurements(seed: int, encoder: FrozenSentenceEncoder,
                       spec: dict = BALANCED_RETRIEVAL_CONTROL_SPEC) -> tuple[dict, dict]:
    episodes = build_balanced_evaluation(seed, spec)
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
    semantic_width = spec["encoder"]["embedding_dim"]
    semantic_query = query_all[:, :semantic_width]
    semantic_candidates = candidate_all[:, :semantic_width].reshape(-1, width, semantic_width)
    content_only_scores = torch.einsum("ed,ewd->ew", semantic_query, semantic_candidates)
    content_only_rankings = [_rank(list(range(width)), row) for row in content_only_scores]

    query_addresses, candidate_addresses, address_audit = _topic_features(episodes, seed, spec)
    pool_size = spec["retrieval"]["address_pool"]
    normal_pools, address_scores = _address_pools(
        query_addresses, candidate_addresses, width, pool_size,
    )
    shuffled_pools, shuffled_address_scores = _address_pools(
        torch.roll(query_addresses, shifts=-1, dims=0), candidate_addresses, width, pool_size,
    )
    split_rankings = _content_rankings(normal_pools, content_scores)
    shuffled_content_scores = torch.roll(content_scores, shifts=-1, dims=0)
    arms = {
        "balanced_split": (split_rankings, content_scores),
        "topic_only": (normal_pools, address_scores),
        "content_only": (content_only_rankings, content_only_scores),
        "shuffled_episode_address": (
            _content_rankings(shuffled_pools, content_scores), shuffled_address_scores,
        ),
        "shuffled_content": (
            _content_rankings(normal_pools, shuffled_content_scores), shuffled_content_scores,
        ),
        "oracle_memory": (
            [[episode["fact_position"]] for episode in episodes], torch.zeros_like(content_scores),
        ),
        "no_memory": ([[] for _ in episodes], torch.zeros_like(content_scores)),
    }
    result = {
        "seed": seed,
        "dataset_audit": dataset_audit(episodes, spec),
        "balance_audit": {
            "fact_kind_side_counts": {
                kind: {
                    str(side): sum(
                        episode["kind"] == kind
                        and episode["fact_position"] % 2 == side
                        for episode in episodes
                    )
                    for side in (0, 1)
                }
                for kind in spec["fact_kinds"]
            },
        },
        "encoder_audit": encoder.audit(),
        "embedding_audit": {
            "calibration": calibration_audit,
            "queries": query_audit,
            "candidates": candidate_audit,
        },
        "fit_audit": fit_audit,
        "address_audit": address_audit,
        "score_audit": {
            "content_scores_sha256": hashlib.sha256(content_scores.contiguous().numpy().tobytes()).hexdigest(),
            "address_scores_sha256": hashlib.sha256(address_scores.contiguous().numpy().tobytes()).hexdigest(),
            "normal_pool_fact_coverage": sum(
                episode["fact_position"] in pool for episode, pool in zip(episodes, normal_pools)
            ) / len(episodes),
        },
        "arms": {
            name: _metrics(
                episodes,
                rankings,
                scores,
                fact_kinds=spec["fact_kinds"],
                fact_positions=spec["fact_positions"],
            )
            for name, (rankings, scores) in arms.items()
        },
    }
    state = {
        "episodes": episodes,
        "content_scores": content_scores,
        "normal_pools": normal_pools,
    }
    return result, state


def run_seed(seed: int, encoder: FrozenSentenceEncoder,
             spec: dict = BALANCED_RETRIEVAL_CONTROL_SPEC) -> dict:
    result, _state = _seed_measurements(seed, encoder, spec)
    return result


def run(spec: dict = BALANCED_RETRIEVAL_CONTROL_SPEC) -> dict:
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
        default=Path("measurement/balanced_retrieval_control_results.json"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run())


if __name__ == "__main__":
    main()
