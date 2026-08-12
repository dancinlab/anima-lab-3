#!/usr/bin/env python3
"""GATE-3: integrated semantic dialogue-memory write and retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate1 import _matched_random
from gate2 import _scores, build_calibration, dataset_audit
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_control2 import _selection_digest, _threshold_rows, match_ranked_counts
from gate_retrieval_control2 import _metrics as retrieval_metrics
from gate_retrieval_control2 import _rank
from gate_retrieval_control3 import build_balanced_evaluation, _topic_features
from memory_gate import fit_canonical_ridge
from measurement.integrated_dialogue_memory_registry import (
    INTEGRATED_DIALOGUE_MEMORY_SPEC,
    spec_sha256,
)
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def stored_rankings(
    selections: list[list[bool]],
    address_scores: torch.Tensor,
    content_scores: torch.Tensor,
    pool_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Rank only candidates admitted by the registered write selector."""
    if (
        address_scores.dim() != 2
        or content_scores.shape != address_scores.shape
        or len(selections) != address_scores.shape[0]
        or pool_size <= 0
    ):
        raise ValueError("stored retrieval inputs have incompatible shapes")
    width = address_scores.shape[1]
    if any(len(row) != width for row in selections):
        raise ValueError("stored selections must match the candidate width")
    if not torch.isfinite(address_scores).all() or not torch.isfinite(content_scores).all():
        raise ValueError("stored retrieval scores must be finite")
    pools = []
    rankings = []
    for selected, address_row, content_row in zip(
        selections, address_scores, content_scores
    ):
        stored = [index for index, keep in enumerate(selected) if bool(keep)]
        pool = _rank(stored, address_row)[:pool_size] if stored else []
        ranking = _rank(pool, content_row) if pool else []
        pools.append(pool)
        rankings.append(ranking)
    return rankings, pools


def _integrated_metrics(
    episodes: list[dict],
    selections: list[list[bool]],
    rankings: list[list[int]],
    content_scores: torch.Tensor,
    spec: dict,
) -> dict:
    base = retrieval_metrics(
        episodes,
        rankings,
        content_scores,
        fact_kinds=spec["fact_kinds"],
        fact_positions=spec["fact_positions"],
    )
    important_kept = distractors_kept = total_kept = stored_hits = stored_total = 0
    distractor_hits = {kind: [0, 0] for kind in spec["distractor_kinds"]}
    for episode, selected, ranking in zip(episodes, selections, rankings):
        fact = episode["fact_position"]
        kept = bool(selected[fact])
        important_kept += kept
        total_kept += sum(bool(value) for value in selected)
        if kept:
            stored_total += 1
            stored_hits += bool(ranking and ranking[0] == fact)
        for row, keep in zip(episode["candidates"], selected):
            if not row["important"]:
                distractors_kept += bool(keep)
                distractor_hits[row["kind"]][0] += bool(keep)
                distractor_hits[row["kind"]][1] += 1
    total = len(episodes)
    distractor_total = total * (spec["candidates_per_episode"] - 1)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / distractor_total,
        "search_size_ratio": total_kept / (total * spec["candidates_per_episode"]),
        "stored": total_kept,
        "recall_at_1": base["recall_at_1"],
        "recall_at_3": base["recall_at_3"],
        "stored_fact_recall_at_1": stored_hits / stored_total if stored_total else 0.0,
        "per_kind_recall_at_1": base["per_kind_recall_at_1"],
        "per_position_recall_at_1": base["per_position_recall_at_1"],
        "per_distractor_storage_rate": {
            kind: hits / count for kind, (hits, count) in distractor_hits.items()
        },
        "mean_fact_rank": base["mean_fact_rank"],
        "mean_fact_margin": base["mean_fact_margin"],
        "rankings_sha256": base["rankings_sha256"],
    }


def _search_audit(
    selections: dict[str, list[list[bool]]],
    rankings: dict[str, list[list[int]]],
    pools: dict[str, list[list[int]]],
) -> dict:
    subset = empty_exact = pool_subset = True
    pool_digests = {}
    for name in selections:
        selected_rows = selections[name]
        rank_rows = rankings[name]
        pool_rows = pools[name]
        for selected, ranking, pool in zip(selected_rows, rank_rows, pool_rows):
            stored = {index for index, keep in enumerate(selected) if keep}
            subset &= set(ranking).issubset(stored)
            pool_subset &= set(pool).issubset(stored)
            empty_exact &= bool(ranking) == bool(stored)
        body = json.dumps(pool_rows, separators=(",", ":"))
        pool_digests[name] = hashlib.sha256(body.encode()).hexdigest()
    return {
        "rankings_subset_of_stored": bool(subset),
        "pools_subset_of_stored": bool(pool_subset),
        "empty_exact_when_no_stored_candidates": bool(empty_exact),
        "pool_sha256": pool_digests,
    }


def run_seed(
    seed: int,
    encoder: FrozenSentenceEncoder,
    checkpoint_dir: Path,
    spec: dict = INTEGRATED_DIALOGUE_MEMORY_SPEC,
) -> dict:
    source_spec = dict(REALISTIC_MEMORY_WRITE_SPEC)
    source_spec["evaluation_episodes"] = spec["evaluation_episodes"]
    source_spec["fact_positions"] = list(spec["fact_positions"])
    calibration = build_calibration(seed, REALISTIC_MEMORY_WRITE_SPEC)
    episodes = build_balanced_evaluation(seed, spec)
    candidates = [row for episode in episodes for row in episode["candidates"]]
    calibration_features, calibration_audit = encoder.encode_rows(calibration)
    candidate_features, candidate_audit = encoder.encode_rows(candidates)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"]
    )
    generator = torch.Generator().manual_seed(
        seed + REALISTIC_MEMORY_WRITE_SPEC["shuffle_seed_offset"]
    )
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, shuffled_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=spec["ridge"]
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_integrated_memory_gate.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_shuffled_integrated_memory_gate.json"
    _atomic_json(checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"]))
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(fake_weight, fake_bias, fake_threshold, spec["encoder"]),
    )

    width = spec["candidates_per_episode"]
    content_scores = _scores(candidate_features, weight, bias).reshape(-1, width)
    fake_scores = _scores(candidate_features, fake_weight, fake_bias).reshape(-1, width)
    semantic = _threshold_rows(content_scores.reshape(-1), threshold, width)
    matched_shuffled = match_ranked_counts(fake_scores.reshape(-1), semantic, width)
    matched_random = [
        _matched_random(
            selection,
            seed + REALISTIC_MEMORY_WRITE_SPEC["random_seed_offset"] + episode_index,
        )
        for episode_index, selection in enumerate(semantic)
    ]
    selections = {
        "semantic_integrated": semantic,
        "store_all_integrated": [[True] * width for _ in episodes],
        "oracle_integrated": [
            [bool(row["important"]) for row in episode["candidates"]]
            for episode in episodes
        ],
        "matched_random_integrated": matched_random,
        "matched_shuffled_integrated": matched_shuffled,
        "no_memory": [[False] * width for _ in episodes],
    }

    query_addresses, candidate_addresses, address_audit = _topic_features(
        episodes, seed, spec
    )
    grouped_addresses = candidate_addresses.reshape(len(episodes), width, -1)
    address_scores = torch.einsum("ed,ewd->ew", query_addresses, grouped_addresses)
    rankings = {}
    pools = {}
    arms = {}
    for name, selected in selections.items():
        rank_rows, pool_rows = stored_rankings(
            selected, address_scores, content_scores, spec["retrieval"]["address_pool"]
        )
        rankings[name] = rank_rows
        pools[name] = pool_rows
        arms[name] = _integrated_metrics(
            episodes, selected, rank_rows, content_scores, spec
        )

    counts = [sum(row) for row in semantic]
    random_counts = [sum(row) for row in matched_random]
    shuffled_counts = [sum(row) for row in matched_shuffled]
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, episodes, source_spec),
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
            }
        },
        "encoder_audit": encoder.audit(),
        "embedding_audit": {
            "calibration": calibration_audit,
            "candidates": candidate_audit,
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
            "method": spec["selection"]["matching"],
            "semantic_counts": counts,
            "matched_random_counts": random_counts,
            "matched_shuffled_counts": shuffled_counts,
            "semantic_selection_sha256": _selection_digest(semantic),
            "matched_random_selection_sha256": _selection_digest(matched_random),
            "matched_shuffled_selection_sha256": _selection_digest(matched_shuffled),
            "fake_scores_sha256": hashlib.sha256(
                fake_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
        },
        "address_audit": {
            **address_audit,
            "address_scores_sha256": hashlib.sha256(
                address_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
        },
        "content_scores_sha256": hashlib.sha256(
            content_scores.contiguous().numpy().tobytes()
        ).hexdigest(),
        "search_audit": _search_audit(selections, rankings, pools),
        "arms": arms,
    }


def run(
    checkpoint_dir: Path,
    spec: dict = INTEGRATED_DIALOGUE_MEMORY_SPEC,
) -> dict:
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
        "seeds": [
            run_seed(seed, encoder, checkpoint_dir, spec) for seed in spec["seeds"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("measurement/integrated_dialogue_memory_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/gate3")
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
