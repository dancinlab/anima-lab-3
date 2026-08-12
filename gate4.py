#!/usr/bin/env python3
"""GATE-4: balanced natural write selection plus stored-candidate retrieval."""
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
from gate2 import _scores
from gate3 import _search_audit, stored_rankings
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_control2 import _selection_digest, _threshold_rows, match_ranked_counts
from gate_retrieval_control2 import _metrics as retrieval_metrics
from gate_retrieval_control3 import _topic_features
from gate_write_control1 import (
    build_balanced_calibration,
    build_balanced_evaluation,
    dataset_audit,
)
from memory_gate import fit_canonical_ridge
from measurement.balanced_integrated_dialogue_registry import (
    BALANCED_INTEGRATED_DIALOGUE_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _text_digest(episodes: list[dict]) -> str:
    texts = [row["text"] for episode in episodes for row in episode["candidates"]]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


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
    kind_storage = {kind: [0, 0] for kind in spec["fact_kinds"]}
    template_storage = {
        f"{kind}:{index}": [0, 0]
        for kind in spec["fact_kinds"]
        for index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
    }
    template_counts = {
        f"{kind}:{index}": [0, 0]
        for kind in spec["fact_kinds"]
        for index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
    }
    distractor_counts = {kind: [0, 0] for kind in spec["distractor_kinds"]}
    important_kept = distractors_kept = total_kept = stored_hits = stored_total = 0
    for episode, selected, ranking in zip(episodes, selections, rankings):
        fact = episode["fact_position"]
        kept = bool(selected[fact])
        correct = bool(ranking and ranking[0] == fact)
        important_kept += kept
        total_kept += sum(bool(value) for value in selected)
        template_key = f'{episode["kind"]}:{episode["fact_template_index"]}'
        kind_storage[episode["kind"]][0] += kept
        kind_storage[episode["kind"]][1] += 1
        template_storage[template_key][0] += kept
        template_storage[template_key][1] += 1
        template_counts[template_key][0] += correct
        template_counts[template_key][1] += 1
        if kept:
            stored_total += 1
            stored_hits += correct
        for row, keep in zip(episode["candidates"], selected):
            if row["important"]:
                continue
            distractors_kept += bool(keep)
            distractor_counts[row["kind"]][0] += bool(keep)
            distractor_counts[row["kind"]][1] += 1
    total = len(episodes)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / (total * (spec["candidates_per_episode"] - 1)),
        "search_size_ratio": total_kept / (total * spec["candidates_per_episode"]),
        "stored": total_kept,
        "recall_at_1": base["recall_at_1"],
        "recall_at_3": base["recall_at_3"],
        "stored_fact_recall_at_1": stored_hits / stored_total if stored_total else 0.0,
        "per_kind_storage_rate": {
            kind: hits / count for kind, (hits, count) in kind_storage.items()
        },
        "per_template_storage_rate": {
            key: hits / count for key, (hits, count) in template_storage.items()
        },
        "per_kind_recall_at_1": base["per_kind_recall_at_1"],
        "per_template_recall_at_1": {
            key: hits / count for key, (hits, count) in template_counts.items()
        },
        "per_position_recall_at_1": base["per_position_recall_at_1"],
        "per_distractor_storage_rate": {
            kind: hits / count for kind, (hits, count) in distractor_counts.items()
        },
        "mean_fact_rank": base["mean_fact_rank"],
        "mean_fact_margin": base["mean_fact_margin"],
        "rankings_sha256": base["rankings_sha256"],
    }


def run_seed(
    seed: int,
    encoder: FrozenSentenceEncoder,
    checkpoint_dir: Path,
    spec: dict = BALANCED_INTEGRATED_DIALOGUE_SPEC,
) -> dict:
    calibration = build_balanced_calibration(seed)
    evaluations = {
        replicate: build_balanced_evaluation(seed, replicate)
        for replicate in spec["replicates"]
    }
    calibration_features, calibration_audit = encoder.encode_rows(calibration)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"]
    )
    generator = torch.Generator().manual_seed(seed + spec["shuffle_seed_offset"])
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, fake_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=spec["ridge"]
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_balanced_integrated_gate.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_balanced_integrated_fake.json"
    _atomic_json(checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"]))
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(fake_weight, fake_bias, fake_threshold, spec["encoder"]),
    )

    replicate_rows = []
    width = spec["candidates_per_episode"]
    for replicate_index, (replicate, episodes) in enumerate(evaluations.items()):
        before_digest = _text_digest(episodes)
        candidates = [row for episode in episodes for row in episode["candidates"]]
        candidate_features, candidate_audit = encoder.encode_rows(candidates)
        content_scores = _scores(candidate_features, weight, bias).reshape(-1, width)
        fake_scores = _scores(candidate_features, fake_weight, fake_bias).reshape(-1, width)
        semantic = _threshold_rows(content_scores.reshape(-1), threshold, width)
        matched_shuffled = match_ranked_counts(fake_scores.reshape(-1), semantic, width)
        matched_random = [
            _matched_random(
                selected,
                seed + spec["random_seed_offset"]
                + replicate_index * spec["evaluation_episodes"] + episode_index,
            )
            for episode_index, selected in enumerate(semantic)
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
        address_seed = seed + replicate_index * spec["retrieval"]["replicate_address_stride"]
        query_addresses, candidate_addresses, address_audit = _topic_features(
            episodes, address_seed, spec
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
        after_digest = _text_digest(episodes)
        replicate_rows.append({
            "name": replicate,
            "embedding_audit": candidate_audit,
            "matching_audit": {
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
            "preservation_audit": {
                "raw_candidate_count_before": len(candidates),
                "raw_candidate_count_after": sum(len(row["candidates"]) for row in episodes),
                "raw_text_sha256_before": before_digest,
                "raw_text_sha256_after": after_digest,
                "long_term_selection_is_separate": True,
            },
            "search_audit": _search_audit(selections, rankings, pools),
            "arms": arms,
        })
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, evaluations, spec),
        "encoder_audit": encoder.audit(),
        "calibration_embedding_audit": calibration_audit,
        "fit_audit": fit_audit,
        "fake_fit_audit": fake_fit_audit,
        "selection_threshold": threshold,
        "fake_selection_threshold": fake_threshold,
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
        "replicates": replicate_rows,
    }


def run(
    checkpoint_dir: Path,
    spec: dict = BALANCED_INTEGRATED_DIALOGUE_SPEC,
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
        "seeds": [run_seed(seed, encoder, checkpoint_dir, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/balanced_integrated_dialogue_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/gate4-balanced-integrated"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
