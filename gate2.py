#!/usr/bin/env python3
"""GATE-2: realistic dialogue and topic-shift memory write selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate1 import _matched_random, _recall
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_control2 import _scores, _selection_digest, _threshold_rows, match_ranked_counts
from memory_gate import fit_canonical_ridge
from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    spec_sha256,
    template_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _token(split: str, family: str, seed: int, index: int) -> str:
    prefix = "보정" if split == "calibration" else "평가"
    return f"{prefix}{family}{seed:04d}{index:06d}"


def _template_entry(entries: list, seed: int, index: int):
    return entries[(seed + index) % len(entries)]


def build_calibration(seed: int, spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> list[dict]:
    rows = []
    half = spec["calibration_rows"] // 2
    fact_templates = spec["templates"]["facts"]["calibration"]
    distractor_templates = spec["templates"]["distractors"]["calibration"]
    for index in range(half):
        kind = spec["fact_kinds"][index % len(spec["fact_kinds"])]
        template = _template_entry(fact_templates[kind], seed, index)
        rows.append({
            "role": "user",
            "text": template.format(
                subject=_token("calibration", "주제", seed, index),
                value=_token("calibration", "값", seed, index),
            ),
            "important": 1,
            "kind": kind,
            "template_index": (seed + index) % len(fact_templates[kind]),
        })
    for index in range(half):
        kind = spec["distractor_kinds"][index % len(spec["distractor_kinds"])]
        entries = distractor_templates[kind]
        role, template = _template_entry(entries, seed, index)
        rows.append({
            "role": role,
            "text": template.format(
                subject=_token("calibration", "보조주제", seed, index),
                value=_token("calibration", "임시값", seed, index),
            ) + " " + _token("calibration", "발화", seed, index),
            "important": 0,
            "kind": kind,
            "template_index": (seed + index) % len(entries),
        })
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    return [rows[index] for index in order]


def build_evaluation(seed: int, spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> list[dict]:
    fact_templates = spec["templates"]["facts"]["evaluation"]
    distractor_templates = spec["templates"]["distractors"]["evaluation"]
    episodes = []
    for index in range(spec["evaluation_episodes"]):
        kind = spec["fact_kinds"][index % len(spec["fact_kinds"])]
        fact_position = spec["fact_positions"][(index // len(spec["fact_kinds"])) % len(spec["fact_positions"])]
        subject = _token("evaluation", "핵심주제", seed, index)
        value = _token("evaluation", "핵심값", seed, index)
        fact_template = _template_entry(fact_templates[kind], seed, index)
        fact_row = {
            "role": "user",
            "text": fact_template.format(subject=subject, value=value),
            "important": 1,
            "kind": kind,
            "value": value,
            "position": fact_position,
            "topic_segment": fact_position // 2,
        }
        distractor_kinds = spec["distractor_kinds"][:]
        rotation = (seed + index) % len(distractor_kinds)
        distractor_kinds = distractor_kinds[rotation:] + distractor_kinds[:rotation]
        candidates = []
        distractor_index = 0
        for position in range(spec["candidates_per_episode"]):
            if position == fact_position:
                candidates.append(fact_row)
                continue
            distractor_kind = distractor_kinds[distractor_index]
            entries = distractor_templates[distractor_kind]
            role, template = _template_entry(entries, seed, index + distractor_index)
            segment = position // 2
            topic = (
                subject if segment == fact_position // 2
                else _token("evaluation", f"보조주제{segment}", seed, index)
            )
            candidates.append({
                "role": role,
                "text": template.format(
                    subject=topic,
                    value=_token("evaluation", "임시값", seed, index * 10 + distractor_index),
                ) + " " + _token("evaluation", "발화", seed, index * 10 + distractor_index),
                "important": 0,
                "kind": distractor_kind,
                "value": None,
                "position": position,
                "topic_segment": segment,
            })
            distractor_index += 1
        episodes.append({
            "kind": kind,
            "subject": subject,
            "value": value,
            "query": f"아까 {subject}에 관해 오래 기억해 달라고 한 내용은 뭐였지?",
            "fact_position": fact_position,
            "topic_switches": spec["topic_switches_per_episode"],
            "candidates": candidates,
        })
    return episodes


def dataset_audit(calibration: list[dict], episodes: list[dict], spec: dict) -> dict:
    calibration_texts = [row["text"] for row in calibration]
    evaluation_rows = [row for episode in episodes for row in episode["candidates"]]
    evaluation_texts = [row["text"] for row in evaluation_rows]
    labels = [row["important"] for row in calibration]
    return {
        "calibration_rows": len(calibration),
        "calibration_unique": len(set(calibration_texts)),
        "calibration_positive": sum(labels),
        "calibration_negative": len(labels) - sum(labels),
        "evaluation_episodes": len(episodes),
        "evaluation_candidates": len(evaluation_rows),
        "evaluation_unique": len(set(evaluation_texts)),
        "overlap": len(set(calibration_texts) & set(evaluation_texts)),
        "fact_counts": {
            kind: sum(episode["kind"] == kind for episode in episodes)
            for kind in spec["fact_kinds"]
        },
        "fact_position_counts": {
            str(position): sum(episode["fact_position"] == position for episode in episodes)
            for position in spec["fact_positions"]
        },
        "distractor_counts": {
            kind: sum(row["kind"] == kind for row in evaluation_rows)
            for kind in spec["distractor_kinds"]
        },
        "topic_switch_counts": {
            str(count): sum(episode["topic_switches"] == count for episode in episodes)
            for count in {episode["topic_switches"] for episode in episodes}
        },
        "template_sha256": template_sha256(spec),
        "calibration_sha256": hashlib.sha256("\n".join(calibration_texts).encode()).hexdigest(),
        "evaluation_sha256": hashlib.sha256("\n".join(evaluation_texts).encode()).hexdigest(),
    }


def _metrics(episodes: list[dict], selections: list[list[bool]], spec: dict) -> dict:
    if len(selections) != len(episodes) or any(
        len(selected) != len(episode["candidates"])
        for episode, selected in zip(episodes, selections)
    ):
        raise ValueError("memory selections must match every dialogue episode")
    important_kept = distractors_kept = total_kept = recalls = 0
    kind_hits = {kind: [0, 0] for kind in spec["fact_kinds"]}
    position_hits = {str(position): [0, 0] for position in spec["fact_positions"]}
    distractor_hits = {kind: [0, 0] for kind in spec["distractor_kinds"]}
    digest_rows = []
    for episode, selected in zip(episodes, selections):
        recalled = _recall(episode, selected, spec["top_k"])
        recalls += recalled
        important_kept += sum(
            bool(keep and row["important"])
            for row, keep in zip(episode["candidates"], selected)
        )
        total_kept += sum(bool(value) for value in selected)
        for row, keep in zip(episode["candidates"], selected):
            if not row["important"]:
                distractors_kept += bool(keep)
                distractor_hits[row["kind"]][0] += bool(keep)
                distractor_hits[row["kind"]][1] += 1
        kind_hits[episode["kind"]][0] += int(recalled)
        kind_hits[episode["kind"]][1] += 1
        position = str(episode["fact_position"])
        position_hits[position][0] += int(recalled)
        position_hits[position][1] += 1
        digest_rows.append("".join("1" if value else "0" for value in selected) + str(int(recalled)))
    total = len(episodes)
    distractor_total = total * (spec["candidates_per_episode"] - 1)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / distractor_total,
        "search_size_ratio": total_kept / (total * spec["candidates_per_episode"]),
        "recall_at_3": recalls / total,
        "stored": total_kept,
        "per_kind_recall": {
            kind: hits / count for kind, (hits, count) in kind_hits.items()
        },
        "per_position_recall": {
            position: hits / count for position, (hits, count) in position_hits.items()
        },
        "per_distractor_storage_rate": {
            kind: hits / count for kind, (hits, count) in distractor_hits.items()
        },
        "records_sha256": hashlib.sha256("\n".join(digest_rows).encode()).hexdigest(),
    }


def run_seed(seed: int, encoder: FrozenSentenceEncoder, checkpoint_dir: Path,
             spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> dict:
    calibration = build_calibration(seed, spec)
    episodes = build_evaluation(seed, spec)
    evaluation_rows = [row for episode in episodes for row in episode["candidates"]]
    calibration_features, calibration_embedding_audit = encoder.encode_rows(calibration)
    evaluation_features, evaluation_embedding_audit = encoder.encode_rows(evaluation_rows)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"],
    )
    generator = torch.Generator().manual_seed(seed + spec["shuffle_seed_offset"])
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, shuffled_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=spec["ridge"],
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_realistic_memory_gate.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_shuffled_realistic_memory_gate.json"
    _atomic_json(checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"]))
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(fake_weight, fake_bias, fake_threshold, spec["encoder"]),
    )
    semantic_scores = _scores(evaluation_features, weight, bias)
    fake_scores = _scores(evaluation_features, fake_weight, fake_bias)
    semantic = _threshold_rows(semantic_scores, threshold, spec["candidates_per_episode"])
    matched_shuffled = match_ranked_counts(
        fake_scores, semantic, spec["candidates_per_episode"],
    )
    matched_random = [
        _matched_random(selection, seed + spec["random_seed_offset"] + index)
        for index, selection in enumerate(semantic)
    ]
    arms = {
        "semantic_gate": semantic,
        "store_all": [[True] * spec["candidates_per_episode"] for _ in episodes],
        "oracle_gate": [
            [bool(row["important"]) for row in episode["candidates"]] for episode in episodes
        ],
        "matched_random": matched_random,
        "matched_shuffled_gate": matched_shuffled,
        "no_memory": [[False] * spec["candidates_per_episode"] for _ in episodes],
    }
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, episodes, spec),
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
            "semantic_counts": [sum(row) for row in semantic],
            "matched_shuffled_counts": [sum(row) for row in matched_shuffled],
            "matched_random_counts": [sum(row) for row in matched_random],
            "semantic_selection_sha256": _selection_digest(semantic),
            "matched_shuffled_selection_sha256": _selection_digest(matched_shuffled),
            "matched_random_selection_sha256": _selection_digest(matched_random),
            "fake_scores_sha256": hashlib.sha256(
                fake_scores.contiguous().numpy().tobytes()
            ).hexdigest(),
        },
        "arms": {name: _metrics(episodes, rows, spec) for name, rows in arms.items()},
    }


def run(checkpoint_dir: Path, spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> dict:
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
        default=Path("measurement/realistic_memory_write_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/gate2"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
