#!/usr/bin/env python3
"""GATE-WRITE-CONTROL-1: balanced templates with natural held-out words."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
from collections import Counter
from pathlib import Path

import torch
import transformers

from gate1 import _matched_random
from gate2 import _scores
from gate_control1 import FrozenSentenceEncoder, _checkpoint_payload
from gate_control2 import _selection_digest, _threshold_rows, match_ranked_counts
from memory_gate import fit_canonical_ridge
from measurement.balanced_natural_write_registry import (
    BALANCED_NATURAL_WRITE_SPEC,
    spec_sha256,
)


SYNTHETIC_PATTERN = re.compile(r"(?:보정|평가)(?:주제|값|핵심|보조|임시|발화)|[0-9]")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _subjects(qualifiers: list[str], heads: list[str]) -> list[str]:
    values = [f"{qualifier} {head}" for qualifier, head in itertools.product(qualifiers, heads)]
    if len(values) != len(set(values)) or any(not value.strip() for value in values):
        raise ValueError("natural subject phrases must be unique and non-empty")
    return values


def _render_distractor(
    role_template: list[str], subject: str, value: str, closer: str
) -> tuple[str, str]:
    role, template = role_template
    text = template.format(subject=subject, value=value)
    return role, f"{text} 참고로 {subject}에서 {value} 이야기는 {closer}"


def build_balanced_calibration(
    seed: int, spec: dict = BALANCED_NATURAL_WRITE_SPEC
) -> list[dict]:
    lexicon = spec["lexicons"]["calibration"]
    facts = spec["templates"]["facts"]["calibration"]
    rows: list[dict] = []
    for kind in spec["fact_kinds"]:
        subjects = _subjects(lexicon["subject_qualifiers"], spec["subject_heads"][kind])
        values = lexicon["values"][kind]
        for template_index, subject, value in itertools.product(
            range(len(facts[kind])), subjects, values
        ):
            rows.append({
                "role": "user",
                "text": facts[kind][template_index].format(subject=subject, value=value),
                "important": 1,
                "kind": kind,
                "template_index": template_index,
            })

    negative_total = spec["calibration_rows"] - len(rows)
    distractor = lexicon["distractor"]
    subjects = _subjects(
        distractor["subject_qualifiers"], distractor["subject_heads"]
    )
    occurrences = Counter()
    for index in range(negative_total):
        kind = spec["distractor_kinds"][index % len(spec["distractor_kinds"])]
        occurrence = occurrences[kind]
        occurrences[kind] += 1
        subject = subjects[occurrence % len(subjects)]
        value = distractor["values"][(occurrence // len(subjects)) % len(distractor["values"])]
        closer = distractor["closers"][
            (occurrence // (len(subjects) * len(distractor["values"])))
            % len(distractor["closers"])
        ]
        entries = spec["templates"]["distractors"]["calibration"][kind]
        template_index = occurrence % len(entries)
        role, text = _render_distractor(
            entries[template_index], subject, value, closer
        )
        rows.append({
            "role": role,
            "text": text,
            "important": 0,
            "kind": kind,
            "template_index": template_index,
        })

    if len(rows) != spec["calibration_rows"]:
        raise ValueError("balanced calibration row count changed")
    order = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return [rows[index] for index in order]


def build_balanced_evaluation(
    seed: int, replicate: str, spec: dict = BALANCED_NATURAL_WRITE_SPEC
) -> list[dict]:
    if replicate not in spec["replicates"]:
        raise ValueError("unknown natural-language replicate")
    lexicon = spec["lexicons"]["evaluation"][replicate]
    facts = spec["templates"]["facts"]["evaluation"]
    closers = spec["lexicons"]["evaluation"]["distractor_closers"]
    distractor_heads = spec["lexicons"]["evaluation"]["distractor_subject_heads"]
    distractor_subjects = _subjects(
        lexicon["distractor"]["subject_qualifiers"], distractor_heads
    )
    distractor_values = lexicon["distractor"]["values"]
    episodes = []
    for kind in spec["fact_kinds"]:
        subjects = _subjects(lexicon["subject_qualifiers"], spec["subject_heads"][kind])
        values = lexicon["values"][kind]
        for template_index, subject_index, value_index in itertools.product(
            range(len(facts[kind])), range(len(subjects)), range(len(values))
        ):
            episode_index = len(episodes)
            position_index = (
                subject_index + value_index + template_index + seed
            ) % len(spec["fact_positions"])
            fact_position = spec["fact_positions"][position_index]
            subject = subjects[subject_index]
            value = values[value_index]
            fact = {
                "role": "user",
                "text": facts[kind][template_index].format(subject=subject, value=value),
                "important": 1,
                "kind": kind,
                "value": value,
                "position": fact_position,
                "topic_segment": fact_position // 2,
                "template_index": template_index,
            }
            distractor_kinds = spec["distractor_kinds"][:]
            rotation = (seed + episode_index) % len(distractor_kinds)
            distractor_kinds = distractor_kinds[rotation:] + distractor_kinds[:rotation]
            candidates = []
            distractor_index = 0
            for position in range(spec["candidates_per_episode"]):
                if position == fact_position:
                    candidates.append(fact)
                    continue
                distractor_kind = distractor_kinds[distractor_index]
                occurrence = episode_index
                side_subject = distractor_subjects[occurrence % len(distractor_subjects)]
                side_value = distractor_values[
                    (occurrence // len(distractor_subjects)) % len(distractor_values)
                ]
                closer = closers[
                    (occurrence // (len(distractor_subjects) * len(distractor_values)))
                    % len(closers)
                ]
                entries = spec["templates"]["distractors"]["evaluation"][distractor_kind]
                distractor_template = (episode_index + distractor_index) % len(entries)
                role, text = _render_distractor(
                    entries[distractor_template], side_subject, side_value, closer
                )
                candidates.append({
                    "role": role,
                    "text": text,
                    "important": 0,
                    "kind": distractor_kind,
                    "value": None,
                    "position": position,
                    "topic_segment": position // 2,
                    "template_index": distractor_template,
                })
                distractor_index += 1
            episodes.append({
                "kind": kind,
                "subject": subject,
                "value": value,
                "query": f"아까 {subject}에 관해 오래 기억해 달라고 한 내용은 뭐였지?",
                "fact_position": fact_position,
                "fact_template_index": template_index,
                "candidates": candidates,
            })
    if len(episodes) != spec["evaluation_episodes"]:
        raise ValueError("balanced evaluation episode count changed")
    return episodes


def dataset_audit(calibration: list[dict], evaluations: dict[str, list[dict]], spec: dict) -> dict:
    calibration_texts = [row["text"] for row in calibration]
    calibration_set = set(calibration_texts)
    evaluation_sets = {
        name: {row["text"] for episode in episodes for row in episode["candidates"]}
        for name, episodes in evaluations.items()
    }
    evaluation_texts = [text for values in evaluation_sets.values() for text in values]
    fact_template_counts = {
        name: {
            f"{kind}:{template_index}": sum(
                episode["kind"] == kind
                and episode["fact_template_index"] == template_index
                for episode in episodes
            )
            for kind in spec["fact_kinds"]
            for template_index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
        }
        for name, episodes in evaluations.items()
    }
    calibration_template_counts = {
        f"{kind}:{template_index}": sum(
            row["important"] and row["kind"] == kind
            and row["template_index"] == template_index
            for row in calibration
        )
        for kind in spec["fact_kinds"]
        for template_index in range(len(spec["templates"]["facts"]["calibration"][kind]))
    }
    replicate_pairs = list(itertools.combinations(spec["replicates"], 2))
    return {
        "calibration_rows": len(calibration),
        "calibration_unique": len(calibration_set),
        "calibration_positive": sum(row["important"] for row in calibration),
        "calibration_negative": sum(not row["important"] for row in calibration),
        "calibration_template_counts": calibration_template_counts,
        "evaluation_episodes": {name: len(rows) for name, rows in evaluations.items()},
        "evaluation_candidates": {
            name: sum(len(row["candidates"]) for row in rows)
            for name, rows in evaluations.items()
        },
        "evaluation_unique": {name: len(evaluation_sets[name]) for name in spec["replicates"]},
        "evaluation_fact_template_counts": fact_template_counts,
        "calibration_evaluation_overlap": {
            name: len(calibration_set & evaluation_sets[name]) for name in spec["replicates"]
        },
        "cross_replicate_overlap": {
            f"{left}:{right}": len(evaluation_sets[left] & evaluation_sets[right])
            for left, right in replicate_pairs
        },
        "synthetic_token_count": sum(
            bool(SYNTHETIC_PATTERN.search(text))
            for text in calibration_texts + evaluation_texts
        ),
        "calibration_sha256": hashlib.sha256("\n".join(calibration_texts).encode()).hexdigest(),
        "evaluation_sha256": {
            name: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
            for name, values in evaluation_sets.items()
        },
    }


def _metrics(
    episodes: list[dict], selections: list[list[bool]], scores: torch.Tensor, spec: dict
) -> dict:
    width = spec["candidates_per_episode"]
    if len(episodes) != len(selections) or scores.shape != (len(episodes), width):
        raise ValueError("balanced natural write metric shapes changed")
    kind_counts = {kind: [0, 0] for kind in spec["fact_kinds"]}
    template_counts = {
        f"{kind}:{index}": [0, 0]
        for kind in spec["fact_kinds"]
        for index in range(len(spec["templates"]["facts"]["evaluation"][kind]))
    }
    distractor_counts = {kind: [0, 0] for kind in spec["distractor_kinds"]}
    important_kept = distractors_kept = stored = 0
    for episode, selected in zip(episodes, selections):
        if len(selected) != width:
            raise ValueError("selection width changed")
        fact_position = episode["fact_position"]
        kept = bool(selected[fact_position])
        important_kept += kept
        stored += sum(bool(value) for value in selected)
        kind_counts[episode["kind"]][0] += kept
        kind_counts[episode["kind"]][1] += 1
        template_key = f'{episode["kind"]}:{episode["fact_template_index"]}'
        template_counts[template_key][0] += kept
        template_counts[template_key][1] += 1
        for row, keep in zip(episode["candidates"], selected):
            if row["important"]:
                continue
            distractors_kept += bool(keep)
            distractor_counts[row["kind"]][0] += bool(keep)
            distractor_counts[row["kind"]][1] += 1
    total = len(episodes)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / (total * (width - 1)),
        "search_size_ratio": stored / (total * width),
        "stored": stored,
        "per_kind_storage_rate": {
            kind: hits / count for kind, (hits, count) in kind_counts.items()
        },
        "per_template_storage_rate": {
            key: hits / count for key, (hits, count) in template_counts.items()
        },
        "per_distractor_storage_rate": {
            kind: hits / count for kind, (hits, count) in distractor_counts.items()
        },
        "selection_sha256": _selection_digest(selections),
        "scores_sha256": hashlib.sha256(scores.contiguous().numpy().tobytes()).hexdigest(),
    }


def run_seed(
    seed: int, encoder: FrozenSentenceEncoder, checkpoint_dir: Path, spec: dict
) -> dict:
    calibration = build_balanced_calibration(seed, spec)
    evaluations = {
        replicate: build_balanced_evaluation(seed, replicate, spec)
        for replicate in spec["replicates"]
    }
    calibration_features, calibration_embedding = encoder.encode_rows(calibration)
    labels = torch.tensor([row["important"] for row in calibration], dtype=torch.float64)
    weight, bias, threshold, fit_audit = fit_canonical_ridge(
        calibration_features, labels, ridge=spec["ridge"]
    )
    generator = torch.Generator().manual_seed(seed + spec["shuffle_seed_offset"])
    shuffled_labels = labels[torch.randperm(len(labels), generator=generator)]
    fake_weight, fake_bias, fake_threshold, fake_fit_audit = fit_canonical_ridge(
        calibration_features, shuffled_labels, ridge=spec["ridge"]
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}_balanced_natural_write.json"
    fake_checkpoint_path = checkpoint_dir / f"seed_{seed}_balanced_natural_fake.json"
    _atomic_json(checkpoint_path, _checkpoint_payload(weight, bias, threshold, spec["encoder"]))
    _atomic_json(
        fake_checkpoint_path,
        _checkpoint_payload(fake_weight, fake_bias, fake_threshold, spec["encoder"]),
    )
    replicate_rows = []
    for replicate, episodes in evaluations.items():
        candidates = [row for episode in episodes for row in episode["candidates"]]
        features, embedding_audit = encoder.encode_rows(candidates)
        width = spec["candidates_per_episode"]
        scores = _scores(features, weight, bias).reshape(-1, width)
        fake_scores = _scores(features, fake_weight, fake_bias).reshape(-1, width)
        semantic = _threshold_rows(scores.reshape(-1), threshold, width)
        matched_fake = match_ranked_counts(fake_scores.reshape(-1), semantic, width)
        matched_random = [
            _matched_random(
                selected,
                seed + spec["random_seed_offset"]
                + spec["replicates"].index(replicate) * spec["evaluation_episodes"] + index,
            )
            for index, selected in enumerate(semantic)
        ]
        arms = {
            "semantic_gate": (semantic, scores),
            "matched_shuffled_gate": (matched_fake, fake_scores),
            "matched_random": (matched_random, torch.zeros_like(scores)),
        }
        replicate_rows.append({
            "name": replicate,
            "embedding_audit": embedding_audit,
            "matching_audit": {
                "semantic_counts": [sum(row) for row in semantic],
                "matched_shuffled_counts": [sum(row) for row in matched_fake],
                "matched_random_counts": [sum(row) for row in matched_random],
                "semantic_selection_sha256": _selection_digest(semantic),
                "matched_shuffled_selection_sha256": _selection_digest(matched_fake),
                "matched_random_selection_sha256": _selection_digest(matched_random),
            },
            "arms": {
                name: _metrics(episodes, selected, arm_scores, spec)
                for name, (selected, arm_scores) in arms.items()
            },
        })
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, evaluations, spec),
        "calibration_embedding_audit": calibration_embedding,
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
    checkpoint_dir: Path, spec: dict = BALANCED_NATURAL_WRITE_SPEC
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
        "encoder_audit": encoder.audit(),
        "seeds": [run_seed(seed, encoder, checkpoint_dir, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/balanced_natural_write_results.json"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/gate-write-control1"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run(args.checkpoint_dir))


if __name__ == "__main__":
    main()
