#!/usr/bin/env python3
"""GATE-1: controlled long-term dialogue-memory write selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from creativity_classifier import text_to_vector
from memory_gate import DialogueMemoryGate, fit_canonical_memory_gate
from measurement.memory_write_gate_registry import MEMORY_WRITE_GATE_SPEC, spec_sha256


FACT_TEMPLATES = {
    "calibration": {
        "preference": "내 {subject} 선호는 {value}야",
        "commitment": "{subject} 약속은 {value}로 정했어",
        "goal": "내 {subject} 목표는 {value}야",
        "profile": "내 {subject} 정보는 {value}야",
    },
    "evaluation": {
        "preference": "{subject}에서는 {value}을 가장 좋아해",
        "commitment": "{subject} 일정으로 {value}하기로 했어",
        "goal": "{subject}에서 이루려는 것은 {value}야",
        "profile": "{subject} 항목은 {value}로 알려둘게",
    },
}

DISTRACTOR_TEMPLATES = {
    "calibration": {
        "greeting": ("user", "안녕, 오늘도 반가워"),
        "thanks": ("user", "설명해 줘서 고마워"),
        "filler": ("user", "음, 그렇구나"),
        "weather": ("user", "지금 바깥 날씨가 잠깐 흐리네"),
        "mood": ("user", "지금은 기분이 조금 차분해"),
        "question": ("user", "{subject} 이야기는 어떻게 생각해?"),
        "ack": ("assistant", "알겠어, 계속 이야기해 보자"),
    },
    "evaluation": {
        "greeting": ("user", "반가워, 잘 지냈어?"),
        "thanks": ("user", "도움이 됐어, 감사해"),
        "filler": ("user", "아하, 그런 셈이네"),
        "weather": ("user", "오늘 공기가 잠시 선선하네"),
        "mood": ("user", "방금은 마음이 약간 편안해졌어"),
        "question": ("user", "그런데 {subject} 쪽은 어떤가?"),
        "ack": ("assistant", "응, 다음 말을 들어볼게"),
    },
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _token(split: str, family: str, index: int) -> str:
    prefix = "보정" if split == "calibration" else "평가"
    return f"{prefix}{family}{index:05d}"


def build_calibration(seed: int, spec: dict = MEMORY_WRITE_GATE_SPEC) -> list[dict]:
    rows = []
    half = spec["calibration_rows"] // 2
    for index in range(half):
        kind = spec["fact_kinds"][index % len(spec["fact_kinds"])]
        subject = _token("calibration", "주제", seed * half + index)
        value = _token("calibration", "값", seed * half + index)
        rows.append({
            "role": "user",
            "text": FACT_TEMPLATES["calibration"][kind].format(subject=subject, value=value),
            "important": 1,
            "kind": kind,
        })
    for index in range(half):
        kind = spec["distractor_kinds"][index % len(spec["distractor_kinds"])]
        role, template = DISTRACTOR_TEMPLATES["calibration"][kind]
        rows.append({
            "role": role,
            "text": (
                template.format(subject=_token("calibration", "잡담", seed * half + index))
                + " " + _token("calibration", "회차", seed * half + index)
            ),
            "important": 0,
            "kind": kind,
        })
    random.Random(seed).shuffle(rows)
    return rows


def build_evaluation(seed: int, spec: dict = MEMORY_WRITE_GATE_SPEC) -> list[dict]:
    episodes = []
    for index in range(spec["evaluation_episodes"]):
        kind = spec["fact_kinds"][index % len(spec["fact_kinds"])]
        subject = _token("evaluation", "주제", seed * spec["evaluation_episodes"] + index)
        value = _token("evaluation", "값", seed * spec["evaluation_episodes"] + index)
        candidates = [{
            "role": "user",
            "text": FACT_TEMPLATES["evaluation"][kind].format(subject=subject, value=value),
            "important": 1,
            "kind": kind,
            "value": value,
        }]
        for distractor_kind in spec["distractor_kinds"]:
            role, template = DISTRACTOR_TEMPLATES["evaluation"][distractor_kind]
            candidates.append({
                "role": role,
                "text": (
                    template.format(subject=subject)
                    + " " + _token(
                        "evaluation", distractor_kind,
                        (seed * spec["evaluation_episodes"] + index) * 10
                        + spec["distractor_kinds"].index(distractor_kind),
                    )
                ),
                "important": 0,
                "kind": distractor_kind,
                "value": None,
            })
        random.Random(seed * 1000003 + index).shuffle(candidates)
        episodes.append({
            "kind": kind,
            "subject": subject,
            "value": value,
            "query": f"{subject}에 대해 내가 알려준 값은 뭐였지?",
            "candidates": candidates,
        })
    return episodes


def dataset_audit(calibration: list[dict], episodes: list[dict], spec: dict) -> dict:
    calibration_texts = [row["text"] for row in calibration]
    evaluation_texts = [row["text"] for episode in episodes for row in episode["candidates"]]
    labels = [row["important"] for row in calibration]
    return {
        "calibration_rows": len(calibration),
        "calibration_unique": len(set(calibration_texts)),
        "calibration_positive": sum(labels),
        "calibration_negative": len(labels) - sum(labels),
        "evaluation_episodes": len(episodes),
        "evaluation_candidates": len(evaluation_texts),
        "evaluation_unique": len(set(evaluation_texts)),
        "overlap": len(set(calibration_texts) & set(evaluation_texts)),
        "fact_counts": {
            kind: sum(episode["kind"] == kind for episode in episodes)
            for kind in spec["fact_kinds"]
        },
        "calibration_sha256": hashlib.sha256("\n".join(calibration_texts).encode()).hexdigest(),
        "evaluation_sha256": hashlib.sha256("\n".join(evaluation_texts).encode()).hexdigest(),
    }


def _selection(gate: DialogueMemoryGate, episode: dict) -> list[bool]:
    return [gate.should_index(row["role"], row["text"]) for row in episode["candidates"]]


def _matched_random(selection: list[bool], seed: int) -> list[bool]:
    count = sum(selection)
    ranked = sorted(
        range(len(selection)),
        key=lambda index: hashlib.sha256(f"{seed}|{index}".encode()).digest(),
    )
    chosen = set(ranked[:count])
    return [index in chosen for index in range(len(selection))]


def _recall(episode: dict, selected: list[bool], top_k: int) -> bool:
    selected_rows = [
        (index, row) for index, (row, keep) in enumerate(zip(episode["candidates"], selected))
        if keep
    ]
    if not selected_rows:
        return False
    query = text_to_vector(episode["query"]).float()
    vectors = torch.stack([text_to_vector(row["text"]).float() for _, row in selected_rows])
    similarities = F.cosine_similarity(query.unsqueeze(0), vectors, dim=1)
    top = similarities.topk(min(top_k, len(selected_rows))).indices.tolist()
    return any(selected_rows[index][1]["important"] == 1 for index in top)


def _metrics(episodes: list[dict], selections: list[list[bool]], top_k: int) -> dict:
    important_kept = distractors_kept = total_kept = recalls = 0
    kind_hits = {kind: [0, 0] for kind in MEMORY_WRITE_GATE_SPEC["fact_kinds"]}
    digest_rows = []
    for episode, selected in zip(episodes, selections):
        important_kept += sum(
            keep and row["important"] for row, keep in zip(episode["candidates"], selected)
        )
        distractors_kept += sum(
            keep and not row["important"] for row, keep in zip(episode["candidates"], selected)
        )
        total_kept += sum(selected)
        recalled = _recall(episode, selected, top_k)
        recalls += recalled
        kind_hits[episode["kind"]][0] += int(recalled)
        kind_hits[episode["kind"]][1] += 1
        digest_rows.append("".join("1" if value else "0" for value in selected) + str(int(recalled)))
    total = len(episodes)
    distractor_total = total * (MEMORY_WRITE_GATE_SPEC["candidates_per_episode"] - 1)
    return {
        "important_storage_rate": important_kept / total,
        "distractor_storage_rate": distractors_kept / distractor_total,
        "search_size_ratio": total_kept / (total * MEMORY_WRITE_GATE_SPEC["candidates_per_episode"]),
        "recall_at_3": recalls / total,
        "stored": total_kept,
        "per_kind_recall": {
            kind: hits / count for kind, (hits, count) in kind_hits.items()
        },
        "records_sha256": hashlib.sha256("\n".join(digest_rows).encode()).hexdigest(),
    }


def run_seed(seed: int, output_dir: Path, spec: dict = MEMORY_WRITE_GATE_SPEC) -> dict:
    calibration = build_calibration(seed, spec)
    episodes = build_evaluation(seed, spec)
    gate, fit_audit = fit_canonical_memory_gate(
        calibration, vector_dim=spec["vector_dim"], ridge=spec["ridge"]
    )
    shuffled = [dict(row) for row in calibration]
    generator = torch.Generator().manual_seed(seed + spec["shuffle_seed_offset"])
    permuted = torch.tensor([row["important"] for row in shuffled])[torch.randperm(len(shuffled), generator=generator)]
    for row, label in zip(shuffled, permuted.tolist()):
        row["important"] = int(label)
    shuffled_gate, shuffled_fit_audit = fit_canonical_memory_gate(
        shuffled, vector_dim=spec["vector_dim"], ridge=spec["ridge"]
    )

    checkpoint_path = output_dir / f"seed_{seed}_memory_gate.json"
    _atomic_json(checkpoint_path, gate.to_payload())
    selective = [_selection(gate, episode) for episode in episodes]
    arms = {
        "selective_gate": selective,
        "store_all": [[True] * spec["candidates_per_episode"] for _ in episodes],
        "oracle_gate": [[bool(row["important"]) for row in episode["candidates"]] for episode in episodes],
        "matched_random": [
            _matched_random(selection, seed + spec["random_seed_offset"] + index)
            for index, selection in enumerate(selective)
        ],
        "shuffled_gate": [_selection(shuffled_gate, episode) for episode in episodes],
        "no_memory": [[False] * spec["candidates_per_episode"] for _ in episodes],
    }
    return {
        "seed": seed,
        "dataset_audit": dataset_audit(calibration, episodes, spec),
        "fit_audit": fit_audit,
        "shuffled_fit_audit": shuffled_fit_audit,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "arms": {name: _metrics(episodes, rows, spec["top_k"]) for name, rows in arms.items()},
    }


def run(output: Path, checkpoint_dir: Path, spec: dict = MEMORY_WRITE_GATE_SPEC) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "seeds": [run_seed(seed, checkpoint_dir, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("measurement/memory_write_gate_results.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gate1"))
    args = parser.parse_args()
    payload = run(args.output, args.checkpoint_dir)
    _atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
