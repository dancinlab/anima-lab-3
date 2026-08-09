#!/usr/bin/env python3
"""CONTROL-1: verify dynamic relation learning without the language path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graft_behavior import sha256_file
from measurement.episode_control_registry import CONTROL_SPEC, spec_sha256
from trinity import VectorMemory


@dataclass(frozen=True)
class RelationEpisode:
    stores: tuple[tuple[int, int], tuple[int, int]]
    distractors: tuple[int, int]
    query_position: int

    @property
    def query_key(self) -> int:
        return self.stores[self.query_position][0]

    @property
    def target(self) -> int:
        return self.stores[self.query_position][1]

    def fingerprint(self) -> str:
        payload = {
            "stores": self.stores,
            "distractors": self.distractors,
            "query_position": self.query_position,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class DynamicRelationGRU(nn.Module):
    """Canonical GRU positive control with a direct value readout."""

    def __init__(self, input_dim: int, hidden_dim: int, values: int):
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, values)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.recurrent(sequence)[0][:, -1]
        return self.readout(hidden)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_split(split: str, spec: dict = CONTROL_SPEC,
                excluded_fingerprints: set[str] | None = None) -> list[RelationEpisode]:
    if split not in spec["splits"]:
        raise ValueError(f"unknown split: {split}")
    count = spec["splits"][split]
    cycle = spec["keys"] * spec["values"] * spec["relations_per_episode"]
    if count % cycle:
        raise ValueError("registered split size must preserve exact query balance")
    rng = random.Random(spec["data_seed"] + spec["split_seed_offsets"][split])
    episodes = []
    seen = set(excluded_fingerprints or ())
    for index in range(count):
        target = index % spec["values"]
        query_position = (index // spec["values"]) % spec["relations_per_episode"]
        query_key = (index // (spec["values"] * spec["relations_per_episode"])) % spec["keys"]
        while True:
            other_key = rng.randrange(spec["keys"])
            other_value = rng.randrange(spec["values"])
            distractors = tuple(rng.randrange(spec["distractors"])
                                for _ in range(spec["distractor_steps"]))
            if other_key == query_key or other_value == target:
                continue
            stores = [None, None]
            stores[query_position] = (query_key, target)
            stores[1 - query_position] = (other_key, other_value)
            episode = RelationEpisode(tuple(stores), distractors, query_position)
            fingerprint = episode.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                episodes.append(episode)
                break
    rng.shuffle(episodes)
    return episodes


def build_splits(spec: dict = CONTROL_SPEC) -> dict[str, list[RelationEpisode]]:
    splits = {}
    seen = set()
    for name in spec["splits"]:
        splits[name] = build_split(name, spec, seen)
        seen.update(row.fingerprint() for row in splits[name])
    return splits


def _input_dim(spec: dict) -> int:
    return 3 + spec["keys"] + spec["values"] + spec["distractors"]


def encode_episodes(episodes: list[RelationEpisode], spec: dict = CONTROL_SPEC) -> torch.Tensor:
    sequence_len = spec["relations_per_episode"] + spec["distractor_steps"] + 1
    encoded = torch.zeros(len(episodes), sequence_len, _input_dim(spec))
    key_start = 3
    value_start = key_start + spec["keys"]
    distractor_start = value_start + spec["values"]
    for row_index, episode in enumerate(episodes):
        for step, (key, value) in enumerate(episode.stores):
            encoded[row_index, step, 0] = 1.0
            encoded[row_index, step, key_start + key] = 1.0
            encoded[row_index, step, value_start + value] = 1.0
        for offset, distractor in enumerate(episode.distractors):
            step = spec["relations_per_episode"] + offset
            encoded[row_index, step, 1] = 1.0
            encoded[row_index, step, distractor_start + distractor] = 1.0
        encoded[row_index, -1, 2] = 1.0
        encoded[row_index, -1, key_start + episode.query_key] = 1.0
    return encoded


def labels(episodes: list[RelationEpisode]) -> torch.Tensor:
    return torch.tensor([episode.target for episode in episodes], dtype=torch.long)


def audit_split(episodes: list[RelationEpisode], spec: dict = CONTROL_SPEC) -> dict:
    def counts(values) -> dict[str, int]:
        counter = Counter(values)
        return {str(index): counter[index] for index in range(max(counter, default=-1) + 1)}

    return {
        "episodes": len(episodes),
        "unique_fingerprints": len({row.fingerprint() for row in episodes}),
        "target_counts": counts(row.target for row in episodes),
        "query_key_counts": counts(row.query_key for row in episodes),
        "query_position_counts": counts(row.query_position for row in episodes),
        "fingerprint_set_sha256": hashlib.sha256(
            "\n".join(sorted(row.fingerprint() for row in episodes)).encode()
        ).hexdigest(),
    }


def dataset_audit(splits: dict[str, list[RelationEpisode]], spec: dict = CONTROL_SPEC) -> dict:
    fingerprints = {
        name: {row.fingerprint() for row in episodes} for name, episodes in splits.items()
    }
    overlap = {}
    names = list(spec["splits"])
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap[f"{left}:{right}"] = len(fingerprints[left] & fingerprints[right])
    return {
        "splits": {name: audit_split(splits[name], spec) for name in names},
        "overlap": overlap,
    }


def _confusion(expected: torch.Tensor, predicted: torch.Tensor, classes: int) -> list[list[int]]:
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    for target, actual in zip(expected.tolist(), predicted.tolist()):
        matrix[target, actual] += 1
    return matrix.tolist()


def _metrics(expected: torch.Tensor, predicted: torch.Tensor, classes: int) -> dict:
    matrix = _confusion(expected, predicted, classes)
    return {
        "accuracy": float((expected == predicted).float().mean()),
        "confusion_matrix": matrix,
        "per_value_recall": [
            matrix[index][index] / max(sum(matrix[index]), 1) for index in range(classes)
        ],
        "selection_counts": torch.bincount(predicted, minlength=classes).tolist(),
    }


@torch.no_grad()
def _evaluate(model: DynamicRelationGRU, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    predicted = model(x).argmax(-1).cpu()
    return _metrics(y.cpu(), predicted, model.readout.out_features)


def _vector_memory_predictions(episodes: list[RelationEpisode], spec: dict) -> torch.Tensor:
    rows = []
    for episode in episodes:
        memory = VectorMemory(capacity=spec["relations_per_episode"], dim=spec["values"])
        for key, value in episode.stores:
            memory.store(F.one_hot(torch.tensor(key), spec["keys"]).float(),
                         F.one_hot(torch.tensor(value), spec["values"]).float())
        query = F.one_hot(torch.tensor(episode.query_key), spec["keys"]).float()
        rows.append(int(memory.retrieve(query, top_k=1)[0].argmax()))
    return torch.tensor(rows)


def _no_memory_predictions(train: list[RelationEpisode], evaluate: list[RelationEpisode],
                           spec: dict) -> torch.Tensor:
    table = torch.zeros(spec["keys"], spec["values"], dtype=torch.long)
    for episode in train:
        table[episode.query_key, episode.target] += 1
    majority = table.argmax(-1)
    return torch.tensor([int(majority[row.query_key]) for row in evaluate])


def _shuffled_labels(expected: torch.Tensor, spec: dict) -> torch.Tensor:
    return (expected + 1) % spec["values"]


def run_seed(seed: int, splits: dict[str, list[RelationEpisode]], output_dir: Path,
             spec: dict = CONTROL_SPEC) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(spec["device"])
    tensors = {name: encode_episodes(rows, spec).to(device) for name, rows in splits.items()}
    targets = {name: labels(rows).to(device) for name, rows in splits.items()}
    model = DynamicRelationGRU(_input_dim(spec), spec["state_dim"], spec["values"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000_000)
    best = {"accuracy": -1.0, "step": 0, "state": None}
    losses = []
    for step in range(1, spec["train_steps"] + 1):
        indices = torch.randint(
            len(splits["train"]), (spec["batch_size"],), generator=generator
        ).to(device)
        model.train()
        logits = model(tensors["train"].index_select(0, indices))
        loss = F.cross_entropy(logits, targets["train"].index_select(0, indices))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % spec["validate_every"] == 0:
            metrics = _evaluate(model, tensors["validation"], targets["validation"])
            if metrics["accuracy"] > best["accuracy"]:
                best = {
                    "accuracy": metrics["accuracy"],
                    "step": step,
                    "state": {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()},
                }
            if step % (spec["validate_every"] * 5) == 0:
                print(
                    f"[seed {seed}] step={step} loss={sum(losses[-50:]) / 50:.4f} "
                    f"validation={metrics['accuracy']:.4f}", flush=True,
                )
    if best["state"] is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best["state"])
    gru = _evaluate(model, tensors["eval"], targets["eval"])
    expected = targets["eval"].cpu()
    vector = _metrics(
        expected, _vector_memory_predictions(splits["eval"], spec), spec["values"]
    )
    no_memory = _metrics(
        expected, _no_memory_predictions(splits["train"], splits["eval"], spec), spec["values"]
    )
    shuffled = _metrics(
        _shuffled_labels(expected, spec), model(tensors["eval"]).argmax(-1).cpu(), spec["values"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"seed_{seed}_gru.pt"
    torch.save({
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "seed": seed,
        "selected_step": best["step"],
        "validation_accuracy": best["accuracy"],
        "model": best["state"],
    }, checkpoint)
    return {
        "seed": seed,
        "selected_step": best["step"],
        "validation_accuracy": best["accuracy"],
        "final_loss_mean_50": sum(losses[-50:]) / min(50, len(losses)),
        "arms": {
            "gru": gru,
            "vector_memory": vector,
            "no_memory": no_memory,
            "shuffled_labels": shuffled,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/episode_control_results.json")
    parser.add_argument("--verdict", default="measurement/episode_control_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/episode_control1")
    args = parser.parse_args()
    spec = CONTROL_SPEC
    splits = build_splits(spec)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(splits, spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [
            run_seed(seed, splits, Path(args.checkpoint_dir), spec) for seed in spec["seeds"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.episode_control_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
