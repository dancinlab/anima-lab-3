#!/usr/bin/env python3
"""META-1: test whether the causal GRAFT state calibrates its own action errors."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import graft_behavior
from graft_behavior import GraftActionChannel, bridge_hub_dim_for_arm, sha256_file
from measurement.graft_behavior_registry import experiment as action_experiment
from measurement.metacognition_registry import METACOGNITION_SPEC, spec_sha256
from trinity import HFDecoder


class ConfidenceReader(nn.Module):
    """A small calibrated probe; it never feeds back into the action path."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @torch.no_grad()
    def fit_normalization(self, features: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)


def _load_checkpoint(path: Path, expected_sha256: str, device: str) -> dict:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    return payload


def _perturb_state(state: torch.Tensor, arm: str, level: float,
                   generator: torch.Generator) -> torch.Tensor:
    if level == 0.0:
        return state.clone()
    noise = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    if arm == "consciousness":
        if state.shape[-1] % 2:
            raise ValueError("phase state must contain equal cosine and sine halves")
        half = state.shape[-1] // 2
        theta = torch.atan2(state[..., half:], state[..., :half])
        theta = theta + level * noise[..., :half]
        return torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1)
    scale = state.pow(2).mean().sqrt().clamp_min(1e-6)
    return state + level * scale * noise


def _expanded_states(examples, arm: str, seed: int, split: str) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    labels = []
    levels = []
    split_offset = 0 if split == "train" else 10_000_000
    for example_index, example in enumerate(examples):
        source = getattr(example, "state" if arm == "consciousness" else "memory")
        for level_index, level in enumerate(METACOGNITION_SPEC["readout_noise_levels"]):
            generator = torch.Generator().manual_seed(
                seed * 100_000 + split_offset + example_index * 100 + level_index
                + (0 if arm == "consciousness" else 50_000_000)
            )
            states.append(_perturb_state(source, arm, level, generator))
            labels.append(example.target)
            levels.append(level)
    return states, torch.tensor(labels), torch.tensor(levels, dtype=torch.float32)


@torch.no_grad()
def _extract_dataset(decoder: HFDecoder, channel: GraftActionChannel, examples, arm: str,
                     seed: int, split: str, action_ids: list[int]) -> dict[str, torch.Tensor]:
    states, labels, levels = _expanded_states(examples, arm, seed, split)
    codes = channel.inference_codes([state.to(decoder.device) for state in states]).float()
    prompt = graft_behavior._prompt_tokens(decoder)
    actions = torch.tensor(action_ids, device=decoder.device)
    logits = []
    batch_size = METACOGNITION_SPEC["reader"]["batch_size"]
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        tokens = prompt.expand(stop - start, -1)
        gate = codes[start:stop].unsqueeze(1).expand(-1, prompt.shape[1], -1)
        row = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
        logits.append(row.index_select(-1, actions).cpu())
    logits = torch.cat(logits)
    codes = codes.cpu()
    predicted = logits.argmax(-1)
    correct = predicted.eq(labels).float()
    return {"codes": codes, "logits": logits, "predicted": predicted,
            "correct": correct, "labels": labels, "levels": levels}


def _reader_features(codes: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    chosen = F.one_hot(predicted, num_classes=4).float()
    return torch.cat((codes.float(), chosen), dim=-1)


def _train_reader(features: torch.Tensor, correct: torch.Tensor, seed: int,
                  device: str) -> ConfidenceReader:
    cfg = METACOGNITION_SPEC["reader"]
    torch.manual_seed(seed)
    reader = ConfidenceReader(features.shape[-1], cfg["hidden_dim"]).to(device)
    train_x = features.to(device)
    train_y = correct.to(device)
    reader.fit_normalization(train_x)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 30_000_000)
    for _ in range(cfg["train_steps"]):
        indices = torch.randint(len(train_x), (cfg["batch_size"],), generator=generator).to(device)
        loss = F.binary_cross_entropy_with_logits(reader(train_x[indices]), train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return reader.eval()


def _auroc(probability: torch.Tensor, correct: torch.Tensor) -> float:
    positive = int(correct.sum())
    negative = len(correct) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(probability)
    ranks = torch.empty(len(probability), dtype=torch.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and probability[order[stop]] == probability[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = ranks[correct.bool()].sum().item()
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def binary_metrics(probability: torch.Tensor, correct: torch.Tensor) -> dict:
    probability = probability.detach().float().cpu()
    correct = correct.detach().float().cpu()
    bins = METACOGNITION_SPEC["reader"]["ece_bins"]
    ece = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        mask = (probability >= lo) & (probability < hi if index + 1 < bins else probability <= hi)
        if mask.any():
            ece += float(mask.float().mean()) * abs(float(probability[mask].mean() - correct[mask].mean()))
    count = max(1, int(len(correct) * METACOGNITION_SPEC["reader"]["selective_fraction"]))
    order = torch.argsort(probability)
    low = float(correct[order[:count]].mean())
    high = float(correct[order[-count:]].mean())
    return {
        "auroc": _auroc(probability, correct),
        "brier": float((probability - correct).pow(2).mean()),
        "ece": ece,
        "selective_accuracy_gap": high - low,
        "high_confidence_accuracy": high,
        "low_confidence_accuracy": low,
        "mean_confidence": float(probability.mean()),
    }


def _shuffle_codes(codes: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    permutation = []
    for index, action in enumerate(predicted.tolist()):
        candidate = (index + 1) % len(codes)
        while candidate != index and predicted[candidate].item() == action:
            candidate = (candidate + 1) % len(codes)
        permutation.append(candidate)
    return codes[torch.tensor(permutation)]


@torch.no_grad()
def _evaluate_readers(state_reader: ConfidenceReader, output_reader: ConfidenceReader,
                      data: dict[str, torch.Tensor], seed: int, device: str) -> dict:
    codes = data["codes"]
    predicted = data["predicted"]
    correct = data["correct"]
    generator = torch.Generator().manual_seed(seed + 40_000_000)
    random_codes = torch.randn(codes.shape, generator=generator)
    random_codes *= codes.norm(dim=-1, keepdim=True) / random_codes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    conditions = {
        "normal": codes,
        "off": torch.zeros_like(codes),
        "shuffle": _shuffle_codes(codes, predicted),
        "noise": random_codes,
        "recovered": codes,
    }
    confidence = {}
    raw_probability = {}
    for name, reader_codes in conditions.items():
        features = _reader_features(reader_codes, predicted).to(device)
        probability = torch.sigmoid(state_reader(features)).cpu()
        raw_probability[name] = probability
        confidence[name] = binary_metrics(probability, correct)
    output_probability = torch.sigmoid(output_reader(data["logits"].to(device))).cpu()
    action_by_level = {}
    for level in METACOGNITION_SPEC["readout_noise_levels"]:
        mask = data["levels"].eq(level)
        action_by_level[str(level)] = {
            "accuracy": float(correct[mask].mean()),
            "examples": int(mask.sum()),
            "mean_confidence": float(raw_probability["normal"][mask].mean()),
        }
    frozen_predictions = {name: predicted.clone() for name in METACOGNITION_SPEC["interventions"]}
    return {
        "action": {
            "accuracy": float(correct.mean()),
            "correct_examples": int(correct.sum()),
            "incorrect_examples": int(len(correct) - correct.sum()),
            "by_noise_level": action_by_level,
        },
        "confidence": confidence,
        "output_only": binary_metrics(output_probability, correct),
        "intervention_actions_identical": all(
            torch.equal(predicted, row) for row in frozen_predictions.values()
        ),
        "recovery_confidence_identical": torch.equal(
            raw_probability["normal"], raw_probability["recovered"]
        ),
    }


def run_arm(decoder: HFDecoder, seed: int, arm: str, checkpoint_path: Path,
            output_dir: Path) -> dict:
    action_spec = action_experiment(METACOGNITION_SPEC["action_experiment"])
    expected_sha = METACOGNITION_SPEC["archive"]["checkpoint_sha256"][str(seed)][arm]
    payload = _load_checkpoint(checkpoint_path, expected_sha, decoder.device)
    if payload.get("seed") != seed or payload.get("arm") != arm:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
    if payload.get("spec_sha256") != METACOGNITION_SPEC["action_spec_sha256"]:
        raise RuntimeError(f"checkpoint action spec mismatch: {checkpoint_path}")
    input_dim = action_spec["state_dim"] if arm == "consciousness" else action_spec["memory_state_dim"]
    channel = GraftActionChannel(
        input_dim, decoder.d_model, action_spec["gate_rho"],
        hub_dim=bridge_hub_dim_for_arm(action_spec, arm),
    ).to(decoder.device)
    channel.load_state_dict(payload["channel"], strict=True)
    channel.eval()
    graft_behavior.BEHAVIOR_SPEC = action_spec
    train_examples = graft_behavior.build_examples(seed, "train")
    eval_examples = graft_behavior.build_examples(seed, "eval")
    action_ids = graft_behavior._action_token_ids(decoder.tokenizer)
    train = _extract_dataset(decoder, channel, train_examples, arm, seed, "train", action_ids)
    evaluate = _extract_dataset(decoder, channel, eval_examples, arm, seed, "eval", action_ids)
    state_reader = _train_reader(
        _reader_features(train["codes"], train["predicted"]), train["correct"],
        seed + (0 if arm == "consciousness" else 1_000_000), decoder.device,
    )
    output_reader = _train_reader(
        train["logits"], train["correct"],
        seed + (100_000 if arm == "consciousness" else 1_100_000), decoder.device,
    )
    metrics = _evaluate_readers(state_reader, output_reader, evaluate, seed, decoder.device)
    reader_path = output_dir / f"seed_{seed}_{arm}_readers.pt"
    torch.save({
        "experiment": METACOGNITION_SPEC["experiment"],
        "spec_sha256": spec_sha256(),
        "seed": seed,
        "arm": arm,
        "source_checkpoint_sha256": expected_sha,
        "state_reader": state_reader.state_dict(),
        "output_reader": output_reader.state_dict(),
    }, reader_path)
    metrics.update({
        "source_checkpoint_sha256": expected_sha,
        "reader_checkpoint": {"path": str(reader_path), "sha256": sha256_file(reader_path)},
    })
    del channel, state_reader, output_reader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="checkpoints/graft_behavior_phase_state_bridge32_repair")
    parser.add_argument("--reader-dir", default="checkpoints/metacognition")
    parser.add_argument("--output", default="measurement/metacognition_results.json")
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    reader_dir = Path(args.reader_dir)
    reader_dir.mkdir(parents=True, exist_ok=True)
    action_spec = action_experiment(METACOGNITION_SPEC["action_experiment"])
    decoder = HFDecoder(action_spec["model"], lora=False, freeze_base=True,
                        gate_strength=action_spec["gate_strength"],
                        gate_rms_max=action_spec["gate_rms_max"])
    decoder.model.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    rows = []
    for seed in METACOGNITION_SPEC["seeds"]:
        arms = {}
        for arm in METACOGNITION_SPEC["arms"]:
            path = source_dir / f"seed_{seed}_{arm}.pt"
            print(f"[META-1] seed={seed} arm={arm} source={path}", flush=True)
            arms[arm] = run_arm(decoder, seed, arm, path, reader_dir)
        rows.append({"seed": seed, "arms": arms})
    result = {
        "experiment": METACOGNITION_SPEC["experiment"],
        "spec": METACOGNITION_SPEC,
        "spec_sha256": spec_sha256(),
        "model": action_spec["model"],
        "seeds": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"[results] {output}")


if __name__ == "__main__":
    main()
