#!/usr/bin/env python3
"""Map hidden-situation information through the canonical QuantumC→bridge path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from measurement.state_survival_registry import STATE_SURVIVAL_SPEC, spec_sha256
from pure import PureMind
from trinity import QuantumC, ThalamicBridge


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sense_vector(payload: dict) -> torch.Tensor:
    theta, resultant = payload["global"]
    return theta * resultant


def _trial_seed(seed: int, target: int, index: int, per_situation: int, split: str) -> int:
    offset = 0 if split == "train" else 1_000_000
    return seed * 10_000 + target * per_situation + index + offset


@torch.no_grad()
def collect_examples(seed: int, split: str, bridge: ThalamicBridge) -> dict[int, dict]:
    spec = STATE_SURVIVAL_SPEC
    per_situation = spec[f"{split}_examples_per_situation"]
    delays = spec["delay_steps"]
    rows = {delay: {channel: [] for channel in spec["channels"]} for delay in delays}
    labels = {delay: [] for delay in delays}

    for target, situation in enumerate(spec["situations"]):
        for index in range(per_situation):
            trial_seed = _trial_seed(seed, target, index, per_situation, split)
            torch.manual_seed(trial_seed)
            engine = QuantumC(nc=spec["cells"], dim=spec["engine_dim"], max_cells=spec["cells"])
            encoder = PureMind(store=None, c_engine=engine)
            for _ in range(spec["warm_steps"]):
                engine.step()
            nuisance = spec["nuisance_words"][trial_seed % len(spec["nuisance_words"])]
            payload = encoder.encode_sense([*situation["words"], nuisance])
            for _ in range(spec["sense_steps"]):
                engine.step(x_input=payload)

            timeline = []
            current_delay = 0
            for delay in delays:
                while current_delay < delay:
                    engine.step()
                    current_delay += 1
                state = engine.get_state_channels()
                timeline.append(state["phase"].clone())
                bridge_trace = bridge.trace(state["phase"], seq_len=1)
                features = {
                    "sense_input": _sense_vector(payload),
                    "phase": state["phase"],
                    "amplitude": state["amplitude"],
                    "phase_velocity": state["phase_velocity"],
                    "tension_frustration": state["tension_frustration"],
                    "full_state": state["full_state"],
                    "temporal_phase": torch.cat(timeline, dim=-1),
                    "bridge_cells": bridge_trace["cells"],
                    "bridge_pooled": bridge_trace["pooled"],
                    "bridge_gate": bridge_trace["gate"],
                }
                for channel, value in features.items():
                    rows[delay][channel].append(value.detach().reshape(-1).float().cpu())
                labels[delay].append(target)

    shuffled = list(range(sum(labels[delays[0]].count(target) for target in range(len(spec["situations"])))))
    random.Random(seed + (0 if split == "train" else 1_000_000)).shuffle(shuffled)
    result = {}
    for delay in delays:
        order = torch.tensor(shuffled)
        result[delay] = {
            "features": {channel: torch.stack(values).index_select(0, order)
                         for channel, values in rows[delay].items()},
            "labels": torch.tensor(labels[delay], dtype=torch.long).index_select(0, order),
        }
    return result


def _ridge_predict(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor,
                   ridge: float) -> torch.Tensor:
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    train = (train_x - mean) / scale
    evaluate = (eval_x - mean) / scale
    dimension = max(train.shape[1], 1)
    kernel = train @ train.T / dimension
    target = F.one_hot(train_y, num_classes=len(STATE_SURVIVAL_SPEC["situations"])).float()
    alpha = torch.linalg.solve(
        kernel + ridge * torch.eye(kernel.shape[0], dtype=kernel.dtype), target
    )
    return evaluate @ train.T / dimension @ alpha


def probe_channel(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor,
                  eval_y: torch.Tensor, seed: int) -> dict:
    ridge = STATE_SURVIVAL_SPEC["probe_ridge"]
    logits = _ridge_predict(train_x, train_y, eval_x, ridge)
    generator = torch.Generator().manual_seed(seed)
    shuffled_y = train_y.index_select(0, torch.randperm(len(train_y), generator=generator))
    shuffled_logits = _ridge_predict(train_x, shuffled_y, eval_x, ridge)
    return {
        "accuracy": float((logits.argmax(-1) == eval_y).float().mean()),
        "shuffled_label_accuracy": float((shuffled_logits.argmax(-1) == eval_y).float().mean()),
    }


def run_seed(seed: int) -> dict:
    spec = STATE_SURVIVAL_SPEC
    torch.manual_seed(seed + 3_000_000)
    bridge = ThalamicBridge(
        c_dim=2 * spec["engine_dim"],
        d_model=spec["bridge"]["output_dim"],
        hub_dim=spec["bridge"]["hub_dim"],
    ).eval()
    train = collect_examples(seed, "train", bridge)
    evaluate = collect_examples(seed, "eval", bridge)
    result = {}
    for delay in spec["delay_steps"]:
        result[str(delay)] = {}
        for channel in spec["channels"]:
            result[str(delay)][channel] = probe_channel(
                train[delay]["features"][channel], train[delay]["labels"],
                evaluate[delay]["features"][channel], evaluate[delay]["labels"],
                seed + delay * 101 + spec["channels"].index(channel),
            )
        print(f"[seed {seed}] delay={delay} " + " ".join(
            f"{channel}={result[str(delay)][channel]['accuracy']:.3f}"
            for channel in spec["channels"]
        ), flush=True)
    return {"seed": seed, "delays": result}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/state_survival_results.json")
    parser.add_argument("--seeds", default=",".join(map(str, STATE_SURVIVAL_SPEC["seeds"])))
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if set(seeds) != set(STATE_SURVIVAL_SPEC["seeds"]):
        raise ValueError("STATE-1 requires the complete registered seed pair")
    behavior_path = Path(STATE_SURVIVAL_SPEC["downstream_behavior"]["verdict_path"])
    behavior = json.loads(behavior_path.read_text())
    payload = {
        "experiment": STATE_SURVIVAL_SPEC["experiment"],
        "spec": STATE_SURVIVAL_SPEC,
        "spec_sha256": spec_sha256(),
        "seeds": [run_seed(seed) for seed in seeds],
        "downstream_behavior": {
            "experiment": behavior.get("experiment"),
            "verdict": behavior.get("verdict"),
            "path": str(behavior_path),
            "sha256": sha256_file(behavior_path),
        },
    }
    output = Path(args.output)
    _atomic_json(output, payload)
    print(f"[results] {output}")


if __name__ == "__main__":
    main()
