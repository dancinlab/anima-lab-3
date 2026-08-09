#!/usr/bin/env python3
"""Ground hidden QuantumC experiences into forced-choice language actions.

This extends the existing GRAFT path rather than inventing another decoder: QuantumC,
PureMind's native word-to-sense encoder, ThalamicBridge, and HFDecoder are the runtime.
The frozen language model sees the same question on every trial; only the hidden state
changes. A direct vector-memory arm is the positive and specificity control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pure import PureMind
from measurement.graft_behavior_registry import BEHAVIOR_SPEC, experiment, spec_sha256
from trinity import HFDecoder, PSI_BALANCE, QuantumC, ThalamicBridge


@dataclass
class Example:
    state: torch.Tensor
    memory: torch.Tensor
    target: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class GraftActionChannel(nn.Module):
    """The existing GRAFT bridge plus an arm-local decoder projector."""

    def __init__(self, state_dim: int, d_model: int, gate_rho: float):
        super().__init__()
        self.bridge = ThalamicBridge(c_dim=state_dim, d_model=d_model, alpha=0.5)
        self.projector = nn.Linear(d_model, d_model)
        nn.init.normal_(self.projector.weight, mean=0.0, std=2e-3)
        nn.init.zeros_(self.projector.bias)
        self.projector.bias.requires_grad_(False)
        self.gate_rho = gate_rho
        self.register_buffer("population_mean", torch.zeros(1, d_model))
        self.register_buffer("mean_ready", torch.tensor(False))

    def raw_code(self, state: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.bridge(state, seq_len=1)[:, 0, :] - PSI_BALANCE)

    def training_codes(self, states: list[torch.Tensor]) -> torch.Tensor:
        raw = torch.cat([self.raw_code(state) for state in states], dim=0)
        mean = raw.mean(0, keepdim=True)
        with torch.no_grad():
            if not bool(self.mean_ready):
                self.population_mean.copy_(mean)
                self.mean_ready.fill_(True)
            else:
                self.population_mean.mul_(0.99).add_(mean, alpha=0.01)
        centered = raw - mean
        return self.gate_rho * centered / centered.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)

    def inference_codes(self, states: list[torch.Tensor]) -> torch.Tensor:
        raw = torch.cat([self.raw_code(state) for state in states], dim=0)
        centered = raw - self.population_mean
        return self.gate_rho * centered / centered.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def _memory_state(payload: dict, cells: int, readout: str) -> torch.Tensor:
    theta, resultant = payload["global"]
    if readout == "amplitude":
        encoded = theta * resultant
    elif readout == "phase":
        encoded = torch.cat((resultant * torch.cos(theta), resultant * torch.sin(theta)))
    else:
        raise ValueError(f"unknown state readout: {readout}")
    return encoded.unsqueeze(0).expand(cells, -1).clone()


def _consciousness_state(c: QuantumC, readout: str) -> torch.Tensor:
    if readout == "amplitude":
        return c.get_states().clone()
    if readout == "phase":
        return c.get_phase_states().clone()
    raise ValueError(f"unknown state readout: {readout}")


def build_examples(seed: int, split: str) -> list[Example]:
    spec = BEHAVIOR_SPEC
    per_situation = spec[f"{split}_examples_per_situation"]
    offset = 0 if split == "train" else 1_000_000
    examples = []
    for target, situation in enumerate(spec["situations"]):
        for index in range(per_situation):
            trial_seed = seed * 10_000 + target * per_situation + index + offset
            torch.manual_seed(trial_seed)
            c = QuantumC(nc=spec["cells"], dim=spec["state_dim"], max_cells=spec["cells"])
            encoder = PureMind(store=None, c_engine=c)
            for _ in range(spec["warm_steps"]):
                c.step()
            nuisance = spec["nuisance_words"][trial_seed % len(spec["nuisance_words"])]
            payload = encoder.encode_sense([*situation["words"], nuisance])
            for _ in range(spec["sense_steps"]):
                c.step(x_input=payload)
            lo, hi = spec["delay_steps"]
            delay = lo + trial_seed % (hi - lo + 1)
            for _ in range(delay):
                c.step()
            readout = spec.get("readout", "amplitude")
            state = _consciousness_state(c, readout)
            memory = _memory_state(payload, spec["cells"], readout)
            if state.shape[-1] != spec["state_dim"] or memory.shape[-1] != spec["state_dim"]:
                raise RuntimeError("registered state_dim does not match the selected readout")
            examples.append(Example(state, memory, target))
    random.Random(seed + offset).shuffle(examples)
    return examples


def _action_token_ids(tokenizer) -> list[int]:
    ids = []
    for row in BEHAVIOR_SPEC["situations"]:
        encoded = tokenizer.encode(" " + row["action"], add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action must be one token: {row['action']} -> {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError("action token ids are not unique")
    return ids


def _prompt_tokens(decoder: HFDecoder) -> torch.Tensor:
    return decoder.tokenizer(BEHAVIOR_SPEC["question"], return_tensors="pt").input_ids.to(decoder.device)


def train_channel(decoder: HFDecoder, channel: GraftActionChannel, examples: list[Example], arm: str,
                  seed: int, action_ids: list[int]) -> list[float]:
    spec = BEHAVIOR_SPEC
    channel.to(decoder.device).train()
    optimizer = torch.optim.AdamW(channel.parameters(), lr=spec["learning_rate"], weight_decay=0.0)
    prompt = _prompt_tokens(decoder)
    actions = torch.tensor(action_ids, device=decoder.device)
    rng = random.Random(seed + (0 if arm == "consciousness" else 100_000))
    losses = []
    neutral_weight = spec.get("language_kl_weight", 0.0)
    neutral = []
    if neutral_weight:
        with torch.no_grad():
            for text in spec["neutral_prompts"]:
                tokens = decoder.tokenizer(text, return_tensors="pt").input_ids.to(decoder.device)
                base = F.log_softmax(decoder(tokens, None)[:, -1, :].float(), -1)
                neutral.append((tokens, base))
    for step in range(1, spec["train_steps"] + 1):
        batch = [examples[rng.randrange(len(examples))] for _ in range(spec["batch_size"])]
        states = [getattr(row, "state" if arm == "consciousness" else "memory").to(decoder.device) for row in batch]
        codes = channel.training_codes(states)
        gate = codes.unsqueeze(1).expand(-1, prompt.shape[1], -1)
        tokens = prompt.expand(len(batch), -1)
        logits = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
        targets = torch.tensor([row.target for row in batch], device=decoder.device)
        action_loss = F.cross_entropy(logits.index_select(-1, actions), targets)
        language_kl = torch.zeros((), device=decoder.device)
        if neutral:
            neutral_tokens, base_lp = neutral[(step - 1) % len(neutral)]
            neutral_gate = codes.unsqueeze(1).expand(-1, neutral_tokens.shape[1], -1)
            neutral_batch = neutral_tokens.expand(len(batch), -1)
            gated_lp = F.log_softmax(
                decoder(neutral_batch, neutral_gate, gate_projector=channel.projector)[:, -1, :].float(), -1
            )
            language_kl = (gated_lp.exp() * (gated_lp - base_lp)).sum(-1).mean()
        loss = action_loss + neutral_weight * language_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(channel.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            print(f"[{seed}:{arm}] step={step} loss={sum(losses[-50:]) / 50:.4f} "
                  f"action={float(action_loss.detach()):.4f} neutralKL={float(language_kl.detach()):.4f}",
                  flush=True)
    return losses


@torch.no_grad()
def evaluate_channel(decoder: HFDecoder, channel: GraftActionChannel, examples: list[Example], arm: str,
                     seed: int, action_ids: list[int]) -> dict:
    channel.eval()
    prompt = _prompt_tokens(decoder)
    actions = torch.tensor(action_ids, device=decoder.device)
    states = [getattr(row, "state" if arm == "consciousness" else "memory").to(decoder.device)
              for row in examples]
    normal_codes = channel.inference_codes(states)
    labels = torch.tensor([row.target for row in examples], device=decoder.device)
    permutation = []
    for index, label in enumerate(labels.tolist()):
        candidate = (index + 1) % len(examples)
        while labels[candidate].item() == label:
            candidate = (candidate + 1) % len(examples)
        permutation.append(candidate)
    shuffled_codes = normal_codes[torch.tensor(permutation, device=decoder.device)]
    generator = torch.Generator(device=decoder.device).manual_seed(seed + 2_000_000)
    noise_codes = torch.randn(normal_codes.shape, generator=generator, device=decoder.device)
    noise_codes = noise_codes * (
        normal_codes.norm(dim=-1, keepdim=True) / noise_codes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )

    def action_logits(codes):
        rows = []
        batch_size = BEHAVIOR_SPEC["batch_size"]
        for start in range(0, len(examples), batch_size):
            stop = min(start + batch_size, len(examples))
            tokens = prompt.expand(stop - start, -1)
            if codes is None:
                logits = decoder(tokens, None)[:, -1, :].float()
            else:
                gate = codes[start:stop].unsqueeze(1).expand(-1, prompt.shape[1], -1)
                logits = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
            rows.append(logits.index_select(-1, actions).cpu())
        return torch.cat(rows)

    normal = action_logits(normal_codes)
    conditions = {
        "normal": normal,
        "off": action_logits(None),
        "shuffle": action_logits(shuffled_codes),
        "noise": action_logits(noise_codes),
        "recovered": action_logits(normal_codes),
    }
    labels_cpu = labels.cpu()
    metrics = {
        mode: {"accuracy": float((logits.argmax(-1) == labels_cpu).float().mean())}
        for mode, logits in conditions.items()
    }
    metrics["recovered"]["logits_identical"] = bool(torch.equal(normal, conditions["recovered"]))

    kl_values = []
    for index, text in enumerate(BEHAVIOR_SPEC["neutral_prompts"]):
        tokens = decoder.tokenizer(text, return_tensors="pt").input_ids.to(decoder.device)
        code = normal_codes[index % len(normal_codes)].view(1, 1, -1).expand(1, tokens.shape[1], -1)
        gated = F.log_softmax(decoder(tokens, code, gate_projector=channel.projector)[:, -1, :].float(), -1)
        base = F.log_softmax(decoder(tokens, None)[:, -1, :].float(), -1)
        kl_values.append(float((gated.exp() * (gated - base)).sum()))
    metrics["neutral_kl_nats"] = sum(kl_values) / len(kl_values)
    return metrics


def run_seed(decoder: HFDecoder, seed: int, output_dir: Path) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    train = build_examples(seed, "train")
    evaluate = build_examples(seed, "eval")
    action_ids = _action_token_ids(decoder.tokenizer)
    arms = {}
    checkpoints = {}
    for arm in ("consciousness", "memory"):
        torch.manual_seed(seed + (0 if arm == "consciousness" else 100_000))
        channel = GraftActionChannel(BEHAVIOR_SPEC["state_dim"], decoder.d_model,
                                     BEHAVIOR_SPEC["gate_rho"]).to(decoder.device)
        losses = train_channel(decoder, channel, train, arm, seed, action_ids)
        arms[arm] = evaluate_channel(decoder, channel, evaluate, arm, seed, action_ids)
        arms[arm]["final_loss_mean_50"] = sum(losses[-50:]) / min(50, len(losses))
        path = output_dir / f"seed_{seed}_{arm}.pt"
        torch.save({"seed": seed, "arm": arm, "spec_sha256": spec_sha256(BEHAVIOR_SPEC),
                    "channel": channel.state_dict()}, path)
        checkpoints[arm] = {"path": str(path), "sha256": sha256_file(path)}
        del channel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"seed": seed, "arms": arms, "checkpoints": checkpoints}


def main() -> None:
    global BEHAVIOR_SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/graft_behavior_results.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/graft_behavior")
    parser.add_argument("--model", default=BEHAVIOR_SPEC["model"])
    parser.add_argument("--experiment", default=BEHAVIOR_SPEC["experiment"])
    parser.add_argument("--seeds", default=",".join(map(str, BEHAVIOR_SPEC["seeds"])))
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    BEHAVIOR_SPEC = experiment(args.experiment)
    if args.train_steps is not None:
        BEHAVIOR_SPEC["train_steps"] = args.train_steps
    seeds = [int(item) for item in args.seeds.split(",") if item]
    output = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    decoder = HFDecoder(args.model, lora=False, freeze_base=True,
                        gate_strength=BEHAVIOR_SPEC["gate_strength"],
                        gate_rms_max=BEHAVIOR_SPEC["gate_rms_max"])
    decoder.model.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    payload = {
        "experiment": BEHAVIOR_SPEC["experiment"],
        "spec": BEHAVIOR_SPEC,
        "spec_sha256": spec_sha256(BEHAVIOR_SPEC),
        "model": args.model,
        "seeds": [run_seed(decoder, seed, checkpoint_dir) for seed in seeds],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[results] {output}")


if __name__ == "__main__":
    main()
