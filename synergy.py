#!/usr/bin/env python3
"""SYNERGY-1: combine independently hidden clues through the canonical GRAFT path."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graft_behavior import GraftActionChannel, _memory_state, sha256_file
from measurement.synergy_registry import (
    SYNERGY_SPEC,
    experiment as synergy_experiment,
    spec_sha256,
)
from pure import PureMind
from trinity import HFDecoder, QuantumC, RecurrentWorkspaceBridge


@dataclass
class SplitCueExample:
    quantum: tuple[torch.Tensor, torch.Tensor]
    memory: tuple[torch.Tensor, torch.Tensor]
    module_a: int
    module_b: int
    target: int


class SynergyActionChannel(nn.Module):
    """GRAFT channel with an optional standard recurrent sensory control."""

    def __init__(self, arm: str, state_dim: int, d_model: int, spec: dict):
        super().__init__()
        self.arm = arm
        self.cells = spec["cells_per_module"]
        workspace_rounds = None
        if "_workspace_" in arm:
            workspace_rounds = int(arm.rsplit("_", 1)[-1])
        bind_roles = arm.endswith("_relation")
        if bind_roles:
            workspace_rounds = spec["relation_rounds"]
        bridge = None
        if workspace_rounds is not None:
            registered_rounds = (
                [spec["relation_rounds"]] if bind_roles else spec["workspace_rounds"]
            )
            if workspace_rounds not in registered_rounds:
                raise ValueError("workspace arm names an unregistered round count")
            bridge = RecurrentWorkspaceBridge(
                c_dim=state_dim,
                d_model=d_model,
                hub_dim=spec["bridge_hub_dim"],
                alpha=0.5,
                rounds=workspace_rounds,
                bind_roles=bind_roles,
            )
        self.action = GraftActionChannel(
            state_dim=state_dim,
            d_model=d_model,
            gate_rho=spec["gate_rho"],
            hub_dim=spec["bridge_hub_dim"],
            bridge=bridge,
        )
        self.recurrent = None
        if arm == "gru":
            if spec["gru_hidden_dim"] != state_dim:
                raise ValueError("registered GRU hidden width must match the bridge state width")
            self.recurrent = nn.GRU(state_dim, spec["gru_hidden_dim"], batch_first=True)

    @property
    def projector(self) -> nn.Linear:
        return self.action.projector

    def _bridge_states(
        self, pairs: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list:
        if "_workspace_" in self.arm or self.arm.endswith("_relation"):
            return pairs
        if self.recurrent is None:
            return [torch.cat(pair, dim=0) for pair in pairs]
        sequence = torch.stack([
            torch.stack((pair[0].mean(0), pair[1].mean(0)), dim=0) for pair in pairs
        ])
        hidden = self.recurrent(sequence)[0][:, -1]
        return [row.unsqueeze(0).expand(self.cells, -1) for row in hidden]

    def training_codes(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.action.training_codes(self._bridge_states(pairs))

    def inference_codes(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.action.inference_codes(self._bridge_states(pairs))


def _module_snapshot(seed: int, split: str, module: int, cue_index: int, repeat: int,
                     spec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    split_offset = 0 if split == "train" else 1_000_000
    module_offset = 0 if module == 0 else 10_000_000
    trial_seed = seed * 100_000_000 + split_offset + module_offset + cue_index * 1_000 + repeat
    torch.manual_seed(trial_seed)
    c = QuantumC(
        nc=spec["cells_per_module"],
        dim=spec["engine_dim"],
        max_cells=spec["cells_per_module"],
    )
    encoder = PureMind(store=None, c_engine=c)
    for _ in range(spec["warm_steps"]):
        c.step()
    cue = spec["module_a_cues" if module == 0 else "module_b_cues"][cue_index]
    nuisance_index = (cue_index * 3 + repeat + module * 5) % len(spec["nuisance_words"])
    payload = encoder.encode_sense([cue, spec["nuisance_words"][nuisance_index]])
    for _ in range(spec["sense_steps"]):
        c.step(x_input=payload)
    lo, hi = spec["delay_steps"]
    delay = lo + trial_seed % (hi - lo + 1)
    for _ in range(delay):
        c.step()
    quantum = c.get_phase_states().clone()
    memory = _memory_state(payload, spec["cells_per_module"], "phase")
    if quantum.shape != (spec["cells_per_module"], spec["state_dim"]):
        raise RuntimeError("QuantumC split-cue state shape drifted from the registered contract")
    if memory.shape != quantum.shape:
        raise RuntimeError("direct-memory split-cue state shape does not match QuantumC")
    return quantum, memory


def _target_index(module_a: int, module_b: int, spec: dict) -> int:
    table = spec.get("target_table")
    if table is None:
        return (module_a + module_b) % len(spec["actions"])
    return int(table[module_a][module_b])


def build_examples(seed: int, split: str, spec: dict = SYNERGY_SPEC) -> list[SplitCueExample]:
    repeats = spec[f"{split}_repeats_per_pair"]
    cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for module, cues in enumerate((spec["module_a_cues"], spec["module_b_cues"])):
        for cue_index in range(len(cues)):
            for repeat in range(repeats):
                cache[(module, cue_index, repeat)] = _module_snapshot(
                    seed, split, module, cue_index, repeat, spec
                )
    examples = []
    for module_a in range(len(spec["module_a_cues"])):
        for module_b in range(len(spec["module_b_cues"])):
            for repeat in range(repeats):
                qa, ma = cache[(0, module_a, repeat)]
                qb, mb = cache[(1, module_b, repeat)]
                examples.append(SplitCueExample(
                    quantum=(qa, qb),
                    memory=(ma, mb),
                    module_a=module_a,
                    module_b=module_b,
                    target=_target_index(module_a, module_b, spec),
                ))
    random.Random(seed + (0 if split == "train" else 1_000_000)).shuffle(examples)
    return examples


def audit_examples(examples: list[SplitCueExample]) -> dict:
    pairs = Counter((row.module_a, row.module_b) for row in examples)
    targets = Counter(row.target for row in examples)
    by_a: dict[int, Counter] = defaultdict(Counter)
    by_b: dict[int, Counter] = defaultdict(Counter)
    for row in examples:
        by_a[row.module_a][row.target] += 1
        by_b[row.module_b][row.target] += 1
    return {
        "examples": len(examples),
        "pair_count": len(pairs),
        "examples_per_pair": min(pairs.values()) if pairs else 0,
        "target_counts": {str(key): targets[key] for key in sorted(targets)},
        "module_a_target_counts": {
            str(key): {str(target): by_a[key][target] for target in sorted(targets)}
            for key in sorted(by_a)
        },
        "module_b_target_counts": {
            str(key): {str(target): by_b[key][target] for target in sorted(targets)}
            for key in sorted(by_b)
        },
    }


def _pairs(examples: list[SplitCueExample], arm: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    source = "quantum" if arm == "quantum_pair" or arm.startswith("quantum_") else "memory"
    return [getattr(row, source) for row in examples]


def _action_token_ids(decoder: HFDecoder, spec: dict) -> list[int]:
    ids = []
    for action in spec["actions"]:
        encoded = decoder.tokenizer.encode(" " + action, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action must be one token: {action} -> {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError("action token ids are not unique")
    return ids


def _prompt_tokens(decoder: HFDecoder, spec: dict) -> torch.Tensor:
    return decoder.tokenizer(spec["question"], return_tensors="pt").input_ids.to(decoder.device)


def _arm_seed(seed: int, arm: str, spec: dict) -> int:
    """Resolve an arm-local seed without coupling repaired runs to roster order."""
    offsets = spec.get("arm_seed_offsets")
    if offsets is None:
        return seed + spec["arms"].index(arm) * 100_000
    if set(offsets) != set(spec["arms"]):
        raise ValueError("registered arm seed offsets do not match the arm roster")
    offset = offsets[arm]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(f"registered arm seed offset is invalid: {arm}")
    return seed + offset


def train_channel(decoder: HFDecoder, channel: SynergyActionChannel,
                  examples: list[SplitCueExample], arm: str, seed: int,
                  action_ids: list[int], spec: dict) -> list[float]:
    channel.to(decoder.device).train()
    optimizer = torch.optim.AdamW(
        channel.parameters(), lr=spec["learning_rate"],
        weight_decay=spec["weight_decay"],
    )
    prompt = _prompt_tokens(decoder, spec)
    actions = torch.tensor(action_ids, device=decoder.device)
    rng = random.Random(_arm_seed(seed, arm, spec))
    neutral = []
    with torch.no_grad():
        for text in spec["neutral_prompts"]:
            tokens = decoder.tokenizer(text, return_tensors="pt").input_ids.to(decoder.device)
            base = F.log_softmax(decoder(tokens, None)[:, -1, :].float(), -1)
            neutral.append((tokens, base))
    losses = []
    source_pairs = _pairs(examples, arm)
    for step in range(1, spec["train_steps"] + 1):
        indices = [rng.randrange(len(examples)) for _ in range(spec["batch_size"])]
        pairs = [tuple(part.to(decoder.device) for part in source_pairs[index]) for index in indices]
        codes = channel.training_codes(pairs)
        gate = codes.unsqueeze(1).expand(-1, prompt.shape[1], -1)
        tokens = prompt.expand(len(indices), -1)
        logits = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
        targets = torch.tensor([examples[index].target for index in indices], device=decoder.device)
        action_loss = F.cross_entropy(logits.index_select(-1, actions), targets)
        neutral_tokens, base_lp = neutral[(step - 1) % len(neutral)]
        neutral_gate = codes.unsqueeze(1).expand(-1, neutral_tokens.shape[1], -1)
        neutral_batch = neutral_tokens.expand(len(indices), -1)
        gated_lp = F.log_softmax(
            decoder(neutral_batch, neutral_gate, gate_projector=channel.projector)[:, -1, :].float(), -1
        )
        language_kl = (gated_lp.exp() * (gated_lp - base_lp)).sum(-1).mean()
        loss = action_loss + spec["language_kl_weight"] * language_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(channel.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            print(
                f"[{seed}:{arm}] step={step} loss={sum(losses[-50:]) / 50:.4f} "
                f"action={float(action_loss.detach()):.4f} neutralKL={float(language_kl.detach()):.4f}",
                flush=True,
            )
    return losses


def _partner_permutation(examples: list[SplitCueExample], spec: dict | int) -> list[int]:
    if isinstance(spec, int):
        target_for = lambda a, b: (a + b) % spec
    else:
        target_for = lambda a, b: _target_index(a, b, spec)
    permutation = []
    for index, row in enumerate(examples):
        candidate = (index + 1) % len(examples)
        while candidate != index:
            other = examples[candidate]
            changed_target = target_for(row.module_a, other.module_b) != row.target
            if other.module_b != row.module_b and changed_target:
                break
            candidate = (candidate + 1) % len(examples)
        if candidate == index:
            raise RuntimeError("could not build registered partner shuffle")
        permutation.append(candidate)
    return permutation


@torch.no_grad()
def evaluate_channel(decoder: HFDecoder, channel: SynergyActionChannel,
                     examples: list[SplitCueExample], arm: str,
                     action_ids: list[int], spec: dict) -> dict:
    channel.eval()
    prompt = _prompt_tokens(decoder, spec)
    actions = torch.tensor(action_ids, device=decoder.device)
    normal = _pairs(examples, arm)
    zero_a = torch.zeros_like(normal[0][0])
    zero_b = torch.zeros_like(normal[0][1])
    partner = _partner_permutation(examples, spec)
    all_conditions = {
        "normal": normal,
        "module_a_only": [(pair[0], zero_b) for pair in normal],
        "module_b_only": [(zero_a, pair[1]) for pair in normal],
        "partner_shuffle": [(pair[0], normal[partner[index]][1]) for index, pair in enumerate(normal)],
        "role_swap": [(pair[1], pair[0]) for pair in normal],
        "recovered": normal,
    }
    conditions = {name: all_conditions[name] for name in spec["interventions"]}

    def action_logits(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rows = []
        for start in range(0, len(pairs), spec["batch_size"]):
            stop = min(start + spec["batch_size"], len(pairs))
            device_pairs = [tuple(part.to(decoder.device) for part in pair) for pair in pairs[start:stop]]
            codes = channel.inference_codes(device_pairs)
            tokens = prompt.expand(stop - start, -1)
            gate = codes.unsqueeze(1).expand(-1, prompt.shape[1], -1)
            logits = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
            rows.append(logits.index_select(-1, actions).cpu())
        return torch.cat(rows)

    logits = {name: action_logits(pairs) for name, pairs in conditions.items()}
    labels = torch.tensor([row.target for row in examples])
    metrics = {
        name: {"accuracy": float((value.argmax(-1) == labels).float().mean())}
        for name, value in logits.items()
    }
    metrics["recovered"]["logits_identical"] = bool(torch.equal(logits["normal"], logits["recovered"]))

    normal_codes = channel.inference_codes([
        tuple(part.to(decoder.device) for part in pair) for pair in normal
    ])
    kl_values = []
    for index, text in enumerate(spec["neutral_prompts"]):
        tokens = decoder.tokenizer(text, return_tensors="pt").input_ids.to(decoder.device)
        code = normal_codes[index].view(1, 1, -1).expand(1, tokens.shape[1], -1)
        gated = F.log_softmax(
            decoder(tokens, code, gate_projector=channel.projector)[:, -1, :].float(), -1
        )
        base = F.log_softmax(decoder(tokens, None)[:, -1, :].float(), -1)
        kl_values.append(float((gated.exp() * (gated - base)).sum()))
    return {"conditions": metrics, "neutral_kl_nats": sum(kl_values) / len(kl_values)}


def run_seed(decoder: HFDecoder, seed: int, output_dir: Path, spec: dict) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    train = build_examples(seed, "train", spec)
    evaluate = build_examples(seed, "eval", spec)
    action_ids = _action_token_ids(decoder, spec)
    arms = {}
    checkpoints = {}
    for arm in spec["arms"]:
        torch.manual_seed(_arm_seed(seed, arm, spec))
        channel = SynergyActionChannel(arm, spec["state_dim"], decoder.d_model, spec).to(decoder.device)
        losses = train_channel(decoder, channel, train, arm, seed, action_ids, spec)
        arms[arm] = evaluate_channel(decoder, channel, evaluate, arm, action_ids, spec)
        arms[arm]["final_loss_mean_50"] = sum(losses[-50:]) / min(50, len(losses))
        path = output_dir / f"seed_{seed}_{arm}.pt"
        torch.save({"seed": seed, "arm": arm, "spec_sha256": spec_sha256(spec),
                    "channel": channel.state_dict()}, path)
        checkpoints[arm] = {"path": str(path), "sha256": sha256_file(path)}
        del channel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "seed": seed,
        "arms": arms,
        "checkpoints": checkpoints,
        "dataset_audit": {"train": audit_examples(train), "eval": audit_examples(evaluate)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/synergy_results.json")
    parser.add_argument("--verdict", default="measurement/synergy_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/synergy1")
    parser.add_argument("--experiment", default=SYNERGY_SPEC["experiment"])
    parser.add_argument("--model")
    parser.add_argument("--seeds")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    try:
        spec = synergy_experiment(args.experiment)
    except ValueError:
        from measurement.relation_registry import experiment as relation_experiment
        spec = relation_experiment(args.experiment)
    if args.train_steps is not None:
        spec["train_steps"] = args.train_steps
    model = args.model or spec["model"]
    seeds = spec["seeds"] if args.seeds is None else [int(item) for item in args.seeds.split(",") if item]
    output = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    decoder = HFDecoder(
        model, lora=False, freeze_base=True,
        gate_strength=spec["gate_strength"], gate_rms_max=spec["gate_rms_max"],
    )
    decoder.model.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    seed_rows = [run_seed(decoder, seed, checkpoint_dir, spec) for seed in seeds]
    audits = [row.pop("dataset_audit") for row in seed_rows]
    if any(audit != audits[0] for audit in audits[1:]):
        raise RuntimeError("seed-specific split-cue audits disagree")
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "model": model,
        "dataset_audit": audits[0],
        "seeds": seed_rows,
    }
    if spec["experiment"].startswith("relation1_"):
        payload["source"] = {
            "results": json.loads(Path(spec["source_results"]).read_text()),
            "verdict": json.loads(Path(spec["source_verdict_path"]).read_text()),
        }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(f"[results] {output}")
    if seeds == spec["seeds"] and args.train_steps is None:
        if spec["experiment"].startswith("relation1_"):
            from measurement.relation_gate import adjudicate
        else:
            from measurement.synergy_gate import adjudicate
        verdict = adjudicate(payload)
        verdict_path = Path(args.verdict)
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_tmp = verdict_path.with_name(verdict_path.name + ".tmp")
        verdict_tmp.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
        verdict_tmp.replace(verdict_path)
        print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
