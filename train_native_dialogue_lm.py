#!/usr/bin/env python3
"""Train the self-owned Anima dialogue model from random initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conscious_lm import build_model_from_config
from measurement.native_dialogue_registry import (
    NATIVE_DIALOGUE_SPEC,
    checkpoint_spec_sha256,
    preset,
    spec_sha256,
)
from native_dialogue_lm import NativeDialogueTokenizer


DOCUMENT_SPLIT = re.compile(r"\n\s*\n(?=user:\s)", re.IGNORECASE)
ROLE_LINE = re.compile(r"^(user|assistant):\s?(.*)$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dialogue_events(document: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    role = None
    content: list[str] = []
    for line in document.splitlines():
        matched = ROLE_LINE.match(line)
        if matched:
            if role is not None and "\n".join(content).strip():
                events.append((role, "\n".join(content).strip()))
            role = matched.group(1).lower()
            content = [matched.group(2)]
        elif role is not None:
            content.append(line)
    if role is not None and "\n".join(content).strip():
        events.append((role, "\n".join(content).strip()))
    return events


def load_dialogue_examples(path: Path, tokenizer: NativeDialogueTokenizer) -> list[tuple[np.ndarray, np.ndarray]]:
    text = path.read_text(encoding="utf-8")
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for document in DOCUMENT_SPLIT.split(text):
        events = dialogue_events(document)
        if not events or events[0][0] != "user":
            continue
        ids = [tokenizer.ids["<bos>"]]
        response_mask = [False]
        for role, content in events:
            marker = tokenizer.ids[f"<{role}>"]
            piece = tokenizer.encode(content + "\n")
            ids.extend((marker, *piece))
            response_mask.extend((False, *([role == "assistant"] * len(piece))))
        ids.append(tokenizer.ids["<eos>"])
        response_mask.append(events[-1][0] == "assistant")
        if len(ids) >= 3 and any(response_mask):
            examples.append((np.asarray(ids, dtype=np.int32), np.asarray(response_mask, dtype=np.bool_)))
    if not examples:
        raise ValueError(f"no valid dialogue examples in {path}")
    return examples


def load_general_tokens(path: Path, tokenizer: NativeDialogueTokenizer) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    ids = [tokenizer.ids["<bos>"], *tokenizer.encode(text), tokenizer.ids["<eos>"]]
    if len(ids) < 3:
        raise ValueError(f"general corpus is empty: {path}")
    return np.asarray(ids, dtype=np.int32)


class BatchSource:
    def __init__(self, general, dialogue, block_size: int, seed: int, dialogue_fraction: float):
        self.general = general
        if dialogue and isinstance(dialogue[0], tuple):
            dialogue = [dialogue]
        self.dialogue = dialogue
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.dialogue_fraction = dialogue_fraction

    def _general(self):
        stream = self.general[int(self.rng.integers(len(self.general)))]
        start = int(self.rng.integers(max(1, len(stream) - self.block_size - 1)))
        seq = stream[start:start + self.block_size + 1]
        mask = np.ones(len(seq), dtype=np.bool_)
        return seq, mask

    def _dialogue(self):
        group = self.dialogue[int(self.rng.integers(len(self.dialogue)))]
        seq, mask = group[int(self.rng.integers(len(group)))]
        if len(seq) > self.block_size + 1:
            assistant_positions = np.flatnonzero(mask)
            end = min(len(seq), int(assistant_positions[-1]) + 1)
            start = max(0, end - self.block_size - 1)
            seq, mask = seq[start:end], mask[start:end]
        return seq, mask

    def batch(self, size: int, response_only: bool, device: torch.device):
        xs, ys = [], []
        for _ in range(size):
            is_dialogue = bool(self.dialogue) and (
                not self.general or self.rng.random() < self.dialogue_fraction
            )
            seq, response_mask = self._dialogue() if is_dialogue else self._general()
            x = np.zeros(self.block_size, dtype=np.int64)
            y = np.full(self.block_size, -100, dtype=np.int64)
            usable = min(self.block_size, len(seq) - 1)
            x[:usable] = seq[:usable]
            y[:usable] = seq[1:usable + 1]
            if is_dialogue and response_only:
                y[:usable][~response_mask[1:usable + 1]] = -100
            xs.append(x)
            ys.append(y)
        return (
            torch.as_tensor(np.stack(xs), dtype=torch.long, device=device),
            torch.as_tensor(np.stack(ys), dtype=torch.long, device=device),
        )


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def validation_loss(model, source: BatchSource, batches: int, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses = []
    state = source.rng.bit_generator.state
    for _ in range(batches):
        x, y = source.batch(batch_size, response_only=False, device=device)
        logits, _, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1)).item())
    source.rng.bit_generator.state = state
    model.train()
    return float(sum(losses) / len(losses))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-general", type=Path, action="append", default=[])
    parser.add_argument("--train-dialogue", type=Path, action="append", default=[])
    parser.add_argument("--validation-general", type=Path, action="append", default=[])
    parser.add_argument("--validation-dialogue", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(NATIVE_DIALOGUE_SPEC["presets"]), default="micro")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    all_paths = args.train_general + args.train_dialogue + args.validation_general + args.validation_dialogue
    if not args.train_dialogue or not all_paths or any(not path.is_file() for path in all_paths):
        parser.error("training dialogue and every declared corpus file must exist")
    if args.steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        parser.error("steps, batch size and gradient accumulation must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    config = preset(args.preset)
    if tokenizer_path.is_file():
        tokenizer = NativeDialogueTokenizer.load(tokenizer_path)
    else:
        tokenizer = NativeDialogueTokenizer.train(
            args.train_general + args.train_dialogue,
            vocab_size=config["vocab_size"],
        )
        tokenizer.save(tokenizer_path)
    config["vocab_size"] = tokenizer.vocab_size

    train_general = [load_general_tokens(path, tokenizer) for path in args.train_general]
    train_dialogue = [load_dialogue_examples(path, tokenizer) for path in args.train_dialogue]
    val_general = [load_general_tokens(path, tokenizer) for path in args.validation_general]
    val_dialogue = [load_dialogue_examples(path, tokenizer) for path in args.validation_dialogue]
    device = select_device(args.device)
    random.seed(NATIVE_DIALOGUE_SPEC["training"]["seed"])
    np.random.seed(NATIVE_DIALOGUE_SPEC["training"]["seed"])
    torch.manual_seed(NATIVE_DIALOGUE_SPEC["training"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(NATIVE_DIALOGUE_SPEC["training"]["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(NATIVE_DIALOGUE_SPEC["training"]["beta1"], NATIVE_DIALOGUE_SPEC["training"]["beta2"]),
        weight_decay=NATIVE_DIALOGUE_SPEC["training"]["weight_decay"],
    )
    start_step = 0
    train_source = BatchSource(
        train_general, train_dialogue, config["block_size"],
        NATIVE_DIALOGUE_SPEC["training"]["seed"],
        NATIVE_DIALOGUE_SPEC["training"]["dialogue_fraction"],
    )
    validation_source = BatchSource(
        val_general, val_dialogue, config["block_size"],
        NATIVE_DIALOGUE_SPEC["training"]["seed"] + 1, 0.5,
    )
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        resume_hash = resumed.get("checkpoint_spec_sha256")
        if resumed["config"] != config or (
            resume_hash is not None and resume_hash != checkpoint_spec_sha256()
        ):
            raise ValueError("resume checkpoint does not match the registered model")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        start_step = int(resumed["step"])
        train_source.rng.bit_generator.state = resumed["sampler_state"]

    warmup = max(1, int(args.steps * NATIVE_DIALOGUE_SPEC["training"]["warmup_fraction"]))
    response_start = int(args.steps * (1.0 - NATIVE_DIALOGUE_SPEC["training"]["response_only_fraction"]))
    initial_val = validation_loss(model, validation_source, args.validation_batches, args.batch_size, device)
    history = []
    started = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        if step <= warmup:
            scale = step / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        response_only = step >= response_start
        for _ in range(args.grad_accum):
            x, y = train_source.batch(args.batch_size, response_only, device)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)
            )
            with context:
                logits, _, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
                scaled = loss / args.grad_accum
            scaled.backward()
            accumulated += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            value = {
                "step": step,
                "train_ce": round(accumulated / args.grad_accum, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "response_only": response_only,
                "elapsed_seconds": round(time.time() - started, 1),
            }
            history.append(value)
            print(json.dumps(value, sort_keys=True), flush=True)
        if step % args.save_every == 0 and step < args.steps:
            atomic_torch_save({
                "format": NATIVE_DIALOGUE_SPEC["checkpoint_format"] + "_resume",
                "spec_sha256": spec_sha256(),
                "checkpoint_spec_sha256": checkpoint_spec_sha256(),
                "config": config,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_state": train_source.rng.bit_generator.state,
            }, args.output_dir / "resume.pt")

    final_val = validation_loss(model, validation_source, args.validation_batches, args.batch_size, device)
    state = {
        key: value.detach().cpu().to(torch.float16) if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    atomic_torch_save({
        "format": NATIVE_DIALOGUE_SPEC["checkpoint_format"],
        "spec_sha256": spec_sha256(),
        "checkpoint_spec_sha256": checkpoint_spec_sha256(),
        "config": config,
        "model_state": state,
        "step": args.steps,
        "training_seed": NATIVE_DIALOGUE_SPEC["training"]["seed"],
    }, args.output_dir / "final.pt")
    manifest = {
        str(path): {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in all_paths
    }
    summary = {
        "format": NATIVE_DIALOGUE_SPEC["checkpoint_format"] + "_summary",
        "spec_sha256": spec_sha256(),
        "preset": args.preset,
        "config": config,
        "parameters": model.count_params(),
        "steps": args.steps,
        "global_batch": args.batch_size * args.grad_accum,
        "initial_validation_ce": initial_val,
        "final_validation_ce": final_val,
        "validation_descended": final_val < initial_val,
        "dialogue_examples": sum(len(group) for group in train_dialogue),
        "dialogue_examples_per_file": {
            str(path): len(group) for path, group in zip(args.train_dialogue, train_dialogue)
        },
        "data_manifest": manifest,
        "history": history,
        "wall_seconds": time.time() - started,
    }
    atomic_json(summary, args.output_dir / "train_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
