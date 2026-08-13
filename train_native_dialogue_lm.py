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
import shutil
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
ROLE_LINE = re.compile(r"^(state|system|user|assistant):\s?(.*)$", re.IGNORECASE)


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
            if role == "system":
                role = "state"
            content = [matched.group(2)]
        elif role is not None:
            content.append(line)
    if role is not None and "\n".join(content).strip():
        events.append((role, "\n".join(content).strip()))
    return events


def jsonl_dialogue_events(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dialogue JSON at {path}:{line_number}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"dialogue row lacks messages at {path}:{line_number}")
            events = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"invalid message at {path}:{line_number}")
                role = message.get("role")
                content = message.get("content")
                if role == "system":
                    role = "state"
                if role not in {"state", "user", "assistant"} or not isinstance(content, str):
                    raise ValueError(f"invalid role or content at {path}:{line_number}")
                if content.strip():
                    events.append((role, content.strip()))
            yield events


def load_dialogue_examples(path: Path, tokenizer: NativeDialogueTokenizer) -> list[tuple[np.ndarray, np.ndarray]]:
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    if path.suffix == ".jsonl":
        event_stream = jsonl_dialogue_events(path)
    else:
        text = path.read_text(encoding="utf-8")
        event_stream = (dialogue_events(document) for document in DOCUMENT_SPLIT.split(text))
    for events in event_stream:
        if not events or not any(role == "user" for role, _ in events):
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

    def batch(self, size: int, response_only: bool, device: torch.device,
              source_mode: str = "mixed"):
        if source_mode not in {"mixed", "dialogue"}:
            raise ValueError("source_mode must be mixed or dialogue")
        if source_mode == "dialogue" and not self.dialogue:
            raise ValueError("dialogue source is required for dialogue-only batches")
        xs, ys = [], []
        for _ in range(size):
            is_dialogue = source_mode == "dialogue" or bool(self.dialogue) and (
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


def prepare_tokenizer(
    output_dir: Path,
    vocab_size: int,
    training_files: list[Path],
    starting_checkpoint: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[NativeDialogueTokenizer, Path]:
    """Create a tokenizer for a new model or preserve it across continuation.

    Token IDs are part of the learned model, even though they do not change a
    tensor's shape.  A continuation checkpoint therefore owns the tokenizer
    beside it.  Rebuilding a same-sized vocabulary would silently attach the
    old embedding rows to different text pieces.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    if starting_checkpoint is not None:
        source_path = starting_checkpoint.parent / "tokenizer.json"
        if not source_path.is_file():
            raise ValueError(f"starting checkpoint tokenizer is missing: {source_path}")
        source_hash = sha256_file(source_path)
        if expected_sha256 and source_hash != expected_sha256:
            raise ValueError("starting checkpoint tokenizer differs from the data manifest")
        if tokenizer_path.is_file():
            if sha256_file(tokenizer_path) != source_hash:
                raise ValueError("output tokenizer differs from the starting checkpoint tokenizer")
        else:
            temporary = tokenizer_path.with_name(tokenizer_path.name + ".tmp")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, tokenizer_path)
    elif not tokenizer_path.is_file():
        tokenizer = NativeDialogueTokenizer.train(training_files, vocab_size=vocab_size)
        tokenizer.save(tokenizer_path)
    tokenizer = NativeDialogueTokenizer.load(tokenizer_path)
    if expected_sha256 and sha256_file(tokenizer_path) != expected_sha256:
        raise ValueError("output tokenizer differs from the data manifest")
    if tokenizer.vocab_size != int(vocab_size):
        raise ValueError("tokenizer vocabulary size differs from the registered model")
    return tokenizer, tokenizer_path


@torch.no_grad()
def validation_loss(model, source: BatchSource, batches: int, batch_size: int,
                    device: torch.device, source_mode: str = "mixed",
                    response_only: bool = False) -> float:
    model.eval()
    losses = []
    state = source.rng.bit_generator.state
    for _ in range(batches):
        x, y = source.batch(batch_size, response_only=response_only, device=device,
                            source_mode=source_mode)
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


@torch.inference_mode()
def assert_registered_model_is_causal() -> None:
    """Fail before corpus loading if the shared model can read future tokens."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(NATIVE_DIALOGUE_SPEC["training"]["seed"])
        config = preset("micro")
        config["block_size"] = 8
        model = build_model_from_config(config, dropout=0.0).eval()
        left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        right = torch.tensor([[1, 2, 3, 4, 101, 102, 103, 104]], dtype=torch.long)
        left_logits = model(left)[0][:, :4]
        right_logits = model(right)[0][:, :4]
        maximum_delta = (left_logits - right_logits).abs().max().item()
        if maximum_delta > 1e-7:
            raise RuntimeError(
                "registered model is non-causal: future tokens changed prefix logits "
                f"by {maximum_delta:.9g}"
            )
    finally:
        torch.random.set_rng_state(state)


def assert_registered_model_has_finite_gradients() -> None:
    """Fail before corpus loading when a registered forward path cannot train."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(NATIVE_DIALOGUE_SPEC["training"]["seed"])
        config = preset("micro")
        config["block_size"] = 8
        model = build_model_from_config(config, dropout=0.0).train()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        logits = model(tokens)[0]
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        loss.backward()
        if not torch.isfinite(loss) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("registered model produced non-finite training gradients")
    finally:
        torch.random.set_rng_state(state)


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
    parser.add_argument("--weights", type=Path,
                        help="Start a new optimizer phase from a completed native checkpoint")
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--dialogue-only", action="store_true")
    parser.add_argument("--response-only", action="store_true")
    parser.add_argument("--response-only-fraction", type=float,
                        default=NATIVE_DIALOGUE_SPEC["training"]["response_only_fraction"])
    parser.add_argument("--reset-schedule", action="store_true")
    args = parser.parse_args()
    if args.resume and args.weights:
        parser.error("resume and weights are mutually exclusive")

    assert_registered_model_is_causal()
    assert_registered_model_has_finite_gradients()

    manifest = None
    if args.data_manifest:
        manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        if manifest.get("format") != "anima_native_dialogue_data_v2":
            parser.error("unsupported native dialogue data manifest")
        root = args.data_manifest.parent
        splits = manifest["splits"]
        tokenizer_files = manifest.get("tokenizer_files", [])
        args.train_general.extend(root / path for path in splits.get("train_general", []))
        args.train_dialogue.extend(root / path for path in splits.get("train_dialogue", []))
        args.validation_general.extend(root / path for path in splits.get("validation_general", []))
        args.validation_dialogue.extend(root / path for path in splits.get("validation_dialogue", []))
    else:
        tokenizer_files = []

    all_paths = args.train_general + args.train_dialogue + args.validation_general + args.validation_dialogue
    if not args.train_dialogue or not all_paths or any(not path.is_file() for path in all_paths):
        parser.error("training dialogue and every declared corpus file must exist")
    if args.steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        parser.error("steps, batch size and gradient accumulation must be positive")
    if not 0.0 <= args.response_only_fraction <= 1.0:
        parser.error("response-only fraction must be between zero and one")
    if not args.resume and not args.weights and any(
        (args.output_dir / name).exists() for name in ("resume.pt", "final.pt")
    ):
        parser.error("refusing to overwrite an existing checkpoint without --resume or --weights")

    config = preset(args.preset)
    tokenizer_training_files = (
        [args.data_manifest.parent / path for path in tokenizer_files]
        if tokenizer_files else args.train_general + args.train_dialogue
    )
    tokenizer, tokenizer_path = prepare_tokenizer(
        args.output_dir,
        vocab_size=config["vocab_size"],
        training_files=tokenizer_training_files,
        starting_checkpoint=args.resume or args.weights,
        expected_sha256=manifest.get("base_tokenizer_sha256") if manifest else None,
    )
    tokenizer_hash = sha256_file(tokenizer_path)

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
        fused=device.type == "cuda",
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
        if resumed.get("tokenizer_sha256", tokenizer_hash) != tokenizer_hash:
            raise ValueError("resume checkpoint tokenizer identity mismatch")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        start_step = int(resumed["step"])
        train_source.rng.bit_generator.state = resumed["sampler_state"]
    elif args.weights:
        weighted = torch.load(args.weights, map_location=device, weights_only=True)
        weight_hash = weighted.get("checkpoint_spec_sha256")
        if weighted["config"] != config or (
            weight_hash is not None and weight_hash != checkpoint_spec_sha256()
        ):
            raise ValueError("weight checkpoint does not match the registered model")
        if weighted.get("tokenizer_sha256", tokenizer_hash) != tokenizer_hash:
            raise ValueError("weight checkpoint tokenizer identity mismatch")
        model.load_state_dict(weighted["model_state"])
        start_step = int(weighted["step"])

    schedule_start = start_step if args.reset_schedule or args.weights else 0
    schedule_steps = args.steps - schedule_start
    if schedule_steps <= 0:
        raise ValueError("steps must exceed the resumed step")
    warmup = max(1, int(schedule_steps * NATIVE_DIALOGUE_SPEC["training"]["warmup_fraction"]))
    response_start = (
        args.steps + 1 if args.response_only_fraction == 0
        else schedule_start + int(schedule_steps * (1.0 - args.response_only_fraction))
    )
    source_mode = "dialogue" if args.dialogue_only else "mixed"
    initial_response_only = args.response_only or start_step + 1 >= response_start
    initial_val = validation_loss(
        model, validation_source, args.validation_batches, args.batch_size, device,
        source_mode=source_mode,
        response_only=initial_response_only,
    )
    history = []
    started = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        schedule_step = step - schedule_start
        if schedule_step <= warmup:
            scale = schedule_step / warmup
        else:
            progress = (schedule_step - warmup) / max(1, schedule_steps - warmup)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        response_only = args.response_only or step >= response_start
        for _ in range(args.grad_accum):
            x, y = train_source.batch(
                args.batch_size, response_only, device, source_mode=source_mode
            )
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)
            )
            with context:
                logits, _, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at step {step}; refusing to update weights"
                    )
                scaled = loss / args.grad_accum
            scaled.backward()
            accumulated += loss.item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at step {step}: {gradient_norm.item()}"
            )
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
                "tokenizer_sha256": tokenizer_hash,
                "config": config,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_state": train_source.rng.bit_generator.state,
            }, args.output_dir / "resume.pt")

    final_response_only = args.response_only or args.steps >= response_start
    final_val = validation_loss(
        model, validation_source, args.validation_batches, args.batch_size, device,
        source_mode=source_mode,
        response_only=final_response_only,
    )
    atomic_torch_save({
        "format": NATIVE_DIALOGUE_SPEC["checkpoint_format"] + "_resume",
        "spec_sha256": spec_sha256(),
        "checkpoint_spec_sha256": checkpoint_spec_sha256(),
        "tokenizer_sha256": tokenizer_hash,
        "config": config,
        "step": args.steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_state": train_source.rng.bit_generator.state,
    }, args.output_dir / "resume.pt")
    state = {
        key: value.detach().cpu().to(torch.float16) if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    atomic_torch_save({
        "format": NATIVE_DIALOGUE_SPEC["checkpoint_format"],
        "spec_sha256": spec_sha256(),
        "checkpoint_spec_sha256": checkpoint_spec_sha256(),
        "tokenizer_sha256": tokenizer_hash,
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
        "source_mode": source_mode,
        "response_only": args.response_only,
        "response_only_fraction": args.response_only_fraction,
        "validation_response_only": final_response_only,
        "schedule_reset": args.reset_schedule,
        "starting_checkpoint": str(args.weights or args.resume) if (args.weights or args.resume) else None,
        "initial_validation_ce": initial_val,
        "final_validation_ce": final_val,
        "validation_descended": final_val < initial_val,
        "dialogue_examples": sum(len(group) for group in train_dialogue),
        "dialogue_examples_per_file": {
            str(path): len(group) for path, group in zip(args.train_dialogue, train_dialogue)
        },
        "data_manifest": manifest,
        "tokenizer_sha256": tokenizer_hash,
        "history": history,
        "wall_seconds": time.time() - started,
    }
    atomic_json(summary, args.output_dir / f"train_summary_step_{args.steps}.json")
    atomic_json(summary, args.output_dir / "train_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
