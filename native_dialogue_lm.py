"""Tokenizer, prompt and checkpoint path for the self-trained Anima model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from measurement.native_dialogue_registry import (
    NATIVE_DIALOGUE_SPEC,
    SPECIAL_TOKENS,
    checkpoint_spec_sha256,
)


class NativeDialogueTokenizer:
    def __init__(self, backend):
        self.backend = backend
        self.ids = {token: backend.token_to_id(token) for token in SPECIAL_TOKENS}
        missing = [token for token, token_id in self.ids.items() if token_id is None]
        if missing:
            raise ValueError(f"tokenizer is missing required tokens: {missing}")

    @classmethod
    def train(cls, files: Iterable[Path | str], vocab_size: int):
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        paths = [Path(path) for path in files]
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError("all tokenizer training files must exist")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(cls._training_texts(paths), trainer)
        return cls(tokenizer)

    @staticmethod
    def _training_texts(paths: Iterable[Path]):
        """Yield text only, never JSON field names, from registered corpora."""
        for path in paths:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"invalid dialogue JSON at {path}:{line_number}") from exc
                        for message in row.get("messages", []):
                            content = message.get("content") if isinstance(message, dict) else None
                            if isinstance(content, str) and content.strip():
                                yield content
            else:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield line

    @classmethod
    def load(cls, path: Path | str):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.backend.decode(list(token_ids), skip_special_tokens=True)

    def format_prompt(self, text: str, state: str, history: list[dict], max_tokens: int) -> list[int]:
        if not text.strip():
            raise ValueError("dialogue text must not be empty")
        pieces = ["<bos>"]
        if state.strip():
            pieces.extend(("<state>", state.strip(), "\n"))
        for row in history:
            if not isinstance(row, dict):
                continue
            content = str(row.get("content", "")).strip()
            role = row.get("role")
            if content and role in {"user", "assistant"}:
                pieces.extend((f"<{role}>", content, "\n"))
        pieces.extend(("<user>", text.strip(), "\n<assistant>"))
        ids = self.encode("".join(pieces))
        return ids[-max_tokens:]

    def trim_response(self, token_ids: Iterable[int]) -> str:
        stopped = []
        stop_ids = {self.ids["<eos>"], self.ids["<user>"]}
        for token_id in token_ids:
            if token_id in stop_ids:
                break
            stopped.append(token_id)
        return self.decode(stopped).strip()


def checkpoint_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_native_model(directory: Path | str, device: str | None = None):
    import torch
    from conscious_lm import build_model_from_config

    directory = Path(directory)
    checkpoint_path = directory / "final.pt"
    tokenizer_path = directory / "tokenizer.json"
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"native model files are incomplete under {directory}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != NATIVE_DIALOGUE_SPEC["checkpoint_format"]:
        raise ValueError("unsupported native dialogue checkpoint format")
    compatibility_hash = payload.get("checkpoint_spec_sha256")
    if compatibility_hash is not None and compatibility_hash != checkpoint_spec_sha256():
        raise ValueError("native dialogue checkpoint spec mismatch")
    tokenizer = NativeDialogueTokenizer.load(tokenizer_path)
    config = payload["config"]
    if tokenizer.vocab_size != int(config["vocab_size"]):
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    model = build_model_from_config(config, dropout=0.0)
    model.load_state_dict(payload["model_state"], strict=True)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device).eval()
    return model, tokenizer, payload
