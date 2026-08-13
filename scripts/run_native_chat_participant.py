#!/usr/bin/env python3
"""Connect the verified native checkpoint to the public chat broker."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import time

import websockets

from model_loader import ModelWrapper
from native_dialogue_lm import checkpoint_sha256, load_native_model


def normalized_history(rows: object, limit: int) -> list[dict[str, str]]:
    """Convert only broker-issued user/anima turns into the native chat format."""
    if not isinstance(rows, list):
        return []
    history: list[dict[str, str]] = []
    for row in rows[-limit:]:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        text = row.get("text")
        if kind not in {"user", "anima"} or not isinstance(text, str) or not text.strip():
            continue
        history.append({
            "role": "user" if kind == "user" else "assistant",
            "content": text.strip(),
        })
    return history


class NativeChatParticipant:
    def __init__(self, model_dir: Path, status_file: Path, history_limit: int) -> None:
        model, tokenizer, payload = load_native_model(model_dir)
        self.wrapper = ModelWrapper("native-dialogue", model, "anima-native")
        self.wrapper.tokenizer = tokenizer
        self.step = int(payload["step"])
        self.checkpoint_sha256 = checkpoint_sha256(model_dir / "final.pt")
        self.status_file = status_file
        self.history: deque[dict[str, str]] = deque(maxlen=history_limit)
        self.generated = 0

    def write_status(self, connected: bool, error: str | None = None) -> None:
        payload = {
            "format": "anima_native_public_chat_status_v1",
            "updated_at": time.time(),
            "connected": connected,
            "backend": "native-dialogue",
            "checkpoint_step": self.step,
            "checkpoint_sha256": self.checkpoint_sha256,
            "claude_fallback": False,
            "generated_responses": self.generated,
            "last_error": error,
            "raw_dialogue_in_status": False,
        }
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, self.status_file)

    async def generate(self, text: str) -> str | None:
        prior = list(self.history)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.wrapper.generate_dialogue(text, "", prior),
        )

    async def serve_connection(self, websocket) -> None:
        hello = json.loads(await websocket.recv())
        self.history.clear()
        self.history.extend(normalized_history(hello.get("history"), self.history.maxlen or 20))
        self.write_status(True)
        async for raw in websocket:
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(message, dict) or message.get("type") != "msg":
                continue
            kind = message.get("kind")
            text = message.get("text")
            if kind == "anima" and isinstance(text, str) and text.strip():
                self.history.append({"role": "assistant", "content": text.strip()})
                continue
            if kind != "user" or not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            answer = await self.generate(text)
            self.history.append({"role": "user", "content": text})
            if not answer:
                self.write_status(True, "empty_native_response")
                continue
            self.generated += 1
            await websocket.send(json.dumps({
                "type": "msg",
                "text": answer,
                "lang": message.get("lang"),
                "reply_to": message.get("id"),
                "factors": {
                    "model": "anima-native-303m",
                    "checkpoint_step": self.step,
                    "claude_fallback": False,
                },
            }, ensure_ascii=False))
            self.write_status(True)

    async def run(self, broker_url: str) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    broker_url,
                    max_size=2**20,
                    ping_interval=20,
                    ping_timeout=60,
                ) as websocket:
                    backoff = 1.0
                    await self.serve_connection(websocket)
            except asyncio.CancelledError:
                self.write_status(False, "stopped")
                raise
            except Exception as exc:
                self.write_status(False, type(exc).__name__)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=20)
    args = parser.parse_args()
    if args.history_limit < 1:
        parser.error("--history-limit must be positive")
    participant = NativeChatParticipant(
        args.model_dir.expanduser().resolve(),
        args.status_file.expanduser().resolve(),
        args.history_limit,
    )
    asyncio.run(participant.run(args.broker_url))


if __name__ == "__main__":
    main()
