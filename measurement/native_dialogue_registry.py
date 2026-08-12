"""Single source of truth for the self-trained Anima dialogue model."""
from __future__ import annotations

import copy
import hashlib
import json


SPECIAL_TOKENS = (
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<user>", "<assistant>", "<state>", "<memory>",
)

NATIVE_DIALOGUE_SPEC = {
    "experiment": "native_dialogue1_self_trained_conversation_model",
    "checkpoint_format": "anima_native_dialogue_v1",
    "tokenizer": {
        "type": "byte_level_bpe",
        "normalizer": "NFKC",
        "vocab_size": 32000,
        "special_tokens": list(SPECIAL_TOKENS),
    },
    "presets": {
        "micro": {
            "vocab_size": 4096,
            "dim": 192,
            "heads": 4,
            "layers": 4,
            "block_size": 256,
            "dropout": 0.1,
            "ffn_type": "standard",
            "gate_strength": 0.001,
            "n_ca_rules": 8,
        },
        "screen": {
            "vocab_size": 16000,
            "dim": 384,
            "heads": 6,
            "layers": 8,
            "block_size": 512,
            "dropout": 0.1,
            "ffn_type": "standard",
            "gate_strength": 0.001,
            "n_ca_rules": 8,
        },
        "target": {
            "vocab_size": 32000,
            "dim": 832,
            "heads": 13,
            "layers": 13,
            "block_size": 1024,
            "dropout": 0.1,
            "ffn_type": "standard",
            "gate_strength": 0.001,
            "n_ca_rules": 8,
        },
    },
    "training": {
        "seed": 20260813,
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "warmup_fraction": 0.05,
        "dialogue_fraction": 0.35,
        "response_only_fraction": 0.25,
    },
    "generation": {
        "max_new_tokens": 192,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    },
    "thresholds": {
        "minimum_semantic_pass_per_language": 6,
        "panel_items_per_language": 7,
        "maximum_empty": 0,
        "maximum_utf8_damage": 0,
        "maximum_role_leaks": 0,
        "manual_review_required": True,
    },
}


def preset(name: str) -> dict:
    if name not in NATIVE_DIALOGUE_SPEC["presets"]:
        raise ValueError(f"unknown native dialogue preset: {name}")
    return copy.deepcopy(NATIVE_DIALOGUE_SPEC["presets"][name])


def canonical_spec(spec: dict = NATIVE_DIALOGUE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = NATIVE_DIALOGUE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def checkpoint_spec() -> dict:
    """Return only fields that determine checkpoint tensor compatibility."""
    return {
        "checkpoint_format": NATIVE_DIALOGUE_SPEC["checkpoint_format"],
        "special_tokens": list(SPECIAL_TOKENS),
    }


def checkpoint_spec_sha256() -> str:
    return hashlib.sha256(canonical_spec(checkpoint_spec()).encode()).hexdigest()
