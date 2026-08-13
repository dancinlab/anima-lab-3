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
            "dim": 896,
            "heads": 14,
            "layers": 11,
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
    "native_dialogue2": {
        "screen_additional_steps": 12000,
        "target_pretrain_steps": 35000,
        "target_dialogue_steps": 10000,
        "global_batch": 64,
        "sources": {
            "dialogue": {
                "repo_id": "lemon-mint/smol-koreantalk",
                "revision": "04eb12c1f999578d74dd87720bc853cedb7fa156",
                "license": "Apache-2.0",
                "files": [f"data/train-{index:05d}-of-00008.parquet" for index in range(8)],
            },
            "general": {
                "repo_id": "seongchaeae/sage-pretrain-corpus",
                "revision": "6b3da140fbb98a90fcc5b855269c85ded9cde1f1",
                "license": "ODC-BY-1.0",
                "en_files": [f"fineweb_edu/fineweb_edu_part_{index:04d}.parquet" for index in range(8)],
                "ko_files": [f"fineweb2_ko/shard_00_part_{index:03d}.parquet" for index in range(4)],
            },
        },
        "validation_percent": 1,
        "tokenizer_sample_characters_per_language_and_kind": 32_000_000,
    },
    "native_dialogue3": {
        "screen_additional_steps": 12000,
        "global_batch": 32,
        "instruction_examples_per_language": 100_000,
        "instruction_candidate_multiplier": 4,
        "instruction_shards_per_language": 4,
        "memory_examples_per_language": 20_000,
        "validation_percent": 1,
        "maximum_screen_tokens": 513,
        "source": {
            "repo_id": "CohereLabs/aya_collection_language_split",
            "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
            "license": "Apache-2.0",
            "files": {
                "en": "english/train-00000-of-00007.parquet",
                "ko": "korean/train-00000-of-00002.parquet",
            },
        },
    },
    "native_dialogue4": {
        "screen_additional_steps": 12000,
        "global_batch": 32,
        "examples_per_language": 100_000,
        "candidate_multiplier": 2,
        "conversation_shards_per_language": 4,
        "memory_examples_per_language": 20_000,
        "validation_percent": 1,
        "maximum_screen_tokens": 513,
        "sources": {
            "en": {
                "repo_id": "HuggingFaceTB/smol-smoltalk",
                "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
                "license": "Apache-2.0",
                "files": [f"data/train-{index:05d}-of-00004.parquet" for index in range(4)],
            },
            "ko": {
                "repo_id": "IkJun1/korean-qa-dataset",
                "revision": "e1f177a7497cf4e55e54d86101c6c522345441d2",
                "license": "MIT",
                "files": ["QA_dataset.jsonl"],
            },
        },
    },
    "native_dialogue5": {
        "architecture": "byte_level_bpe_attention_causal_local_mix",
        "parameters": 303_628_504,
        "pretrain_steps": 35_000,
        "dialogue_steps": 10_000,
        "global_batch": 64,
        "micro_batch": 16,
        "gradient_accumulation": 4,
        "context_tokens": 1_024,
        "external_pretrained_weights": 0,
        "quality_data_profile": "native_dialogue4",
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
