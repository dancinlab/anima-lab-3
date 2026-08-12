"""Single source of truth for GATE-CONTROL-1."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.memory_write_gate_registry import MEMORY_WRITE_GATE_SPEC, spec_sha256 as gate1_spec_sha256


SEMANTIC_MEMORY_WRITE_SPEC = {
    "experiment": "gate_control1_semantic_write_positive_control",
    "preregistration_commit": "6a7f03c2e",
    "gate1_spec": copy.deepcopy(MEMORY_WRITE_GATE_SPEC),
    "gate1_spec_sha256": gate1_spec_sha256(),
    "encoder": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "pooling": "attention_mask_mean",
        "normalize": True,
        "max_length": 128,
        "embedding_dim": 384,
        "role_dim": 2,
        "feature_dim": 386,
        "batch_size": 64,
        "device": "cpu",
    },
    "runtime": {
        "python": "3.13",
        "torch": "2.8.0",
        "transformers": "4.55.4",
    },
    "fit_method": "canonical_ridge",
    "ridge": 0.001,
    "selector_arm": "semantic_gate",
    "arms": [
        "semantic_gate", "store_all", "oracle_gate", "matched_random",
        "shuffled_gate", "no_memory",
    ],
    "thresholds": copy.deepcopy(MEMORY_WRITE_GATE_SPEC["thresholds"]),
}


def canonical_spec(spec: dict = SEMANTIC_MEMORY_WRITE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SEMANTIC_MEMORY_WRITE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
