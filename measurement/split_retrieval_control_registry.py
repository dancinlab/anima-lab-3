"""Single source of truth for GATE-RETRIEVAL-CONTROL-2."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    spec_sha256 as gate2_spec_sha256,
)


SPLIT_RETRIEVAL_CONTROL_SPEC = {
    "experiment": "gate_retrieval_control2_split_topic_content",
    "preregistration_commit": "1c61f9e36",
    "gate2_spec_sha256": gate2_spec_sha256(),
    "seeds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["seeds"]),
    "evaluation_episodes": REALISTIC_MEMORY_WRITE_SPEC["evaluation_episodes"],
    "candidates_per_episode": REALISTIC_MEMORY_WRITE_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["fact_kinds"]),
    "fact_positions": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["fact_positions"]),
    "topic_switches_per_episode": REALISTIC_MEMORY_WRITE_SPEC["topic_switches_per_episode"],
    "encoder": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["runtime"]),
    "ridge": REALISTIC_MEMORY_WRITE_SPEC["ridge"],
    "retrieval": {
        "address": "deterministic_character_trigram_vector",
        "address_pool": 2,
        "content": "registered_semantic_importance_ridge_score",
        "content_only": "whole_sentence_cosine",
        "top_k": 3,
        "tie_break": "candidate_index_ascending",
        "shuffled_topic": "cyclic_next_episode",
        "shuffled_content": "cyclic_next_episode",
    },
    "arms": [
        "split_topic_content", "topic_only", "content_only", "shuffled_topic",
        "shuffled_content", "oracle_memory", "no_memory",
    ],
    "thresholds": {
        "split_recall_at_3": 0.95,
        "split_recall_at_1": 0.90,
        "minimum_per_kind_recall_at_1": 0.85,
        "minimum_per_position_recall_at_1": 0.85,
        "oracle_recall_at_3": 0.99,
        "maximum_no_memory_recall_at_3": 0.05,
        "maximum_shuffled_topic_recall_at_3": 0.35,
        "maximum_shuffled_content_recall_at_1": 0.60,
        "minimum_shuffled_content_gap": 0.30,
        "maximum_topic_only_recall_at_1": 0.10,
        "minimum_topic_only_recall_at_3": 0.95,
    },
}


def canonical_spec(spec: dict = SPLIT_RETRIEVAL_CONTROL_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SPLIT_RETRIEVAL_CONTROL_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
