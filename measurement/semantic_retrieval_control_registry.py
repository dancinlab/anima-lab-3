"""Single source of truth for GATE-RETRIEVAL-CONTROL-1."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    spec_sha256 as gate2_spec_sha256,
)


SEMANTIC_RETRIEVAL_CONTROL_SPEC = {
    "experiment": "gate_retrieval_control1_semantic_retrieval",
    "preregistration_commit": "b9bd07c6a",
    "gate2_spec_sha256": gate2_spec_sha256(),
    "seeds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["seeds"]),
    "evaluation_episodes": REALISTIC_MEMORY_WRITE_SPEC["evaluation_episodes"],
    "candidates_per_episode": REALISTIC_MEMORY_WRITE_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["fact_kinds"]),
    "fact_positions": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["fact_positions"]),
    "topic_switches_per_episode": REALISTIC_MEMORY_WRITE_SPEC["topic_switches_per_episode"],
    "top_k": REALISTIC_MEMORY_WRITE_SPEC["top_k"],
    "encoder": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["runtime"]),
    "retrieval": {
        "feature_slice": "sentence_embedding_only",
        "feature_dim": REALISTIC_MEMORY_WRITE_SPEC["encoder"]["embedding_dim"],
        "similarity": "cosine_of_l2_normalized_vectors",
        "descending": True,
        "tie_break": "candidate_index_ascending",
        "shuffled_query": "cyclic_next_episode",
    },
    "arms": [
        "semantic_retrieval", "character_retrieval", "oracle_memory",
        "shuffled_query", "no_memory",
    ],
    "thresholds": {
        "semantic_recall_at_3": 0.95,
        "minimum_per_kind_recall": 0.90,
        "minimum_per_position_recall": 0.90,
        "oracle_recall_at_3": 0.99,
        "maximum_shuffled_recall_at_3": 0.25,
        "maximum_no_memory_recall_at_3": 0.05,
        "minimum_shuffled_gap": 0.50,
        "maximum_drop_from_character": 0.05,
    },
}


def canonical_spec(spec: dict = SEMANTIC_RETRIEVAL_CONTROL_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = SEMANTIC_RETRIEVAL_CONTROL_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
