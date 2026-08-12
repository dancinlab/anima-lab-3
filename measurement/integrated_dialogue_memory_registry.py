"""Single source of truth for GATE-3."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.content_swap_retrieval_control_registry import (
    CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
    spec_sha256 as retrieval_spec_sha256,
)
from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    spec_sha256 as write_spec_sha256,
)


INTEGRATED_DIALOGUE_MEMORY_SPEC = {
    "experiment": "gate3_integrated_dialogue_memory",
    "preregistration_commit": "0d7b4094b",
    "write_spec_sha256": write_spec_sha256(),
    "retrieval_spec_sha256": retrieval_spec_sha256(),
    "seeds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["seeds"]),
    "calibration_rows": REALISTIC_MEMORY_WRITE_SPEC["calibration_rows"],
    "evaluation_episodes": REALISTIC_MEMORY_WRITE_SPEC["evaluation_episodes"],
    "candidates_per_episode": REALISTIC_MEMORY_WRITE_SPEC["candidates_per_episode"],
    "important_per_episode": REALISTIC_MEMORY_WRITE_SPEC["important_per_episode"],
    "fact_kinds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["fact_kinds"]),
    "distractor_kinds": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["distractor_kinds"]),
    "fact_positions": list(range(REALISTIC_MEMORY_WRITE_SPEC["candidates_per_episode"])),
    "topic_switches_per_episode": REALISTIC_MEMORY_WRITE_SPEC["topic_switches_per_episode"],
    "encoder": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["runtime"]),
    "fit_method": REALISTIC_MEMORY_WRITE_SPEC["fit_method"],
    "ridge": REALISTIC_MEMORY_WRITE_SPEC["ridge"],
    "selection": {
        "normal": "semantic_importance_threshold",
        "fake": "shuffled_label_scores_matched_per_episode",
        "random": "random_matched_per_episode",
        "matching": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["matching"]),
    },
    "retrieval": {
        "address": "seeded_episode_segment_unit_vector",
        "address_dim": 64,
        "address_seed_offset": 310000,
        "address_pool": 2,
        "content": "registered_semantic_importance_ridge_score",
        "top_k": 3,
        "tie_break": "candidate_index_ascending",
        "stored_candidates_only": True,
    },
    "arms": [
        "semantic_integrated",
        "store_all_integrated",
        "oracle_integrated",
        "matched_random_integrated",
        "matched_shuffled_integrated",
        "no_memory",
    ],
    "thresholds": {
        "minimum_store_all_recall_at_1": 0.90,
        "minimum_store_all_recall_at_3": 0.95,
        "minimum_oracle_recall_at_1": 0.99,
        "minimum_oracle_recall_at_3": 0.99,
        "maximum_no_memory_recall_at_1": 0.05,
        "maximum_no_memory_recall_at_3": 0.05,
        "minimum_fake_recall_gap": 0.25,
        "minimum_important_storage_rate": 0.90,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_per_distractor_storage_rate": 0.50,
        "maximum_search_size_ratio": 0.50,
        "minimum_integrated_recall_at_1": 0.90,
        "minimum_integrated_recall_at_3": 0.90,
        "minimum_stored_fact_recall_at_1": 0.95,
        "minimum_per_kind_recall_at_1": 0.85,
        "minimum_per_position_recall_at_1": 0.85,
        "maximum_recall_drop_from_store_all": 0.05,
    },
}


def canonical_spec(spec: dict = INTEGRATED_DIALOGUE_MEMORY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = INTEGRATED_DIALOGUE_MEMORY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
