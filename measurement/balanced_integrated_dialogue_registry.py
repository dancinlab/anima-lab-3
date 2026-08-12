"""Single source of truth for GATE-4."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.balanced_natural_write_registry import (
    BALANCED_NATURAL_WRITE_SPEC,
    spec_sha256 as write_spec_sha256,
)
from measurement.content_swap_retrieval_control_registry import (
    CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
    spec_sha256 as retrieval_spec_sha256,
)


BALANCED_INTEGRATED_DIALOGUE_SPEC = {
    "experiment": "gate4_balanced_natural_integrated_dialogue_memory",
    "preregistration_commit": "e52b419c5",
    "write_spec_sha256": write_spec_sha256(),
    "retrieval_spec_sha256": retrieval_spec_sha256(),
    "seeds": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["seeds"]),
    "replicates": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["replicates"]),
    "calibration_rows": BALANCED_NATURAL_WRITE_SPEC["calibration_rows"],
    "evaluation_episodes": BALANCED_NATURAL_WRITE_SPEC["evaluation_episodes"],
    "candidates_per_episode": BALANCED_NATURAL_WRITE_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["fact_kinds"]),
    "distractor_kinds": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["distractor_kinds"]),
    "fact_positions": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["fact_positions"]),
    "templates": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["templates"]),
    "encoder": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["runtime"]),
    "fit_method": BALANCED_NATURAL_WRITE_SPEC["fit_method"],
    "ridge": BALANCED_NATURAL_WRITE_SPEC["ridge"],
    "shuffle_seed_offset": BALANCED_NATURAL_WRITE_SPEC["shuffle_seed_offset"],
    "random_seed_offset": BALANCED_NATURAL_WRITE_SPEC["random_seed_offset"],
    "selection": {
        "normal": "balanced_natural_semantic_importance_threshold",
        "fake": "shuffled_label_scores_matched_per_episode",
        "random": "random_matched_per_episode",
        "matching": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["matching"]),
    },
    "retrieval": {
        "address": "seeded_episode_segment_unit_vector",
        "address_dim": 64,
        "address_seed_offset": 410000,
        "replicate_address_stride": 100000,
        "address_pool": CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC["retrieval"]["address_pool"],
        "content": "balanced_natural_semantic_importance_ridge_score",
        "top_k": 3,
        "tie_break": "candidate_index_ascending",
        "stored_candidates_only": True,
    },
    "preservation": {
        "raw_transcript": "immutable_full_candidate_texts",
        "long_term_memory": "separate_selected_candidate_indices",
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
        "expected_selection_threshold": 0.5,
        "selection_threshold_tolerance": 1e-12,
        "minimum_store_all_recall_at_1": 0.90,
        "minimum_store_all_recall_at_3": 0.95,
        "minimum_oracle_recall_at_1": 0.99,
        "maximum_no_memory_recall_at_1": 0.05,
        "minimum_fake_recall_gap": 0.25,
        "minimum_important_storage_rate": 0.90,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_per_distractor_storage_rate": 0.50,
        "maximum_search_size_ratio": 0.50,
        "minimum_integrated_recall_at_1": 0.90,
        "minimum_integrated_recall_at_3": 0.90,
        "minimum_stored_fact_recall_at_1": 0.95,
        "minimum_per_kind_recall_at_1": 0.90,
        "minimum_per_template_recall_at_1": 0.90,
        "minimum_per_position_recall_at_1": 0.85,
        "maximum_recall_drop_from_store_all": 0.05,
    },
}


def canonical_spec(spec: dict = BALANCED_INTEGRATED_DIALOGUE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = BALANCED_INTEGRATED_DIALOGUE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
