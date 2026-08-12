"""Single source of truth for GATE-RETRIEVAL-CONTROL-4."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.balanced_retrieval_control_registry import (
    BALANCED_RETRIEVAL_CONTROL_SPEC,
    spec_sha256 as balanced_spec_sha256,
)


CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC = {
    "experiment": "gate_retrieval_control4_within_pool_content_swap",
    "preregistration_commit": "bde992d00",
    "balanced_spec_sha256": balanced_spec_sha256(),
    "seeds": copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC["seeds"]),
    "runtime": copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC["runtime"]),
    "encoder": copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC["encoder"]),
    "evaluation_episodes": BALANCED_RETRIEVAL_CONTROL_SPEC["evaluation_episodes"],
    "candidates_per_episode": BALANCED_RETRIEVAL_CONTROL_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC["fact_kinds"]),
    "fact_positions": copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC["fact_positions"]),
    "retrieval": {
        "address_pool": BALANCED_RETRIEVAL_CONTROL_SPEC["retrieval"]["address_pool"],
        "intervention": "swap_two_scores_within_normal_address_pool",
        "uses_labels": False,
        "uses_episode_order": False,
        "recovery": "apply_identical_swap_twice",
    },
    "arms": [
        "within_pool_content_swap",
        "restored_content",
    ],
    "thresholds": {
        "normal_recall_at_3": 0.95,
        "normal_recall_at_1": 0.90,
        "minimum_per_kind_recall_at_1": 0.85,
        "minimum_per_position_recall_at_1": 0.85,
        "minimum_topic_only_recall_at_1": 0.45,
        "maximum_topic_only_recall_at_1": 0.55,
        "minimum_topic_only_recall_at_3": 0.95,
        "maximum_shuffled_episode_recall_at_3": 0.35,
        "maximum_swapped_recall_at_1": 0.10,
        "minimum_swap_gap": 0.80,
        "oracle_recall_at_3": 0.99,
        "maximum_no_memory_recall_at_3": 0.05,
        "minimum_normal_pool_fact_coverage": 0.99,
    },
}


def canonical_spec(spec: dict = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
