"""Single source of truth for GATE-1."""
from __future__ import annotations

import hashlib
import json


MEMORY_WRITE_GATE_SPEC = {
    "experiment": "gate1_dialogue_memory_write_selection",
    "preregistration_commit": "092b818e3",
    "seeds": [1337, 7331],
    "calibration_rows": 4096,
    "evaluation_episodes": 1024,
    "candidates_per_episode": 8,
    "important_per_episode": 1,
    "fact_kinds": ["preference", "commitment", "goal", "profile"],
    "distractor_kinds": [
        "greeting", "thanks", "filler", "weather", "mood", "question", "ack",
    ],
    "vector_dim": 128,
    "role_dim": 2,
    "feature_dim": 130,
    "fit_method": "canonical_ridge",
    "ridge": 0.001,
    "top_k": 3,
    "shuffle_seed_offset": 50000,
    "random_seed_offset": 70000,
    "arms": [
        "selective_gate", "store_all", "oracle_gate", "matched_random",
        "shuffled_gate", "no_memory",
    ],
    "thresholds": {
        "important_storage_rate": 0.90,
        "recall_at_3": 0.90,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_search_size_ratio": 0.50,
        "maximum_recall_drop_from_all": 0.02,
        "oracle_important_storage_rate": 0.99,
        "oracle_recall_at_3": 0.99,
        "store_all_recall_at_3": 0.90,
        "no_memory_max_recall_at_3": 0.05,
        "minimum_fake_recall_gap": 0.25,
    },
}


def canonical_spec(spec: dict = MEMORY_WRITE_GATE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = MEMORY_WRITE_GATE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
