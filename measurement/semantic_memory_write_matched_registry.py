"""Single source of truth for GATE-CONTROL-2."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.semantic_memory_write_registry import (
    SEMANTIC_MEMORY_WRITE_SPEC,
    spec_sha256 as control1_spec_sha256,
)


MATCHED_SEMANTIC_MEMORY_WRITE_SPEC = {
    "experiment": "gate_control2_matched_semantic_write_control",
    "preregistration_commit": "9c7dda4a6",
    "control1_spec": copy.deepcopy(SEMANTIC_MEMORY_WRITE_SPEC),
    "control1_spec_sha256": control1_spec_sha256(),
    "matching": {
        "method": "per_episode_score_rank",
        "descending": True,
        "tie_break": "candidate_index_ascending",
    },
    "arms": [
        "semantic_gate", "store_all", "oracle_gate", "matched_random",
        "matched_shuffled_gate", "no_memory",
    ],
}


def canonical_spec(spec: dict = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
