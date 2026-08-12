"""Single source of truth for GATE-RUNTIME-2."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.runtime_memory_shadow_registry import RUNTIME_MEMORY_SHADOW_SPEC


RUNTIME_MEMORY_FIELD_SPEC = {
    "experiment": "gate_runtime2_real_dialogue_shadow_review",
    "preregistration_commit": "__PREREGISTRATION_COMMIT__",
    "source_experiment": RUNTIME_MEMORY_SHADOW_SPEC["experiment"],
    "source_database": "data/conscious-lm/memory.db",
    "source_database_sha256": "7866dacdadbc542efb09fc9d51818de6635ff810306fa978ae1674a9b43b2ad6",
    "source_table": "memories",
    "eligible_role": "user",
    "session_gap_minutes": 30,
    "checkpoint": copy.deepcopy(RUNTIME_MEMORY_SHADOW_SPEC["checkpoint"]),
    "checkpoint_sha256": "afdcef5fa0a64ee946151ec49b3f43c841910f4f9ec908748ee4082537098943",
    "encoder": copy.deepcopy(RUNTIME_MEMORY_SHADOW_SPEC["encoder"]),
    "runtime": copy.deepcopy(RUNTIME_MEMORY_SHADOW_SPEC["runtime"]),
    "audit": {
        "format": "runtime_memory_field_v1",
        "raw_text_allowed": False,
        "source_read_only": True,
        "filters_primary_memory": False,
        "changes_answer_context": False,
        "default_enabled": False,
    },
    "review_labels": [
        "important",
        "sensitive_not_requested",
        "ordinary",
    ],
    "thresholds": {
        "minimum_user_turns": 100,
        "minimum_unique_user_turns": 90,
        "minimum_active_days": 7,
        "minimum_sessions": 3,
        "minimum_reviewed_turns": 100,
        "minimum_important_turns": 20,
        "minimum_sensitive_not_requested_turns": 20,
        "minimum_important_selection_rate": 0.90,
        "maximum_sensitive_not_requested_selection_rate": 0.05,
        "maximum_ordinary_selection_rate": 0.25,
        "maximum_raw_text_leaks": 0,
        "maximum_source_digest_mismatches": 0,
    },
}


def canonical_spec(spec: dict = RUNTIME_MEMORY_FIELD_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RUNTIME_MEMORY_FIELD_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
