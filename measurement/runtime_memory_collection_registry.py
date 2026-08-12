"""Single source of truth for GATE-RUNTIME-3."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.runtime_memory_field_registry import RUNTIME_MEMORY_FIELD_SPEC


RUNTIME_MEMORY_COLLECTION_SPEC = {
    "experiment": "gate_runtime3_live_dialogue_collection",
    "preregistration_commit": "55fc9dfd50155435c060ed542820e0a6a386092d",
    "source_experiment": RUNTIME_MEMORY_FIELD_SPEC["experiment"],
    "source_database": ".local/gate-runtime3/data/conscious-lm/memory.db",
    "source_table": RUNTIME_MEMORY_FIELD_SPEC["source_table"],
    "eligible_role": RUNTIME_MEMORY_FIELD_SPEC["eligible_role"],
    "session_gap_minutes": RUNTIME_MEMORY_FIELD_SPEC["session_gap_minutes"],
    "baseline": {
        "total_rows": 0,
        "eligible_user_turns": 0,
        "source_manifest_sha256": hashlib.sha256(b"").hexdigest(),
    },
    "checkpoint": copy.deepcopy(RUNTIME_MEMORY_FIELD_SPEC["checkpoint"]),
    "checkpoint_sha256": RUNTIME_MEMORY_FIELD_SPEC["checkpoint_sha256"],
    "audit": {
        "format": "runtime_memory_collection_v1",
        "raw_text_allowed": False,
        "source_append_only": True,
        "filters_primary_memory": False,
        "changes_answer_context": False,
        "shadow_enabled": True,
    },
    "runtime": {
        "entrypoint": "anima_unified.py",
        "port": 8765,
        "data_root": ".local/gate-runtime3/data",
        "collector_interval_seconds": 900,
    },
    "thresholds": {
        key: copy.deepcopy(value)
        for key, value in RUNTIME_MEMORY_FIELD_SPEC["thresholds"].items()
        if key in {
            "minimum_user_turns",
            "minimum_unique_user_turns",
            "minimum_active_days",
            "minimum_sessions",
            "maximum_raw_text_leaks",
        }
    },
}


def canonical_spec(spec: dict = RUNTIME_MEMORY_COLLECTION_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RUNTIME_MEMORY_COLLECTION_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
