"""Single source of truth for GATE-RUNTIME-1."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.balanced_natural_write_registry import BALANCED_NATURAL_WRITE_SPEC


RUNTIME_MEMORY_SHADOW_SPEC = {
    "experiment": "gate_runtime1_answer_inert_memory_shadow",
    "preregistration_commit": "__PREREGISTRATION_COMMIT__",
    "source_experiment": "gate4_balanced_natural_integrated_dialogue_memory",
    "replicates": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["replicates"]),
    "evaluation_seed": 20260812,
    "calibration_seed": 20260811,
    "calibration_rows": BALANCED_NATURAL_WRITE_SPEC["calibration_rows"],
    "evaluation_episodes": BALANCED_NATURAL_WRITE_SPEC["evaluation_episodes"],
    "candidates_per_episode": BALANCED_NATURAL_WRITE_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["fact_kinds"]),
    "distractor_kinds": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["distractor_kinds"]),
    "encoder": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC["runtime"]),
    "fit_method": "canonical_ridge_sorted_calibration",
    "ridge": BALANCED_NATURAL_WRITE_SPEC["ridge"],
    "checkpoint": "checkpoints/gate-runtime1/canonical_semantic_memory_gate.json",
    "audit": {
        "format": "runtime_memory_shadow_v1",
        "path": "per_model_data/memory_shadow.jsonl",
        "raw_text_allowed": False,
        "write_after_primary_store": True,
        "search_after_primary_context": True,
        "filters_primary_memory": False,
        "changes_answer_context": False,
        "default_enabled": False,
    },
    "faults": ["initialization", "search_audit", "write_audit"],
    "thresholds": {
        "minimum_important_selection_rate": 0.90,
        "maximum_distractor_selection_rate": 0.25,
        "maximum_selection_ratio": 0.50,
        "minimum_per_kind_selection_rate": 0.90,
        "maximum_answer_digest_mismatches": 0,
        "maximum_primary_store_digest_mismatches": 0,
        "maximum_primary_search_digest_mismatches": 0,
        "maximum_raw_text_leaks": 0,
        "maximum_missing_write_audits": 0,
        "maximum_missing_search_audits": 0,
    },
}


def canonical_spec(spec: dict = RUNTIME_MEMORY_SHADOW_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = RUNTIME_MEMORY_SHADOW_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
