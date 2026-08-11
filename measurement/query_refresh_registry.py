"""Single source of truth for QUERY-REFRESH-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.cue_history_registry import CUE_HISTORY_SPEC


QUERY_REFRESH_SPEC = {
    "experiment": "query_refresh1_query_state_refresh",
    "preregistration_commit": "5ecc3697e",
    "source_experiment": CUE_HISTORY_SPEC["experiment"],
    "source_verdict": "CH4_HISTORY_SENSITIVE_NOT_SUFFICIENT",
    "source_results": "measurement/cue_history_results.json",
    "source_verdict_path": "measurement/cue_history_verdict.json",
    "evaluation_combinations": deepcopy(CUE_HISTORY_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_HISTORY_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "minimum_cells", "maximum_cells",
            "missing_fraction", "device",
        )
    },
    "histories": ["original", "event_reversed"],
    "conditions": ["query_full", "query_quarter_missing"],
    "query_context_steps": [6, 8, 12, 16],
    "baseline_query_context_steps": 6,
    "event_order_rule": CUE_HISTORY_SPEC["event_order_rule"],
    "thresholds": {
        "category_accuracy": 0.90,
        "minimum_category_recall": 0.75,
        "maximum_history_disagreement": 0.10,
        "minimum_history_disagreement_reduction": 0.05,
        "minimum_processing_disagreement": 0.10,
    },
}


def canonical_spec(spec: dict = QUERY_REFRESH_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = QUERY_REFRESH_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
