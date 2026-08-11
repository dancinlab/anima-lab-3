"""Single source of truth for CUE-HISTORY-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.cue_align_registry import CUE_ALIGN_SPEC


CUE_HISTORY_SPEC = {
    "experiment": "cue_history1_episode_processing_history",
    "preregistration_commit": "f90ce65e5",
    "source_experiment": CUE_ALIGN_SPEC["experiment"],
    "source_verdict": "CA4_NONLINEAR_OR_EPISODE_SHIFT",
    "source_results": "measurement/cue_align_results.json",
    "source_verdict_path": "measurement/cue_align_verdict.json",
    "evaluation_combinations": deepcopy(CUE_ALIGN_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_ALIGN_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "minimum_cells", "maximum_cells",
            "missing_fraction", "device",
        )
    },
    "histories": [
        "original", "original_repeat", "event_reversed",
        "distractor_swapped", "both_changed",
    ],
    "judged_histories": ["event_reversed", "distractor_swapped", "both_changed"],
    "conditions": ["query_full", "query_quarter_missing"],
    "event_order_rule": "reverse_all_event_triples_and_remap_query_position",
    "distractor_swap_rule": "cyclic_next_episode_within_query_context",
    "thresholds": {
        "category_accuracy": 0.90,
        "minimum_category_recall": 0.75,
        "minimum_damaged_gain": 0.02,
        "minimum_prediction_disagreement": 0.10,
        "minimum_distractor_changed_fraction": 0.90,
    },
}


def canonical_spec(spec: dict = CUE_HISTORY_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CUE_HISTORY_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
