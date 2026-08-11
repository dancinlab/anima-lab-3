"""Single source of truth for CUE-ALIGN-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.cue_context_registry import CUE_CONTEXT_SPEC


CUE_ALIGN_SPEC = {
    "experiment": "cue_align1_storage_query_alignment",
    "preregistration_commit": "407384ca3",
    "source_experiment": CUE_CONTEXT_SPEC["experiment"],
    "source_verdict": "CC3_QUERY_REFIT_INSUFFICIENT",
    "source_results": "measurement/cue_context_results.json",
    "source_verdict_path": "measurement/cue_context_verdict.json",
    "checkpoint_path": "checkpoints/cue_align1/storage_query_alignment.pt",
    "evaluation_combinations": deepcopy(CUE_CONTEXT_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_CONTEXT_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "minimum_cells", "maximum_cells",
            "calibration_episodes", "calibration_engine_seeds", "calibration_data_seed",
            "calibration_seed_base", "missing_fraction", "device",
        )
    },
    "fit_method": "canonical_affine_ridge",
    "ridge": 1e-3,
    "fit_conditions": ["full_pair", "quarter_missing_pair"],
    "models": ["source", "global_affine", "category_oracle", "wrong_pair"],
    "conditions": ["query_full", "query_quarter_missing"],
    "wrong_pair_rule": "same_ordinal_next_context",
    "global_alignment_uses_labels": False,
    "category_oracle_uses_true_label": True,
    "thresholds": {
        "category_accuracy": 0.90,
        "minimum_category_recall": 0.75,
        "minimum_damaged_gain": 0.02,
        "wrong_pair_max_accuracy": 0.25,
    },
}


def canonical_spec(spec: dict = CUE_ALIGN_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CUE_ALIGN_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def calibration_pairs(spec: dict = CUE_ALIGN_SPEC) -> int:
    return spec["calibration_episodes"] * len(spec["calibration_engine_seeds"])

