"""Single source of truth for CUE-CONTEXT-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.component2_registry import COMPONENT2_SPEC
from measurement.cue_robust_registry import CUE_ROBUST_SPEC


CUE_CONTEXT_SPEC = {
    "experiment": "cue_context1_storage_query_state_shift",
    "preregistration_commit": "90808831e",
    "source_experiment": CUE_ROBUST_SPEC["experiment"],
    "source_verdict": "CR2_CONTEXT_CATEGORY_NOT_RECOVERED",
    "source_results": "measurement/cue_robust_results.json",
    "source_verdict_path": "measurement/cue_robust_verdict.json",
    "checkpoint_path": "checkpoints/cue_context1/storage_query_components.pt",
    "evaluation_combinations": deepcopy(CUE_ROBUST_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_ROBUST_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "minimum_cells", "maximum_cells", "temperature", "bias", "device",
        )
    },
    "calibration_episodes": COMPONENT2_SPEC["calibration_episodes"],
    "calibration_engine_seeds": deepcopy(COMPONENT2_SPEC["calibration_engine_seeds"]),
    "calibration_data_seed": COMPONENT2_SPEC["calibration_data_seed"],
    "calibration_seed_base": COMPONENT2_SPEC["calibration_seed_base"],
    "fit_method": COMPONENT2_SPEC["fit_method"],
    "model_class": COMPONENT2_SPEC["model_class"],
    "input_dim": COMPONENT2_SPEC["input_dim"],
    "address_dim": COMPONENT2_SPEC["address_dim"],
    "weight_decay": COMPONENT2_SPEC["weight_decay"],
    "missing_fraction": 0.25,
    "training_mask_salt": "cue_context1_train",
    "combined_schedule": "even_storage_odd_query",
    "models": ["source", "storage_only", "query_only", "combined", "fake_query"],
    "conditions": ["storage_full", "storage_quarter_missing",
                   "query_full", "query_quarter_missing"],
    "fake_label_offset": 1,
    "thresholds": {
        "category_accuracy": 0.90,
        "minimum_category_recall": 0.75,
        "maximum_storage_full_regression": 0.02,
        "fake_category_max_accuracy": 0.25,
    },
}


def canonical_spec(spec: dict = CUE_CONTEXT_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CUE_CONTEXT_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def calibration_pairs(spec: dict = CUE_CONTEXT_SPEC) -> int:
    return spec["calibration_episodes"] * len(spec["calibration_engine_seeds"])


def training_mask_indices(index: int, spec: dict = CUE_CONTEXT_SPEC) -> tuple[int, ...]:
    total = calibration_pairs(spec)
    if not 0 <= index < total:
        raise ValueError("training pair index is outside the registered calibration")
    prefix = (
        f'{spec["training_mask_salt"]}|{spec["calibration_data_seed"]}|'
        f'{index}|context|{spec["missing_fraction"]:.2f}|'
    )
    ranked = [
        (hashlib.sha256(f"{prefix}{dimension}".encode()).digest(), dimension)
        for dimension in range(spec["state_dim"])
    ]
    count = int(round(spec["state_dim"] * spec["missing_fraction"]))
    return tuple(sorted(dimension for _, dimension in sorted(ranked)[:count]))


def training_mask_plan_audit(spec: dict = CUE_CONTEXT_SPEC) -> dict:
    masks = [training_mask_indices(index, spec) for index in range(calibration_pairs(spec))]
    encoded = "\n".join(",".join(map(str, row)) for row in masks).encode()
    return {
        "pairs": len(masks),
        "removed_per_state": int(round(spec["state_dim"] * spec["missing_fraction"])),
        "unique_masks": len(set(masks)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
