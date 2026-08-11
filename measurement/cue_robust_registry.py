"""Single source of truth for CUE-ROBUST-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.component2_registry import COMPONENT2_SPEC
from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC


CUE_ROBUST_SPEC = {
    "experiment": "cue_robust1_damage_augmented_readout",
    "preregistration_commit": "eee24d2f2",
    "source_experiment": CUE_MECHANISM_SPEC["experiment"],
    "source_verdict": "CM3_DUAL_CATEGORY_LOSS",
    "source_results": "measurement/cue_mechanism_results.json",
    "source_verdict_path": "measurement/cue_mechanism_verdict.json",
    "source_component_results": "measurement/component2_results.json",
    "source_component_verdict_path": "measurement/component2_verdict.json",
    "source_component_verdict": "CS2_COMPONENT_FIT_INVALID",
    "checkpoint_path": "checkpoints/cue_robust1/damage_augmented_components.pt",
    "evaluation_combinations": deepcopy(CUE_MECHANISM_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CUE_MECHANISM_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
            "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode", "conditions", "arms",
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
    "training_mask_salt": "cue_robust1_train",
    "training_missing_fraction": 0.25,
    "training_mask_components": ["context", "key"],
    "fake_label_offset": 1,
    "thresholds": {
        "full_category_accuracy": 0.90,
        "full_minimum_category_recall": 0.75,
        "maximum_full_category_regression": 0.02,
        "partial_category_accuracy": 0.90,
        "partial_minimum_category_recall": 0.75,
        "full_selection_accuracy": 0.90,
        "full_final_accuracy": 0.90,
        "full_minimum_value_recall": 0.75,
        "single_quarter_selection_accuracy": 0.90,
        "single_quarter_final_accuracy": 0.90,
        "both_quarter_selection_accuracy": 0.90,
        "both_quarter_final_accuracy": 0.90,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
        "fake_category_max_accuracy": 0.25,
    },
}


def canonical_spec(spec: dict = CUE_ROBUST_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = CUE_ROBUST_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def training_examples_per_component(spec: dict = CUE_ROBUST_SPEC) -> int:
    return (
        spec["calibration_episodes"]
        * len(spec["calibration_engine_seeds"])
        * spec["events_per_episode"]
    )


def training_mask_indices(index: int, component: str,
                          spec: dict = CUE_ROBUST_SPEC) -> tuple[int, ...]:
    total = training_examples_per_component(spec)
    if component not in spec["training_mask_components"]:
        raise ValueError("unregistered training cue component")
    if not 0 <= index < total:
        raise ValueError("training cue index is outside the registered calibration")
    prefix = (
        f'{spec["training_mask_salt"]}|{spec["calibration_data_seed"]}|'
        f'{index}|{component}|{spec["training_missing_fraction"]:.2f}|'
    )
    ranked = [
        (hashlib.sha256(f"{prefix}{dimension}".encode()).digest(), dimension)
        for dimension in range(spec["state_dim"])
    ]
    count = int(round(spec["state_dim"] * spec["training_missing_fraction"]))
    return tuple(sorted(dimension for _, dimension in sorted(ranked)[:count]))


def training_mask_plan_audit(spec: dict = CUE_ROBUST_SPEC) -> dict:
    total = training_examples_per_component(spec)
    result = {}
    for component in spec["training_mask_components"]:
        masks = [training_mask_indices(index, component, spec) for index in range(total)]
        encoded = "\n".join(",".join(map(str, row)) for row in masks).encode()
        result[component] = {
            "examples": total,
            "removed_per_example": int(round(
                spec["state_dim"] * spec["training_missing_fraction"]
            )),
            "unique_masks": len(set(masks)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return result
