"""Single source of truth for COMPLETION-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.address_center2_registry import ADDRESS_CENTER2_SPEC


COMPLETION_SPEC = {
    "experiment": "completion1_partial_cue_retrieval",
    "preregistration_commit": "eafd591b4",
    "source_experiment": ADDRESS_CENTER2_SPEC["experiment"],
    "source_verdict": "AC1_CONTEXT_CENTER_INTEGRATED_NOT_UNIQUE",
    "source_results": "measurement/address_center2_results.json",
    "source_verdict_path": "measurement/address_center2_verdict.json",
    "evaluation_combinations": deepcopy(ADDRESS_CENTER2_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(ADDRESS_CENTER2_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
            "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode",
        )
    },
    "mask_salt": "completion1",
    "missing_fractions": [0.25, 0.50, 0.75],
    "mask_components": ["context", "key"],
    "arms": [
        "full_cue", "context_half_cue", "key_half_cue",
        "both_quarter_missing", "both_half_missing",
        "both_three_quarters_missing", "exact_context_key_control",
        "exact_context_key_partner_swap",
    ],
    "thresholds": {
        "full_selection_accuracy": 0.90,
        "full_final_accuracy": 0.90,
        "full_minimum_value_recall": 0.75,
        "quarter_selection_accuracy": 0.90,
        "quarter_final_accuracy": 0.90,
        "single_half_selection_accuracy": 0.80,
        "single_half_final_accuracy": 0.80,
        "both_half_selection_accuracy": 0.75,
        "both_half_final_accuracy": 0.75,
        "both_half_minimum_value_recall": 0.60,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = COMPLETION_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = COMPLETION_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def cue_mask_indices(episode_index: int, component: str, missing_fraction: float,
                     spec: dict = COMPLETION_SPEC) -> tuple[int, ...]:
    if component not in spec["mask_components"]:
        raise ValueError("unregistered cue component")
    if missing_fraction not in spec["missing_fractions"]:
        raise ValueError("unregistered missing fraction")
    if not 0 <= episode_index < spec["eval_episodes"]:
        raise ValueError("episode index is outside the registered evaluation")
    ranked = []
    prefix = (
        f'{spec["mask_salt"]}|{spec["data_seed"]}|{episode_index}|{component}|'
        f'{missing_fraction:.2f}|'
    )
    for dimension in range(spec["state_dim"]):
        ranked.append((hashlib.sha256(f"{prefix}{dimension}".encode()).digest(), dimension))
    count = int(round(spec["state_dim"] * missing_fraction))
    return tuple(sorted(dimension for _, dimension in sorted(ranked)[:count]))


def mask_plan_audit(spec: dict = COMPLETION_SPEC) -> dict:
    rows = {}
    for component in spec["mask_components"]:
        for fraction in spec["missing_fractions"]:
            masks = [
                cue_mask_indices(index, component, fraction, spec)
                for index in range(spec["eval_episodes"])
            ]
            encoded = "\n".join(
                ",".join(map(str, indices)) for indices in masks
            ).encode()
            rows[f"{component}:{fraction:.2f}"] = {
                "removed_per_episode": int(round(spec["state_dim"] * fraction)),
                "episodes": len(masks),
                "unique_masks": len(set(masks)),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
    return rows
