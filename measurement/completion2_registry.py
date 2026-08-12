"""Single source of truth for COMPLETION-2."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC


COMPLETION2_SPEC = {
    "experiment": "completion2_extended_partial_cue_boundary",
    "preregistration_commit": "fb6dcdfe0",
    "source_experiment": KEY_REFRESH2_SPEC["experiment"],
    "source_verdict": "KR2I_FULL_PATH_RECOVERED",
    "source_results": "measurement/key_refresh2_results.json",
    "source_verdict_path": "measurement/key_refresh2_verdict.json",
    "source_condition": KEY_REFRESH2_SPEC["integrated_condition"],
    "evaluation_combinations": deepcopy(KEY_REFRESH2_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(KEY_REFRESH2_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
            "components_per_key", "stores_per_episode", "retrievals_per_episode",
            "transform_calls_per_episode", "query_context_sense_steps",
            "query_key_sense_steps",
        )
    },
    "mask_salt": "completion1",
    "mask_components": ["context", "key"],
    "missing_fractions": [0.25, 0.50, 0.75, 1.0],
    "conditions": {
        "full_cue": [0.0, 0.0],
        "context_quarter_missing": [0.25, 0.0],
        "key_quarter_missing": [0.0, 0.25],
        "both_quarter_missing": [0.25, 0.25],
        "context_half_missing": [0.50, 0.0],
        "key_half_missing": [0.0, 0.50],
        "both_half_missing": [0.50, 0.50],
        "context_three_quarters_missing": [0.75, 0.0],
        "key_three_quarters_missing": [0.0, 0.75],
        "both_three_quarters_missing": [0.75, 0.75],
        "context_absent": [1.0, 0.0],
        "key_absent": [0.0, 1.0],
        "both_absent": [1.0, 1.0],
    },
    "arms": [
        "full_cue", "context_quarter_missing", "key_quarter_missing",
        "both_quarter_missing", "context_half_missing", "key_half_missing",
        "both_half_missing", "context_three_quarters_missing",
        "key_three_quarters_missing", "both_three_quarters_missing",
        "context_absent", "key_absent", "both_absent",
        "exact_context_key_control", "exact_context_key_partner_swap",
    ],
    "thresholds": {
        "damage_selection_accuracy": 0.90,
        "damage_final_accuracy": 0.90,
        "damage_minimum_value_recall": 0.75,
        "absent_max_selection_accuracy": 0.40,
        "absent_max_final_accuracy": 0.40,
        "exact_selection_accuracy": 0.99,
        "exact_final_accuracy": 0.90,
        "exact_minimum_value_recall": 0.75,
        "partner_swap_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = COMPLETION2_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = COMPLETION2_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def cue_mask_indices(episode_index: int, component: str, missing_fraction: float,
                     spec: dict = COMPLETION2_SPEC) -> tuple[int, ...]:
    if component not in spec["mask_components"]:
        raise ValueError("unregistered cue component")
    if missing_fraction not in spec["missing_fractions"]:
        raise ValueError("unregistered missing fraction")
    if not 0 <= episode_index < spec["eval_episodes"]:
        raise ValueError("episode index is outside the registered evaluation")
    prefix = (
        f'{spec["mask_salt"]}|{spec["data_seed"]}|{episode_index}|{component}|'
        f'{missing_fraction:.2f}|'
    )
    ranked = [
        (hashlib.sha256(f"{prefix}{dimension}".encode()).digest(), dimension)
        for dimension in range(spec["state_dim"])
    ]
    count = int(round(spec["state_dim"] * missing_fraction))
    return tuple(sorted(dimension for _, dimension in sorted(ranked)[:count]))


def mask_plan_audit(spec: dict = COMPLETION2_SPEC) -> dict:
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
