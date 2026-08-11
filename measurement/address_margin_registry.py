"""Single source of truth for ADDRESS-MARGIN-1."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC


ADDRESS_MARGIN_SPEC = {
    "experiment": "address_margin1_composite_address_margin",
    "preregistration_commit": "799e48bad",
    "source_experiment": CONTEXT_SETTLE2_SPEC["experiment"],
    "source_verdict": "CT2I_COMPOSITION_LOSS",
    "source_results": "measurement/context_settle2_results.json",
    "source_verdict_path": "measurement/context_settle2_verdict.json",
    "evaluation_combinations": deepcopy(CONTEXT_SETTLE2_SPEC["evaluation_combinations"]),
    **{
        name: deepcopy(CONTEXT_SETTLE2_SPEC[name])
        for name in (
            "eval_episodes", "events_per_episode", "distractor_steps", "contexts",
            "keys", "values", "data_seed", "episode_seed_base", "seed_stride",
            "settled_context_steps", "key_sense_steps", "value_sense_steps",
            "distractor_sense_steps", "state_dim", "component_address_dim",
            "composite_address_dim", "value_address_dim", "minimum_cells",
            "maximum_cells", "component_weight", "temperature", "bias", "device",
        )
    },
    "arms": [
        "continuous_frozen", "context_centered", "key_centered",
        "predicted_centers", "oracle_centers", "shifted_center_control",
    ],
    "thresholds": {
        "source_selection_max_accuracy": 0.85,
        "category_accuracy": 0.90,
        "minimum_category_recall": 0.75,
        "center_selection_accuracy": 0.90,
        "center_final_accuracy": 0.90,
        "minimum_value_recall": 0.75,
        "minimum_selection_gain": 0.08,
        "oracle_selection_accuracy": 1.0,
        "oracle_final_accuracy": 0.90,
        "shifted_selection_max_accuracy": 0.05,
    },
}


def canonical_spec(spec: dict = ADDRESS_MARGIN_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = ADDRESS_MARGIN_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
