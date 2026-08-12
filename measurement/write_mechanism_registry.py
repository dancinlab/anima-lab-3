"""Single source of truth for GATE-WRITE-MECHANISM-1."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.integrated_dialogue_memory_registry import (
    INTEGRATED_DIALOGUE_MEMORY_SPEC,
    spec_sha256 as integrated_spec_sha256,
)


WRITE_MECHANISM_SPEC = {
    "experiment": "gate_write_mechanism1_seed_factor_decomposition",
    "preregistration_commit": "25f626b3f",
    "integrated_spec_sha256": integrated_spec_sha256(),
    "seeds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["seeds"]),
    "calibration_rows": INTEGRATED_DIALOGUE_MEMORY_SPEC["calibration_rows"],
    "evaluation_episodes": INTEGRATED_DIALOGUE_MEMORY_SPEC["evaluation_episodes"],
    "candidates_per_episode": INTEGRATED_DIALOGUE_MEMORY_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["fact_kinds"]),
    "distractor_kinds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["distractor_kinds"]),
    "fact_positions": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["fact_positions"]),
    "encoder": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["encoder"]),
    "runtime": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["runtime"]),
    "fit_method": INTEGRATED_DIALOGUE_MEMORY_SPEC["fit_method"],
    "ridge": INTEGRATED_DIALOGUE_MEMORY_SPEC["ridge"],
    "factors": ["template", "identifier", "layout"],
    "arms": {
        "baseline": [],
        "template_swap": ["template"],
        "identifier_swap": ["identifier"],
        "layout_swap": ["layout"],
        "all_swap": ["template", "identifier", "layout"],
    },
    "expected_baselines": {
        "1337": {
            "important_storage_rate": 0.7626953125,
            "distractor_storage_rate": 0.0,
            "commitment_storage_rate": 0.21484375,
            "selection_sha256": "5029440e81289ee5a9794f3735a07c6f11d5fab158dd5da3a208f30f3b024f92",
        },
        "7331": {
            "important_storage_rate": 1.0,
            "distractor_storage_rate": 0.0,
            "commitment_storage_rate": 1.0,
            "selection_sha256": "953b1824a9270067ae9e4254b7656829b02d37c3fd78c3b14d9820a6b2a3c563",
        },
    },
    "thresholds": {
        "expected_selection_threshold": 0.5,
        "selection_threshold_tolerance": 1e-12,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_per_distractor_storage_rate": 0.50,
        "minimum_peer_gap_fraction": 0.80,
        "maximum_inactive_factor_change": 0.10,
    },
}


def canonical_spec(spec: dict = WRITE_MECHANISM_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = WRITE_MECHANISM_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
