from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from measurement.mechanism_gate import _classify, _controls_pass, adjudicate
from measurement.mechanism_registry import MECHANISM_SPEC, canonical_spec, spec_sha256
from quantum_engine_fast import CANONICAL_DYNAMICS_COMPONENTS, QuantumConsciousnessEngineFast
from reset_experiment import build_reset_episodes, trace_reset_episode


def test_mechanism_registry_matches_preregistration_and_runtime_ssot():
    assert MECHANISM_SPEC["preregistration_commit"] == "76a7141d1"
    assert MECHANISM_SPEC["components"] == list(CANONICAL_DYNAMICS_COMPONENTS)
    assert [row["name"] for row in MECHANISM_SPEC["interventions"]] == [
        "intact",
        "frozen",
        *[f"without_{name}" for name in CANONICAL_DYNAMICS_COMPONENTS],
    ]
    assert MECHANISM_SPEC["update_steps"] == [8]
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_default_step_is_bit_exact_and_unknown_ablation_fails():
    torch.manual_seed(741)
    left = QuantumConsciousnessEngineFast(dim=16, initial_cells=4, max_cells=4)
    torch.manual_seed(741)
    right = QuantumConsciousnessEngineFast(dim=16, initial_cells=4, max_cells=4)
    torch.manual_seed(852)
    left.step()
    torch.manual_seed(852)
    right.step(dynamics_ablation=())
    for name in ("_amplitudes", "_phases", "_phase_velocities", "_frustrations"):
        assert torch.equal(getattr(left, name), getattr(right, name))
    with pytest.raises(ValueError, match="unknown dynamics components"):
        left.step(dynamics_ablation=("not_registered",))


def test_intervention_arms_share_the_same_start_and_question_rng():
    episode = build_reset_episodes(0, MECHANISM_SPEC)[0]
    rows = []
    for intervention in MECHANISM_SPEC["interventions"]:
        spec = dict(MECHANISM_SPEC)
        spec["dynamics_ablation"] = intervention["disabled"]
        rows.append(trace_reset_episode(
            episode,
            123456789,
            MECHANISM_SPEC["update_steps"][-1],
            intervention["mode"],
            spec,
        ))
    assert len({row["state_before_digest"] for row in rows}) == 1
    assert len({row["query_rng_digest"] for row in rows}) == 1
    frozen = rows[1]
    assert frozen["state_before_digest"] == frozen["state_after_digest"]
    assert frozen["performed_updates"] == 0
    assert all(row["performed_updates"] == 8 for row in rows[2:])


def test_mechanism_verdict_shapes():
    assert _classify({"1337": [], "7331": []})[0] == "MC3_NO_SINGLE_COMPONENT_NECESSARY"
    assert _classify({"1337": ["phase_rotation"], "7331": ["phase_rotation"]})[0] == "MC1_SINGLE_COMPONENT_NECESSARY"
    assert _classify({
        "1337": ["phase_rotation", "frustration_regulation"],
        "7331": ["phase_rotation", "frustration_regulation"],
    })[0] == "MC2_DISTRIBUTED_COMPONENTS_NECESSARY"
    assert _classify({"1337": ["phase_rotation"], "7331": []})[0] == "MC4_SEED_CONDITIONAL_COMPONENT"


def test_partner_swap_threshold_is_applied_to_the_registered_pooled_sample():
    thresholds = MECHANISM_SPEC["thresholds"]
    arms = {
        "exact_three_candidates": {
            "selection_accuracy": 1.0,
            "accuracy": 1.0,
            "per_value_recall": [1.0] * MECHANISM_SPEC["values"],
        },
        "exact_three_partner_swap": {"accuracy": 0.05},
        "exact_three_recovered": {"prediction_match": 1.0},
    }
    assert _controls_pass(arms, thresholds)
    arms["exact_three_partner_swap"]["accuracy"] = 0.05001
    assert not _controls_pass(arms, thresholds)


def test_committed_mechanism_result_replays_and_fails_closed():
    results_path = Path("measurement/mechanism_results.json")
    verdict_path = Path("measurement/mechanism_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        pytest.skip("MECHANISM-1 result is not committed yet")
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["seeds"][0]["interventions"][0]["disabled"] = ["phase_rotation"]
    assert adjudicate(tampered)["verdict"] == "MC0_INVALID"
