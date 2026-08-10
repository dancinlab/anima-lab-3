from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from context import _calibration_spec, _composite, collect_context_states
from key_stability import StableKeyProjector
from measurement.context_gate import adjudicate
from measurement.context_registry import CONTEXT_SPEC, spec_sha256
from separation import build_episodes, dataset_audit, trace_similar_episode


def _projector(classes: int) -> StableKeyProjector:
    model = StableKeyProjector(4, 4, classes, 0.1, True)
    model.eval()
    model.requires_grad_(False)
    return model


def test_context_registry_and_datasets_match_preregistration():
    assert CONTEXT_SPEC["preregistration_commit"] == "db66f1163"
    assert CONTEXT_SPEC["fit_method"] == "canonical_ridge"
    assert CONTEXT_SPEC["component_address_dim"] == 32
    assert CONTEXT_SPEC["composite_address_dim"] == 64
    assert CONTEXT_SPEC["component_weight"] == 1.0
    assert len(CONTEXT_SPEC["evaluation_combinations"]) == 4
    assert len(set(CONTEXT_SPEC["evaluation_names"])) == 4
    assert len(spec_sha256()) == 64
    calibration_spec = _calibration_spec(CONTEXT_SPEC)
    calibration = build_episodes(calibration_spec)
    evaluation = build_episodes(CONTEXT_SPEC)
    assert not ({row.fingerprint() for row in calibration} & {row.fingerprint() for row in evaluation})
    for rows, spec, total in (
        (calibration, calibration_spec, CONTEXT_SPEC["calibration_episodes"]),
        (evaluation, CONTEXT_SPEC, CONTEXT_SPEC["eval_episodes"]),
    ):
        audit = dataset_audit(rows, spec)
        assert audit["episodes"] == total
        assert audit["unique_fingerprints"] == total
        assert len(set(audit["target_counts"].values())) == 1
        assert len(set(audit["query_position_counts"].values())) == 1
        assert len(set(audit["shared_key_counts"].values())) == 1
        assert len(set(audit["query_context_counts"].values())) == 1


def test_composite_address_has_fixed_equal_components_and_masks():
    context_projector = _projector(4)
    key_projector = _projector(4)
    context = torch.randn(3, 4)
    key = torch.randn(2, 4)
    spec = {"component_weight": 1.0, "composite_address_dim": 8}
    normal = _composite(context_projector, key_projector, context, key, spec)
    no_context = _composite(
        context_projector, key_projector, context, key, spec, mask_context=True
    )
    no_key = _composite(
        context_projector, key_projector, context, key, spec, mask_key=True
    )
    assert normal.shape == (8,)
    assert torch.equal(no_context[:4], torch.zeros(4))
    assert torch.equal(no_context[4:], normal[4:])
    assert torch.equal(no_key[:4], normal[:4])
    assert torch.equal(no_key[4:], torch.zeros(4))


def test_trace_exposes_context_states_without_changing_registered_shapes():
    spec = deepcopy(CONTEXT_SPEC)
    spec["eval_episodes"] = 8
    spec["exact_marginal_balance"] = True
    episode = build_episodes(spec)[0]
    trace = trace_similar_episode(episode, 12345, distinct=False, spec=spec)
    assert len(trace["contexts"]) == spec["events_per_episode"]
    assert trace["query_context"].dim() == 2
    assert all(state.shape[1] == spec["state_dim"] for state in trace["contexts"])
    assert trace["query_context"].shape[1] == spec["state_dim"]


def test_committed_context_result_replays_and_fails_closed():
    results_path = Path("measurement/context_results.json")
    verdict_path = Path("measurement/context_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    tampered = deepcopy(payload)
    tampered["evaluations"][0]["integration_audit"]["component_weight"] = 2.0
    assert adjudicate(tampered)["verdict"] == "CX0_INVALID"

    context_loss = deepcopy(payload)
    context_loss["evaluations"][0]["context_classification"]["accuracy"] = 0.5
    assert adjudicate(context_loss)["verdict"] == "CX2_CONTEXT_CODE_LOSS"

    composition_loss = deepcopy(payload)
    for row in composition_loss["evaluations"]:
        normal = row["arms"]["composite_context_key_normal"]
        normal["selection_accuracy"] = 0.5
        normal["accuracy"] = 0.5
    assert adjudicate(composition_loss)["verdict"] == "CX3_COMPOSITION_LOSS"

    value_loss = deepcopy(payload)
    for row in value_loss["evaluations"]:
        normal = row["arms"]["composite_context_key_normal"]
        normal["selection_accuracy"] = 1.0
        normal["accuracy"] = 0.5
        normal["per_value_recall"] = [0.5] * CONTEXT_SPEC["values"]
    assert adjudicate(value_loss)["verdict"] == "CX4_VALUE_READOUT_LOSS"
