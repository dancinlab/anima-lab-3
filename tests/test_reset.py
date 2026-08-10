from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from measurement.reset_gate import _classify, adjudicate
from measurement.reset_registry import RESET_SPEC, canonical_spec, spec_sha256
from reset import build_reset_episodes, reset_dataset_audit, trace_reset_episode


def test_reset_registry_is_canonical_and_preregistered():
    assert RESET_SPEC["preregistration_commit"] == "d6be8b58e"
    assert RESET_SPEC["update_steps"] == [0, 2, 4, 8]
    assert RESET_SPEC["update_modes"] == [
        "varied_sensory", "repeated_sensory", "autonomous",
    ]
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_reset_datasets_are_balanced_deterministic_disjoint_and_varied():
    first = {replicate: build_reset_episodes(replicate) for replicate in RESET_SPEC["replicates"]}
    second = {replicate: build_reset_episodes(replicate) for replicate in RESET_SPEC["replicates"]}
    assert first == second
    audit = reset_dataset_audit(first)
    assert audit["combined_unique_fingerprints"] == (
        RESET_SPEC["episodes_per_replicate"] * len(RESET_SPEC["replicates"])
    )
    assert not any(audit["cross_replicate_overlap"].values())
    assert audit["varied_input_distinct_count_minimum"] == 8
    assert audit["varied_input_distinct_count_maximum"] == 8


def test_reset_modes_share_zero_update_start_and_apply_registered_inputs():
    episode = build_reset_episodes(0)[0]
    seed = 123456789
    zero = [trace_reset_episode(episode, seed, 0, mode) for mode in RESET_SPEC["update_modes"]]
    for row in zero[1:]:
        assert all(torch.equal(left, right) for left, right in zip(zero[0]["keys"], row["keys"]))
        assert all(torch.equal(left, right) for left, right in zip(zero[0]["values"], row["values"]))
        assert torch.equal(zero[0]["query"], row["query"])

    varied = trace_reset_episode(episode, seed, 8, "varied_sensory")
    repeated = trace_reset_episode(episode, seed, 8, "repeated_sensory")
    autonomous = trace_reset_episode(episode, seed, 8, "autonomous")
    assert len(set(varied["applied_inputs"])) == 8
    assert repeated["applied_inputs"] == [RESET_SPEC["repeated_neutral_word"]] * 8
    assert autonomous["applied_inputs"] == [None] * 8
    assert {varied["performed_updates"], repeated["performed_updates"], autonomous["performed_updates"]} == {8}


def _recovery(varied: tuple[bool, bool], repeated: tuple[bool, bool], autonomous: tuple[bool, bool]):
    return {
        "varied_sensory": {str(seed): value for seed, value in zip(RESET_SPEC["seeds"], varied)},
        "repeated_sensory": {str(seed): value for seed, value in zip(RESET_SPEC["seeds"], repeated)},
        "autonomous": {str(seed): value for seed, value in zip(RESET_SPEC["seeds"], autonomous)},
    }


@pytest.mark.parametrize(
    ("recovery", "verdict"),
    [
        (_recovery((True, True), (True, True), (True, True)), "RS1_AUTONOMOUS_SETTLING"),
        (_recovery((True, True), (True, True), (False, False)), "RS2_SENSORY_FORCING_RESET"),
        (_recovery((True, True), (False, False), (False, False)), "RS3_VARIED_INPUT_RESET"),
        (_recovery((False, False), (False, False), (False, False)), "RS4_NO_REGISTERED_RECOVERY"),
        (_recovery((True, False), (False, False), (False, False)), "RS5_MIXED_MECHANISM"),
    ],
)
def test_reset_registered_verdict_shapes(recovery, verdict):
    assert _classify(recovery)[0] == verdict


def test_committed_reset_result_replays_and_fails_closed():
    results_path = Path("measurement/reset_results.json")
    verdict_path = Path("measurement/reset_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        pytest.skip("RESET-1 result is not committed yet")
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["seeds"][0]["modes"][0]["updates"][0]["replicates"][0]["update_audit"]["performed_updates_maximum"] = 1
    assert adjudicate(tampered)["verdict"] == "RS0_INVALID"
