from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from measurement.settle_gate import _classify, adjudicate
from measurement.settle_registry import SETTLE_SPEC, canonical_spec, spec_sha256
from reset_experiment import build_reset_episodes, reset_dataset_audit, trace_reset_episode
from settle import _exact_p, _paired


def test_settle_registry_matches_preregistration():
    assert SETTLE_SPEC["preregistration_commit"] == "0a1bffc6a"
    assert SETTLE_SPEC["replicates"] == [0, 1, 2, 3, 4, 5]
    assert SETTLE_SPEC["update_steps"] == [0, 2, 4, 8]
    assert SETTLE_SPEC["update_modes"] == ["autonomous", "frozen"]
    assert SETTLE_SPEC["preserve_query_rng"] is True
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_settle_datasets_are_balanced_deterministic_and_disjoint():
    first = {replicate: build_reset_episodes(replicate, SETTLE_SPEC) for replicate in SETTLE_SPEC["replicates"]}
    second = {replicate: build_reset_episodes(replicate, SETTLE_SPEC) for replicate in SETTLE_SPEC["replicates"]}
    assert first == second
    audit = reset_dataset_audit(first, SETTLE_SPEC)
    assert audit["combined_unique_fingerprints"] == SETTLE_SPEC["episodes_per_replicate"] * 6
    assert not any(audit["cross_replicate_overlap"].values())


def test_frozen_state_and_question_rng_are_exactly_controlled():
    episode = build_reset_episodes(0, SETTLE_SPEC)[0]
    seed = 123456789
    frozen = [trace_reset_episode(episode, seed, count, "frozen", SETTLE_SPEC) for count in SETTLE_SPEC["update_steps"]]
    active_zero = trace_reset_episode(episode, seed, 0, "autonomous", SETTLE_SPEC)
    assert all(row["state_before_digest"] == row["state_after_digest"] for row in frozen)
    assert len({row["query_rng_digest"] for row in frozen}) == 1
    assert all(torch.equal(frozen[0]["query"], row["query"]) for row in frozen[1:])
    assert torch.equal(frozen[0]["query"], active_zero["query"])
    assert frozen[-1]["performed_updates"] == 0
    assert active_zero["performed_updates"] == 0


def test_exact_paired_test_and_verdict_shapes():
    assert _exact_p(0, 0) == 1.0
    assert _exact_p(10, 0) == pytest.approx(2 / 1024)
    value = _paired([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1])
    assert value["autonomous_only_correct"] == 1
    assert value["frozen_only_correct"] == 1
    assert value["net_accuracy_delta"] == 0.0
    assert _classify({"1337": True, "7331": True})[0] == "ST1_AUTONOMOUS_SETTLING_CAUSAL"
    assert _classify({"1337": True, "7331": False})[0] == "ST2_SEED_CONDITIONAL_SETTLING"
    assert _classify({"1337": False, "7331": False})[0] == "ST3_NO_AUTONOMOUS_SETTLING"


def test_committed_settle_result_replays_and_fails_closed():
    results_path = Path("measurement/settle_results.json")
    verdict_path = Path("measurement/settle_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        pytest.skip("SETTLE-1 result is not committed yet")
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["seeds"][0]["modes"][0]["updates"][0]["replicates"][0]["update_audit"]["performed_updates_maximum"] = 1
    assert adjudicate(tampered)["verdict"] == "ST0_INVALID"
