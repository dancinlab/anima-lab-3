from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from context_settle import build_evaluation_episodes, extended_dataset_audit
from measurement.context_settle_gate import _classify, adjudicate
from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, canonical_spec, spec_sha256


def test_context_settle_registry_matches_preregistration():
    assert CONTEXT_SETTLE_SPEC["preregistration_commit"] == "aa3330d6e"
    assert CONTEXT_SETTLE_SPEC["context_steps"] == [3, 4, 6, 9]
    assert CONTEXT_SETTLE_SPEC["transition_positions"] == [8, 12]
    assert CONTEXT_SETTLE_SPEC["baseline_steps"] == 3
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_context_settle_dataset_is_balanced_deterministic_and_disjoint():
    first = build_evaluation_episodes()
    second = build_evaluation_episodes()
    assert first == second
    audit = extended_dataset_audit(first)
    assert audit["episodes"] == CONTEXT_SETTLE_SPEC["eval_episodes"]
    assert audit["unique_fingerprints"] == CONTEXT_SETTLE_SPEC["eval_episodes"]
    assert not any(audit["source_overlap"].values())
    assert set(audit["event_context_counts"].values()) == {1024}
    assert set(audit["event_key_counts"].values()) == {1024}


def test_context_settle_classification_order_is_fail_closed():
    assert _classify(False, {4: True, 6: True, 9: True})[0] == "CT3_BASELINE_ALREADY_STABLE"
    assert _classify(True, {4: False, 6: True, 9: True}) == (
        "CT1_MINIMUM_SETTLING_FOUND",
        "all context positions recovered at 6 settling steps",
        6,
    )
    assert _classify(True, {4: False, 6: False, 9: False})[0] == "CT2_NO_REGISTERED_SETTLING_RECOVERY"


def test_committed_context_settle_result_replays_and_fails_closed():
    results = Path("measurement/context_settle_results.json")
    verdict = Path("measurement/context_settle_verdict.json")
    if not results.is_file() or not verdict.is_file():
        return
    payload = json.loads(results.read_text())
    expected = json.loads(verdict.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["engines"][0]["candidates"][0]["state_audit"]["context_step_calls"] += 1
    assert adjudicate(changed)["verdict"] == "CT0_INVALID"
