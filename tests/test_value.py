from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from measurement.value_gate import adjudicate, boundary_verdict
from measurement.value_registry import VALUE_SPEC, spec_sha256
from value import build_value_episodes, prefix_episode, value_dataset_audit


def test_value_registry_matches_preregistration():
    assert VALUE_SPEC["preregistration_commit"] == "ec4dca3fc"
    assert VALUE_SPEC["event_counts"] == [4, 8, 12, 16]
    assert VALUE_SPEC["eval_episodes"] == 512
    assert len(VALUE_SPEC["evaluation_combinations"]) == 4
    assert len(spec_sha256()) == 64


def test_value_prefixes_are_nested_balanced_and_keep_the_query():
    episodes = build_value_episodes()
    assert episodes == build_value_episodes()
    audit = value_dataset_audit(episodes)
    assert audit["base"]["episodes"] == VALUE_SPEC["eval_episodes"]
    assert set(audit["base"]["query_triple_counts"].values()) == {1}
    for episode in episodes[:64]:
        previous = None
        for count in VALUE_SPEC["event_counts"]:
            row = prefix_episode(episode, count)
            assert row.query_position == 0
            assert row.target == episode.target
            assert len(set(zip(row.contexts, row.keys))) == count
            assert set(row.values) == set(row.active_values)
            assert all(row.values.count(value) == count // 4 for value in row.active_values)
            if previous is not None:
                assert row.contexts[:len(previous.contexts)] == previous.contexts
                assert row.keys[:len(previous.keys)] == previous.keys
                assert row.values[:len(previous.values)] == previous.values
            previous = row


def test_value_boundary_verdicts_are_fail_closed_by_order():
    assert boundary_verdict([True, True, True, True])[0] == "VB1_READOUT_VALID_THROUGH_16"
    assert boundary_verdict([True, True, True, False])[0] == "VB2_BOUNDARY_12"
    assert boundary_verdict([True, True, False, False])[0] == "VB3_BOUNDARY_8"
    assert boundary_verdict([True, False, False, False])[0] == "VB4_BOUNDARY_4"
    assert boundary_verdict([False, False, False, False])[0] == "VB5_BELOW_4"
    assert boundary_verdict([True, False, True, False])[0] == "VB6_NON_MONOTONIC"


def test_committed_value_result_replays_and_rejects_spec_changes():
    results_path = Path("measurement/value_results.json")
    verdict_path = Path("measurement/value_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["dataset_audit"]["prefixes"]["4"]["value_balanced"] -= 1
    assert adjudicate(changed)["verdict"] == "VB0_INVALID"
