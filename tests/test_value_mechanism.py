from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from measurement.value_mechanism_gate import adjudicate, position_verdict
from measurement.value_mechanism_registry import VALUE_MECHANISM_SPEC, spec_sha256
from value import build_value_episodes
from value_mechanism import position_dataset_audit, position_episode


def test_value_mechanism_registry_matches_preregistration():
    assert VALUE_MECHANISM_SPEC["preregistration_commit"] == "61cf10232"
    assert VALUE_MECHANISM_SPEC["query_positions"] == [0, 4, 8, 12, 15]
    assert VALUE_MECHANISM_SPEC["events_per_episode"] == 16
    assert len(spec_sha256()) == 64


def test_position_move_changes_only_order_and_keeps_query_identity():
    episodes = build_value_episodes()
    audit = position_dataset_audit(episodes)
    assert len(audit["shared_event_set_sha256"]) == 64
    for episode in episodes[:64]:
        baseline = sorted(zip(episode.contexts, episode.keys, episode.values))
        for position in VALUE_MECHANISM_SPEC["query_positions"]:
            row = position_episode(episode, position)
            assert row.query_position == position
            assert row.target == episode.target
            assert row.query_context == episode.query_context
            assert row.query_key == episode.query_key
            assert sorted(zip(row.contexts, row.keys, row.values)) == baseline


def test_position_verdicts_cover_registered_outcomes():
    assert position_verdict([True] * 5, [1.0] * 5, 0.05)[0] == "VP1_POSITION_INVARIANT"
    assert position_verdict([True, True, False, False, False], [1.0, .95, .8, .7, .6], .05)[0] == "VP2_LATE_POSITION_LOSS"
    assert position_verdict([True, False, True, False, False], [1.0, .7, .95, .6, .5], .05)[0] == "VP3_POSITION_SPECIFIC"
    assert position_verdict([False] * 5, [.80, .81, .80, .81, .80], .05)[0] == "VP4_POSITION_NOT_CAUSAL"


def test_committed_position_result_replays_and_rejects_event_changes():
    results_path = Path("measurement/value_mechanism_results.json")
    verdict_path = Path("measurement/value_mechanism_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["dataset_audit"]["positions"]["0"]["minimum_unique_pairs"] -= 1
    assert adjudicate(changed)["verdict"] == "VP0_INVALID"
