from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from capacity import build_capacity_episodes, count_spec
from measurement.capacity2_gate import adjudicate
from measurement.capacity2_registry import CAPACITY2_SPEC, canonical_spec, spec_sha256
from quantum_engine_fast import CANONICAL_DYNAMICS_COMPONENTS
from separation import trace_similar_episode


def test_capacity2_registry_matches_preregistration_and_runtime_ssot():
    assert CAPACITY2_SPEC["preregistration_commit"] == "0f05e9066"
    assert CAPACITY2_SPEC["event_counts"] == [2, 3, 4]
    assert CAPACITY2_SPEC["settling_updates"] == 8
    assert CAPACITY2_SPEC["mechanism_component"] == "frustration_regulation"
    assert CAPACITY2_SPEC["canonical_dynamics_components"] == list(CANONICAL_DYNAMICS_COMPONENTS)
    assert [row["name"] for row in CAPACITY2_SPEC["conditions"]] == [
        "baseline", "settled", "without_frustration_regulation",
    ]
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_capacity2_conditions_share_pre_query_state_and_question_rng():
    event_count = 3
    episode = build_capacity_episodes(event_count)[0]
    rows = []
    for condition in CAPACITY2_SPEC["conditions"]:
        spec = count_spec(event_count)
        spec["pre_query_updates"] = condition["updates"]
        spec["pre_query_dynamics_ablation"] = condition["disabled"]
        rows.append(trace_similar_episode(
            episode, 987654321, distinct=True, spec=spec,
        )["update_audit"])
    assert len({row["state_before_sha256"] for row in rows}) == 1
    assert len({row["query_rng_sha256"] for row in rows}) == 1
    assert rows[0]["state_before_sha256"] == rows[0]["state_after_sha256"]
    assert rows[0]["performed_updates"] == 0
    assert rows[1]["performed_updates"] == rows[2]["performed_updates"] == 8
    assert rows[2]["disabled"] == ["frustration_regulation"]


def test_committed_capacity2_result_replays_and_fails_closed():
    results_path = Path("measurement/capacity2_results.json")
    verdict_path = Path("measurement/capacity2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["seeds"][0]["conditions"][1]["updates"] = 7
    assert adjudicate(tampered)["verdict"] == "CP0_INVALID"
