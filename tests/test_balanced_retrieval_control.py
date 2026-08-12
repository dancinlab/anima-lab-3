import copy
import json
from pathlib import Path

import pytest
import torch

from gate_retrieval_control3 import (
    _episode_addresses,
    _topic_features,
    build_balanced_evaluation,
)
from gate_retrieval_control2 import _metrics
from measurement.balanced_retrieval_control_gate import adjudicate
from measurement.balanced_retrieval_control_registry import (
    BALANCED_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def test_balanced_retrieval_control_is_preregistered_and_pinned():
    spec = BALANCED_RETRIEVAL_CONTROL_SPEC
    assert spec["preregistration_commit"] == "b26e3587a"
    assert spec["fact_positions"] == list(range(8))
    assert spec["retrieval"]["address"] == "seeded_episode_segment_unit_vector"
    assert spec["thresholds"]["split_recall_at_1"] == 0.90
    assert len(spec_sha256()) == 64


def test_fact_position_and_kind_side_are_exactly_balanced():
    spec = BALANCED_RETRIEVAL_CONTROL_SPEC
    episodes = build_balanced_evaluation(1337, spec)
    assert len(episodes) == spec["evaluation_episodes"]
    assert {
        position: sum(row["fact_position"] == position for row in episodes)
        for position in spec["fact_positions"]
    } == {position: 128 for position in spec["fact_positions"]}
    for kind in spec["fact_kinds"]:
        assert {
            side: sum(
                row["kind"] == kind and row["fact_position"] % 2 == side
                for row in episodes
            )
            for side in (0, 1)
        } == {0: 128, 1: 128}


def test_episode_addresses_are_unique_deterministic_and_normalized():
    first = _episode_addresses(1337, 32, 4, 64, 310000)
    second = _episode_addresses(1337, 32, 4, 64, 310000)
    other = _episode_addresses(7331, 32, 4, 64, 310000)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.allclose(first.norm(dim=2), torch.ones(32, 4))
    assert len({row.numpy().tobytes() for row in first.reshape(-1, 64)}) == 128


def test_topic_features_assign_one_address_per_episode_segment():
    spec = copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_balanced_evaluation(1337, spec)
    queries, candidates, audit = _topic_features(episodes, 1337, spec)
    assert queries.shape == (16, 64)
    assert candidates.shape == (128, 64)
    assert audit["episode_segment_addresses"] == 64
    assert audit["unique_episode_segment_addresses"] == 64
    grouped = candidates.reshape(16, 8, 64)
    for episode_index, episode in enumerate(episodes):
        segment = episode["fact_position"] // 2
        assert torch.equal(queries[episode_index], grouped[episode_index, 2 * segment])
        assert torch.equal(queries[episode_index], grouped[episode_index, 2 * segment + 1])


def test_episode_address_builder_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        _episode_addresses(1337, 0, 4, 64, 310000)


def test_shared_metrics_accept_registered_balanced_positions():
    spec = copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC)
    spec["evaluation_episodes"] = 32
    episodes = build_balanced_evaluation(1337, spec)
    rankings = [[episode["fact_position"]] for episode in episodes]
    metrics = _metrics(
        episodes,
        rankings,
        torch.zeros(32, 8),
        fact_kinds=spec["fact_kinds"],
        fact_positions=spec["fact_positions"],
    )
    assert metrics["recall_at_1"] == 1.0
    assert set(metrics["per_position_recall_at_1"]) == set(map(str, range(8)))


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(BALANCED_RETRIEVAL_CONTROL_SPEC)
    changed["retrieval"]["address_pool"] = 3
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GRC3_0_INVALID"


def test_recorded_result_replays_registered_content_loss_verdict():
    payload = json.loads(
        Path("measurement/balanced_retrieval_control_results.json").read_text()
    )
    verdict = adjudicate(payload)
    assert verdict["verdict"] == "GRC3B_CONTENT_RANKING_LOSS"
    assert [row["seed"] for row in verdict["seeds"]] == [1337, 7331]
