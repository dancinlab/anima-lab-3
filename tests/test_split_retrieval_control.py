import copy

import pytest
import torch

from gate2 import build_evaluation
from gate_retrieval_control2 import (
    _address_pools,
    _candidate_topic,
    _content_rankings,
    _metrics,
)
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC
from measurement.split_retrieval_control_gate import adjudicate
from measurement.split_retrieval_control_registry import (
    SPLIT_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def test_split_retrieval_control_is_preregistered_and_pinned():
    spec = SPLIT_RETRIEVAL_CONTROL_SPEC
    assert spec["preregistration_commit"] == "1c61f9e36"
    assert spec["retrieval"]["address_pool"] == 2
    assert spec["retrieval"]["top_k"] == 3
    assert spec["thresholds"]["split_recall_at_1"] == 0.90
    assert len(spec_sha256()) == 64


def test_candidate_topics_preserve_two_rows_per_segment():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_evaluation(1337, spec)
    for index, episode in enumerate(episodes):
        topics = [
            _candidate_topic(episode, 1337, index, row)
            for row in episode["candidates"]
        ]
        assert len(set(topics)) == 4
        assert topics.count(episode["subject"]) == 2


def test_address_pool_and_content_ranking_are_deterministic():
    queries = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pools, scores = _address_pools(queries, candidates, width=4, pool_size=2)
    assert pools == [[0, 1]]
    assert _content_rankings(pools, torch.tensor([[0.2, 0.8, 0.0, 0.0]])) == [[1, 0]]
    assert scores.shape == (1, 4)


def test_metrics_record_top1_and_top3_separately():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_evaluation(1337, spec)
    rankings = []
    scores = torch.zeros(16, 8)
    for row, episode in enumerate(episodes):
        fact = episode["fact_position"]
        other = fact + 1
        rankings.append([other, fact])
        scores[row, fact] = 1
    metrics = _metrics(episodes, rankings, scores)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0


def test_address_pool_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        _address_pools(torch.ones(2, 3), torch.ones(5, 3), width=3, pool_size=2)


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(SPLIT_RETRIEVAL_CONTROL_SPEC)
    changed["retrieval"]["address_pool"] = 3
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GRC2_0_INVALID"
