import copy

import pytest
import torch

from gate_retrieval_control1 import _metrics, _rank, dataset_audit
from gate2 import build_evaluation
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC
from measurement.semantic_retrieval_control_gate import adjudicate
from measurement.semantic_retrieval_control_registry import (
    SEMANTIC_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def test_semantic_retrieval_control_is_preregistered_and_pinned():
    spec = SEMANTIC_RETRIEVAL_CONTROL_SPEC
    assert spec["preregistration_commit"] == "b9bd07c6a"
    assert spec["encoder"]["revision"] == "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    assert spec["retrieval"]["feature_dim"] == 384
    assert spec["top_k"] == 3
    assert len(spec_sha256()) == 64


def test_retrieval_dataset_reuses_gate2_and_is_balanced():
    episodes = build_evaluation(1337, REALISTIC_MEMORY_WRITE_SPEC)
    audit = dataset_audit(episodes, REALISTIC_MEMORY_WRITE_SPEC)
    assert audit["evaluation_episodes"] == audit["query_unique"] == 1024
    assert audit["evaluation_candidates"] == audit["evaluation_unique"] == 8192
    assert audit["query_candidate_overlap"] == 0
    assert set(audit["fact_counts"].values()) == {256}
    assert set(audit["fact_position_counts"].values()) == {256}
    assert audit["topic_switch_counts"] == {"3": 1024}


def test_rank_is_descending_with_candidate_index_tie_break():
    assert _rank(torch.tensor([0.2, 0.7, 0.7, -0.1])) == [1, 2, 0, 3]
    with pytest.raises(ValueError):
        _rank(torch.tensor([0.1, float("nan")]))


def test_metrics_require_complete_candidate_permutations():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_evaluation(1337, spec)
    with pytest.raises(ValueError):
        _metrics(episodes, [[0, 0]] * 4, [[1.0, 1.0]] * 4, 3)


def test_metrics_use_candidate_order_scores_for_margin():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_evaluation(1337, spec)
    rankings = []
    scores = []
    for episode in episodes:
        fact = episode["fact_position"]
        row_scores = [0.0] * 8
        row_scores[fact] = 2.0
        ranking = [fact] + [index for index in range(8) if index != fact]
        rankings.append(ranking)
        scores.append(row_scores)
    metrics = _metrics(episodes, rankings, scores, 3)
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mean_fact_margin"] == 2.0


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(SEMANTIC_RETRIEVAL_CONTROL_SPEC)
    changed["top_k"] = 4
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GRC0_INVALID"
