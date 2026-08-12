import copy

import pytest
import torch

from gate2 import _metrics, build_calibration, build_evaluation, dataset_audit
from measurement.realistic_memory_write_gate import _rate_map_valid, adjudicate
from measurement.realistic_memory_write_registry import (
    REALISTIC_MEMORY_WRITE_SPEC,
    spec_sha256,
    template_sha256,
)


def test_gate2_is_preregistered_and_pinned():
    spec = REALISTIC_MEMORY_WRITE_SPEC
    assert spec["preregistration_commit"] == "a9e1a1a82"
    assert spec["encoder"]["revision"] == "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    assert spec["fact_positions"] == [0, 2, 4, 6]
    assert spec["topic_switches_per_episode"] == 3
    assert len(spec_sha256()) == len(template_sha256()) == 64


def test_gate2_dataset_is_balanced_disjoint_and_deterministic():
    spec = REALISTIC_MEMORY_WRITE_SPEC
    first_calibration = build_calibration(1337, spec)
    first_episodes = build_evaluation(1337, spec)
    second_calibration = build_calibration(1337, spec)
    second_episodes = build_evaluation(1337, spec)
    assert first_calibration == second_calibration
    assert first_episodes == second_episodes
    audit = dataset_audit(first_calibration, first_episodes, spec)
    assert audit["calibration_rows"] == audit["calibration_unique"] == 4096
    assert audit["calibration_positive"] == audit["calibration_negative"] == 2048
    assert audit["evaluation_episodes"] == 1024
    assert audit["evaluation_candidates"] == audit["evaluation_unique"] == 8192
    assert audit["overlap"] == 0
    assert set(audit["fact_counts"].values()) == {256}
    assert set(audit["fact_position_counts"].values()) == {256}
    assert set(audit["distractor_counts"].values()) == {1024}
    assert audit["topic_switch_counts"] == {"3": 1024}


def test_gate2_episode_preserves_order_and_one_fact_per_episode():
    spec = REALISTIC_MEMORY_WRITE_SPEC
    for episode in build_evaluation(7331, spec):
        assert len(episode["candidates"]) == spec["candidates_per_episode"]
        important = [row for row in episode["candidates"] if row["important"]]
        assert len(important) == 1
        assert important[0]["position"] == episode["fact_position"]
        assert [row["position"] for row in episode["candidates"]] == list(range(8))
        assert {row["topic_segment"] for row in episode["candidates"]} == {0, 1, 2, 3}


def test_gate2_metrics_report_kind_position_and_distractor_slices():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 16
    episodes = build_evaluation(1337, spec)
    oracle = [[bool(row["important"]) for row in episode["candidates"]] for episode in episodes]
    metrics = _metrics(episodes, oracle, spec)
    assert metrics["important_storage_rate"] == metrics["recall_at_3"] == 1.0
    assert set(metrics["per_kind_recall"].values()) == {1.0}
    assert set(metrics["per_position_recall"].values()) == {1.0}
    assert set(metrics["per_distractor_storage_rate"].values()) == {0.0}
    assert metrics["search_size_ratio"] == 0.125


def test_gate2_rate_map_rejects_missing_and_nonfinite_values():
    assert _rate_map_valid({"a": 0.0, "b": 1.0}, ["a", "b"])
    assert not _rate_map_valid({"a": 0.0}, ["a", "b"])
    assert not _rate_map_valid({"a": float("nan"), "b": 0.0}, ["a", "b"])
    assert not _rate_map_valid({"a": True, "b": 0.0}, ["a", "b"])


def test_gate2_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    changed["fact_positions"] = [0, 1, 2, 3]
    payload = {
        "experiment": REALISTIC_MEMORY_WRITE_SPEC["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "G2R0_INVALID"


def test_gate2_builders_reject_malformed_width_through_metrics():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["evaluation_episodes"] = 4
    episodes = build_evaluation(1337, spec)
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        _metrics(episodes, [[True]] * len(episodes), spec)
