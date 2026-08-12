import copy

import torch

from gate_control2 import match_ranked_counts
from measurement.semantic_memory_write_matched_gate import adjudicate
from measurement.semantic_memory_write_matched_registry import (
    MATCHED_SEMANTIC_MEMORY_WRITE_SPEC,
    spec_sha256,
)


def test_matched_control_is_preregistered_and_pinned():
    assert MATCHED_SEMANTIC_MEMORY_WRITE_SPEC["preregistration_commit"] == "9c7dda4a6"
    assert MATCHED_SEMANTIC_MEMORY_WRITE_SPEC["matching"] == {
        "method": "per_episode_score_rank",
        "descending": True,
        "tie_break": "candidate_index_ascending",
    }
    assert len(spec_sha256()) == 64


def test_ranked_fake_matches_each_episode_count_and_breaks_ties_by_index():
    scores = torch.tensor([0.1, 0.9, 0.4, 0.3, 1.0, 1.0, 0.2, 0.1])
    reference = [[True, False, True, False], [False, True, False, False]]
    matched = match_ranked_counts(scores, reference, width=4)
    assert matched == [[False, True, True, False], [True, False, False, False]]
    assert [sum(row) for row in matched] == [sum(row) for row in reference]


def test_ranked_fake_rejects_invalid_shapes_and_nonfinite_scores():
    try:
        match_ranked_counts(torch.ones(3), [[True, False]], width=2)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")
    try:
        match_ranked_counts(torch.tensor([0.0, float("nan")]), [[True, False]], width=2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite scores must fail")


def test_matched_adjudicator_fails_closed_on_changed_registration():
    changed = copy.deepcopy(MATCHED_SEMANTIC_MEMORY_WRITE_SPEC)
    changed["matching"]["tie_break"] = "changed"
    payload = {
        "experiment": MATCHED_SEMANTIC_MEMORY_WRITE_SPEC["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GCM0_INVALID"


def test_matched_adjudicator_rejects_per_episode_count_mismatch():
    spec = MATCHED_SEMANTIC_MEMORY_WRITE_SPEC
    base = spec["control1_spec"]
    width = base["gate1_spec"]["evaluation_episodes"]
    audit = {
        "method": spec["matching"],
        "semantic_counts": [1] * width,
        "matched_shuffled_counts": [1] * width,
        "matched_random_counts": [1] * width,
        "semantic_selection_sha256": "0" * 64,
        "matched_shuffled_selection_sha256": "1" * 64,
        "matched_random_selection_sha256": "2" * 64,
        "fake_scores_sha256": "3" * 64,
    }
    rows = []
    for seed in base["gate1_spec"]["seeds"]:
        rows.append({"seed": seed, "matching_audit": copy.deepcopy(audit)})
    rows[0]["matching_audit"]["matched_shuffled_counts"][0] = 2
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(),
        "seeds": rows,
    }
    assert adjudicate(payload)["verdict"] == "GCM0_INVALID"
