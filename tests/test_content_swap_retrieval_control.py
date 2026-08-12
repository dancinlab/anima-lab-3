import copy
import json
from pathlib import Path

import pytest
import torch

from gate_retrieval_control4 import swap_pool_scores
from measurement.content_swap_retrieval_control_gate import adjudicate
from measurement.content_swap_retrieval_control_registry import (
    CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def test_content_swap_control_is_preregistered_and_pinned():
    spec = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC
    assert spec["preregistration_commit"] == "bde992d00"
    assert spec["retrieval"]["uses_labels"] is False
    assert spec["retrieval"]["uses_episode_order"] is False
    assert spec["thresholds"]["maximum_swapped_recall_at_1"] == 0.10
    assert len(spec_sha256()) == 64


def test_pool_swap_changes_only_the_registered_pair_and_is_involutive():
    scores = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ])
    pools = [[0, 1], [2, 3]]
    swapped = swap_pool_scores(scores, pools)
    assert torch.equal(swapped, torch.tensor([
        [1.0, 0.0, 2.0, 3.0],
        [4.0, 5.0, 7.0, 6.0],
    ]))
    assert torch.equal(swap_pool_scores(swapped, pools), scores)
    assert torch.equal(torch.sort(swapped, dim=1).values, torch.sort(scores, dim=1).values)


@pytest.mark.parametrize("pools", [[], [[0]], [[0, 0]], [[0, 4]]])
def test_pool_swap_rejects_invalid_registered_pools(pools):
    scores = torch.zeros(1, 4)
    with pytest.raises(ValueError):
        swap_pool_scores(scores, pools)


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC)
    changed["thresholds"]["maximum_swapped_recall_at_1"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GRC4_0_INVALID"


def test_recorded_result_replays_registered_verdict():
    path = Path("measurement/content_swap_retrieval_control_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    verdict = adjudicate(json.loads(path.read_text()))
    assert verdict["verdict"] == "GRC4A_CONTENT_ALIGNMENT_CAUSAL"
    assert [row["seed"] for row in verdict["seeds"]] == [1337, 7331]


def test_adjudicator_rejects_a_broken_restoration_audit():
    payload = json.loads(
        Path("measurement/content_swap_retrieval_control_results.json").read_text()
    )
    payload["seeds"][0]["swap_audit"]["restored_scores_exact"] = False
    verdict = adjudicate(payload)
    assert verdict["verdict"] == "GRC4_0_INVALID"
    assert "restoration audit" in verdict["reason"]
