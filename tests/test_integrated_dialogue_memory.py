import copy
import json
from pathlib import Path

import pytest
import torch

from gate3 import stored_rankings
from measurement.integrated_dialogue_memory_gate import adjudicate
from measurement.integrated_dialogue_memory_registry import (
    INTEGRATED_DIALOGUE_MEMORY_SPEC,
    spec_sha256,
)


def test_gate3_is_preregistered_and_pinned():
    spec = INTEGRATED_DIALOGUE_MEMORY_SPEC
    assert spec["preregistration_commit"] == "0d7b4094b"
    assert spec["retrieval"]["stored_candidates_only"] is True
    assert spec["fact_positions"] == list(range(8))
    assert spec["thresholds"]["minimum_integrated_recall_at_1"] == 0.90
    assert len(spec_sha256()) == 64


def test_stored_rankings_exclude_unstored_candidates_and_use_content_order():
    address = torch.tensor([[0.9, 1.0, 0.8, 0.7]], dtype=torch.float64)
    content = torch.tensor([[0.0, 0.1, 0.7, 0.8]], dtype=torch.float64)
    rankings, pools = stored_rankings(
        [[False, True, True, False]], address, content, pool_size=2
    )
    assert pools == [[1, 2]]
    assert rankings == [[2, 1]]


def test_stored_rankings_return_empty_when_nothing_was_stored():
    rankings, pools = stored_rankings(
        [[False, False]], torch.zeros(1, 2), torch.ones(1, 2), pool_size=2
    )
    assert rankings == [[]]
    assert pools == [[]]


@pytest.mark.parametrize(
    "selection,address,content,pool_size",
    [
        ([[True]], torch.zeros(2, 1), torch.zeros(2, 1), 1),
        ([[True]], torch.zeros(1, 2), torch.zeros(1, 2), 1),
        ([[True]], torch.tensor([[float("nan")]]), torch.zeros(1, 1), 1),
        ([[True]], torch.zeros(1, 1), torch.zeros(1, 1), 0),
    ],
)
def test_stored_rankings_reject_invalid_inputs(selection, address, content, pool_size):
    with pytest.raises(ValueError):
        stored_rankings(selection, address, content, pool_size)


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC)
    changed["thresholds"]["minimum_integrated_recall_at_1"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "G3_0_INVALID"


def test_recorded_result_replays_registered_verdict():
    path = Path("measurement/integrated_dialogue_memory_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    verdict = adjudicate(json.loads(path.read_text()))
    assert verdict["verdict"] in {
        "G3A_INTEGRATED_DIALOGUE_MEMORY_VALID_NOT_UNIQUE",
        "G3B_WRITE_SELECTION_LOSS",
        "G3C_RETRIEVAL_LOSS",
    }
    assert [row["seed"] for row in verdict["seeds"]] == [1337, 7331]


def test_adjudicator_rejects_broken_stored_candidate_audit():
    path = Path("measurement/integrated_dialogue_memory_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    payload = json.loads(path.read_text())
    payload["seeds"][0]["search_audit"]["rankings_subset_of_stored"] = False
    verdict = adjudicate(payload)
    assert verdict["verdict"] == "G3_0_INVALID"
    assert "stored-candidate" in verdict["reason"]
