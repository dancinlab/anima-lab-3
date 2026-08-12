import copy
import json
from pathlib import Path

import pytest
import torch

from gate4 import _text_digest
from measurement.balanced_integrated_dialogue_gate import adjudicate
from measurement.balanced_integrated_dialogue_registry import (
    BALANCED_INTEGRATED_DIALOGUE_SPEC,
    spec_sha256,
)


def test_gate4_is_preregistered_and_pinned():
    spec = BALANCED_INTEGRATED_DIALOGUE_SPEC
    assert spec["preregistration_commit"] != "__PREREGISTRATION_COMMIT__"
    assert spec["replicates"] == ["daily", "work"]
    assert spec["retrieval"]["stored_candidates_only"] is True
    assert spec["thresholds"]["minimum_per_template_recall_at_1"] == 0.90
    assert len(spec_sha256()) == 64


def test_text_digest_tracks_raw_candidate_order_and_content():
    episodes = [{"candidates": [{"text": "가"}, {"text": "나"}]}]
    digest = _text_digest(episodes)
    assert len(digest) == 64
    changed = copy.deepcopy(episodes)
    changed[0]["candidates"].reverse()
    assert _text_digest(changed) != digest


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(BALANCED_INTEGRATED_DIALOGUE_SPEC)
    changed["thresholds"]["minimum_integrated_recall_at_1"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(changed),
        "runtime": changed["runtime"],
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "G4_0_INVALID"


def test_recorded_result_replays_registered_verdict():
    path = Path("measurement/balanced_integrated_dialogue_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    verdict = adjudicate(json.loads(path.read_text()))
    assert verdict["verdict"] in {
        "G4A_BALANCED_INTEGRATED_MEMORY_VALID_NOT_UNIQUE",
        "G4B_WRITE_SELECTION_LOSS",
        "G4C_RETRIEVAL_LOSS",
    }
