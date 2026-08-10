from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from key_stability import DEFAULT_STABLE_KEY_FIT_METHOD, fit_stable_key_projector
from measurement.canonical2_gate import adjudicate
from measurement.canonical2_registry import CANONICAL2_SPEC, canonical_spec, spec_sha256
from measurement.key_registry import KEY_SPEC


def test_canonical2_registry_matches_preregistration_and_public_default():
    assert CANONICAL2_SPEC["preregistration_commit"] == "3f02c9441"
    assert CANONICAL2_SPEC["fit_method"] == DEFAULT_STABLE_KEY_FIT_METHOD == "canonical_ridge"
    assert CANONICAL2_SPEC["event_counts"] == [2, 3, 4]
    assert len(CANONICAL2_SPEC["evaluation_combinations"]) == 4
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_public_stable_key_fit_is_canonical_and_rejects_unknown_method():
    generator = torch.Generator().manual_seed(818)
    states = torch.randn(64, KEY_SPEC["input_dim"], generator=generator)
    labels = torch.arange(64) % KEY_SPEC["keys"]
    left, left_audit = fit_stable_key_projector(states, labels, KEY_SPEC)
    right, right_audit = fit_stable_key_projector(states, labels, KEY_SPEC, method="canonical_ridge")
    assert left_audit == right_audit
    assert all(torch.equal(left.state_dict()[name], right.state_dict()[name]) for name in left.state_dict())
    with pytest.raises(ValueError, match="unknown stable key fit method"):
        fit_stable_key_projector(states, labels, KEY_SPEC, method="random_good_seed")


def test_committed_canonical2_result_replays_and_fails_closed():
    results_path = Path("measurement/canonical2_results.json")
    verdict_path = Path("measurement/canonical2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["canonical1_pooled_match"] = False
    assert adjudicate(tampered)["verdict"] == "CI0_INVALID"
