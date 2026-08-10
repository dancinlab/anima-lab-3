from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from key_stability import fit_canonical_projector
from measurement.canonical_gate import adjudicate
from measurement.canonical_registry import CANONICAL_SPEC, canonical_spec, spec_sha256
from measurement.key_registry import KEY_SPEC


def test_canonical_registry_matches_preregistration():
    assert CANONICAL_SPEC["preregistration_commit"] == "ad8009478"
    assert [row["name"] for row in CANONICAL_SPEC["calibration_arms"]] == [
        "calibration_1337", "calibration_7331", "pooled",
    ]
    assert CANONICAL_SPEC["method"] == "ridge_fixed_orthogonal_targets"
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_canonical_fit_is_repeatable_and_order_stable():
    generator = torch.Generator().manual_seed(717)
    states = torch.randn(96, KEY_SPEC["input_dim"], generator=generator)
    labels = torch.arange(96) % KEY_SPEC["keys"]
    left, audit = fit_canonical_projector(states, labels, KEY_SPEC)
    right, _ = fit_canonical_projector(states, labels, KEY_SPEC)
    reversed_model, _ = fit_canonical_projector(states.flip(0), labels.flip(0), KEY_SPEC)
    assert audit["method"] == "ridge_fixed_orthogonal_targets"
    assert all(torch.equal(left.state_dict()[name], right.state_dict()[name]) for name in left.state_dict())
    assert max(float((left.state_dict()[name] - reversed_model.state_dict()[name]).abs().max())
               for name in left.state_dict()) <= CANONICAL_SPEC["order_tolerance"]


def test_canonical_fit_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fit_canonical_projector(torch.zeros(4, 3), torch.zeros(4, dtype=torch.long), KEY_SPEC)
    bad = torch.zeros(4, KEY_SPEC["input_dim"])
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit_canonical_projector(bad, torch.zeros(4, dtype=torch.long), KEY_SPEC)


def test_committed_canonical_result_replays_and_fails_closed():
    results_path = Path("measurement/canonical_results.json")
    verdict_path = Path("measurement/canonical_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["canonical_projectors"][0]["repeat_equal"] = False
    assert adjudicate(tampered)["verdict"] == "CN0_INVALID"
