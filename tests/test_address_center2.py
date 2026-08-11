from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from measurement.address_center2_gate import adjudicate
from measurement.address_center2_registry import (
    ADDRESS_CENTER2_SPEC, canonical_spec, spec_sha256,
)


def test_address_center2_registry_matches_preregistration():
    assert ADDRESS_CENTER2_SPEC["preregistration_commit"] == "09c3ea80e"
    assert ADDRESS_CENTER2_SPEC["settled_context_steps"] == 6
    assert ADDRESS_CENTER2_SPEC["components_per_key"] == 2
    assert ADDRESS_CENTER2_SPEC["transform_calls_per_episode"] == 17
    assert ADDRESS_CENTER2_SPEC["composite_address_dim"] == 64
    assert ADDRESS_CENTER2_SPEC["thresholds"]["minimum_selection_gain"] == 0.08
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_committed_address_center2_result_replays_and_fails_closed():
    results_path = Path("measurement/address_center2_results.json")
    verdict_path = Path("measurement/address_center2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["memory_path_audit"][
        "integrated_context_center"
    ]["maximum_calls"] += 1
    assert adjudicate(changed)["verdict"] == "AC0_INVALID"

    path_loss = deepcopy(payload)
    for row in path_loss["evaluations"]:
        arm = row["arms"]["integrated_context_center"]
        arm["selection_accuracy"] = 0.5
        arm["accuracy"] = 0.5
        row["arms"]["integrated_context_center_recovered"] = deepcopy(arm)
        row["reference_audit"]["center_metric_match"] = False
    assert adjudicate(path_loss)["verdict"] == "AC0_INVALID"
