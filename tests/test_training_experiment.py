from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from key_stability import train_projector
from measurement.key_registry import KEY_SPEC
from measurement.training_gate import _classify, adjudicate
from measurement.training_registry import TRAINING_SPEC, canonical_spec, spec_sha256, training_name


def test_training_registry_matches_preregistration():
    assert TRAINING_SPEC["preregistration_commit"] == "aff5bee25"
    assert len(TRAINING_SPEC["training_combinations"]) == 4
    assert len(TRAINING_SPEC["evaluation_combinations"]) == 4
    assert TRAINING_SPEC["calibration_seed"] == 1337
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_default_batch_seed_remains_bit_exact():
    generator = torch.Generator().manual_seed(991)
    states = torch.randn(64, KEY_SPEC["input_dim"], generator=generator)
    labels = torch.arange(64) % KEY_SPEC["keys"]
    local = {**KEY_SPEC, "train_steps": 4, "batch_size": 16}
    default, _ = train_projector(states, labels, 1337, False, local)
    explicit, _ = train_projector(states, labels, 1337, False, local, batch_seed=1337)
    assert all(torch.equal(default.state_dict()[name], explicit.state_dict()[name]) for name in default.state_dict())


def test_training_grid_verdicts():
    low, high = TRAINING_SPEC["factor_seeds"]
    names = {(i, b): training_name({"initialization_seed": i, "batch_seed": b}) for i in (low, high) for b in (low, high)}
    def grid(a, b, c, d):
        return {names[(low, low)]: a, names[(low, high)]: b, names[(high, low)]: c, names[(high, high)]: d}
    assert _classify(grid(False, False, True, True), low, high)[0] == "TR1_INITIALIZATION_CAUSAL"
    assert _classify(grid(False, True, False, True), low, high)[0] == "TR2_BATCH_ORDER_CAUSAL"
    assert _classify(grid(False, True, True, True), low, high)[0] == "TR3_EITHER_FACTOR_SUFFICIENT"
    assert _classify(grid(False, False, False, True), low, high)[0] == "TR4_BOTH_FACTORS_REQUIRED"
    assert _classify(grid(True, False, True, False), low, high)[0] == "TR5_FACTOR_INTERACTION_OR_MIXED"


def test_committed_training_result_replays_and_fails_closed():
    results_path = Path("measurement/training_results.json")
    verdict_path = Path("measurement/training_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["training_combinations"][0]["batch_seed"] = 99
    assert adjudicate(tampered)["verdict"] == "TR0_INVALID"
