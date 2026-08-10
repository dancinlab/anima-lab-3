from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from measurement.seedmap_gate import _classify, adjudicate
from measurement.seedmap_registry import SEEDMAP_SPEC, canonical_spec, combination_name, spec_sha256


def test_seedmap_registry_matches_preregistration():
    assert SEEDMAP_SPEC["preregistration_commit"] == "3e828054a"
    assert SEEDMAP_SPEC["event_count"] == 4
    assert SEEDMAP_SPEC["settling_updates"] == 8
    assert SEEDMAP_SPEC["factors"] == ["projector_seed", "prototype_seed", "engine_seed"]
    assert len(SEEDMAP_SPEC["combinations"]) == 8
    assert len({combination_name(row) for row in SEEDMAP_SPEC["combinations"]}) == 8
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_seedmap_verdict_classification():
    none = {name: {"rescue": False, "reverse": False} for name in SEEDMAP_SPEC["factors"]}
    single = deepcopy(none)
    single["projector_seed"] = {"rescue": True, "reverse": True}
    assert _classify(single, [False, True])[0] == "SM1_SINGLE_FACTOR_CAUSAL"
    multiple = deepcopy(single)
    multiple["engine_seed"] = {"rescue": True, "reverse": True}
    assert _classify(multiple, [False, True])[0] == "SM2_MULTIPLE_FACTORS_CAUSAL"
    asymmetric = deepcopy(none)
    asymmetric["prototype_seed"] = {"rescue": True, "reverse": False}
    assert _classify(asymmetric, [False, True])[0] == "SM3_ASYMMETRIC_FACTOR_EFFECT"
    assert _classify(none, [False, True])[0] == "SM4_FACTOR_INTERACTION"
    assert _classify(none, [True] * 6)[0] == "SM5_NO_FACTOR_EFFECT"


def test_committed_seedmap_result_replays_and_fails_closed():
    results_path = Path("measurement/seedmap_results.json")
    verdict_path = Path("measurement/seedmap_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    tampered = deepcopy(payload)
    tampered["combinations"][0]["engine_seed"] = 99
    assert adjudicate(tampered)["verdict"] == "SM0_INVALID"
