import copy
import json
from pathlib import Path

import pytest

from gate2 import build_calibration, build_evaluation
from gate_write_mechanism1 import factor_sources
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC
from measurement.write_mechanism_gate import adjudicate
from measurement.write_mechanism_registry import WRITE_MECHANISM_SPEC, spec_sha256


def test_write_mechanism_is_preregistered_and_pinned():
    spec = WRITE_MECHANISM_SPEC
    assert spec["preregistration_commit"] == "25f626b3f"
    assert spec["thresholds"]["minimum_peer_gap_fraction"] == 0.80
    assert set(spec["arms"]) == {
        "baseline", "template_swap", "identifier_swap", "layout_swap", "all_swap",
    }
    assert len(spec_sha256()) == 64


def test_generation_controls_reproduce_peer_only_when_all_sources_swap():
    spec = copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC)
    spec["calibration_rows"] = 32
    spec["evaluation_episodes"] = 16
    baseline = build_calibration(1337, spec)
    peer = build_calibration(7331, spec)
    swapped = build_calibration(
        1337, spec, template_seed=7331, identifier_seed=7331, layout_seed=7331
    )
    assert swapped == peer
    assert baseline != peer
    eval_peer = build_evaluation(7331, spec)
    eval_swapped = build_evaluation(
        1337, spec, template_seed=7331, identifier_seed=7331, layout_seed=7331
    )
    assert eval_swapped == eval_peer


@pytest.mark.parametrize("factor", ["template", "identifier", "layout"])
def test_one_factor_swap_changes_only_registered_source(factor):
    sources = factor_sources(1337, 7331, [factor])
    assert sources[factor] == 7331
    assert all(sources[name] == 1337 for name in sources if name != factor)


def test_factor_sources_reject_unknown_or_same_seed():
    with pytest.raises(ValueError):
        factor_sources(1337, 7331, ["unknown"])
    with pytest.raises(ValueError):
        factor_sources(1337, 1337, [])


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(WRITE_MECHANISM_SPEC)
    changed["thresholds"]["minimum_peer_gap_fraction"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(),
        "runtime": changed["runtime"],
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GWM0_INVALID"


def test_recorded_result_replays_registered_verdict():
    path = Path("measurement/write_mechanism_results.json")
    if not path.exists():
        pytest.skip("recorded result is created only after the preregistered run")
    verdict = adjudicate(json.loads(path.read_text()))
    assert verdict["verdict"] in {
        "GWM1_TEMPLATE_CAUSAL", "GWM2_IDENTIFIER_CAUSAL", "GWM3_LAYOUT_CAUSAL",
        "GWM4_MULTIFACTOR", "GWM5_UNEXPLAINED",
    }
