import json
from copy import deepcopy
from pathlib import Path

from measurement.component_gate import adjudicate
from measurement.component_registry import COMPONENT_SPEC, spec_sha256


def test_component_registry_matches_preregistration():
    assert COMPONENT_SPEC["preregistration_commit"] == "23078054b"
    assert COMPONENT_SPEC["positions"] == list(range(16))
    assert COMPONENT_SPEC["eval_episodes"] == 512
    assert len(spec_sha256()) == 64


def test_committed_component_result_replays_and_fails_closed():
    results = Path("measurement/component_results.json"); verdict = Path("measurement/component_verdict.json")
    if not results.is_file() or not verdict.is_file(): return
    payload = json.loads(results.read_text()); expected = json.loads(verdict.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload); changed["engines"][0]["positions"][0]["position_label"] += 1
    assert adjudicate(changed)["verdict"] == "AC0_INVALID"
