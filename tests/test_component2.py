import json
from copy import deepcopy
from pathlib import Path

from component2 import balance_components
from conjunction import build_episodes
from measurement.component2_gate import adjudicate
from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256
from measurement.conjunction2_registry import CONJUNCTION2_SPEC


def test_component2_registry_and_balanced_calibration():
    assert COMPONENT2_SPEC["preregistration_commit"] == "3a3ea619c"
    local = {**CONJUNCTION2_SPEC, "eval_episodes": 512, "data_seed": COMPONENT2_SPEC["calibration_data_seed"]}
    episodes = balance_components(build_episodes(local))
    contexts = [v for row in episodes for v in row.contexts]; keys = [v for row in episodes for v in row.keys]
    assert [contexts.count(i) for i in range(8)] == [1024] * 8
    assert [keys.count(i) for i in range(8)] == [1024] * 8
    assert len(spec_sha256()) == 64


def test_committed_component2_result_replays_and_fails_closed():
    results = Path("measurement/component2_results.json"); verdict = Path("measurement/component2_verdict.json")
    if not results.is_file() or not verdict.is_file(): return
    payload = json.loads(results.read_text()); expected = json.loads(verdict.read_text()); assert adjudicate(payload) == expected
    changed = deepcopy(payload); changed["deterministic"] = False
    assert adjudicate(changed)["verdict"] == "CS0_INVALID"
