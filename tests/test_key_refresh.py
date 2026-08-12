from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

import conjunction
from conjunction import ConjunctionEpisode, trace_episode
from key_refresh import _runtime_spec
from measurement.key_refresh_gate import _classify, adjudicate
from measurement.key_refresh_registry import (
    KEY_REFRESH_SPEC, canonical_spec, spec_sha256,
)


def test_registry_matches_preregistration():
    assert KEY_REFRESH_SPEC["preregistration_commit"] == "804c5f4ac"
    assert KEY_REFRESH_SPEC["query_context_sense_steps"] == 8
    assert KEY_REFRESH_SPEC["key_sense_steps"] == 3
    assert KEY_REFRESH_SPEC["query_key_steps"] == [3, 4, 6, 8, 12]
    assert KEY_REFRESH_SPEC["baseline_query_key_steps"] == 3
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_runtime_spec_changes_only_registered_query_key_steps():
    baseline = _runtime_spec(3)
    refreshed = _runtime_spec(8)
    assert baseline["settled_context_steps"] == refreshed["settled_context_steps"] == 6
    assert baseline["query_context_sense_steps"] == 8
    assert refreshed["query_context_sense_steps"] == 8
    assert baseline["key_sense_steps"] == refreshed["key_sense_steps"] == 3
    assert baseline["query_key_sense_steps"] == 3
    assert refreshed["query_key_sense_steps"] == 8
    try:
        _runtime_spec(5)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered query-key step count must fail")


def test_common_trace_separates_storage_and_query_key_steps(monkeypatch):
    calls = []

    class Engine:
        n_cells = 1

        def get_phase_states(self):
            return torch.zeros(1, 96)

        def step(self, **_):
            return None

    def sense(_engine, _encoder, word, steps, _spec):
        calls.append((word, steps))
        return torch.zeros(1, 96)

    monkeypatch.setattr(conjunction, "_new_engine", lambda *_: (Engine(), object()))
    monkeypatch.setattr(conjunction, "_sense_separation_token", sense)
    episode = ConjunctionEpisode(
        contexts=(0,), keys=(0,), values=(0,), active_contexts=(0,), active_keys=(0,),
        active_values=(0,), distractors=(1,), query_position=0,
    )
    from context_settle2 import _runtime_spec as common_runtime_spec
    spec = common_runtime_spec(6, _runtime_spec(8))
    spec["pre_query_updates"] = 0
    trace = trace_episode(episode, 123, spec)
    assert [steps for _, steps in calls] == [6, 3, 3, 1, 8, 8]
    assert trace["sense_audit"]["context_step_calls"] == 14
    assert trace["sense_audit"]["key_step_calls"] == 11


def test_existing_runtime_defaults_query_key_to_storage_key():
    spec = deepcopy(KEY_REFRESH_SPEC)
    spec.pop("query_key_sense_steps", None)
    from context_settle2 import _runtime_spec as common_runtime_spec
    runtime = common_runtime_spec(6, spec)
    assert runtime["query_key_sense_steps"] == runtime["key_sense_steps"] == 3


def test_closed_verdict_order():
    assert _classify(6, True, True, True)[0] == "KRF1_KEY_PATH_RECOVERED_AND_SUSTAINED"
    assert _classify(None, True, False, True)[0] == "KRF2_RECOVERED_NOT_SUSTAINED"
    assert _classify(None, False, False, True)[0] == "KRF3_IMPROVED_NOT_RECOVERED"
    assert _classify(None, False, False, False)[0] == "KRF4_KEY_REFRESH_NOT_CAUSAL"


def test_committed_result_replays_and_fails_closed():
    results_path = Path("measurement/key_refresh_results.json")
    verdict_path = Path("measurement/key_refresh_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["candidates"]["8"]["state_audit"][
        "key_step_calls"
    ] += 1
    assert adjudicate(changed)["verdict"] == "KRF0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["candidates"]["6"]["trace_digests"][
        "storage_key"
    ] = "0" * 64
    assert adjudicate(changed)["verdict"] == "KRF0_INVALID"
