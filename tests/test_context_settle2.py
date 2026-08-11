from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

import conjunction
from conjunction import ConjunctionEpisode, trace_episode
from context_settle2 import _runtime_spec
from measurement.context_settle2_gate import adjudicate
from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC, canonical_spec, spec_sha256


def test_context_settle2_registry_matches_preregistration():
    assert CONTEXT_SETTLE2_SPEC["preregistration_commit"] == "e5023e810"
    assert CONTEXT_SETTLE2_SPEC["baseline_context_steps"] == 3
    assert CONTEXT_SETTLE2_SPEC["settled_context_steps"] == 6
    assert CONTEXT_SETTLE2_SPEC["key_sense_steps"] == 3
    assert CONTEXT_SETTLE2_SPEC["value_sense_steps"] == 3
    assert CONTEXT_SETTLE2_SPEC["conditions"] == ["baseline_3", "settled_6"]
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_common_trace_routes_only_context_through_registered_six_steps(monkeypatch):
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
    spec = _runtime_spec(6)
    spec["pre_query_updates"] = 0
    trace = trace_episode(episode, 123, spec)
    assert [steps for _, steps in calls] == [6, 3, 3, 1, 6, 3]
    assert trace["sense_audit"] == {
        "context_sense_steps": 6, "key_sense_steps": 3,
        "value_sense_steps": 3, "distractor_sense_steps": 1,
        "context_step_calls": 12, "key_step_calls": 6,
        "value_step_calls": 3, "distractor_step_calls": 1,
    }


def test_common_trace_rejects_invalid_sense_counts():
    episode = ConjunctionEpisode(
        contexts=(0,), keys=(0,), values=(0,), active_contexts=(0,), active_keys=(0,),
        active_values=(0,), distractors=(), query_position=0,
    )
    spec = _runtime_spec(6)
    spec["context_sense_steps"] = 0
    try:
        trace_episode(episode, 123, spec)
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("invalid context sense count was accepted")


def test_committed_context_settle2_result_replays_and_fails_closed():
    results_path = Path("measurement/context_settle2_results.json")
    verdict_path = Path("measurement/context_settle2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected
    changed = deepcopy(payload)
    changed["evaluations"][0]["conditions"]["settled_6"]["state_audit"][
        "context_step_calls"
    ] += 1
    assert adjudicate(changed)["verdict"] == "CT2I0_INVALID"
