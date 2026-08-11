from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from conjunction import ConjunctionEpisode, trace_episode
from context_settle2 import _runtime_spec
from measurement.query_refresh_gate import _classify, _first_sustained, adjudicate
from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, canonical_spec, spec_sha256


def _episode(context: int) -> ConjunctionEpisode:
    return ConjunctionEpisode(
        contexts=(context, (context + 1) % 8),
        keys=(0, 1),
        values=(2, 3),
        active_contexts=(context, (context + 1) % 8),
        active_keys=(0, 1),
        active_values=(2, 3),
        distractors=(0, 1),
        query_position=0,
    )


def test_query_refresh_registry_matches_preregistration():
    assert QUERY_REFRESH_SPEC["preregistration_commit"] == "5ecc3697e"
    assert QUERY_REFRESH_SPEC["query_context_steps"] == [6, 8, 12, 16]
    assert QUERY_REFRESH_SPEC["baseline_query_context_steps"] == 6
    assert QUERY_REFRESH_SPEC["histories"] == ["original", "event_reversed"]
    assert len(canonical_spec()) > 100
    assert len(spec_sha256()) == 64


def test_runtime_spec_defaults_query_steps_to_storage_steps():
    runtime = _runtime_spec(6, QUERY_REFRESH_SPEC)
    assert runtime["context_sense_steps"] == 6
    assert runtime["query_context_sense_steps"] == 6
    changed = _runtime_spec(6, {**QUERY_REFRESH_SPEC, "query_context_sense_steps": 12})
    assert changed["context_sense_steps"] == 6
    assert changed["query_context_sense_steps"] == 12


def test_trace_rejects_invalid_query_step_count():
    spec = _runtime_spec(6, {**QUERY_REFRESH_SPEC, "query_context_sense_steps": 0})
    try:
        trace_episode(_episode(0), 123, spec)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive query-context sensing must fail")


def test_sustained_profiles_and_closed_verdict_order():
    assert _first_sustained({6: False, 8: True, 12: True, 16: True}) == 8
    assert _first_sustained({6: False, 8: True, 12: False, 16: True}) == 16
    assert _first_sustained({6: False, 8: False}) is None
    assert _classify(8, 12, True)[0] == "QR1_REFRESH_RECOVERS_AND_CONVERGES"
    assert _classify(8, None, True)[0] == "QR2_REFRESH_RECOVERS_NOT_CONVERGED"
    assert _classify(None, 12, True)[0] == "QR3_HISTORY_CONVERGES_NOT_RECOVERED"
    assert _classify(None, None, True)[0] == "QR4_REFRESH_CHANGES_NOT_SUFFICIENT"
    assert _classify(None, None, False)[0] == "QR5_REFRESH_NOT_PRIMARY"


def test_committed_query_refresh_result_replays_and_fails_closed():
    results_path = Path("measurement/query_refresh_results.json")
    verdict_path = Path("measurement/query_refresh_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    changed = deepcopy(payload)
    changed["evaluations"][0]["pair_audits"]["8"]["original"][
        "query_context_step_calls"
    ] += 1
    assert adjudicate(changed)["verdict"] == "QR0_INVALID"

    changed = deepcopy(payload)
    changed["evaluations"][0]["source_reference_audit"]["original"][
        "pair_digest_match"
    ] = False
    assert adjudicate(changed)["verdict"] == "QR0_INVALID"
