from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from measurement.separation2_gate import adjudicate
from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256
from separation import build_episodes, dataset_audit


def test_separation2_registry_and_dataset_match_preregistration():
    assert SEPARATION2_SPEC["preregistration_commit"] == "805623f16"
    assert SEPARATION2_SPEC["fit_method"] == "canonical_ridge"
    assert SEPARATION2_SPEC["settling_updates"] == 8
    assert len(SEPARATION2_SPEC["evaluation_combinations"]) == 4
    assert len(set(SEPARATION2_SPEC["evaluation_names"])) == 4
    assert len(spec_sha256()) == 64
    first = build_episodes(SEPARATION2_SPEC)
    second = build_episodes(SEPARATION2_SPEC)
    assert first == second
    audit = dataset_audit(first, SEPARATION2_SPEC)
    assert audit["episodes"] == SEPARATION2_SPEC["eval_episodes"]
    assert audit["unique_fingerprints"] == SEPARATION2_SPEC["eval_episodes"]
    assert len(set(audit["target_counts"].values())) == 1
    assert len(set(audit["query_position_counts"].values())) == 1
    assert len(set(audit["shared_key_counts"].values())) == 1
    assert len(set(audit["query_context_counts"].values())) == 1


def test_committed_separation2_result_replays_and_fails_closed():
    results_path = Path("measurement/separation2_results.json")
    verdict_path = Path("measurement/separation2_verdict.json")
    if not results_path.is_file() or not verdict_path.is_file():
        return
    payload = json.loads(results_path.read_text())
    expected = json.loads(verdict_path.read_text())
    assert adjudicate(payload) == expected

    tampered = deepcopy(payload)
    tampered["evaluations"][0]["update_audit"]["requested_updates"] = 7
    assert adjudicate(tampered)["verdict"] == "SP0_INVALID"

    stable_pass = deepcopy(payload)
    for row in stable_pass["evaluations"]:
        row["arms"]["stable_similar_normal"] = deepcopy(
            row["arms"]["stable_distinct_key_control"]
        )
    assert adjudicate(stable_pass)["verdict"] == "SP1_SIMILAR_EPISODES_SEPARATED_NOT_UNIQUE"

    raw_only = deepcopy(payload)
    for row in raw_only["evaluations"]:
        row["arms"]["raw_similar_normal"] = deepcopy(
            row["arms"]["stable_distinct_key_control"]
        )
    assert adjudicate(raw_only)["verdict"] == "SP2_CANONICAL_KEY_COLLISION"

    value_loss = deepcopy(stable_pass)
    for row in value_loss["evaluations"]:
        metric = row["arms"]["stable_similar_normal"]
        metric["accuracy"] = 0.25
        metric["per_value_recall"] = [0.25] * SEPARATION2_SPEC["values"]
    assert adjudicate(value_loss)["verdict"] == "SP4_VALUE_READOUT_LOSS"
