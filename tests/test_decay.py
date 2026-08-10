from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from decay import build_decay_episodes, decay_dataset_audit, _trace, _same_prefix
from episode_control import _metrics
from key_stability import StableKeyProjector
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
from measurement.decay_gate import adjudicate
from measurement.decay_registry import DECAY_SPEC, spec_sha256
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(passed: bool = True) -> dict:
    total = DECAY_SPEC["eval_episodes_per_delay"]
    expected = torch.arange(total) % DECAY_SPEC["values"]
    predicted = expected.clone() if passed else (expected + 1) % DECAY_SPEC["values"]
    value = _metrics(expected, predicted, DECAY_SPEC["values"])
    value.update({
        "selection_accuracy": float(passed),
        "correct_content_accuracy": 1.0,
        "retrieval_api_match": 1.0,
        "key_margin_mean": 0.5,
        "key_margin_min": 0.1,
    })
    return value


def test_decay_dataset_is_balanced_unique_and_deterministic():
    first = build_decay_episodes()
    second = build_decay_episodes()
    assert first == second
    assert all(len(row.values) == DECAY_SPEC["prepared_events"] for row in first)
    assert all(row.query_position < DECAY_SPEC["queryable_events"] for row in first)
    audit = decay_dataset_audit(first)
    assert audit["episodes"] == DECAY_SPEC["eval_episodes_per_delay"]
    assert audit["unique_fingerprints"] == DECAY_SPEC["eval_episodes_per_delay"]
    for name in ("target_counts", "query_position_counts", "query_key_counts", "query_context_counts"):
        assert len(set(audit[name].values())) == 1


def test_two_and_three_event_streams_share_exact_prefix():
    episode = build_decay_episodes()[0]
    two = _trace(episode, 123456, 2, 2)
    three = _trace(episode, 123456, 3, 2)
    assert _same_prefix(two, three)
    assert len(two["keys"]) == len(two["values"]) == 2
    assert len(three["keys"]) == len(three["values"]) == 3


def _passing_payload(tmp_path: Path) -> dict:
    source_rows, result_rows, checkpoints, prototypes = [], [], {}, {}
    total = DECAY_SPEC["eval_episodes_per_delay"]
    for seed in DECAY_SPEC["seeds"]:
        projector = StableKeyProjector(
            KEY_SPEC["input_dim"], KEY_SPEC["address_dim"], KEY_SPEC["keys"],
            KEY_SPEC["temperature"], KEY_SPEC["bias"],
        )
        projector_path = tmp_path / f"key_{seed}.pt"
        torch.save({
            "experiment": KEY_SPEC["experiment"],
            "spec_sha256": key_spec_sha256(),
            "seed": seed,
            "model_class": KEY_SPEC["model_class"],
            "projector": projector.state_dict(),
        }, projector_path)
        projector_receipt = {"path": str(projector_path), "sha256": _sha(projector_path)}
        prototype_path = tmp_path / f"prototype_{seed}.pt"
        torch.save({
            "prototypes": {
                "quantum": torch.ones(DECAY_SPEC["values"], DECAY_SPEC["state_dim"]),
                "sensory": torch.ones(DECAY_SPEC["values"], DECAY_SPEC["state_dim"]),
            },
        }, prototype_path)
        prototype_receipt = {"path": str(prototype_path), "sha256": _sha(prototype_path)}
        checkpoints[str(seed)] = projector_receipt
        prototypes[str(seed)] = prototype_receipt
        source_rows.append({
            "seed": seed,
            "source_checkpoint": projector_receipt,
            "prototype_checkpoint": prototype_receipt,
        })
        delays = []
        for delay in DECAY_SPEC["distractor_steps"]:
            arms = {name: _metric() for name in DECAY_SPEC["arms"]}
            arms["exact_three_partner_swap"] = _metric(False)
            arms["exact_three_recovered"]["prediction_match"] = 1.0
            delays.append({
                "distractor_steps": delay,
                "arms": arms,
                "integration_audit": {
                    "stable_transform_calls": {
                        name: {
                            "episodes": total,
                            "total": total * calls,
                            "minimum": calls,
                            "maximum": calls,
                        }
                        for name, calls in DECAY_SPEC["expected_transform_calls"].items()
                    },
                    "address_width_minimum": DECAY_SPEC["address_dim"],
                    "address_width_maximum": DECAY_SPEC["address_dim"],
                },
                "state_audit": {
                    "episodes": total,
                    "unique_episode_seeds": total,
                    "episode_seed_sha256": "a" * 64,
                    "prefix_state_matches": total,
                    "minimum_cells": DECAY_SPEC["minimum_cells"],
                    "maximum_cells": DECAY_SPEC["maximum_cells"],
                },
            })
        result_rows.append({
            "seed": seed,
            "delays": delays,
            "projector_frozen": True,
            "projector_unchanged": True,
            "source_checkpoint": projector_receipt,
            "prototype_checkpoint": prototype_receipt,
        })
    source_results_path = tmp_path / "capacity_results.json"
    source_results_path.write_text(json.dumps({
        "experiment": CAPACITY_SPEC["experiment"],
        "spec": CAPACITY_SPEC,
        "spec_sha256": capacity_spec_sha256(),
        "seeds": source_rows,
    }))
    source_verdict_path = tmp_path / "capacity_verdict.json"
    source_verdict_path.write_text(json.dumps({
        "verdict": DECAY_SPEC["source_verdict"],
        "spec_sha256": capacity_spec_sha256(),
    }))
    episodes = build_decay_episodes()
    return {
        "experiment": DECAY_SPEC["experiment"],
        "spec": deepcopy(DECAY_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": decay_dataset_audit(episodes),
        "source_capacity": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": DECAY_SPEC["source_verdict"],
            "source_spec_sha256": capacity_spec_sha256(),
            "checkpoints": checkpoints,
            "prototype_checkpoints": prototypes,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def _set(payload: dict, arm: str, delays, passed: bool, seeds=None) -> None:
    selected = set(DECAY_SPEC["seeds"] if seeds is None else seeds)
    for row in payload["seeds"]:
        if row["seed"] not in selected:
            continue
        for item in row["delays"]:
            if item["distractor_steps"] in delays:
                item["arms"][arm] = _metric(passed)


def test_decay_gate_verdicts_and_fail_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "D4_BOUNDARY_NOT_REPRODUCED"

    competition = deepcopy(value)
    _set(competition, "three_stream_three_candidates", DECAY_SPEC["distractor_steps"], False)
    assert adjudicate(competition)["verdict"] == "D1_RETRIEVAL_COMPETITION"

    history = deepcopy(value)
    for arm in ("three_stream_two_candidates", "three_stream_three_candidates"):
        _set(history, arm, DECAY_SPEC["distractor_steps"], False)
    assert adjudicate(history)["verdict"] == "D2_STREAM_HISTORY_LOSS"

    interaction = deepcopy(value)
    _set(interaction, "three_stream_three_candidates", [2, 4, 8], False)
    assert adjudicate(interaction)["verdict"] == "D3_DELAY_INTERACTION"

    mixed = deepcopy(value)
    _set(mixed, "three_stream_three_candidates", [2], False, seeds=[DECAY_SPEC["seeds"][0]])
    assert adjudicate(mixed)["verdict"] == "D5_NON_MONOTONIC_OR_MIXED"

    invalid = deepcopy(value)
    _set(invalid, "two_stream_two_candidates", [2], False)
    assert adjudicate(invalid)["verdict"] == "D0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["delays"][0]["state_audit"]["prefix_state_matches"] -= 1
    assert adjudicate(invalid)["verdict"] == "D0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["delays"][0]["state_audit"]["episode_seed_sha256"] = "b" * 64
    assert adjudicate(invalid)["verdict"] == "D0_INVALID"
