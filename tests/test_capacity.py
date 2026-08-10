from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from capacity import build_capacity_episodes, capacity_dataset_audit
from episode_control import _metrics
from key_stability import StableKeyProjector
from measurement.capacity_gate import adjudicate
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(accuracy: float = 1.0) -> dict:
    total = CAPACITY_SPEC["eval_episodes_per_count"]
    expected = torch.arange(total) % CAPACITY_SPEC["values"]
    predicted = expected.clone() if accuracy == 1.0 else (expected + 1) % CAPACITY_SPEC["values"]
    value = _metrics(expected, predicted, CAPACITY_SPEC["values"])
    value.update({
        "selection_accuracy": accuracy,
        "correct_content_accuracy": 1.0,
        "retrieval_api_match": 1.0,
        "key_margin_mean": 0.5,
        "key_margin_min": 0.1,
    })
    return value


def test_capacity_datasets_are_balanced_unique_and_deterministic():
    for count in CAPACITY_SPEC["event_counts"]:
        first = build_capacity_episodes(count)
        second = build_capacity_episodes(count)
        assert first == second
        assert all(len(row.values) == count for row in first)
        audit = capacity_dataset_audit(count)
        assert audit["episodes"] == CAPACITY_SPEC["eval_episodes_per_count"]
        assert audit["unique_fingerprints"] == CAPACITY_SPEC["eval_episodes_per_count"]
        assert len(set(audit["target_counts"].values())) == 1
        assert len(set(audit["query_position_counts"].values())) == 1
        assert len(set(audit["shared_key_counts"].values())) == 1


def _passing_payload(tmp_path: Path) -> dict:
    source_rows, result_rows, checkpoints, prototypes = [], [], {}, {}
    total = CAPACITY_SPEC["eval_episodes_per_count"]
    for seed in CAPACITY_SPEC["seeds"]:
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
                "quantum": torch.ones(CAPACITY_SPEC["values"], CAPACITY_SPEC["state_dim"]),
                "sensory": torch.ones(CAPACITY_SPEC["values"], CAPACITY_SPEC["state_dim"]),
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
        counts = []
        for count in CAPACITY_SPEC["event_counts"]:
            arms = {name: _metric() for name in CAPACITY_SPEC["arms"]}
            arms["exact_key_partner_swap"] = _metric(0.0)
            arms["exact_key_recovered"]["prediction_match"] = 1.0
            calls = count + 1
            counts.append({
                "event_count": count,
                "arms": arms,
                "integration_audit": {
                    "stable_transform_calls": {
                        "episodes": total,
                        "total": total * calls,
                        "minimum": calls,
                        "maximum": calls,
                    },
                    "address_width_minimum": CAPACITY_SPEC["address_dim"],
                    "address_width_maximum": CAPACITY_SPEC["address_dim"],
                },
                "state_audit": {
                    "episodes": total,
                    "unique_episode_seeds": total,
                    "episode_seed_sha256": "a" * 64,
                    "minimum_cells": CAPACITY_SPEC["minimum_cells"],
                    "maximum_cells": CAPACITY_SPEC["maximum_cells"],
                },
            })
        result_rows.append({
            "seed": seed,
            "counts": counts,
            "projector_frozen": True,
            "projector_unchanged": True,
            "source_checkpoint": projector_receipt,
            "prototype_checkpoint": prototype_receipt,
        })
    source_results_path = tmp_path / "episode2_results.json"
    source_results_path.write_text(json.dumps({
        "experiment": EPISODE2_SPEC["experiment"],
        "spec": EPISODE2_SPEC,
        "spec_sha256": episode2_spec_sha256(),
        "seeds": source_rows,
    }))
    source_verdict_path = tmp_path / "episode2_verdict.json"
    source_verdict_path.write_text(json.dumps({
        "verdict": CAPACITY_SPEC["source_verdict"],
        "spec_sha256": episode2_spec_sha256(),
    }))
    return {
        "experiment": CAPACITY_SPEC["experiment"],
        "spec": deepcopy(CAPACITY_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": {
            str(count): capacity_dataset_audit(count)
            for count in CAPACITY_SPEC["event_counts"]
        },
        "source_episode2": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": CAPACITY_SPEC["source_verdict"],
            "source_spec_sha256": episode2_spec_sha256(),
            "checkpoints": checkpoints,
            "prototype_checkpoints": prototypes,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def _set_count_accuracy(payload: dict, count: int, accuracy: float) -> None:
    for seed in payload["seeds"]:
        row = next(item for item in seed["counts"] if item["event_count"] == count)
        row["arms"]["stable_distinct_normal"] = _metric(accuracy)


def test_capacity_gate_verdicts_and_fail_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "C1_CAPACITY_AT_LEAST_4"

    boundary_three = deepcopy(value)
    _set_count_accuracy(boundary_three, 4, 0.0)
    assert adjudicate(boundary_three)["verdict"] == "C2_CAPACITY_BOUNDARY_3"

    boundary_two = deepcopy(boundary_three)
    _set_count_accuracy(boundary_two, 3, 0.0)
    assert adjudicate(boundary_two)["verdict"] == "C3_CAPACITY_BOUNDARY_2"

    below_two = deepcopy(boundary_two)
    _set_count_accuracy(below_two, 2, 0.0)
    assert adjudicate(below_two)["verdict"] == "C4_CAPACITY_BELOW_2"

    non_monotonic = deepcopy(value)
    _set_count_accuracy(non_monotonic, 3, 0.0)
    assert adjudicate(non_monotonic)["verdict"] == "C5_NON_MONOTONIC"

    invalid = deepcopy(value)
    invalid["dataset_audit"]["2"]["unique_fingerprints"] -= 1
    assert adjudicate(invalid)["verdict"] == "C0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["projector_unchanged"] = False
    assert adjudicate(invalid)["verdict"] == "C0_INVALID"
