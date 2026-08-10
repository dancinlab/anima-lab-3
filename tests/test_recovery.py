from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from episode_control import _metrics
from key_stability import StableKeyProjector
from measurement.decay_registry import DECAY_SPEC, spec_sha256 as decay_spec_sha256
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from measurement.recovery_gate import _close, adjudicate
from measurement.recovery_registry import RECOVERY_SPEC, spec_sha256
from recovery import build_recovery_episodes, recovery_dataset_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(passed: bool = True, total: int | None = None) -> dict:
    total = total or RECOVERY_SPEC["episodes_per_replicate"]
    expected = torch.arange(total) % RECOVERY_SPEC["values"]
    predicted = expected.clone() if passed else (expected + 1) % RECOVERY_SPEC["values"]
    value = _metrics(expected, predicted, RECOVERY_SPEC["values"])
    value.update({
        "selection_accuracy": float(passed),
        "correct_content_accuracy": 1.0,
        "retrieval_api_match": 1.0,
        "key_margin_mean": 0.5 if passed else -0.5,
        "key_margin_min": 0.1 if passed else -0.9,
    })
    return value


def _geometry(passed: bool, total: int) -> dict:
    half = total // 2
    return {
        "episodes": total,
        "target_rank_counts": [total, 0, 0] if passed else [0, total, 0],
        "selection_position_confusion": (
            [[half, 0, 0], [0, half, 0], [0, 0, 0]]
            if passed else [[0, half, 0], [0, 0, half], [0, 0, 0]]
        ),
        "target_similarity_mean": 0.8 if passed else 0.2,
        "strongest_wrong_similarity_mean": 0.2 if passed else 0.8,
        "third_candidate_similarity_mean": 0.1 if passed else 0.7,
        "target_minus_strongest_wrong_mean": 0.6 if passed else -0.6,
        "target_minus_third_candidate_mean": 0.7 if passed else -0.5,
    }


def _arms(main_passed: bool, total: int) -> dict:
    value = {name: _metric(True, total) for name in RECOVERY_SPEC["arms"]}
    value["stable_three_candidates"] = _metric(main_passed, total)
    value["exact_three_partner_swap"] = _metric(False, total)
    value["exact_three_recovered"]["prediction_match"] = 1.0
    return value


def _replicate_row(replicate: int, delay: int, main_passed: bool, digest: str) -> dict:
    total = RECOVERY_SPEC["episodes_per_replicate"]
    return {
        "replicate": replicate,
        "arms": _arms(main_passed, total),
        "geometry": _geometry(main_passed, total),
        "integration_audit": {
            "stable_transform_calls": {
                name: {
                    "episodes": total,
                    "total": total * calls,
                    "minimum": calls,
                    "maximum": calls,
                }
                for name, calls in RECOVERY_SPEC["expected_transform_calls"].items()
            },
            "address_width_minimum": RECOVERY_SPEC["address_dim"],
            "address_width_maximum": RECOVERY_SPEC["address_dim"],
        },
        "state_audit": {
            "episodes": total,
            "unique_episode_seeds": total,
            "episode_seed_sha256": digest,
            "minimum_cells": RECOVERY_SPEC["minimum_cells"],
            "maximum_cells": RECOVERY_SPEC["maximum_cells"],
        },
    }


def _pooled(replicates: list[dict]) -> dict:
    total = RECOVERY_SPEC["episodes_per_replicate"] * len(replicates)
    arms = {}
    for name in RECOVERY_SPEC["arms"]:
        passed = all(row["arms"][name]["accuracy"] == 1.0 for row in replicates)
        # Test verdict fixtures only use all-pass or all-fail main rows, except where
        # the mixed fixture replaces the pooled row explicitly below.
        arms[name] = _metric(passed, total)
        if name == "exact_three_partner_swap":
            arms[name] = _metric(False, total)
        if name == "exact_three_recovered":
            arms[name]["prediction_match"] = 1.0
    geometry_passed = all(row["geometry"]["target_rank_counts"][0] for row in replicates)
    return {"arms": arms, "geometry": _geometry(geometry_passed, total)}


def _passing_payload(tmp_path: Path) -> dict:
    source_rows, result_rows, checkpoints, prototypes = [], [], {}, {}
    total = RECOVERY_SPEC["episodes_per_replicate"]
    for seed in RECOVERY_SPEC["seeds"]:
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
                "quantum": torch.ones(RECOVERY_SPEC["values"], RECOVERY_SPEC["state_dim"]),
                "sensory": torch.ones(RECOVERY_SPEC["values"], RECOVERY_SPEC["state_dim"]),
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
        for delay in RECOVERY_SPEC["distractor_steps"]:
            passed = delay >= 3
            replicate_rows = [
                _replicate_row(replicate, delay, passed, chr(97 + replicate) * 64)
                for replicate in RECOVERY_SPEC["replicates"]
            ]
            delays.append({
                "distractor_steps": delay,
                "pooled": _pooled(replicate_rows),
                "replicates": replicate_rows,
            })
        result_rows.append({
            "seed": seed,
            "delays": delays,
            "projector_frozen": True,
            "projector_unchanged": True,
            "source_checkpoint": projector_receipt,
            "prototype_checkpoint": prototype_receipt,
        })
    source_results_path = tmp_path / "decay_results.json"
    source_results_path.write_text(json.dumps({
        "experiment": DECAY_SPEC["experiment"],
        "spec": DECAY_SPEC,
        "spec_sha256": decay_spec_sha256(),
        "seeds": source_rows,
    }))
    source_verdict_path = tmp_path / "decay_verdict.json"
    source_verdict_path.write_text(json.dumps({
        "verdict": RECOVERY_SPEC["source_verdict"],
        "spec_sha256": decay_spec_sha256(),
    }))
    episode_sets = {
        replicate: build_recovery_episodes(replicate)
        for replicate in RECOVERY_SPEC["replicates"]
    }
    return {
        "experiment": RECOVERY_SPEC["experiment"],
        "spec": deepcopy(RECOVERY_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": recovery_dataset_audit(episode_sets),
        "source_decay": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": RECOVERY_SPEC["source_verdict"],
            "source_spec_sha256": decay_spec_sha256(),
            "checkpoints": checkpoints,
            "prototype_checkpoints": prototypes,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def _set_all(payload: dict, delay: int, passed: bool) -> None:
    total = RECOVERY_SPEC["episodes_per_replicate"]
    for seed in payload["seeds"]:
        item = next(row for row in seed["delays"] if row["distractor_steps"] == delay)
        for replicate in item["replicates"]:
            replicate["arms"] = _arms(passed, total)
            replicate["geometry"] = _geometry(passed, total)
        item["pooled"] = _pooled(item["replicates"])


def test_recovery_datasets_are_balanced_deterministic_and_disjoint():
    first = {replicate: build_recovery_episodes(replicate) for replicate in RECOVERY_SPEC["replicates"]}
    second = {replicate: build_recovery_episodes(replicate) for replicate in RECOVERY_SPEC["replicates"]}
    assert first == second
    audit = recovery_dataset_audit(first)
    assert audit["combined_unique_fingerprints"] == (
        RECOVERY_SPEC["episodes_per_replicate"] * len(RECOVERY_SPEC["replicates"])
    )
    assert not any(audit["cross_replicate_overlap"].values())


def test_pooled_float32_rounding_tolerance_is_narrow():
    assert _close(0.8645833134651184, 0.8645833333333334)
    assert not _close(0.8645833, 0.8645843)


def test_recovery_gate_verdicts_and_fail_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "RC1_ORDERED_RECOVERY_REPRODUCED"

    mixed = deepcopy(value)
    total = RECOVERY_SPEC["episodes_per_replicate"]
    for seed in mixed["seeds"]:
        start = next(row for row in seed["delays"] if row["distractor_steps"] == 0)
        row = start["replicates"][0]
        row["arms"] = _arms(True, total)
        row["geometry"] = _geometry(True, total)
        # Pooled 1/3 remains below the registered pass threshold and preserves
        # the exact summed confusion required by the fail-closed gate.
        pooled = start["pooled"]
        main = pooled["arms"]["stable_three_candidates"]
        main["accuracy"] = main["selection_accuracy"] = 1 / 3
        main["key_margin_mean"] = -1 / 6
        main["confusion_matrix"] = [
            [64, 128, 0, 0, 0, 0, 0, 0],
            [0, 64, 128, 0, 0, 0, 0, 0],
            [0, 0, 64, 128, 0, 0, 0, 0],
            [0, 0, 0, 64, 128, 0, 0, 0],
            [0, 0, 0, 0, 64, 128, 0, 0],
            [0, 0, 0, 0, 0, 64, 128, 0],
            [0, 0, 0, 0, 0, 0, 64, 128],
            [128, 0, 0, 0, 0, 0, 0, 64],
        ]
        pooled["geometry"] = {
            "episodes": total * 3,
            "target_rank_counts": [total, total * 2, 0],
            "selection_position_confusion": [[256, 512, 0], [0, 256, 512], [0, 0, 0]],
            "target_similarity_mean": 0.4,
            "strongest_wrong_similarity_mean": 0.6,
            "third_candidate_similarity_mean": 0.5,
            "target_minus_strongest_wrong_mean": -0.2,
            "target_minus_third_candidate_mean": -0.1,
        }
    assert adjudicate(mixed)["verdict"] == "RC2_RECOVERY_REPRODUCED_MIXED"

    no_recovery = deepcopy(value)
    _set_all(no_recovery, 0, True)
    assert adjudicate(no_recovery)["verdict"] == "RC3_RECOVERY_NOT_REPRODUCED"

    delay_loss = deepcopy(value)
    _set_all(delay_loss, 0, True)
    for delay in RECOVERY_SPEC["distractor_steps"][1:]:
        _set_all(delay_loss, delay, False)
    assert adjudicate(delay_loss)["verdict"] == "RC4_DELAY_LOSS"

    invalid = deepcopy(value)
    invalid["seeds"][0]["delays"][1]["replicates"][0]["state_audit"]["episode_seed_sha256"] = "f" * 64
    assert adjudicate(invalid)["verdict"] == "RC0_INVALID"
