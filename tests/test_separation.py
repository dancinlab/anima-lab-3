from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from episode_control import _metrics
from key_stability import StableKeyProjector
from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from measurement.separation_gate import adjudicate
from measurement.separation_registry import SEPARATION_SPEC, spec_sha256
from separation import build_episodes, dataset_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(accuracy: float = 1.0) -> dict:
    total = SEPARATION_SPEC["eval_episodes"]
    expected = torch.arange(total) % SEPARATION_SPEC["values"]
    if accuracy == 1.0:
        predicted = expected.clone()
    elif accuracy == 0.0:
        predicted = (expected + 1) % SEPARATION_SPEC["values"]
    else:
        predicted = torch.zeros_like(expected)
    value = _metrics(expected, predicted, SEPARATION_SPEC["values"])
    value.update({
        "selection_accuracy": accuracy,
        "correct_content_accuracy": 1.0,
        "retrieval_api_match": 1.0,
        "key_margin_mean": 0.5,
        "key_margin_min": 0.1,
    })
    return value


def test_separation_dataset_is_unique_balanced_and_deterministic():
    first = build_episodes()
    second = build_episodes()
    assert first == second
    audit = dataset_audit(first)
    assert audit["episodes"] == SEPARATION_SPEC["eval_episodes"]
    assert audit["unique_fingerprints"] == SEPARATION_SPEC["eval_episodes"]
    assert len(set(audit["target_counts"].values())) == 1
    assert len(set(audit["query_position_counts"].values())) == 1
    assert all(len(set(row.contexts)) == SEPARATION_SPEC["events_per_episode"] for row in first)
    assert all(len(set(row.values)) == SEPARATION_SPEC["events_per_episode"] for row in first)
    assert all(len(set(row.distinct_keys)) == SEPARATION_SPEC["events_per_episode"] for row in first)


def _passing_payload(tmp_path: Path) -> dict:
    source_rows, result_rows, checkpoints, prototypes = [], [], {}, {}
    for seed in SEPARATION_SPEC["seeds"]:
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
                "quantum": torch.ones(SEPARATION_SPEC["values"], SEPARATION_SPEC["state_dim"]),
                "sensory": torch.ones(SEPARATION_SPEC["values"], SEPARATION_SPEC["state_dim"]),
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
        arms = {name: _metric() for name in SEPARATION_SPEC["arms"]}
        arms["context_removed_control"] = _metric(0.25)
        arms["exact_context_key_partner_swap"] = _metric(0.0)
        arms["exact_context_key_recovered"]["prediction_match"] = 1.0
        calls = SEPARATION_SPEC["expected_stable_transform_calls_per_episode"]
        result_rows.append({
            "seed": seed,
            "arms": arms,
            "integration_audit": {
                "stable_transform_calls": {
                    "episodes": SEPARATION_SPEC["eval_episodes"],
                    "total": SEPARATION_SPEC["eval_episodes"] * calls,
                    "minimum": calls,
                    "maximum": calls,
                },
                "address_width_minimum": SEPARATION_SPEC["address_dim"],
                "address_width_maximum": SEPARATION_SPEC["address_dim"],
                "projector_frozen": True,
                "projector_unchanged": True,
            },
            "state_audit": {
                "episodes": SEPARATION_SPEC["eval_episodes"],
                "unique_episode_seeds": SEPARATION_SPEC["eval_episodes"],
                "episode_seed_sha256": "a" * 64,
            },
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
        "verdict": SEPARATION_SPEC["source_verdict"],
        "spec_sha256": episode2_spec_sha256(),
    }))
    return {
        "experiment": SEPARATION_SPEC["experiment"],
        "spec": deepcopy(SEPARATION_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": dataset_audit(build_episodes()),
        "source_episode2": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": SEPARATION_SPEC["source_verdict"],
            "source_spec_sha256": episode2_spec_sha256(),
            "checkpoints": checkpoints,
            "prototype_checkpoints": prototypes,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def test_separation_gate_verdicts_and_fail_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "S1_SIMILAR_EPISODES_SEPARATED_NOT_UNIQUE"

    stable_collision = deepcopy(value)
    stable_collision["seeds"][0]["arms"]["stable_similar_normal"] = _metric(0.25)
    assert adjudicate(stable_collision)["verdict"] == "S2_STABLE_ADDRESS_COLLISION"

    context_loss = deepcopy(stable_collision)
    context_loss["seeds"][0]["arms"]["raw_similar_normal"] = _metric(0.25)
    assert adjudicate(context_loss)["verdict"] == "S3_CONTEXT_ADDRESS_LOSS"

    value_loss = deepcopy(value)
    failed = _metric(0.25)
    failed["selection_accuracy"] = 1.0
    value_loss["seeds"][0]["arms"]["stable_similar_normal"] = failed
    assert adjudicate(value_loss)["verdict"] == "S4_VALUE_READOUT_LOSS"

    invalid = deepcopy(value)
    invalid["dataset_audit"]["unique_fingerprints"] -= 1
    assert adjudicate(invalid)["verdict"] == "S0_INVALID"
