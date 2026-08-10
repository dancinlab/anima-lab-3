from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from episode_control import _metrics, build_reference_splits, dataset_audit
from key_stability import StableKeyProjector, key_classification_metrics, train_projector
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.episode_registry import EPISODE_SPEC, spec_sha256 as episode_spec_sha256
from measurement.key_gate import adjudicate
from measurement.key_registry import KEY_SPEC, spec_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(accuracy: float) -> dict:
    total = KEY_SPEC["eval_episodes"]
    expected = torch.arange(total) % KEY_SPEC["values"]
    predicted = expected.clone() if accuracy == 1.0 else torch.zeros_like(expected)
    result = _metrics(expected, predicted, KEY_SPEC["values"])
    result.update({
        "selection_accuracy": accuracy,
        "correct_content_accuracy": 1.0,
        "retrieval_api_match": 1.0,
        "key_margin_mean": 0.5,
        "key_margin_min": 0.1,
    })
    return result


def _key_metric(accuracy: float) -> dict:
    matrix = [[0] * KEY_SPEC["keys"] for _ in range(KEY_SPEC["keys"])]
    for key in range(KEY_SPEC["keys"]):
        matrix[key][key if accuracy == 1.0 else 0] = 1
    return {
        "accuracy": accuracy,
        "per_key_recall": [accuracy] * KEY_SPEC["keys"],
        "confusion_matrix": matrix,
    }


def test_projector_shapes_and_metric_learning_are_deterministic():
    states = torch.eye(8).repeat_interleave(8, dim=0)
    states = torch.nn.functional.pad(states, (0, KEY_SPEC["input_dim"] - 8))
    labels = torch.arange(8).repeat_interleave(8)
    local_spec = deepcopy(KEY_SPEC)
    local_spec.update({"train_steps": 100, "batch_size": 64})
    model, audit = train_projector(states, labels, 123, False, local_spec)
    assert isinstance(model, StableKeyProjector)
    assert model.projection.weight.shape == (KEY_SPEC["address_dim"], KEY_SPEC["input_dim"])
    assert audit["steps"] == 100
    assert key_classification_metrics(model, states, labels, 8)["accuracy"] == 1.0


def _passing_payload(tmp_path: Path) -> dict:
    perfect = _metric(1.0)
    chance = _metrics(
        torch.arange(KEY_SPEC["eval_episodes"]) % KEY_SPEC["values"],
        torch.zeros(KEY_SPEC["eval_episodes"], dtype=torch.long), KEY_SPEC["values"],
    )
    source_rows, result_rows, source_checkpoints = [], [], {}
    for seed in KEY_SPEC["seeds"]:
        source_checkpoint_path = tmp_path / f"source_{seed}.pt"
        torch.save({
            "experiment": EPISODE_SPEC["experiment"],
            "spec_sha256": episode_spec_sha256(),
            "seed": seed,
            "prototypes": {"quantum": torch.ones(8, 96), "sensory": torch.ones(8, 96)},
        }, source_checkpoint_path)
        source_checkpoint = {"path": str(source_checkpoint_path), "sha256": _sha(source_checkpoint_path)}
        source_checkpoints[str(seed)] = source_checkpoint
        source_normal = deepcopy(perfect)
        source_rows.append({
            "seed": seed,
            "arms": {
                "quantum_memory_normal": deepcopy(source_normal),
                "sensory_memory_normal": deepcopy(source_normal),
                "keyed_attention": deepcopy(perfect),
                "no_memory": deepcopy(chance),
            },
            "checkpoint": source_checkpoint,
        })
        projector = StableKeyProjector(96, 32, 8, 0.1)
        training = {
            "examples": KEY_SPEC["calibration_episodes"] * 3,
            "steps": KEY_SPEC["train_steps"],
            "shuffled": False,
            "final_loss_mean_50": 0.1,
            "training_label_sha256": "a" * 64,
        }
        shuffled_training = {**training, "shuffled": True, "training_label_sha256": "b" * 64}
        checkpoint_path = tmp_path / f"key_{seed}.pt"
        torch.save({
            "experiment": KEY_SPEC["experiment"],
            "spec_sha256": spec_sha256(),
            "seed": seed,
            "model_class": KEY_SPEC["model_class"],
            "projector": projector.state_dict(),
            "shuffled_label_projector": projector.state_dict(),
            "training_audit": training,
            "shuffled_training_audit": shuffled_training,
        }, checkpoint_path)
        recovered = deepcopy(perfect)
        recovered["prediction_match"] = 1.0
        result_rows.append({
            "seed": seed,
            "key_classification": _key_metric(1.0),
            "arms": {
                "stabilized_memory_normal": deepcopy(perfect),
                "stabilized_memory_partner_swap": deepcopy(chance),
                "stabilized_memory_recovered": recovered,
                "raw_quantum_memory": deepcopy(source_normal),
                "sensory_memory": deepcopy(source_normal),
                "keyed_attention": deepcopy(perfect),
                "no_memory": deepcopy(chance),
                "shuffled_label_projector": _key_metric(0.125),
            },
            "calibration_audit": {
                "episodes": KEY_SPEC["calibration_episodes"],
                "states": KEY_SPEC["calibration_episodes"] * 3,
                "unique_engine_seeds": KEY_SPEC["calibration_episodes"],
                "engine_seed_sha256": "c" * 64,
                "key_counts": {str(key): 384 for key in range(8)},
            },
            "eval_state_audit": {
                "episodes": KEY_SPEC["eval_episodes"],
                "states": KEY_SPEC["eval_episodes"] * 3,
                "unique_engine_seeds": KEY_SPEC["eval_episodes"],
                "engine_seed_sha256": "d" * 64,
                "key_counts": {str(key): 768 for key in range(8)},
                "calibration_engine_seed_overlap": 0,
            },
            "training_audit": training,
            "shuffled_training_audit": shuffled_training,
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha(checkpoint_path)},
            "source_checkpoint": source_checkpoint,
        })
    audit = dataset_audit(build_reference_splits(ATTENTION_CONTROL_SPEC), ATTENTION_CONTROL_SPEC)
    source_results_path = tmp_path / "episode_results.json"
    source_results_path.write_text(json.dumps({
        "experiment": EPISODE_SPEC["experiment"],
        "spec": EPISODE_SPEC,
        "spec_sha256": episode_spec_sha256(),
        "dataset_audit": audit,
        "seeds": source_rows,
    }))
    source_verdict_path = tmp_path / "episode_verdict.json"
    source_verdict_path.write_text(json.dumps({
        "verdict": KEY_SPEC["source_verdict"], "spec_sha256": episode_spec_sha256(),
    }))
    return {
        "experiment": KEY_SPEC["experiment"],
        "spec": deepcopy(KEY_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": audit,
        "source_episode": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": KEY_SPEC["source_verdict"],
            "source_spec_sha256": episode_spec_sha256(),
            "checkpoints": source_checkpoints,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def test_key_gate_passes_and_fails_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "K1_STABLE_KEY_VALID_NOT_UNIQUE"

    alignment_loss = deepcopy(value)
    alignment_loss["seeds"][0]["key_classification"]["accuracy"] = 0.5
    assert adjudicate(alignment_loss)["verdict"] == "K2_KEY_ALIGNMENT_LOSS"

    invalid = deepcopy(value)
    invalid["seeds"][0]["arms"]["shuffled_label_projector"]["accuracy"] = 0.5
    assert adjudicate(invalid)["verdict"] == "K0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["checkpoint"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "K0_INVALID"
