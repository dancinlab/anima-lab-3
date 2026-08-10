from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from episode_control import _metrics, build_reference_splits, dataset_audit
from key_stability import StableKeyProjector
from measurement.episode2_gate import adjudicate
from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from trinity import VectorMemory


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(accuracy: float, diagnostics: bool = False) -> dict:
    total = EPISODE2_SPEC["eval_episodes"]
    expected = torch.arange(total) % EPISODE2_SPEC["values"]
    predicted = expected.clone() if accuracy == 1.0 else torch.zeros_like(expected)
    result = _metrics(expected, predicted, EPISODE2_SPEC["values"])
    if diagnostics:
        result.update({
            "selection_accuracy": 1.0,
            "correct_content_accuracy": 1.0,
            "retrieval_api_match": 1.0,
            "key_margin_mean": 0.5,
            "key_margin_min": 0.1,
        })
    return result


def test_vector_memory_optional_transform_is_shared_and_default_is_unchanged():
    calls = []

    def transform(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.clone())
        return value[:2]

    memory = VectorMemory(capacity=2, dim=3, key_transform=transform)
    memory.store(torch.tensor([[1.0, 0.0, 9.0]]), torch.tensor([1.0, 2.0, 3.0]))
    memory.store(torch.tensor([[0.0, 1.0, 9.0]]), torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(
        memory.retrieve(torch.tensor([[1.0, 0.0, -9.0]]), top_k=1)[0],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert len(calls) == 3
    assert all(key.shape == (2,) for key in memory.keys)

    raw = VectorMemory(capacity=1, dim=2)
    key = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    raw.store(key, key)
    assert torch.equal(raw.keys[0], key.float().mean(0))
    assert torch.equal(raw.retrieve(key, top_k=1)[0], key.float().mean(0))


@pytest.mark.parametrize("bad", [
    lambda value: [1.0],
    lambda value: torch.tensor([[1.0]]),
    lambda value: torch.tensor([float("nan")]),
])
def test_vector_memory_rejects_invalid_transformed_addresses(bad):
    memory = VectorMemory(key_transform=bad)
    with pytest.raises((TypeError, ValueError)):
        memory.store(torch.ones(2), torch.ones(2))


def _passing_payload(tmp_path: Path) -> dict:
    perfect = _metric(1.0)
    normal = _metric(1.0, diagnostics=True)
    chance = _metric(0.125)
    audit = dataset_audit(build_reference_splits(ATTENTION_CONTROL_SPEC), ATTENTION_CONTROL_SPEC)
    source_rows, result_rows, checkpoints, prototypes = [], [], {}, {}
    for seed in EPISODE2_SPEC["seeds"]:
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
                "quantum": torch.ones(EPISODE2_SPEC["values"], EPISODE2_SPEC["state_dim"]),
                "sensory": torch.ones(EPISODE2_SPEC["values"], EPISODE2_SPEC["state_dim"]),
            },
        }, prototype_path)
        prototype_receipt = {"path": str(prototype_path), "sha256": _sha(prototype_path)}
        checkpoints[str(seed)] = projector_receipt
        prototypes[str(seed)] = prototype_receipt
        source_rows.append({
            "seed": seed,
            "arms": {
                "stabilized_memory_normal": deepcopy(normal),
                "raw_quantum_memory": deepcopy(normal),
                "sensory_memory": deepcopy(normal),
                "keyed_attention": deepcopy(perfect),
                "no_memory": deepcopy(chance),
            },
            "eval_state_audit": {"engine_seed_sha256": "a" * 64},
            "checkpoint": projector_receipt,
            "source_checkpoint": prototype_receipt,
        })
        recovered = deepcopy(perfect)
        recovered["prediction_match"] = 1.0
        call_audit = {
            "episodes": EPISODE2_SPEC["eval_episodes"],
            "total": EPISODE2_SPEC["eval_episodes"] * 3,
            "minimum": 3,
            "maximum": 3,
        }
        result_rows.append({
            "seed": seed,
            "arms": {
                "integrated_stable_normal": deepcopy(normal),
                "integrated_stable_partner_swap": deepcopy(chance),
                "integrated_stable_recovered": recovered,
                "manual_stable_reference": deepcopy(normal),
                "transform_disabled": deepcopy(normal),
                "sensory_memory": deepcopy(normal),
                "keyed_attention": deepcopy(perfect),
                "no_memory": deepcopy(chance),
            },
            "integration_audit": {
                "normal_transform_calls": deepcopy(call_audit),
                "partner_swap_transform_calls": deepcopy(call_audit),
                "recovery_transform_calls": deepcopy(call_audit),
                "address_width_minimum": EPISODE2_SPEC["address_dim"],
                "address_width_maximum": EPISODE2_SPEC["address_dim"],
                "manual_prediction_match": 1.0,
                "manual_selection_match": 1.0,
                "projector_frozen": True,
                "projector_unchanged": True,
            },
            "state_audit": {
                "episodes": EPISODE2_SPEC["eval_episodes"],
                "unique_episode_seeds": EPISODE2_SPEC["eval_episodes"],
                "episode_seed_sha256": "a" * 64,
            },
            "source_checkpoint": projector_receipt,
            "prototype_checkpoint": prototype_receipt,
        })
    source_results_path = tmp_path / "key_results.json"
    source_results_path.write_text(json.dumps({
        "experiment": KEY_SPEC["experiment"],
        "spec": KEY_SPEC,
        "spec_sha256": key_spec_sha256(),
        "dataset_audit": audit,
        "seeds": source_rows,
    }))
    source_verdict_path = tmp_path / "key_verdict.json"
    source_verdict_path.write_text(json.dumps({
        "verdict": EPISODE2_SPEC["source_verdict"],
        "spec_sha256": key_spec_sha256(),
    }))
    return {
        "experiment": EPISODE2_SPEC["experiment"],
        "spec": deepcopy(EPISODE2_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": audit,
        "source_key": {
            "results": {"path": str(source_results_path), "sha256": _sha(source_results_path)},
            "verdict": {"path": str(source_verdict_path), "sha256": _sha(source_verdict_path)},
            "source_verdict": EPISODE2_SPEC["source_verdict"],
            "source_spec_sha256": key_spec_sha256(),
            "checkpoints": checkpoints,
            "prototype_checkpoints": prototypes,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": result_rows,
    }


def test_episode2_gate_passes_and_fails_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "E2I_PATH_RECOVERED_NOT_UNIQUE"

    memory_loss = deepcopy(value)
    memory_loss["seeds"][0]["integration_audit"]["manual_prediction_match"] = 0.5
    assert adjudicate(memory_loss)["verdict"] == "E2I_MEMORY_INTEGRATION_LOSS"

    behavior_loss = deepcopy(value)
    behavior_loss["seeds"][0]["arms"]["integrated_stable_normal"] = _metric(0.125, diagnostics=True)
    assert adjudicate(behavior_loss)["verdict"] == "E2I_BEHAVIOR_LOSS"

    invalid = deepcopy(value)
    invalid["source_key"]["results"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "E2I0_INVALID"
