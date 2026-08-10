from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from episode import _decode, _memory_prediction, build_value_prototypes, trace_episode_states
from episode_control import _metrics, build_reference_splits, dataset_audit
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC, spec_sha256 as source_spec_sha256
from measurement.episode_gate import adjudicate
from measurement.episode_registry import EPISODE_SPEC, spec_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(accuracy: float) -> dict:
    total = EPISODE_SPEC["splits"]["eval"]
    expected = torch.arange(total) % EPISODE_SPEC["values"]
    predicted = expected.clone() if accuracy == 1.0 else torch.zeros_like(expected)
    return _metrics(expected, predicted, EPISODE_SPEC["values"])


def test_registered_words_and_source_shapes_are_canonical():
    assert len(EPISODE_SPEC["key_words"]) == EPISODE_SPEC["keys"] == 8
    assert len(EPISODE_SPEC["value_words"]) == EPISODE_SPEC["values"] == 8
    assert len(set(EPISODE_SPEC["key_words"] + EPISODE_SPEC["value_words"])) == 16
    assert EPISODE_SPEC["splits"] == ATTENTION_CONTROL_SPEC["splits"]
    assert len(spec_sha256()) == 64


def test_value_prototypes_are_disjoint_finite_and_registered_shape():
    prototypes, audit = build_value_prototypes(1337)
    assert audit["states"] == audit["unique_seeds"] == 128
    assert len(audit["seed_sha256"]) == 64
    for value in prototypes.values():
        assert value.shape == (EPISODE_SPEC["values"], EPISODE_SPEC["state_dim"])
        assert torch.isfinite(value).all()


def test_existing_vector_memory_retrieval_and_partner_swap_are_causal():
    prototypes = torch.eye(2)
    keys = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    values = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    normal, selected, api_match, _ = _memory_prediction(
        keys, values, keys[0], prototypes, swap=False
    )
    swapped, _, swapped_api_match, _ = _memory_prediction(
        keys, values, keys[0], prototypes, swap=True
    )
    assert (normal, selected, api_match) == (0, 0, True)
    assert (swapped, swapped_api_match) == (1, True)
    assert _decode(values[1], prototypes) == 1

    transformed, transformed_selected, transformed_api, _ = _memory_prediction(
        keys, values, keys[0], prototypes, key_transform=lambda value: value.flip(0)
    )
    assert (transformed, transformed_selected, transformed_api) == (0, 0, True)


def test_episode_trace_uses_the_registered_sequence_once():
    episode = build_reference_splits(ATTENTION_CONTROL_SPEC)["validation"][0]
    trace = trace_episode_states(episode, 987654321)
    assert set(trace) == {
        "quantum_keys", "quantum_values", "quantum_query",
        "sensory_keys", "sensory_values", "sensory_query",
    }
    assert len(trace["quantum_keys"]) == len(trace["quantum_values"]) == 2
    assert trace["quantum_query"].shape == (48, 96)


def _passing_payload(tmp_path: Path) -> dict:
    source_checkpoints = {}
    source_rows = []
    for seed in EPISODE_SPEC["seeds"]:
        path = tmp_path / f"source_{seed}.pt"
        torch.save({
            "experiment": EPISODE_SPEC["source_experiment"],
            "spec_sha256": source_spec_sha256(ATTENTION_CONTROL_SPEC),
            "seed": seed,
            "model_class": ATTENTION_CONTROL_SPEC["model_class"],
            "model": {},
        }, path)
        receipt = {"path": str(path), "sha256": _sha(path)}
        source_checkpoints[str(seed)] = receipt
        source_rows.append({"seed": seed, "checkpoint": receipt})
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    audit = dataset_audit(splits, ATTENTION_CONTROL_SPEC)
    source_results = tmp_path / "source_results.json"
    source_results.write_text(json.dumps({
        "experiment": EPISODE_SPEC["source_experiment"],
        "spec": ATTENTION_CONTROL_SPEC,
        "spec_sha256": source_spec_sha256(ATTENTION_CONTROL_SPEC),
        "dataset_audit": audit,
        "seeds": source_rows,
    }))
    source_verdict = tmp_path / "source_verdict.json"
    source_verdict.write_text(json.dumps({
        "verdict": EPISODE_SPEC["source_verdict"],
        "spec_sha256": source_spec_sha256(ATTENTION_CONTROL_SPEC),
    }))
    perfect = _metric(1.0)
    chance = _metric(0.125)
    seeds = []
    prototype_count = EPISODE_SPEC["values"] * EPISODE_SPEC["prototype_repeats_per_value"]
    for seed in EPISODE_SPEC["seeds"]:
        prototype_audit = {
            "states": prototype_count,
            "unique_seeds": prototype_count,
            "seed_sha256": "a" * 64,
        }
        checkpoint = tmp_path / f"episode_{seed}.pt"
        torch.save({
            "experiment": EPISODE_SPEC["experiment"],
            "spec_sha256": spec_sha256(),
            "seed": seed,
            "prototype_audit": prototype_audit,
            "prototypes": {
                "quantum": torch.ones(8, 96),
                "sensory": torch.ones(8, 96),
            },
        }, checkpoint)
        normal = deepcopy(perfect)
        normal.update({
            "selection_accuracy": 1.0,
            "correct_content_accuracy": 1.0,
            "retrieval_api_match": 1.0,
            "key_margin_mean": 0.5,
            "key_margin_min": 0.1,
        })
        recovered = deepcopy(perfect)
        recovered["prediction_match"] = 1.0
        seeds.append({
            "seed": seed,
            "arms": {
                "quantum_memory_normal": deepcopy(normal),
                "quantum_memory_partner_swap": deepcopy(chance),
                "quantum_memory_recovered": recovered,
                "sensory_memory_normal": deepcopy(normal),
                "sensory_memory_partner_swap": deepcopy(chance),
                "keyed_attention": deepcopy(perfect),
                "no_memory": deepcopy(chance),
            },
            "state_audit": {
                "episodes": EPISODE_SPEC["splits"]["eval"],
                "unique_episode_seeds": EPISODE_SPEC["splits"]["eval"],
                "episode_seed_sha256": "b" * 64,
                "prototype": prototype_audit,
                "prototype_episode_seed_overlap": 0,
            },
            "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
            "source_checkpoint": source_checkpoints[str(seed)],
        })
    return {
        "experiment": EPISODE_SPEC["experiment"],
        "spec": deepcopy(EPISODE_SPEC),
        "spec_sha256": spec_sha256(),
        "dataset_audit": audit,
        "source_control": {
            "results": {"path": str(source_results), "sha256": _sha(source_results)},
            "verdict": {"path": str(source_verdict), "sha256": _sha(source_verdict)},
            "source_verdict": EPISODE_SPEC["source_verdict"],
            "source_spec_sha256": source_spec_sha256(ATTENTION_CONTROL_SPEC),
            "checkpoints": source_checkpoints,
        },
        "runtime": {"python": "test", "torch": "test", "device": "cpu"},
        "seeds": seeds,
    }


def test_episode_gate_passes_and_fails_closed(tmp_path):
    value = _passing_payload(tmp_path)
    assert adjudicate(value)["verdict"] == "E1_STATE_MEMORY_VALID_NOT_UNIQUE"

    key_loss = deepcopy(value)
    key_loss["seeds"][0]["arms"]["quantum_memory_normal"]["selection_accuracy"] = 0.5
    assert adjudicate(key_loss)["verdict"] == "E2_KEY_RETRIEVAL_LOSS"

    invalid = deepcopy(value)
    invalid["seeds"][0]["arms"]["sensory_memory_normal"]["accuracy"] = 0.5
    assert adjudicate(invalid)["verdict"] == "E0_INVALID"

    invalid = deepcopy(value)
    invalid["source_control"]["results"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "E0_INVALID"

    invalid = deepcopy(value)
    invalid["seeds"][0]["checkpoint"]["sha256"] = "0" * 64
    assert adjudicate(invalid)["verdict"] == "E0_INVALID"
