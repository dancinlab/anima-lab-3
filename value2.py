#!/usr/bin/env python3
"""VALUE-2: integrate a deterministic time-stable value representation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from conjunction import (
    _direct_prediction,
    _exact_addresses,
    _memory_outcome,
    _receipt,
    _wrong_value_states,
    trace_episode,
)
from episode import _decode
from graft_behavior import sha256_file
from key_stability import (
    StableKeyProjector,
    fit_stable_key_projector,
    key_classification_metrics,
)
from measurement.projector_registry import evaluation_name
from measurement.value2_registry import VALUE2_SPEC, spec_sha256
from measurement.value_mechanism_gate import adjudicate as adjudicate_mechanism
from measurement.value_mechanism_registry import (
    VALUE_MECHANISM_SPEC,
    spec_sha256 as mechanism_spec_sha256,
)
from measurement.value_registry import VALUE_SPEC
from separation import _arm_metrics
from trinity import VectorMemory
from value import build_value_episodes, value_dataset_audit
from value_mechanism import position_episode


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_receipt(spec: dict = VALUE2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = mechanism_spec_sha256(VALUE_MECHANISM_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != VALUE_MECHANISM_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_mechanism(results) != verdict
    ):
        raise RuntimeError("registered VALUE-MECHANISM-1 source changed")
    inherited = results["source_value1"]
    for receipt in inherited["prototype_checkpoints"].values():
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered VALUE-MECHANISM-1 prototype changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "prototype_checkpoints": {
            key: dict(value) for key, value in inherited["prototype_checkpoints"].items()
        },
    }


def calibration_episode_spec(spec: dict = VALUE2_SPEC) -> dict:
    value = dict(VALUE_SPEC)
    value["eval_episodes"] = spec["calibration_episodes"]
    value["data_seed"] = spec["calibration_data_seed"]
    return value


def balance_calibration_values(episodes, spec: dict = VALUE2_SPEC):
    """Assign an exact balanced value roster without changing event pairs or Latin structure."""
    balanced = []
    for index, episode in enumerate(episodes):
        target = index % spec["values"]
        assigned = [
            (target + offset) % spec["values"]
            for offset in range(spec["active_values_per_episode"])
        ]
        original = [
            episode.target,
            *sorted(value for value in episode.active_values if value != episode.target),
        ]
        mapping = dict(zip(original, assigned))
        balanced.append(type(episode)(
            contexts=episode.contexts,
            keys=episode.keys,
            values=tuple(mapping[value] for value in episode.values),
            active_contexts=episode.active_contexts,
            active_keys=episode.active_keys,
            active_values=tuple(sorted(assigned)),
            distractors=episode.distractors,
            query_position=episode.query_position,
        ))
    return balanced


def _canonical_rows(states: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = sorted(
        range(len(states)),
        key=lambda index: (int(labels[index]), states[index].contiguous().numpy().tobytes()),
    )
    indices = torch.tensor(order, dtype=torch.long)
    return states[indices], labels[indices]


def collect_calibration_states(episodes, spec: dict = VALUE2_SPEC):
    states, labels, used_seeds, cell_counts = [], [], [], []
    for engine_seed in spec["calibration_engine_seeds"]:
        base = spec["calibration_seed_base"] + engine_seed * spec["seed_stride"]
        for index, episode in enumerate(episodes):
            trial_seed = base + index
            trace = trace_episode(episode, trial_seed, spec)
            states.extend(value.mean(0) for value in trace["values"])
            labels.extend(episode.values)
            used_seeds.append(trial_seed)
            cell_counts.extend(trace["cell_counts"])
    tensor_states = torch.stack(states)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_states, tensor_labels = _canonical_rows(tensor_states, tensor_labels)
    return tensor_states, tensor_labels, {
        "episodes": len(episodes) * len(spec["calibration_engine_seeds"]),
        "states": len(tensor_states),
        "unique_engine_seeds": len(set(used_seeds)),
        "engine_seed_sha256": hashlib.sha256(
            "\n".join(map(str, sorted(used_seeds))).encode()
        ).hexdigest(),
        "label_counts": {
            str(value): int((tensor_labels == value).sum())
            for value in range(spec["values"])
        },
        "state_sha256": hashlib.sha256(tensor_states.numpy().tobytes()).hexdigest(),
        "label_sha256": hashlib.sha256(tensor_labels.numpy().tobytes()).hexdigest(),
        "minimum_cells": min(cell_counts),
        "maximum_cells": max(cell_counts),
    }


def _state_dict_equal(first: StableKeyProjector, second: StableKeyProjector) -> bool:
    return all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def fit_value_projector(states: torch.Tensor, labels: torch.Tensor,
                        output_path: Path, spec: dict = VALUE2_SPEC):
    projector, fit_audit = fit_stable_key_projector(
        states, labels, spec, method=spec["fit_method"]
    )
    repeated, _ = fit_stable_key_projector(
        states, labels, spec, method=spec["fit_method"]
    )
    reverse_states, reverse_labels = _canonical_rows(states.flip(0), labels.flip(0))
    reversed_projector, _ = fit_stable_key_projector(
        reverse_states, reverse_labels, spec, method=spec["fit_method"]
    )
    deterministic = {
        "repeat_equal": _state_dict_equal(projector, repeated),
        "reverse_order_equal": _state_dict_equal(projector, reversed_projector),
    }
    _atomic_torch(output_path, {
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "model_class": spec["model_class"],
        "projector": projector.state_dict(),
        "fit_audit": fit_audit,
        "deterministic": deterministic,
    })
    return projector, fit_audit, deterministic, _receipt(output_path)


class StableValueTransform:
    def __init__(self, projector: StableKeyProjector):
        self.projector = projector
        self.calls = 0
        self.input_widths: list[int] = []
        self.output_widths: list[int] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.input_widths.append(value.numel())
        output = self.projector.address(value.unsqueeze(0))[0].detach()
        self.output_widths.append(output.numel())
        return output


def _integrated_outcome(addresses, values, query, prototypes, projector,
                        *, stored_values=None):
    transform = StableValueTransform(projector)
    memory = VectorMemory(
        capacity=len(addresses), dim=addresses[0].numel(), value_transform=transform
    )
    payloads = values if stored_values is None else stored_values
    for address, value in zip(addresses, payloads):
        memory.store(address, value)
    outcome = _memory_outcome(memory, query, payloads, prototypes)
    return outcome, {
        "calls": transform.calls,
        "minimum_input_width": min(transform.input_widths),
        "maximum_input_width": max(transform.input_widths),
        "minimum_output_width": min(transform.output_widths),
        "maximum_output_width": max(transform.output_widths),
        "stores": len(memory.values),
        "retrievals": 1,
    }


def _external_outcome(addresses, values, query, prototypes, projector):
    transformed = [projector.address(value.mean(0).unsqueeze(0))[0].detach() for value in values]
    return _direct_prediction(addresses, transformed, query, prototypes)


def _run_position(prototype_seed: int, engine_seed: int, query_position: int,
                  episodes, raw_prototypes: torch.Tensor, projector: StableKeyProjector,
                  spec: dict = VALUE2_SPEC) -> dict:
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [query_position] * len(episodes)
    canonical_prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    episode_seeds, cell_counts = [], []
    classification_states, classification_labels = [], []
    path_rows = []
    base = spec["eval_seed_base"] + engine_seed * spec["seed_stride"]
    for index, base_episode in enumerate(episodes):
        episode = position_episode(base_episode, query_position, VALUE_MECHANISM_SPEC)
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, spec)
        cell_counts.extend(trace["cell_counts"])
        classification_states.extend(value.mean(0) for value in trace["values"])
        classification_labels.extend(episode.values)
        exact, query = _exact_addresses(episode, spec=spec)
        normal, path_audit = _integrated_outcome(
            exact, trace["values"], query, canonical_prototypes, projector
        )
        fake, _ = _integrated_outcome(
            exact, trace["values"], query, canonical_prototypes, projector,
            stored_values=_wrong_value_states(episode, trace),
        )
        recovered, _ = _integrated_outcome(
            exact, trace["values"], query, canonical_prototypes, projector
        )
        outcomes = {
            "integrated_stable_value_normal": normal,
            "external_stable_value_reference": _external_outcome(
                exact, trace["values"], query, canonical_prototypes, projector
            ),
            "raw_value_control": _direct_prediction(
                exact, trace["values"], query, raw_prototypes
            ),
            "integrated_stable_value_partner_swap": fake,
            "integrated_stable_value_recovered": recovered,
        }
        canonical_content = int(projector(torch.stack([
            trace["values"][query_position].mean(0)
        ])).argmax(1)[0])
        raw_content = _decode(trace["values"][query_position], raw_prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(
                raw_content if name == "raw_value_control" else canonical_content
            )
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        path_rows.append(path_audit)
        if (index + 1) % 128 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed} position {query_position + 1}] "
                f"evaluated {index + 1}/{len(episodes)} episodes",
                flush=True,
            )
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    normal_records = records["integrated_stable_value_normal"]
    reference_records = records["external_stable_value_reference"]
    arms["integrated_stable_value_normal"]["reference_prediction_match"] = float(
        normal_records["predictions"] == reference_records["predictions"]
    )
    arms["integrated_stable_value_recovered"]["prediction_match"] = float(
        records["integrated_stable_value_recovered"]["predictions"]
        == normal_records["predictions"]
    )
    classification = key_classification_metrics(
        projector, torch.stack(classification_states),
        torch.tensor(classification_labels, dtype=torch.long), spec["values"],
    )
    return {
        "query_position": query_position,
        "query_position_label": query_position + 1,
        "arms": arms,
        "value_classification": classification,
        "path_audit": {
            "minimum_calls": min(row["calls"] for row in path_rows),
            "maximum_calls": max(row["calls"] for row in path_rows),
            "minimum_input_width": min(row["minimum_input_width"] for row in path_rows),
            "maximum_input_width": max(row["maximum_input_width"] for row in path_rows),
            "minimum_output_width": min(row["minimum_output_width"] for row in path_rows),
            "maximum_output_width": max(row["maximum_output_width"] for row in path_rows),
            "minimum_stores": min(row["stores"] for row in path_rows),
            "maximum_stores": max(row["stores"] for row in path_rows),
            "minimum_retrievals": min(row["retrievals"] for row in path_rows),
            "maximum_retrievals": max(row["retrievals"] for row in path_rows),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts), "maximum_cells": max(cell_counts),
        },
    }


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   projector: StableKeyProjector, spec: dict = VALUE2_SPEC) -> dict:
    receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    raw_prototypes = checkpoint["prototypes"]["quantum"]
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "positions": [
            _run_position(
                prototype_seed, engine_seed, position, episodes,
                raw_prototypes, projector, spec,
            )
            for position in spec["query_positions"]
        ],
        "prototype_checkpoint": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/value2_results.json")
    parser.add_argument("--verdict", default="measurement/value2_verdict.json")
    args = parser.parse_args()
    spec = VALUE2_SPEC
    _, source = _source_receipt(spec)
    calibration_spec = calibration_episode_spec(spec)
    calibration_episodes = balance_calibration_values(
        build_value_episodes(calibration_spec), spec
    )
    eval_episodes = build_value_episodes(VALUE_SPEC)
    calibration_states, calibration_labels, calibration_state_audit = collect_calibration_states(
        calibration_episodes, spec
    )
    checkpoint_path = Path(spec["checkpoint_path"])
    projector, fit_audit, deterministic, checkpoint_receipt = fit_value_projector(
        calibration_states, calibration_labels, checkpoint_path, spec
    )
    payload = {
        "experiment": spec["experiment"], "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_value_mechanism1": source,
        "calibration_dataset_audit": value_dataset_audit(
            calibration_episodes, calibration_spec
        ),
        "eval_dataset_audit": value_dataset_audit(eval_episodes, VALUE_SPEC),
        "calibration_state_audit": calibration_state_audit,
        "fit_audit": fit_audit,
        "deterministic_audit": deterministic,
        "checkpoint": checkpoint_receipt,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": [
            {
                "name": evaluation_name(row),
                **run_evaluation(
                    row["prototype_seed"], row["engine_seed"], eval_episodes,
                    source, projector, spec,
                ),
            }
            for row in spec["evaluation_combinations"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.value2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
