#!/usr/bin/env python3
"""EPISODE-2: run frozen stable keys through the shared VectorMemory path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from episode import _decode, _memory_prediction, _with_diagnostics, trace_episode_states
from episode_control import _metrics, build_reference_splits, dataset_audit, labels
from graft_behavior import sha256_file
from key_stability import StableKeyProjector
from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.episode_registry import EPISODE_SPEC
from measurement.key_registry import KEY_SPEC, spec_sha256 as key_spec_sha256
from trinity import VectorMemory


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = EPISODE2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = key_spec_sha256(KEY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != KEY_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered KEY-1 source is not the K1 result")
    checkpoints = {}
    prototype_checkpoints = {}
    for row in results["seeds"]:
        seed = row["seed"]
        for name, receipt in (
            ("projector", row["checkpoint"]),
            ("prototype", row["source_checkpoint"]),
        ):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"KEY-1 {name} checkpoint changed for seed {seed}")
        checkpoints[str(seed)] = dict(row["checkpoint"])
        prototype_checkpoints[str(seed)] = dict(row["source_checkpoint"])
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": checkpoints,
        "prototype_checkpoints": prototype_checkpoints,
    }


def _load_frozen_projector(seed: int, receipt: dict,
                           spec: dict = EPISODE2_SPEC) -> StableKeyProjector:
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != spec["source_experiment"]
        or checkpoint.get("spec_sha256") != key_spec_sha256(KEY_SPEC)
        or checkpoint.get("seed") != seed
        or checkpoint.get("model_class") != spec["model_class"]
    ):
        raise RuntimeError(f"KEY-1 projector identity changed for seed {seed}")
    projector = StableKeyProjector(
        KEY_SPEC["input_dim"], KEY_SPEC["address_dim"], KEY_SPEC["keys"],
        KEY_SPEC["temperature"], KEY_SPEC["bias"],
    )
    projector.load_state_dict(checkpoint["projector"])
    projector.eval()
    projector.requires_grad_(False)
    return projector


class _CountingTransform:
    def __init__(self, projector: StableKeyProjector):
        self.projector = projector
        self.calls = 0
        self.outputs: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        with torch.no_grad():
            output = self.projector.address(value)
        self.outputs.append(output.detach().clone())
        return output


def _integrated_memory_prediction(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    query: torch.Tensor,
    prototypes: torch.Tensor,
    projector: StableKeyProjector | None,
    *,
    swap: bool = False,
) -> tuple[int, int, bool, float, int, int]:
    counter = _CountingTransform(projector) if projector is not None else None
    memory = VectorMemory(
        capacity=len(keys), dim=prototypes.shape[1],
        key_transform=counter,
    )
    stored_values = list(reversed(values)) if swap else values
    for key, value in zip(keys, stored_values):
        memory.store(key, value)
    retrieved = memory.retrieve(query, top_k=1)[0]
    if counter is None:
        query_address = query.detach().float().mean(0) if query.dim() > 1 else query.detach().float()
        calls = 0
    else:
        query_address = counter.outputs[-1]
        calls = counter.calls
    similarities = torch.stack([
        F.cosine_similarity(query_address, key, dim=0) for key in memory.keys
    ])
    selected = int(similarities.argmax())
    api_match = bool(torch.equal(retrieved, stored_values[selected].mean(0)))
    prediction = _decode(retrieved, prototypes)
    margin = float(similarities.max() - similarities.min())
    return prediction, selected, api_match, margin, calls, memory.keys[0].numel()


def _call_audit(values: list[int]) -> dict:
    return {
        "episodes": len(values),
        "total": sum(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def run_seed(seed: int, episodes, source_row: dict, source_receipt: dict,
             spec: dict = EPISODE2_SPEC) -> dict:
    projector_receipt = source_receipt["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, spec)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source_receipt["prototype_checkpoints"][str(seed)]
    prototype_checkpoint = torch.load(
        prototype_receipt["path"], map_location="cpu", weights_only=True
    )
    prototypes = prototype_checkpoint["prototypes"]
    expected = labels(episodes)

    integrated, swapped, recovered, manual, disabled = [], [], [], [], []
    integrated_select, manual_select, disabled_select, positions = [], [], [], []
    integrated_content, manual_content, disabled_content = [], [], []
    integrated_api, manual_api, disabled_api = [], [], []
    integrated_margins, manual_margins, disabled_margins = [], [], []
    normal_calls, swap_calls, recovery_calls, address_widths = [], [], [], []
    episode_seeds = []
    base = EPISODE_SPEC["episode_seed_base"] + seed * EPISODE_SPEC["seed_stride"]

    def transform(value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return projector.address(value)

    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode_states(episode, trial_seed, EPISODE_SPEC)
        q_keys, q_values, q_query = (
            trace["quantum_keys"], trace["quantum_values"], trace["quantum_query"]
        )
        normal = _integrated_memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], projector
        )
        partner_swap = _integrated_memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], projector, swap=True
        )
        recovery = _integrated_memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], projector
        )
        manual_result = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], key_transform=transform
        )
        disabled_result = _integrated_memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], None
        )

        integrated.append(normal[0])
        swapped.append(partner_swap[0])
        recovered.append(recovery[0])
        manual.append(manual_result[0])
        disabled.append(disabled_result[0])
        integrated_select.append(normal[1])
        manual_select.append(manual_result[1])
        disabled_select.append(disabled_result[1])
        positions.append(episode.query_position)
        content = _decode(q_values[episode.query_position], prototypes["quantum"])
        integrated_content.append(content)
        manual_content.append(content)
        disabled_content.append(content)
        integrated_api.append(normal[2])
        manual_api.append(manual_result[2])
        disabled_api.append(disabled_result[2])
        integrated_margins.append(normal[3])
        manual_margins.append(manual_result[3])
        disabled_margins.append(disabled_result[3])
        normal_calls.append(normal[4])
        swap_calls.append(partner_swap[4])
        recovery_calls.append(recovery[4])
        address_widths.extend([normal[5], partner_swap[5], recovery[5]])
        if (index + 1) % 256 == 0:
            print(f"[seed {seed}] evaluated {index + 1}/{len(episodes)} episodes", flush=True)

    integrated_tensor = torch.tensor(integrated)
    recovered_tensor = torch.tensor(recovered)
    manual_tensor = torch.tensor(manual)
    disabled_tensor = torch.tensor(disabled)
    integrated_metrics = _with_diagnostics(
        _metrics(expected, integrated_tensor, spec["values"]), integrated_select, positions,
        integrated_content, expected, integrated_api, integrated_margins,
    )
    manual_metrics = _with_diagnostics(
        _metrics(expected, manual_tensor, spec["values"]), manual_select, positions,
        manual_content, expected, manual_api, manual_margins,
    )
    disabled_metrics = _with_diagnostics(
        _metrics(expected, disabled_tensor, spec["values"]), disabled_select, positions,
        disabled_content, expected, disabled_api, disabled_margins,
    )
    arms = {
        "integrated_stable_normal": integrated_metrics,
        "integrated_stable_partner_swap": _metrics(
            expected, torch.tensor(swapped), spec["values"]
        ),
        "integrated_stable_recovered": {
            **_metrics(expected, recovered_tensor, spec["values"]),
            "prediction_match": float(torch.equal(integrated_tensor, recovered_tensor)),
        },
        "manual_stable_reference": manual_metrics,
        "transform_disabled": disabled_metrics,
        "sensory_memory": source_row["arms"]["sensory_memory"],
        "keyed_attention": source_row["arms"]["keyed_attention"],
        "no_memory": source_row["arms"]["no_memory"],
    }
    after = projector.state_dict()
    return {
        "seed": seed,
        "arms": arms,
        "integration_audit": {
            "normal_transform_calls": _call_audit(normal_calls),
            "partner_swap_transform_calls": _call_audit(swap_calls),
            "recovery_transform_calls": _call_audit(recovery_calls),
            "address_width_minimum": min(address_widths),
            "address_width_maximum": max(address_widths),
            "manual_prediction_match": float(torch.equal(integrated_tensor, manual_tensor)),
            "manual_selection_match": float(integrated_select == manual_select),
            "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
            "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
        },
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/episode2_results.json")
    parser.add_argument("--verdict", default="measurement/episode2_verdict.json")
    args = parser.parse_args()
    spec = EPISODE2_SPEC
    source_results, source_receipt = _source_receipt(spec)
    source_rows = {row["seed"]: row for row in source_results["seeds"]}
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    episodes = splits[spec["eval_split"]]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(splits, ATTENTION_CONTROL_SPEC),
        "source_key": source_receipt,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [
            run_seed(seed, episodes, source_rows[seed], source_receipt, spec)
            for seed in spec["seeds"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.episode2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
