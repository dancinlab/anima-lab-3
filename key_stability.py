#!/usr/bin/env python3
"""KEY-1: learn and test a minimal time-stable address for existing memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode import _decode, _memory_prediction, _with_diagnostics, trace_episode_states
from episode_control import _metrics, build_reference_splits, dataset_audit, labels
from graft_behavior import sha256_file
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC
from measurement.episode_registry import EPISODE_SPEC, spec_sha256 as episode_spec_sha256
from measurement.key_registry import KEY_SPEC, spec_sha256


class StableKeyProjector(nn.Module):
    """Linear metric-learning head; only the projection is used by memory."""

    def __init__(self, input_dim: int, address_dim: int, keys: int, temperature: float,
                 bias: bool = True):
        super().__init__()
        self.projection = nn.Linear(input_dim, address_dim, bias=bias)
        self.prototypes = nn.Parameter(torch.empty(keys, address_dim))
        self.temperature = temperature
        nn.init.normal_(self.prototypes, std=address_dim ** -0.5)

    def address(self, states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(states), dim=-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        addresses = self.address(states)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return addresses @ prototypes.T / self.temperature


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = KEY_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = episode_spec_sha256(EPISODE_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != EPISODE_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered EPISODE-1 source is not the E2 result")
    checkpoints = {}
    for row in results["seeds"]:
        receipt = row["checkpoint"]
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"EPISODE-1 checkpoint changed for seed {row['seed']}")
        checkpoints[str(row["seed"])] = dict(receipt)
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": checkpoints,
    }


def _state_labels(episode) -> list[int]:
    return [episode.stores[0][0], episode.stores[1][0], episode.query_key]


def collect_calibration_states(episodes, seed: int, spec: dict = KEY_SPEC):
    rows, key_labels, used_seeds = [], [], []
    base = spec["calibration_seed_base"] + seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        trace = trace_episode_states(episode, trial_seed, EPISODE_SPEC)
        rows.extend([state.mean(0) for state in trace["quantum_keys"]])
        rows.append(trace["quantum_query"].mean(0))
        key_labels.extend(_state_labels(episode))
        used_seeds.append(trial_seed)
    tensor_labels = torch.tensor(key_labels, dtype=torch.long)
    return torch.stack(rows), tensor_labels, {
        "episodes": len(episodes),
        "states": len(rows),
        "unique_engine_seeds": len(set(used_seeds)),
        "engine_seed_sha256": hashlib.sha256(
            "\n".join(map(str, used_seeds)).encode()
        ).hexdigest(),
        "key_counts": {
            str(key): int((tensor_labels == key).sum()) for key in range(spec["keys"])
        },
    }


def train_projector(states: torch.Tensor, key_labels: torch.Tensor, seed: int,
                    shuffled: bool = False, spec: dict = KEY_SPEC):
    torch.manual_seed(seed)
    model = StableKeyProjector(
        spec["input_dim"], spec["address_dim"], spec["keys"], spec["temperature"],
        spec["bias"],
    )
    training_labels = key_labels.clone()
    if shuffled:
        generator = torch.Generator().manual_seed(spec["shuffle_seed_base"] + seed)
        training_labels = training_labels[torch.randperm(len(training_labels), generator=generator)]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    generator = torch.Generator().manual_seed(spec["calibration_seed_base"] + seed)
    losses = []
    for _ in range(spec["train_steps"]):
        indices = torch.randint(
            len(states), (spec["batch_size"],), generator=generator
        )
        loss = F.cross_entropy(model(states[indices]), training_labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, {
        "examples": len(states),
        "steps": spec["train_steps"],
        "shuffled": shuffled,
        "final_loss_mean_50": sum(losses[-50:]) / 50,
        "training_label_sha256": hashlib.sha256(training_labels.numpy().tobytes()).hexdigest(),
    }


@torch.no_grad()
def key_classification_metrics(model: StableKeyProjector, states: torch.Tensor,
                               key_labels: torch.Tensor, keys: int) -> dict:
    predictions = model(states).argmax(1)
    return {
        "accuracy": float((predictions == key_labels).float().mean()),
        "per_key_recall": [
            float((predictions[key_labels == key] == key).float().mean())
            for key in range(keys)
        ],
        "confusion_matrix": torch.bincount(
            key_labels * keys + predictions, minlength=keys * keys
        ).reshape(keys, keys).tolist(),
    }


@torch.no_grad()
def run_seed(seed: int, calibration_episodes, eval_episodes, source_row: dict,
             source_receipt: dict, output_dir: Path, spec: dict = KEY_SPEC) -> dict:
    calibration_states, calibration_labels, calibration_audit = collect_calibration_states(
        calibration_episodes, seed, spec
    )
    projector, training_audit = train_projector(calibration_states, calibration_labels, seed, False, spec)
    fake_projector, fake_training_audit = train_projector(
        calibration_states, calibration_labels, seed, True, spec
    )
    prototype_receipt = source_receipt["checkpoints"][str(seed)]
    prototype_checkpoint = torch.load(
        prototype_receipt["path"], map_location="cpu", weights_only=True
    )
    prototypes = prototype_checkpoint["prototypes"]
    expected = labels(eval_episodes)
    stable, stable_swap, stable_recovered, raw, sensory = [], [], [], [], []
    stable_select, raw_select, sensory_select, positions = [], [], [], []
    stable_content, raw_content, sensory_content = [], [], []
    stable_api, raw_api, sensory_api = [], [], []
    stable_margins, raw_margins, sensory_margins = [], [], []
    eval_states, eval_labels, used_eval_seeds = [], [], []
    base = spec["eval_seed_base"] + seed * spec["seed_stride"]
    transform = lambda value: projector.address(value.unsqueeze(0))[0]
    for index, episode in enumerate(eval_episodes):
        trial_seed = base + index
        trace = trace_episode_states(episode, trial_seed, EPISODE_SPEC)
        q_keys, q_values, q_query = (
            trace["quantum_keys"], trace["quantum_values"], trace["quantum_query"]
        )
        s_keys, s_values, s_query = (
            trace["sensory_keys"], trace["sensory_values"], trace["sensory_query"]
        )
        normal = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], key_transform=transform
        )
        swapped = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], swap=True,
            key_transform=transform,
        )
        recovered = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], key_transform=transform
        )
        raw_result = _memory_prediction(q_keys, q_values, q_query, prototypes["quantum"])
        sensory_result = _memory_prediction(s_keys, s_values, s_query, prototypes["sensory"])
        stable.append(normal[0])
        stable_swap.append(swapped[0])
        stable_recovered.append(recovered[0])
        raw.append(raw_result[0])
        sensory.append(sensory_result[0])
        stable_select.append(normal[1])
        raw_select.append(raw_result[1])
        sensory_select.append(sensory_result[1])
        positions.append(episode.query_position)
        stable_content.append(_decode(
            q_values[episode.query_position], prototypes["quantum"]
        ))
        raw_content.append(stable_content[-1])
        sensory_content.append(_decode(
            s_values[episode.query_position], prototypes["sensory"]
        ))
        stable_api.append(normal[2])
        raw_api.append(raw_result[2])
        sensory_api.append(sensory_result[2])
        stable_margins.append(normal[3])
        raw_margins.append(raw_result[3])
        sensory_margins.append(sensory_result[3])
        eval_states.extend([state.mean(0) for state in q_keys])
        eval_states.append(q_query.mean(0))
        eval_labels.extend(_state_labels(episode))
        used_eval_seeds.append(trial_seed)
        if (index + 1) % 256 == 0:
            print(f"[seed {seed}] evaluated {index + 1}/{len(eval_episodes)} episodes", flush=True)
    eval_states_tensor = torch.stack(eval_states)
    eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.long)
    stable_tensor = torch.tensor(stable)
    recovered_tensor = torch.tensor(stable_recovered)
    arms = {
        "stabilized_memory_normal": _with_diagnostics(
            _metrics(expected, stable_tensor, spec["values"]), stable_select, positions,
            stable_content, expected, stable_api, stable_margins,
        ),
        "stabilized_memory_partner_swap": _metrics(
            expected, torch.tensor(stable_swap), spec["values"]
        ),
        "stabilized_memory_recovered": {
            **_metrics(expected, recovered_tensor, spec["values"]),
            "prediction_match": float(torch.equal(stable_tensor, recovered_tensor)),
        },
        "raw_quantum_memory": _with_diagnostics(
            _metrics(expected, torch.tensor(raw), spec["values"]), raw_select, positions,
            raw_content, expected, raw_api, raw_margins,
        ),
        "sensory_memory": _with_diagnostics(
            _metrics(expected, torch.tensor(sensory), spec["values"]), sensory_select, positions,
            sensory_content, expected, sensory_api, sensory_margins,
        ),
        "keyed_attention": source_row["arms"]["keyed_attention"],
        "no_memory": source_row["arms"]["no_memory"],
        "shuffled_label_projector": key_classification_metrics(
            fake_projector, eval_states_tensor, eval_labels_tensor, spec["keys"]
        ),
    }
    key_metrics = key_classification_metrics(
        projector, eval_states_tensor, eval_labels_tensor, spec["keys"]
    )
    eval_audit = {
        "episodes": len(eval_episodes),
        "states": len(eval_states),
        "unique_engine_seeds": len(set(used_eval_seeds)),
        "engine_seed_sha256": hashlib.sha256(
            "\n".join(map(str, used_eval_seeds)).encode()
        ).hexdigest(),
        "key_counts": {
            str(key): int((eval_labels_tensor == key).sum()) for key in range(spec["keys"])
        },
        "calibration_engine_seed_overlap": len(
            set(range(
                spec["calibration_seed_base"] + seed * spec["seed_stride"],
                spec["calibration_seed_base"] + seed * spec["seed_stride"] + len(calibration_episodes),
            )) & set(used_eval_seeds)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"seed_{seed}_key_projectors.pt"
    torch.save({
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "seed": seed,
        "model_class": spec["model_class"],
        "projector": projector.state_dict(),
        "shuffled_label_projector": fake_projector.state_dict(),
        "training_audit": training_audit,
        "shuffled_training_audit": fake_training_audit,
    }, checkpoint_path)
    return {
        "seed": seed,
        "key_classification": key_metrics,
        "arms": arms,
        "calibration_audit": calibration_audit,
        "eval_state_audit": eval_audit,
        "training_audit": training_audit,
        "shuffled_training_audit": fake_training_audit,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "source_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/key_results.json")
    parser.add_argument("--verdict", default="measurement/key_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/key1")
    args = parser.parse_args()
    source_results, source_receipt = _source_receipt(KEY_SPEC)
    source_rows = {row["seed"]: row for row in source_results["seeds"]}
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    calibration = splits[KEY_SPEC["calibration_split"]]
    evaluation = splits[KEY_SPEC["eval_split"]]
    payload = {
        "experiment": KEY_SPEC["experiment"],
        "spec": KEY_SPEC,
        "spec_sha256": spec_sha256(KEY_SPEC),
        "dataset_audit": dataset_audit(splits, ATTENTION_CONTROL_SPEC),
        "source_episode": source_receipt,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": KEY_SPEC["device"],
        },
        "seeds": [
            run_seed(
                seed, calibration, evaluation, source_rows[seed], source_receipt,
                Path(args.checkpoint_dir), KEY_SPEC,
            )
            for seed in KEY_SPEC["seeds"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.key_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
