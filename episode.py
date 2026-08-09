#!/usr/bin/env python3
"""EPISODE-1: test one-shot relation recall through the existing memory path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from episode_control import (
    KeyedRelationAttention,
    _evaluate_attention,
    _metrics,
    build_reference_splits,
    dataset_audit,
    labels,
    relation_tensors,
)
from graft_behavior import _memory_state, sha256_file
from measurement.episode_control_registry import ATTENTION_CONTROL_SPEC, spec_sha256 as source_spec_sha256
from measurement.episode_registry import EPISODE_SPEC, spec_sha256
from pure import PureMind
from trinity import QuantumC, VectorMemory


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _source_receipt(spec: dict = EPISODE_SPEC) -> dict:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != ATTENTION_CONTROL_SPEC
        or results.get("spec_sha256") != source_spec_sha256(ATTENTION_CONTROL_SPEC)
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != source_spec_sha256(ATTENTION_CONTROL_SPEC)
    ):
        raise RuntimeError("registered CONTROL-3 source is not the validated positive control")
    checkpoints = {}
    for row in results["seeds"]:
        seed = row["seed"]
        receipt = row["checkpoint"]
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"CONTROL-3 checkpoint changed for seed {seed}")
        checkpoints[str(seed)] = dict(receipt)
    return {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": results["spec_sha256"],
        "checkpoints": checkpoints,
    }


def _sense_token(c: QuantumC, encoder: PureMind, word: str, steps: int,
                 spec: dict = EPISODE_SPEC) -> tuple[torch.Tensor, torch.Tensor]:
    payload = encoder.encode_sense([word])
    for _ in range(steps):
        c.step(x_input=payload)
    quantum = c.get_phase_states().clone()
    sensory = _memory_state(payload, spec["cells"], "phase")
    expected = (spec["cells"], spec["state_dim"])
    if quantum.shape != expected or sensory.shape != expected:
        raise RuntimeError("registered EPISODE-1 state shape changed")
    return quantum, sensory


def _new_engine(seed: int, spec: dict = EPISODE_SPEC) -> tuple[QuantumC, PureMind]:
    torch.manual_seed(seed)
    c = QuantumC(nc=spec["cells"], dim=spec["engine_dim"], max_cells=spec["cells"])
    encoder = PureMind(store=None, c_engine=c)
    for _ in range(spec["warm_steps"]):
        c.step()
    return c, encoder


def build_value_prototypes(seed: int, spec: dict = EPISODE_SPEC) -> tuple[dict, dict]:
    quantum_rows = [[] for _ in range(spec["values"])]
    sensory_rows = [[] for _ in range(spec["values"])]
    used_seeds = []
    base = spec["prototype_seed_base"] + seed * spec["seed_stride"]
    for value, word in enumerate(spec["value_words"]):
        for repeat in range(spec["prototype_repeats_per_value"]):
            trial_seed = base + value * spec["prototype_repeats_per_value"] + repeat
            c, encoder = _new_engine(trial_seed, spec)
            quantum, sensory = _sense_token(c, encoder, word, spec["sense_steps"], spec)
            quantum_rows[value].append(quantum.mean(0))
            sensory_rows[value].append(sensory.mean(0))
            used_seeds.append(trial_seed)
    prototypes = {
        "quantum": torch.stack([torch.stack(rows).mean(0) for rows in quantum_rows]),
        "sensory": torch.stack([torch.stack(rows).mean(0) for rows in sensory_rows]),
    }
    audit = {
        "states": len(used_seeds),
        "unique_seeds": len(set(used_seeds)),
        "seed_sha256": hashlib.sha256(
            "\n".join(map(str, used_seeds)).encode()
        ).hexdigest(),
    }
    return prototypes, audit


def _decode(state: torch.Tensor, prototypes: torch.Tensor) -> int:
    vector = state.mean(0) if state.dim() > 1 else state
    return int(F.cosine_similarity(vector.unsqueeze(0), prototypes, dim=1).argmax())


def _memory_prediction(keys: list[torch.Tensor], values: list[torch.Tensor],
                       query: torch.Tensor, prototypes: torch.Tensor,
                       swap: bool = False) -> tuple[int, int, bool, float]:
    memory = VectorMemory(capacity=len(keys), dim=prototypes.shape[-1])
    stored_values = list(reversed(values)) if swap else values
    for key, value in zip(keys, stored_values):
        memory.store(key, value)
    similarities = torch.stack([
        F.cosine_similarity(query.mean(0), key.mean(0), dim=0) for key in keys
    ])
    selected = int(similarities.argmax())
    retrieved = memory.retrieve(query, top_k=1)[0]
    api_match = bool(torch.equal(retrieved, stored_values[selected].mean(0)))
    prediction = _decode(retrieved, prototypes)
    margin = float(similarities.max() - similarities.min())
    return prediction, selected, api_match, margin


def _with_diagnostics(metrics: dict, selections: list[int], positions: list[int],
                      content_predictions: list[int], expected: torch.Tensor,
                      api_matches: list[bool], margins: list[float]) -> dict:
    value = dict(metrics)
    value.update({
        "selection_accuracy": sum(a == b for a, b in zip(selections, positions)) / len(positions),
        "correct_content_accuracy": float((torch.tensor(content_predictions) == expected).float().mean()),
        "retrieval_api_match": sum(api_matches) / len(api_matches),
        "key_margin_mean": sum(margins) / len(margins),
        "key_margin_min": min(margins),
    })
    return value


@torch.no_grad()
def _attention_metrics(seed: int, episodes, receipt: dict, spec: dict) -> dict:
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    model = KeyedRelationAttention(
        ATTENTION_CONTROL_SPEC["keys"], ATTENTION_CONTROL_SPEC["values"],
        ATTENTION_CONTROL_SPEC["state_dim"], ATTENTION_CONTROL_SPEC["attention_heads"],
        ATTENTION_CONTROL_SPEC["attention_dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    tensors = relation_tensors(episodes)
    return _evaluate_attention(model, tensors, labels(episodes), spec["values"])


def run_seed(seed: int, episodes, source: dict, output_dir: Path,
             spec: dict = EPISODE_SPEC) -> dict:
    prototypes, prototype_audit = build_value_prototypes(seed, spec)
    expected = labels(episodes)
    quantum_normal, quantum_swap, quantum_recovered = [], [], []
    sensory_normal, sensory_swap = [], []
    q_select, s_select, positions = [], [], []
    q_content, s_content, q_api, s_api, q_margins, s_margins = [], [], [], [], [], []
    episode_seeds = []
    base = spec["episode_seed_base"] + seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        c, encoder = _new_engine(trial_seed, spec)
        q_keys, q_values, s_keys, s_values = [], [], [], []
        for key, value in episode.stores:
            q_key, s_key = _sense_token(
                c, encoder, spec["key_words"][key], spec["sense_steps"], spec
            )
            q_value, s_value = _sense_token(
                c, encoder, spec["value_words"][value], spec["sense_steps"], spec
            )
            q_keys.append(q_key)
            q_values.append(q_value)
            s_keys.append(s_key)
            s_values.append(s_value)
        for distractor in episode.distractors:
            _sense_token(
                c, encoder,
                spec["distractor_words"][distractor % len(spec["distractor_words"])],
                spec["distractor_sense_steps"], spec,
            )
        q_query, s_query = _sense_token(
            c, encoder, spec["key_words"][episode.query_key], spec["sense_steps"], spec
        )
        qn, qs, qa, qm = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"]
        )
        qw, _, _, _ = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"], swap=True
        )
        qr, _, _, _ = _memory_prediction(
            q_keys, q_values, q_query, prototypes["quantum"]
        )
        sn, ss, sa, sm = _memory_prediction(
            s_keys, s_values, s_query, prototypes["sensory"]
        )
        sw, _, _, _ = _memory_prediction(
            s_keys, s_values, s_query, prototypes["sensory"], swap=True
        )
        quantum_normal.append(qn)
        quantum_swap.append(qw)
        quantum_recovered.append(qr)
        sensory_normal.append(sn)
        sensory_swap.append(sw)
        q_select.append(qs)
        s_select.append(ss)
        positions.append(episode.query_position)
        q_content.append(_decode(q_values[episode.query_position], prototypes["quantum"]))
        s_content.append(_decode(s_values[episode.query_position], prototypes["sensory"]))
        q_api.append(qa)
        s_api.append(sa)
        q_margins.append(qm)
        s_margins.append(sm)
        if (index + 1) % 256 == 0:
            print(f"[seed {seed}] evaluated {index + 1}/{len(episodes)} episodes", flush=True)
    q_normal_tensor = torch.tensor(quantum_normal)
    s_normal_tensor = torch.tensor(sensory_normal)
    q_recovered_tensor = torch.tensor(quantum_recovered)
    arms = {
        "quantum_memory_normal": _with_diagnostics(
            _metrics(expected, q_normal_tensor, spec["values"]), q_select, positions,
            q_content, expected, q_api, q_margins,
        ),
        "quantum_memory_partner_swap": _metrics(
            expected, torch.tensor(quantum_swap), spec["values"]
        ),
        "quantum_memory_recovered": {
            **_metrics(expected, q_recovered_tensor, spec["values"]),
            "prediction_match": float(torch.equal(q_normal_tensor, q_recovered_tensor)),
        },
        "sensory_memory_normal": _with_diagnostics(
            _metrics(expected, s_normal_tensor, spec["values"]), s_select, positions,
            s_content, expected, s_api, s_margins,
        ),
        "sensory_memory_partner_swap": _metrics(
            expected, torch.tensor(sensory_swap), spec["values"]
        ),
        "keyed_attention": _attention_metrics(
            seed, episodes, source["checkpoints"][str(seed)], spec
        ),
        "no_memory": _metrics(expected, torch.zeros_like(expected), spec["values"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"seed_{seed}_prototypes.pt"
    torch.save({
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "seed": seed,
        "prototype_audit": prototype_audit,
        "prototypes": prototypes,
    }, checkpoint_path)
    return {
        "seed": seed,
        "arms": arms,
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "prototype": prototype_audit,
            "prototype_episode_seed_overlap": len(set(episode_seeds) & {
                spec["prototype_seed_base"] + seed * spec["seed_stride"] + index
                for index in range(spec["values"] * spec["prototype_repeats_per_value"])
            }),
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "source_checkpoint": source["checkpoints"][str(seed)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/episode_results.json")
    parser.add_argument("--verdict", default="measurement/episode_verdict.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/episode1")
    args = parser.parse_args()
    spec = EPISODE_SPEC
    source = _source_receipt(spec)
    splits = build_reference_splits(ATTENTION_CONTROL_SPEC)
    episodes = splits[spec["eval_split"]]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(splits, ATTENTION_CONTROL_SPEC),
        "source_control": source,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [
            run_seed(seed, episodes, source, Path(args.checkpoint_dir), spec)
            for seed in spec["seeds"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.episode_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
