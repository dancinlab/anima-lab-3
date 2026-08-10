#!/usr/bin/env python3
"""CONTEXT-1: test a canonical context+key composite memory address."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from episode import _decode
from graft_behavior import sha256_file
from key_stability import StableKeyProjector, fit_stable_key_projector, key_classification_metrics
from measurement.canonical2_registry import CANONICAL2_SPEC
from measurement.context_registry import CONTEXT_SPEC, spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.separation2_gate import adjudicate as adjudicate_separation2
from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256 as separation2_spec_sha256
from separation import (
    _arm_metrics,
    _direct_prediction,
    _exact_addresses,
    build_episodes,
    dataset_audit,
    trace_similar_episode,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = CONTEXT_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = separation2_spec_sha256(SEPARATION2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != SEPARATION2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_separation2(results) != verdict
    ):
        raise RuntimeError("registered SEPARATION-2 source changed")
    canonical = results["source_canonical2"]
    for receipt in (canonical["checkpoint"], *canonical["prototype_checkpoints"].values()):
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered canonical source checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "canonical_checkpoint": dict(canonical["checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in canonical["prototype_checkpoints"].items()
        },
    }


def _load_key_projector(receipt: dict, spec: dict = CONTEXT_SPEC) -> StableKeyProjector:
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != CANONICAL2_SPEC["experiment"]
        or checkpoint.get("fit_method") != spec["fit_method"]
        or checkpoint.get("model_class") != spec["model_class"]
    ):
        raise RuntimeError("canonical key projector identity changed")
    model = StableKeyProjector(
        spec["state_dim"], spec["component_address_dim"], spec["keys"],
        spec["temperature"], spec["bias"],
    )
    model.load_state_dict(checkpoint["projector"])
    model.eval()
    model.requires_grad_(False)
    return model


def _calibration_spec(spec: dict = CONTEXT_SPEC) -> dict:
    result = deepcopy(spec)
    result["eval_episodes"] = spec["calibration_episodes"]
    result["data_seed"] = spec["calibration_data_seed"]
    result["exact_marginal_balance"] = spec["calibration_exact_marginal_balance"]
    return result


def collect_context_states(episodes, spec: dict = CONTEXT_SPEC):
    rows, labels, seed_rows, cell_counts = [], [], {}, []
    for engine_seed in spec["calibration_engine_seeds"]:
        used = []
        base = spec["calibration_episode_seed_base"] + engine_seed * spec["seed_stride"]
        for index, episode in enumerate(episodes):
            trial_seed = base + index
            trace = trace_similar_episode(episode, trial_seed, distinct=False, spec=spec)
            rows.extend(state.mean(0) for state in trace["contexts"])
            rows.append(trace["query_context"].mean(0))
            labels.extend((*episode.contexts, episode.query_context))
            cell_counts.extend(trace["cell_counts"])
            used.append(trial_seed)
        seed_rows[str(engine_seed)] = {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(used)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, used)).encode()
            ).hexdigest(),
        }
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(rows), tensor_labels, {
        "engine_seeds": seed_rows,
        "states": len(rows),
        "context_counts": {
            str(index): int((tensor_labels == index).sum())
            for index in range(spec["contexts"])
        },
        "minimum_cells": min(cell_counts),
        "maximum_cells": max(cell_counts),
    }


def _component_address(projector: StableKeyProjector, state: torch.Tensor) -> torch.Tensor:
    return projector.address(state.mean(0).unsqueeze(0))[0].detach()


def _composite(context_projector, key_projector, context_state, key_state, spec,
               *, mask_context: bool = False, mask_key: bool = False) -> torch.Tensor:
    context = _component_address(context_projector, context_state) * spec["component_weight"]
    key = _component_address(key_projector, key_state) * spec["component_weight"]
    if mask_context:
        context = torch.zeros_like(context)
    if mask_key:
        key = torch.zeros_like(key)
    address = torch.cat((context, key))
    if address.numel() != spec["composite_address_dim"] or not torch.isfinite(address).all():
        raise RuntimeError("composite memory address changed shape or became non-finite")
    return address


def _composite_prediction(trace, prototypes, context_projector, key_projector, spec,
                          *, mask_context: bool = False, mask_key: bool = False):
    addresses = [
        _composite(context_projector, key_projector, context, key, spec,
                   mask_context=mask_context, mask_key=mask_key)
        for context, key in zip(trace["contexts"], trace["keys"])
    ]
    query = _composite(
        context_projector, key_projector, trace["query_context"], trace["query"], spec,
        mask_context=mask_context, mask_key=mask_key,
    )
    return _direct_prediction(addresses, trace["values"], query, prototypes)


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   context_projector: StableKeyProjector, spec: dict = CONTEXT_SPEC) -> dict:
    key_projector = _load_key_projector(source["canonical_checkpoint"], spec)
    before_context = {name: value.detach().clone() for name, value in context_projector.state_dict().items()}
    before_key = {name: value.detach().clone() for name, value in key_projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    context_states, context_labels = [], []
    episode_seeds, cell_counts = [], []
    before_digests, after_digests, query_rng_digests = [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        similar = trace_similar_episode(episode, trial_seed, distinct=False, spec=spec)
        distinct = trace_similar_episode(episode, trial_seed, distinct=True, spec=spec)
        cell_counts.extend(similar["cell_counts"])
        cell_counts.extend(distinct["cell_counts"])
        for trace in (similar, distinct):
            before_digests.append(trace["update_audit"]["state_before_sha256"])
            after_digests.append(trace["update_audit"]["state_after_sha256"])
            query_rng_digests.append(trace["update_audit"]["query_rng_sha256"])
        context_states.extend(state.mean(0) for state in similar["contexts"])
        context_states.append(similar["query_context"].mean(0))
        context_labels.extend((*episode.contexts, episode.query_context))
        exact, exact_query = _exact_addresses(episode, spec=spec)
        outcomes = {
            "composite_context_key_normal": _composite_prediction(
                similar, prototypes, context_projector, key_projector, spec
            ),
            "context_masked_control": _composite_prediction(
                similar, prototypes, context_projector, key_projector, spec, mask_context=True
            ),
            "key_masked_control": _composite_prediction(
                similar, prototypes, context_projector, key_projector, spec, mask_key=True
            ),
            "composite_distinct_key_control": _composite_prediction(
                distinct, prototypes, context_projector, key_projector, spec
            ),
            "exact_context_key_control": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "exact_context_key_partner_swap": _direct_prediction(
                exact, similar["values"], exact_query, prototypes, rotate=True
            ),
            "composite_context_key_recovered": _composite_prediction(
                similar, prototypes, context_projector, key_projector, spec
            ),
        }
        content = _decode(similar["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        if (index + 1) % 256 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed}] "
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
    arms["composite_context_key_recovered"]["prediction_match"] = float(
        records["composite_context_key_recovered"]["predictions"]
        == records["composite_context_key_normal"]["predictions"]
    )
    context_metrics = key_classification_metrics(
        context_projector, torch.stack(context_states),
        torch.tensor(context_labels, dtype=torch.long), spec["contexts"],
    )
    after_context = context_projector.state_dict()
    after_key = key_projector.state_dict()
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "context_classification": context_metrics,
        "integration_audit": {
            "component_weight": spec["component_weight"],
            "component_address_dim": spec["component_address_dim"],
            "composite_address_dim": spec["composite_address_dim"],
            "context_projector_frozen": not any(
                parameter.requires_grad for parameter in context_projector.parameters()
            ),
            "context_projector_unchanged": all(
                torch.equal(before_context[name], after_context[name]) for name in before_context
            ),
            "key_projector_frozen": not any(
                parameter.requires_grad for parameter in key_projector.parameters()
            ),
            "key_projector_unchanged": all(
                torch.equal(before_key[name], after_key[name]) for name in before_key
            ),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
        },
        "update_audit": {
            "requested_updates": spec["settling_updates"],
            "performed_updates_minimum": spec["settling_updates"],
            "performed_updates_maximum": spec["settling_updates"],
            "disabled": list(spec["pre_query_dynamics_ablation"]),
            "state_before_sha256": hashlib.sha256("\n".join(before_digests).encode()).hexdigest(),
            "state_after_sha256": hashlib.sha256("\n".join(after_digests).encode()).hexdigest(),
            "query_rng_sha256": hashlib.sha256("\n".join(query_rng_digests).encode()).hexdigest(),
        },
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/context_results.json")
    parser.add_argument("--verdict", default="measurement/context_verdict.json")
    parser.add_argument("--checkpoint", default="checkpoints/context1/context_projector.pt")
    args = parser.parse_args()
    spec = CONTEXT_SPEC
    _, source = _source_receipt(spec)
    calibration_spec = _calibration_spec(spec)
    calibration_episodes = build_episodes(calibration_spec)
    evaluation_episodes = build_episodes(spec)
    calibration_fingerprints = {row.fingerprint() for row in calibration_episodes}
    evaluation_fingerprints = {row.fingerprint() for row in evaluation_episodes}
    calibration_states, calibration_labels, calibration_state_audit = collect_context_states(
        calibration_episodes, spec
    )
    fit_spec = {
        "input_dim": spec["state_dim"],
        "address_dim": spec["component_address_dim"],
        "keys": spec["contexts"],
        "temperature": spec["temperature"],
        "bias": spec["bias"],
        "weight_decay": spec["weight_decay"],
    }
    context_projector, fit_audit = fit_stable_key_projector(
        calibration_states, calibration_labels, fit_spec, method=spec["fit_method"]
    )
    context_projector.requires_grad_(False)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": spec["experiment"],
        "spec_sha256": spec_sha256(spec),
        "fit_method": spec["fit_method"],
        "model_class": spec["model_class"],
        "projector": context_projector.state_dict(),
        "fit_audit": fit_audit,
        "calibration_state_audit": calibration_state_audit,
    }, checkpoint_path)
    evaluations = [
        {
            "name": evaluation_name(row),
            **run_evaluation(
                row["prototype_seed"], row["engine_seed"], evaluation_episodes,
                source, context_projector, spec,
            ),
        }
        for row in spec["evaluation_combinations"]
    ]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_separation2": source,
        "calibration_dataset_audit": dataset_audit(calibration_episodes, calibration_spec),
        "evaluation_dataset_audit": dataset_audit(evaluation_episodes, spec),
        "calibration_evaluation_overlap": len(calibration_fingerprints & evaluation_fingerprints),
        "calibration_state_audit": calibration_state_audit,
        "fit_audit": fit_audit,
        "context_checkpoint": _receipt(checkpoint_path),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.context_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
