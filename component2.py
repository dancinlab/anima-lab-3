#!/usr/bin/env python3
"""COMPONENT-2: fit stable context/key components and rerun conjunction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from component import run_engine
from conjunction import ConjunctionEpisode, _atomic_json, build_episodes, dataset_audit, trace_episode
from conjunction2 import _source_receipt as conjunction2_source_receipt, run_evaluation
from key_stability import fit_stable_key_projector
from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256
from measurement.component_gate import adjudicate as adjudicate_component
from measurement.component_registry import COMPONENT_SPEC, spec_sha256 as component_spec_sha256
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.projector_registry import evaluation_name
from value2 import _atomic_torch, _canonical_rows


def _source_receipt(spec=COMPONENT2_SPEC):
    results_path = Path(spec["source_results"]); verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text()); verdict = json.loads(verdict_path.read_text())
    sha = component_spec_sha256(COMPONENT_SPEC)
    if (results.get("experiment") != spec["source_experiment"] or results.get("spec") != COMPONENT_SPEC
        or results.get("spec_sha256") != sha or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != sha or adjudicate_component(results) != verdict):
        raise RuntimeError("registered COMPONENT-1 source changed")
    conjunction_source = conjunction2_source_receipt(CONJUNCTION2_SPEC)
    return {"results": {"path": str(results_path), "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest()},
            "verdict": {"path": str(verdict_path), "sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest()},
            "source_spec_sha256": sha, "conjunction_source": conjunction_source}


def balance_components(episodes, spec=COMPONENT2_SPEC):
    rows = []
    for index, episode in enumerate(episodes):
        target_context = index % spec["contexts"]
        target_key = (index // spec["contexts"]) % spec["keys"]
        assigned_contexts = [(target_context + offset) % spec["contexts"] for offset in range(4)]
        assigned_keys = [(target_key + offset) % spec["keys"] for offset in range(4)]
        old_contexts = [episode.query_context, *sorted(value for value in episode.active_contexts if value != episode.query_context)]
        old_keys = [episode.query_key, *sorted(value for value in episode.active_keys if value != episode.query_key)]
        context_map = dict(zip(old_contexts, assigned_contexts)); key_map = dict(zip(old_keys, assigned_keys))
        rows.append(ConjunctionEpisode(
            contexts=tuple(context_map[value] for value in episode.contexts),
            keys=tuple(key_map[value] for value in episode.keys), values=episode.values,
            active_contexts=tuple(sorted(assigned_contexts)), active_keys=tuple(sorted(assigned_keys)),
            active_values=episode.active_values, distractors=episode.distractors,
            query_position=episode.query_position,
        ))
    return rows


def collect_states(episodes, spec=COMPONENT2_SPEC):
    context_states, context_labels, key_states, key_labels, seeds = [], [], [], [], []
    for engine_seed in spec["calibration_engine_seeds"]:
        base = spec["calibration_seed_base"] + engine_seed * spec["seed_stride"]
        for index, episode in enumerate(episodes):
            trial_seed = base + index; seeds.append(trial_seed)
            trace = trace_episode(episode, trial_seed, {**CONJUNCTION2_SPEC, **spec})
            context_states.extend(row.mean(0) for row in trace["contexts"]); context_labels.extend(episode.contexts)
            key_states.extend(row.mean(0) for row in trace["keys"]); key_labels.extend(episode.keys)
    cs, cl = _canonical_rows(torch.stack(context_states), torch.tensor(context_labels, dtype=torch.long))
    ks, kl = _canonical_rows(torch.stack(key_states), torch.tensor(key_labels, dtype=torch.long))
    return cs, cl, ks, kl, {
        "states_per_component": len(cs), "unique_engine_seeds": len(set(seeds)),
        "context_counts": {str(i): int((cl == i).sum()) for i in range(spec["contexts"])},
        "key_counts": {str(i): int((kl == i).sum()) for i in range(spec["keys"])},
        "context_sha256": hashlib.sha256(cs.numpy().tobytes()).hexdigest(),
        "key_sha256": hashlib.sha256(ks.numpy().tobytes()).hexdigest(),
    }


def fit_components(cs, cl, ks, kl, path, spec=COMPONENT2_SPEC):
    context, ca = fit_stable_key_projector(cs, cl, spec, method=spec["fit_method"])
    key, ka = fit_stable_key_projector(ks, kl, spec, method=spec["fit_method"])
    context2, _ = fit_stable_key_projector(cs, cl, spec, method=spec["fit_method"])
    key2, _ = fit_stable_key_projector(ks, kl, spec, method=spec["fit_method"])
    rcs, rcl = _canonical_rows(cs.flip(0), cl.flip(0)); rks, rkl = _canonical_rows(ks.flip(0), kl.flip(0))
    context3, _ = fit_stable_key_projector(rcs, rcl, spec, method=spec["fit_method"])
    key3, _ = fit_stable_key_projector(rks, rkl, spec, method=spec["fit_method"])
    deterministic = (
        all(torch.equal(context.state_dict()[n], context2.state_dict()[n]) and torch.equal(context.state_dict()[n], context3.state_dict()[n]) for n in context.state_dict())
        and all(torch.equal(key.state_dict()[n], key2.state_dict()[n]) and torch.equal(key.state_dict()[n], key3.state_dict()[n]) for n in key.state_dict())
    )
    _atomic_torch(path, {"experiment": spec["experiment"], "spec_sha256": spec_sha256(spec),
        "context_projector": context.state_dict(), "key_projector": key.state_dict(),
        "context_fit": ca, "key_fit": ka, "deterministic": deterministic})
    return context, key, ca, ka, deterministic, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="measurement/component2_results.json"); parser.add_argument("--verdict", default="measurement/component2_verdict.json"); args = parser.parse_args()
    spec = COMPONENT2_SPEC; source = _source_receipt(spec)
    cal_spec = {**CONJUNCTION2_SPEC, "eval_episodes": spec["calibration_episodes"], "data_seed": spec["calibration_data_seed"]}
    calibration = balance_components(build_episodes(cal_spec), spec)
    cs, cl, ks, kl, state_audit = collect_states(calibration, spec)
    context, key, ca, ka, deterministic, checkpoint = fit_components(cs, cl, ks, kl, Path(spec["checkpoint_path"]), spec)
    diagnostic_episodes = build_episodes(COMPONENT_SPEC)
    diagnostics = [run_engine(seed, diagnostic_episodes, source["conjunction_source"], COMPONENT_SPEC,
        context_projector_override=context, key_projector_override=key) for seed in spec["engine_seeds"]]
    eval_episodes = build_episodes(CONJUNCTION2_SPEC)
    evaluations = [{"name": evaluation_name(row), **run_evaluation(row["prototype_seed"], row["engine_seed"], eval_episodes,
        source["conjunction_source"], CONJUNCTION2_SPEC, context_projector_override=context, key_projector_override=key)} for row in spec["evaluation_combinations"]]
    payload = {"experiment": spec["experiment"], "spec": spec, "spec_sha256": spec_sha256(spec),
        "source_component1": source, "calibration_dataset_audit": dataset_audit(calibration, cal_spec),
        "calibration_state_audit": state_audit, "context_fit": ca, "key_fit": ka,
        "deterministic": deterministic, "checkpoint": checkpoint,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "diagnostics": diagnostics, "evaluations": evaluations}
    _atomic_json(Path(args.output), payload)
    from measurement.component2_gate import adjudicate
    verdict = adjudicate(payload); _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__": main()
