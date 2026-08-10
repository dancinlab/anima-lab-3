#!/usr/bin/env python3
"""RESET-1: separate sensory reset from autonomous state settling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from dataclasses import replace
from pathlib import Path

import torch

from decay import _exact_addresses
from episode import _decode, _new_engine
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from graft_behavior import sha256_file
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.episode_registry import EPISODE_SPEC
from measurement.recovery_registry import RECOVERY_SPEC, spec_sha256 as recovery_spec_sha256
from measurement.reset_registry import RESET_SPEC, spec_sha256
from recovery import (
    _extend_records,
    _geometry,
    _metrics,
    _records,
    _similarities,
    build_recovery_episodes,
    recovery_dataset_audit,
)
from separation import _direct_prediction, _sense_separation_token


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_reset_episodes(replicate: int, spec: dict = RESET_SPEC):
    """Reuse RECOVERY-1 relations but preregister eight distinct neutral inputs."""
    rows = build_recovery_episodes(replicate, RECOVERY_SPEC)
    rng = random.Random(spec["distractor_seed_base"] + replicate * spec["replicate_seed_stride"])
    result = []
    for row in rows:
        distractors = list(range(spec["contexts"]))
        rng.shuffle(distractors)
        result.append(replace(row, distractors=tuple(distractors)))
    return result


def reset_dataset_audit(episode_sets: dict[int, list], spec: dict = RESET_SPEC) -> dict:
    value = recovery_dataset_audit(episode_sets, spec)
    distinct = [
        len(set(row.distractors[:max(spec["update_steps"])]))
        for rows in episode_sets.values()
        for row in rows
    ]
    value["varied_input_distinct_count_minimum"] = min(distinct)
    value["varied_input_distinct_count_maximum"] = max(distinct)
    value["repeated_neutral_word"] = spec["repeated_neutral_word"]
    return value


def _source_receipt(spec: dict = RESET_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = recovery_spec_sha256(RECOVERY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != RECOVERY_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered RECOVERY-1 source is not the ordered recovery result")
    checkpoints, prototypes = {}, {}
    for row in results["seeds"]:
        seed = row["seed"]
        for receipt in (row["source_checkpoint"], row["prototype_checkpoint"]):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"RECOVERY-1 source checkpoint changed for seed {seed}")
        checkpoints[str(seed)] = dict(row["source_checkpoint"])
        prototypes[str(seed)] = dict(row["prototype_checkpoint"])
    return results, {
        "results": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "verdict": {"path": str(verdict_path), "sha256": sha256_file(verdict_path)},
        "source_verdict": verdict["verdict"],
        "source_spec_sha256": expected_sha,
        "checkpoints": checkpoints,
        "prototype_checkpoints": prototypes,
    }


def _phase_state(c, spec: dict) -> torch.Tensor:
    state = c.get_phase_states().clone()
    if (
        state.dim() != 2
        or state.shape[1] != spec["state_dim"]
        or not spec["minimum_cells"] <= state.shape[0] <= spec["maximum_cells"]
        or not torch.isfinite(state).all()
    ):
        raise RuntimeError("registered RESET-1 state range changed")
    return state


def trace_reset_episode(episode, trial_seed: int, updates: int, mode: str,
                        spec: dict = RESET_SPEC) -> dict:
    if updates not in spec["update_steps"]:
        raise ValueError(f"unregistered update count {updates}")
    if mode not in spec["update_modes"]:
        raise ValueError(f"unregistered update mode {mode}")
    c, encoder = _new_engine(trial_seed, EPISODE_SPEC)
    keys, values, cell_counts = [], [], []
    for context, key, value in zip(
        episode.contexts[:spec["prepared_events"]],
        episode.distinct_keys[:spec["prepared_events"]],
        episode.values[:spec["prepared_events"]],
    ):
        for word, steps in (
            (EPISODE_SPEC["distractor_words"][context], EPISODE_SPEC["sense_steps"]),
            (EPISODE_SPEC["key_words"][key], EPISODE_SPEC["sense_steps"]),
            (EPISODE_SPEC["value_words"][value], EPISODE_SPEC["sense_steps"]),
        ):
            state = _sense_separation_token(c, encoder, word, steps, spec)
            cell_counts.append(state.shape[0])
            if word == EPISODE_SPEC["key_words"][key]:
                keys.append(state)
            elif word == EPISODE_SPEC["value_words"][value]:
                values.append(state)

    applied_inputs = []
    for index in range(updates):
        if mode == "autonomous":
            c.step()
            state = _phase_state(c, spec)
            applied_inputs.append(None)
        else:
            context = (
                episode.distractors[index]
                if mode == "varied_sensory"
                else spec["repeated_neutral_context"]
            )
            word = EPISODE_SPEC["distractor_words"][context]
            state = _sense_separation_token(c, encoder, word, 1, spec)
            applied_inputs.append(word)
        cell_counts.append(state.shape[0])

    context_state = _sense_separation_token(
        c, encoder, EPISODE_SPEC["distractor_words"][episode.query_context],
        EPISODE_SPEC["sense_steps"], spec,
    )
    query = _sense_separation_token(
        c, encoder, EPISODE_SPEC["key_words"][episode.distinct_keys[episode.query_position]],
        EPISODE_SPEC["sense_steps"], spec,
    )
    cell_counts.extend((context_state.shape[0], query.shape[0]))
    return {
        "keys": keys, "values": values, "query": query,
        "cell_counts": cell_counts, "applied_inputs": applied_inputs,
        "performed_updates": updates,
    }


def _run_replicate(seed: int, replicate: int, mode: str, updates: int, episodes,
                   projector, prototypes, spec: dict = RESET_SPEC):
    records = _records(spec)
    geometry_rows = []
    call_rows = {name: [] for name in spec["stable_arms"]}
    widths, cell_counts, episode_seeds = [], [], []
    input_counts, distinct_input_counts, performed_updates = [], [], []
    base = spec["episode_seed_base"] + seed * spec["seed_stride"] + replicate * spec["replicate_seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_reset_episode(episode, trial_seed, updates, mode, spec)
        cell_counts.extend(trace["cell_counts"])
        input_values = [value for value in trace["applied_inputs"] if value is not None]
        input_counts.append(len(input_values))
        distinct_input_counts.append(len(set(input_values)))
        performed_updates.append(trace["performed_updates"])
        exact, exact_query = _exact_addresses(episode, spec)
        stable_three = _integrated_memory_prediction(
            trace["keys"], trace["values"], trace["query"], prototypes, projector
        )
        outcomes = {
            "stable_three_candidates": stable_three,
            "stable_two_candidates": _integrated_memory_prediction(
                trace["keys"][:2], trace["values"][:2], trace["query"], prototypes, projector
            ),
            "exact_three_candidates": _direct_prediction(
                exact, trace["values"], exact_query, prototypes
            ),
            "exact_three_partner_swap": _direct_prediction(
                exact, trace["values"], exact_query, prototypes, rotate=True
            ),
            "exact_three_recovered": _direct_prediction(
                exact, trace["values"], exact_query, prototypes
            ),
        }
        content = _decode(trace["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["positions"].append(episode.query_position)
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
            if name in call_rows:
                call_rows[name].append(outcome[4])
                widths.append(outcome[5])
        similarities = _similarities(trace["keys"], trace["query"], projector)
        target = episode.query_position
        wrong = [value for position, value in enumerate(similarities) if position != target]
        geometry_rows.append({
            "similarities": similarities,
            "target_position": target,
            "selected_position": stable_three[1],
            "target_similarity": similarities[target],
            "strongest_wrong_similarity": max(wrong),
            "third_candidate_similarity": similarities[2],
            "target_minus_strongest_wrong": similarities[target] - max(wrong),
            "target_minus_third_candidate": similarities[target] - similarities[2],
        })
        if (index + 1) % 256 == 0:
            print(
                f"[seed {seed} replicate {replicate} {mode} updates {updates}] "
                f"evaluated {index + 1}/{len(episodes)} episodes", flush=True,
            )
    expected = torch.tensor([episode.target for episode in episodes])
    metrics = _metrics(records, expected, spec)
    metrics["exact_three_recovered"]["prediction_match"] = float(
        records["exact_three_recovered"]["predictions"]
        == records["exact_three_candidates"]["predictions"]
    )
    return {
        "replicate": replicate,
        "arms": metrics,
        "geometry": _geometry(geometry_rows),
        "integration_audit": {
            "stable_transform_calls": {
                name: {"episodes": len(values), "total": sum(values), "minimum": min(values), "maximum": max(values)}
                for name, values in call_rows.items()
            },
            "address_width_minimum": min(widths),
            "address_width_maximum": max(widths),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256("\n".join(map(str, episode_seeds)).encode()).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
        },
        "update_audit": {
            "requested_updates": updates,
            "performed_updates_minimum": min(performed_updates),
            "performed_updates_maximum": max(performed_updates),
            "sensory_inputs_minimum": min(input_counts),
            "sensory_inputs_maximum": max(input_counts),
            "distinct_sensory_inputs_minimum": min(distinct_input_counts),
            "distinct_sensory_inputs_maximum": max(distinct_input_counts),
        },
    }, records, geometry_rows


def run_seed(seed: int, episode_sets: dict[int, list], source: dict,
             spec: dict = RESET_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    mode_rows = []
    for mode in spec["update_modes"]:
        update_rows = []
        for updates in spec["update_steps"]:
            pooled_records = _records(spec)
            pooled_geometry, replicate_rows = [], []
            for replicate in spec["replicates"]:
                public, records, geometry = _run_replicate(
                    seed, replicate, mode, updates, episode_sets[replicate],
                    projector, prototypes, spec,
                )
                replicate_rows.append(public)
                _extend_records(pooled_records, records)
                pooled_geometry.extend(geometry)
            pooled_expected = torch.tensor([
                episode.target for replicate in spec["replicates"]
                for episode in episode_sets[replicate]
            ])
            pooled_metrics = _metrics(pooled_records, pooled_expected, spec)
            pooled_metrics["exact_three_recovered"]["prediction_match"] = float(
                pooled_records["exact_three_recovered"]["predictions"]
                == pooled_records["exact_three_candidates"]["predictions"]
            )
            update_rows.append({
                "update_steps": updates,
                "pooled": {"arms": pooled_metrics, "geometry": _geometry(pooled_geometry)},
                "replicates": replicate_rows,
            })
        mode_rows.append({"mode": mode, "updates": update_rows})
    after = projector.state_dict()
    return {
        "seed": seed, "modes": mode_rows,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/reset_results.json")
    parser.add_argument("--verdict", default="measurement/reset_verdict.json")
    args = parser.parse_args()
    spec = RESET_SPEC
    _, source = _source_receipt(spec)
    episode_sets = {replicate: build_reset_episodes(replicate, spec) for replicate in spec["replicates"]}
    payload = {
        "experiment": spec["experiment"], "spec": spec, "spec_sha256": spec_sha256(spec),
        "dataset_audit": reset_dataset_audit(episode_sets, spec),
        "source_recovery": source,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": spec["device"]},
        "seeds": [run_seed(seed, episode_sets, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.reset_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
