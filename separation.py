#!/usr/bin/env python3
"""SEPARATION-1: locate collisions between similar one-shot episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from episode import _decode, _new_engine, _with_diagnostics
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from episode_control import _metrics
from graft_behavior import sha256_file
from measurement.episode2_registry import EPISODE2_SPEC, spec_sha256 as episode2_spec_sha256
from measurement.episode_registry import EPISODE_SPEC
from measurement.separation_registry import SEPARATION_SPEC, spec_sha256
from trinity import VectorMemory


@dataclass(frozen=True)
class SimilarEpisode:
    contexts: tuple[int, ...]
    values: tuple[int, ...]
    shared_key: int
    distinct_keys: tuple[int, ...]
    distractors: tuple[int, ...]
    query_position: int

    @property
    def target(self) -> int:
        return self.values[self.query_position]

    @property
    def query_context(self) -> int:
        return self.contexts[self.query_position]

    def fingerprint(self) -> str:
        payload = {
            "contexts": self.contexts,
            "values": self.values,
            "shared_key": self.shared_key,
            "distinct_keys": self.distinct_keys,
            "distractors": self.distractors,
            "query_position": self.query_position,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _ordered_distinct(first: int, categories: int, count: int,
                      rng: random.Random) -> tuple[int, ...]:
    others = [value for value in range(categories) if value != first]
    rng.shuffle(others)
    return (first, *others[:count - 1])


def build_episodes(spec: dict = SEPARATION_SPEC) -> list[SimilarEpisode]:
    cycle = spec["values"] * spec["events_per_episode"] * spec["keys"] * spec["contexts"]
    exact_marginals = spec.get("exact_marginal_balance", False)
    marginal_categories = (
        spec["values"], spec["events_per_episode"], spec["keys"], spec["contexts"]
    )
    if exact_marginals and any(
        spec["eval_episodes"] % categories for categories in marginal_categories
    ):
        raise ValueError("registered episode count must preserve every marginal balance")
    if not exact_marginals and spec["eval_episodes"] % cycle:
        raise ValueError("registered episode count must preserve exact balance")
    rng = random.Random(spec["data_seed"])
    episodes: list[SimilarEpisode] = []
    seen: set[str] = set()
    for index in range(spec["eval_episodes"]):
        target = index % spec["values"]
        query_position = (index // spec["values"]) % spec["events_per_episode"]
        shared_key = (
            index // (spec["values"] * spec["events_per_episode"])
        ) % spec["keys"]
        if exact_marginals:
            query_context = (index // spec["events_per_episode"]) % spec["contexts"]
        else:
            query_context = (
                index // (spec["values"] * spec["events_per_episode"] * spec["keys"])
            ) % spec["contexts"]
        while True:
            values = list(_ordered_distinct(
                target, spec["values"], spec["events_per_episode"], rng
            ))
            contexts = list(_ordered_distinct(
                query_context, spec["contexts"], spec["events_per_episode"], rng
            ))
            distinct_keys = list(_ordered_distinct(
                shared_key, spec["keys"], spec["events_per_episode"], rng
            ))
            for rows in (values, contexts, distinct_keys):
                rows[0], rows[query_position] = rows[query_position], rows[0]
            episode = SimilarEpisode(
                contexts=tuple(contexts),
                values=tuple(values),
                shared_key=shared_key,
                distinct_keys=tuple(distinct_keys),
                distractors=tuple(
                    rng.randrange(spec["contexts"])
                    for _ in range(spec["distractor_steps"])
                ),
                query_position=query_position,
            )
            fingerprint = episode.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                episodes.append(episode)
                break
    rng.shuffle(episodes)
    return episodes


def dataset_audit(episodes: list[SimilarEpisode], spec: dict = SEPARATION_SPEC) -> dict:
    def counts(values, categories: int) -> dict[str, int]:
        counter = Counter(values)
        return {str(index): counter[index] for index in range(categories)}

    fingerprints = [episode.fingerprint() for episode in episodes]
    return {
        "episodes": len(episodes),
        "unique_fingerprints": len(set(fingerprints)),
        "target_counts": counts((row.target for row in episodes), spec["values"]),
        "query_position_counts": counts(
            (row.query_position for row in episodes), spec["events_per_episode"]
        ),
        "shared_key_counts": counts((row.shared_key for row in episodes), spec["keys"]),
        "query_context_counts": counts(
            (row.query_context for row in episodes), spec["contexts"]
        ),
        "fingerprint_set_sha256": hashlib.sha256(
            "\n".join(sorted(fingerprints)).encode()
        ).hexdigest(),
    }


def _source_receipt(spec: dict = SEPARATION_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = episode2_spec_sha256(EPISODE2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != EPISODE2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered EPISODE-2 source is not the recovered path")
    checkpoints, prototypes = {}, {}
    for row in results["seeds"]:
        seed = row["seed"]
        for receipt in (row["source_checkpoint"], row["prototype_checkpoint"]):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"EPISODE-2 source checkpoint changed for seed {seed}")
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


def _sense_separation_token(c, encoder, word: str, steps: int,
                            spec: dict = SEPARATION_SPEC) -> torch.Tensor:
    payload = encoder.encode_sense([word])
    for _ in range(steps):
        c.step(x_input=payload)
    state = c.get_phase_states().clone()
    if (
        state.dim() != 2
        or state.shape[1] != spec["state_dim"]
        or not spec["minimum_cells"] <= state.shape[0] <= spec["maximum_cells"]
        or not torch.isfinite(state).all()
    ):
        raise RuntimeError("registered SEPARATION-1 state range changed")
    return state


def trace_similar_episode(episode: SimilarEpisode, trial_seed: int, *, distinct: bool,
                          spec: dict = SEPARATION_SPEC) -> dict:
    c, encoder = _new_engine(trial_seed, EPISODE_SPEC)
    keys, values, cell_counts = [], [], []
    episode_keys = episode.distinct_keys if distinct else (episode.shared_key,) * len(episode.values)
    for context, key, value in zip(episode.contexts, episode_keys, episode.values):
        state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][context],
            EPISODE_SPEC["sense_steps"], spec,
        )
        cell_counts.append(state.shape[0])
        key_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["key_words"][key],
            EPISODE_SPEC["sense_steps"], spec,
        )
        value_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["value_words"][value],
            EPISODE_SPEC["sense_steps"], spec,
        )
        cell_counts.extend((key_state.shape[0], value_state.shape[0]))
        keys.append(key_state)
        values.append(value_state)
    for distractor in episode.distractors:
        state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][distractor],
            EPISODE_SPEC["distractor_sense_steps"], spec,
        )
        cell_counts.append(state.shape[0])
    pre_query_updates = spec.get("pre_query_updates", 0)
    if not isinstance(pre_query_updates, int) or pre_query_updates < 0:
        raise ValueError("pre-query update count must be a non-negative integer")
    query_rng = torch.get_rng_state().clone()
    state_before = c.get_phase_states().clone()
    before_digest = hashlib.sha256(state_before.contiguous().numpy().tobytes()).hexdigest()
    for _ in range(pre_query_updates):
        c.step(dynamics_ablation=spec.get("pre_query_dynamics_ablation", ()))
        cell_counts.append(c.n_cells)
    state_after = c.get_phase_states().clone()
    after_digest = hashlib.sha256(state_after.contiguous().numpy().tobytes()).hexdigest()
    torch.set_rng_state(query_rng)
    state = _sense_separation_token(
        c, encoder, EPISODE_SPEC["distractor_words"][episode.query_context],
        EPISODE_SPEC["sense_steps"], spec,
    )
    cell_counts.append(state.shape[0])
    query_key = episode.distinct_keys[episode.query_position] if distinct else episode.shared_key
    query = _sense_separation_token(
        c, encoder, EPISODE_SPEC["key_words"][query_key],
        EPISODE_SPEC["sense_steps"], spec,
    )
    cell_counts.append(query.shape[0])
    return {
        "keys": keys,
        "values": values,
        "query": query,
        "cell_counts": cell_counts,
        "update_audit": {
            "requested_updates": pre_query_updates,
            "performed_updates": pre_query_updates,
            "disabled": list(spec.get("pre_query_dynamics_ablation", ())),
            "state_before_sha256": before_digest,
            "state_after_sha256": after_digest,
            "query_rng_sha256": hashlib.sha256(query_rng.numpy().tobytes()).hexdigest(),
        },
    }


def _exact_addresses(episode: SimilarEpisode, *, remove_context: bool = False,
                     spec: dict = SEPARATION_SPEC):
    rows = []
    for context in episode.contexts:
        context_code = torch.zeros(spec["contexts"]) if remove_context else F.one_hot(
            torch.tensor(context), spec["contexts"]
        ).float()
        key_code = F.one_hot(torch.tensor(episode.shared_key), spec["keys"]).float()
        rows.append(torch.cat((context_code, key_code)))
    query_context = torch.zeros(spec["contexts"]) if remove_context else F.one_hot(
        torch.tensor(episode.query_context), spec["contexts"]
    ).float()
    query_key = F.one_hot(torch.tensor(episode.shared_key), spec["keys"]).float()
    return rows, torch.cat((query_context, query_key))


def _direct_prediction(addresses, values, query, prototypes, *, rotate: bool = False):
    memory = VectorMemory(capacity=len(addresses), dim=addresses[0].numel())
    stored_values = values[1:] + values[:1] if rotate else values
    for address, value in zip(addresses, stored_values):
        memory.store(address, value)
    similarities = torch.stack([
        F.cosine_similarity(query, address, dim=0) for address in memory.keys
    ])
    selected = int(similarities.argmax())
    retrieved = memory.retrieve(query, top_k=1)[0]
    return (
        _decode(retrieved, prototypes), selected,
        bool(torch.equal(retrieved, stored_values[selected].mean(0))),
        float(similarities.max() - similarities.min()),
    )


def _arm_metrics(expected, predictions, selections, positions, contents, api, margins,
                 spec: dict = SEPARATION_SPEC) -> dict:
    return _with_diagnostics(
        _metrics(expected, torch.tensor(predictions), spec["values"]),
        selections, positions, contents, expected, api, margins,
    )


def run_seed(seed: int, episodes: list[SimilarEpisode], source: dict,
             spec: dict = SEPARATION_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    names = (
        "stable_similar_normal", "raw_similar_normal", "stable_distinct_key_control",
        "exact_context_key_control", "exact_context_key_partner_swap",
        "exact_context_key_recovered", "context_removed_control",
    )
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in names
    }
    calls, address_widths, episode_seeds, cell_counts = [], [], [], []
    base = spec["episode_seed_base"] + seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        similar = trace_similar_episode(episode, trial_seed, distinct=False, spec=spec)
        distinct = trace_similar_episode(episode, trial_seed, distinct=True, spec=spec)
        cell_counts.extend(similar["cell_counts"])
        cell_counts.extend(distinct["cell_counts"])
        exact, exact_query = _exact_addresses(episode)
        removed, removed_query = _exact_addresses(episode, remove_context=True)
        outcomes = {
            "stable_similar_normal": _integrated_memory_prediction(
                similar["keys"], similar["values"], similar["query"], prototypes, projector
            ),
            "raw_similar_normal": _integrated_memory_prediction(
                similar["keys"], similar["values"], similar["query"], prototypes, None
            ),
            "stable_distinct_key_control": _integrated_memory_prediction(
                distinct["keys"], distinct["values"], distinct["query"], prototypes, projector
            ),
            "exact_context_key_control": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "exact_context_key_partner_swap": _direct_prediction(
                exact, similar["values"], exact_query, prototypes, rotate=True
            ),
            "exact_context_key_recovered": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "context_removed_control": _direct_prediction(
                removed, similar["values"], removed_query, prototypes
            ),
        }
        content = _decode(similar["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            record = records[name]
            record["predictions"].append(outcome[0])
            record["selections"].append(outcome[1])
            record["contents"].append(content)
            record["api"].append(outcome[2])
            record["margins"].append(outcome[3])
        stable = outcomes["stable_similar_normal"]
        calls.append(stable[4])
        address_widths.append(stable[5])
        if (index + 1) % 256 == 0:
            print(f"[seed {seed}] evaluated {index + 1}/{len(episodes)} episodes", flush=True)
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    recovered_predictions = records["exact_context_key_recovered"]["predictions"]
    arms["exact_context_key_recovered"]["prediction_match"] = float(
        recovered_predictions == records["exact_context_key_control"]["predictions"]
    )
    after = projector.state_dict()
    return {
        "seed": seed,
        "arms": arms,
        "integration_audit": {
            "stable_transform_calls": {
                "episodes": len(calls), "total": sum(calls),
                "minimum": min(calls), "maximum": max(calls),
            },
            "address_width_minimum": min(address_widths),
            "address_width_maximum": max(address_widths),
            "projector_frozen": not any(
                parameter.requires_grad for parameter in projector.parameters()
            ),
            "projector_unchanged": all(
                torch.equal(before[name], after[name]) for name in before
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
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/separation_results.json")
    parser.add_argument("--verdict", default="measurement/separation_verdict.json")
    args = parser.parse_args()
    spec = SEPARATION_SPEC
    _, source = _source_receipt(spec)
    episodes = build_episodes(spec)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": dataset_audit(episodes, spec),
        "source_episode2": source,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episodes, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.separation_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
