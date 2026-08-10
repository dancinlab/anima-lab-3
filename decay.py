#!/usr/bin/env python3
"""DECAY-1: separate stored-candidate competition from state-stream delay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from episode import _decode, _new_engine
from episode2 import _integrated_memory_prediction, _load_frozen_projector
from measurement.capacity_registry import CAPACITY_SPEC, spec_sha256 as capacity_spec_sha256
from measurement.decay_registry import DECAY_SPEC, spec_sha256
from measurement.episode2_registry import EPISODE2_SPEC
from measurement.episode_registry import EPISODE_SPEC
from separation import (
    SimilarEpisode,
    _arm_metrics,
    _direct_prediction,
    _ordered_distinct,
    _sense_separation_token,
)
from graft_behavior import sha256_file


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_decay_episodes(spec: dict = DECAY_SPEC) -> list[SimilarEpisode]:
    total = spec["eval_episodes_per_delay"]
    for categories in (spec["values"], spec["queryable_events"], spec["keys"], spec["contexts"]):
        if total % categories:
            raise ValueError("registered episode count must preserve every marginal balance")
    rng = random.Random(spec["data_seed"])
    episodes, seen = [], set()
    for index in range(total):
        target = index % spec["values"]
        query_position = (index // spec["values"]) % spec["queryable_events"]
        query_key = (index // (spec["values"] * spec["queryable_events"])) % spec["keys"]
        query_context = (index // spec["queryable_events"]) % spec["contexts"]
        while True:
            values = list(_ordered_distinct(target, spec["values"], spec["prepared_events"], rng))
            keys = list(_ordered_distinct(query_key, spec["keys"], spec["prepared_events"], rng))
            contexts = list(_ordered_distinct(
                query_context, spec["contexts"], spec["prepared_events"], rng
            ))
            for rows in (values, keys, contexts):
                rows[0], rows[query_position] = rows[query_position], rows[0]
            episode = SimilarEpisode(
                contexts=tuple(contexts),
                values=tuple(values),
                shared_key=query_key,
                distinct_keys=tuple(keys),
                distractors=tuple(
                    rng.randrange(spec["contexts"])
                    for _ in range(max(spec["distractor_steps"]))
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


def decay_dataset_audit(episodes: list[SimilarEpisode], spec: dict = DECAY_SPEC) -> dict:
    def counts(values, categories: int) -> dict[str, int]:
        counter = Counter(values)
        return {str(index): counter[index] for index in range(categories)}

    fingerprints = [row.fingerprint() for row in episodes]
    return {
        "episodes": len(episodes),
        "unique_fingerprints": len(set(fingerprints)),
        "target_counts": counts((row.target for row in episodes), spec["values"]),
        "query_position_counts": counts(
            (row.query_position for row in episodes), spec["queryable_events"]
        ),
        "query_key_counts": counts(
            (row.distinct_keys[row.query_position] for row in episodes), spec["keys"]
        ),
        "query_context_counts": counts(
            (row.query_context for row in episodes), spec["contexts"]
        ),
        "fingerprint_set_sha256": hashlib.sha256(
            "\n".join(sorted(fingerprints)).encode()
        ).hexdigest(),
    }


def _source_receipt(spec: dict = DECAY_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = capacity_spec_sha256(CAPACITY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CAPACITY_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
    ):
        raise RuntimeError("registered CAPACITY-1 source is not the event boundary")
    checkpoints, prototypes = {}, {}
    for row in results["seeds"]:
        seed = row["seed"]
        for receipt in (row["source_checkpoint"], row["prototype_checkpoint"]):
            path = Path(receipt["path"])
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"CAPACITY-1 source checkpoint changed for seed {seed}")
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


def _trace(episode: SimilarEpisode, trial_seed: int, event_count: int, delay: int,
           spec: dict = DECAY_SPEC) -> dict:
    c, encoder = _new_engine(trial_seed, EPISODE_SPEC)
    keys, values, cell_counts = [], [], []
    for context, key, value in zip(
        episode.contexts[:event_count],
        episode.distinct_keys[:event_count],
        episode.values[:event_count],
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
    for distractor in episode.distractors[:delay]:
        state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][distractor],
            EPISODE_SPEC["distractor_sense_steps"], spec,
        )
        cell_counts.append(state.shape[0])
    context_state = _sense_separation_token(
        c, encoder, EPISODE_SPEC["distractor_words"][episode.query_context],
        EPISODE_SPEC["sense_steps"], spec,
    )
    query = _sense_separation_token(
        c, encoder,
        EPISODE_SPEC["key_words"][episode.distinct_keys[episode.query_position]],
        EPISODE_SPEC["sense_steps"], spec,
    )
    cell_counts.extend((context_state.shape[0], query.shape[0]))
    return {"keys": keys, "values": values, "query": query, "cell_counts": cell_counts}


def _exact_addresses(episode: SimilarEpisode, spec: dict = DECAY_SPEC):
    addresses = [
        F.one_hot(torch.tensor(key), spec["keys"]).float()
        for key in episode.distinct_keys
    ]
    query = F.one_hot(
        torch.tensor(episode.distinct_keys[episode.query_position]), spec["keys"]
    ).float()
    return addresses, query


def _same_prefix(two: dict, three: dict) -> bool:
    return all(
        torch.equal(left, right)
        for name in ("keys", "values")
        for left, right in zip(two[name], three[name][:2])
    )


def _run_delay(seed: int, delay: int, episodes, projector, prototypes,
               spec: dict = DECAY_SPEC) -> dict:
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    call_rows = {name: [] for name in spec["stable_arms"]}
    widths, episode_seeds, cell_counts, prefix_matches = [], [], [], 0
    base = spec["episode_seed_base"] + delay * spec["delay_seed_stride"] + seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        two = _trace(episode, trial_seed, 2, delay, spec)
        three = _trace(episode, trial_seed, 3, delay, spec)
        prefix_matches += int(_same_prefix(two, three))
        cell_counts.extend(two["cell_counts"])
        cell_counts.extend(three["cell_counts"])
        exact, exact_query = _exact_addresses(episode, spec)
        outcomes = {
            "two_stream_two_candidates": _integrated_memory_prediction(
                two["keys"], two["values"], two["query"], prototypes, projector
            ),
            "three_stream_two_candidates": _integrated_memory_prediction(
                three["keys"][:2], three["values"][:2], three["query"], prototypes, projector
            ),
            "three_stream_three_candidates": _integrated_memory_prediction(
                three["keys"], three["values"], three["query"], prototypes, projector
            ),
            "raw_three_stream_three_candidates": _integrated_memory_prediction(
                three["keys"], three["values"], three["query"], prototypes, None
            ),
            "exact_three_candidates": _direct_prediction(
                exact, three["values"], exact_query, prototypes
            ),
            "exact_three_partner_swap": _direct_prediction(
                exact, three["values"], exact_query, prototypes, rotate=True
            ),
            "exact_three_recovered": _direct_prediction(
                exact, three["values"], exact_query, prototypes
            ),
        }
        content = _decode(three["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
            if name in call_rows:
                call_rows[name].append(outcome[4])
                widths.append(outcome[5])
        if (index + 1) % 256 == 0:
            print(f"[seed {seed} delay {delay}] evaluated {index + 1}/{len(episodes)} episodes", flush=True)
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    arms["exact_three_recovered"]["prediction_match"] = float(
        records["exact_three_recovered"]["predictions"]
        == records["exact_three_candidates"]["predictions"]
    )
    return {
        "distractor_steps": delay,
        "arms": arms,
        "integration_audit": {
            "stable_transform_calls": {
                name: {
                    "episodes": len(values), "total": sum(values),
                    "minimum": min(values), "maximum": max(values),
                }
                for name, values in call_rows.items()
            },
            "address_width_minimum": min(widths),
            "address_width_maximum": max(widths),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "prefix_state_matches": prefix_matches,
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
        },
    }


def run_seed(seed: int, episodes, source: dict, spec: dict = DECAY_SPEC) -> dict:
    projector_receipt = source["checkpoints"][str(seed)]
    projector = _load_frozen_projector(seed, projector_receipt, EPISODE2_SPEC)
    before = {name: value.detach().clone() for name, value in projector.state_dict().items()}
    prototype_receipt = source["prototype_checkpoints"][str(seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    delays = [
        _run_delay(seed, delay, episodes, projector, prototypes, spec)
        for delay in spec["distractor_steps"]
    ]
    after = projector.state_dict()
    return {
        "seed": seed,
        "delays": delays,
        "projector_frozen": not any(parameter.requires_grad for parameter in projector.parameters()),
        "projector_unchanged": all(torch.equal(before[name], after[name]) for name in before),
        "source_checkpoint": projector_receipt,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/decay_results.json")
    parser.add_argument("--verdict", default="measurement/decay_verdict.json")
    args = parser.parse_args()
    spec = DECAY_SPEC
    _, source = _source_receipt(spec)
    episodes = build_decay_episodes(spec)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "dataset_audit": decay_dataset_audit(episodes, spec),
        "source_capacity": source,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "seeds": [run_seed(seed, episodes, source, spec) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.decay_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
