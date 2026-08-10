#!/usr/bin/env python3
"""CONJUNCTION-1: require both context and key to retrieve an episodic value."""
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

from context import _composite, _load_key_projector
from context2 import CompositeStateTransform, _load_context_projector
from episode import _decode, _new_engine
from graft_behavior import sha256_file
from measurement.conjunction_registry import CONJUNCTION_SPEC, spec_sha256
from measurement.context2_gate import adjudicate as adjudicate_context2
from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256 as context2_spec_sha256
from measurement.episode_registry import EPISODE_SPEC
from measurement.projector_registry import evaluation_name
from separation import _arm_metrics, _sense_separation_token
from trinity import VectorMemory


@dataclass(frozen=True)
class ConjunctionEpisode:
    contexts: tuple[int, ...]
    keys: tuple[int, ...]
    values: tuple[int, ...]
    active_contexts: tuple[int, ...]
    active_keys: tuple[int, ...]
    active_values: tuple[int, ...]
    distractors: tuple[int, ...]
    query_position: int

    @property
    def query_context(self) -> int:
        return self.contexts[self.query_position]

    @property
    def query_key(self) -> int:
        return self.keys[self.query_position]

    @property
    def target(self) -> int:
        return self.values[self.query_position]

    def fingerprint(self) -> str:
        payload = {
            "contexts": self.contexts,
            "keys": self.keys,
            "values": self.values,
            "active_contexts": self.active_contexts,
            "active_keys": self.active_keys,
            "active_values": self.active_values,
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


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = CONJUNCTION_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = context2_spec_sha256(CONTEXT2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CONTEXT2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_context2(results) != verdict
    ):
        raise RuntimeError("registered CONTEXT-2 source changed")
    context1 = results["source_context1"]
    receipts = (
        context1["context_checkpoint"], context1["canonical_checkpoint"],
        *context1["prototype_checkpoints"].values(),
    )
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered CONTEXT-2 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "context_checkpoint": dict(context1["context_checkpoint"]),
        "canonical_checkpoint": dict(context1["canonical_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in context1["prototype_checkpoints"].items()
        },
    }


def _active_set(required: int, categories: int, count: int,
                rng: random.Random) -> list[int]:
    others = [value for value in range(categories) if value != required]
    rng.shuffle(others)
    selected = [required, *others[:count - 1]]
    rng.shuffle(selected)
    return selected


def build_episodes(spec: dict = CONJUNCTION_SPEC) -> list[ConjunctionEpisode]:
    query_combinations = spec["contexts"] * spec["keys"] * spec["values"]
    if spec["eval_episodes"] % query_combinations:
        raise ValueError("registered episode count must balance every query triple")
    if spec["events_per_episode"] != (
        spec["active_contexts_per_episode"] * spec["active_keys_per_episode"]
    ):
        raise ValueError("registered event count must cover the active Cartesian product")
    active_count = spec["active_values_per_episode"]
    if not (
        spec["active_contexts_per_episode"]
        == spec["active_keys_per_episode"]
        == active_count
    ):
        raise ValueError("registered Latin square requires equal active category counts")

    rng = random.Random(spec["data_seed"])
    episodes: list[ConjunctionEpisode] = []
    seen: set[str] = set()
    for index in range(spec["eval_episodes"]):
        query_context = index % spec["contexts"]
        query_key = (index // spec["contexts"]) % spec["keys"]
        target = (index // (spec["contexts"] * spec["keys"])) % spec["values"]
        while True:
            active_contexts = _active_set(
                query_context, spec["contexts"], spec["active_contexts_per_episode"], rng
            )
            active_keys = _active_set(
                query_key, spec["keys"], spec["active_keys_per_episode"], rng
            )
            active_values = _active_set(target, spec["values"], active_count, rng)
            context_rank = {value: rank for rank, value in enumerate(active_contexts)}
            key_rank = {value: rank for rank, value in enumerate(active_keys)}
            offset = rng.randrange(active_count)
            target_slot = (
                context_rank[query_context] + key_rank[query_key] + offset
            ) % active_count
            active_values.remove(target)
            rng.shuffle(active_values)
            value_slots: list[int | None] = [None] * active_count
            value_slots[target_slot] = target
            remaining_slots = [slot for slot, value in enumerate(value_slots) if value is None]
            for slot, value in zip(remaining_slots, active_values):
                value_slots[slot] = value
            contexts, keys, values = [], [], []
            for context in active_contexts:
                for key in active_keys:
                    contexts.append(context)
                    keys.append(key)
                    slot = (context_rank[context] + key_rank[key] + offset) % active_count
                    values.append(int(value_slots[slot]))
            query_position = next(
                position for position, pair in enumerate(zip(contexts, keys))
                if pair == (query_context, query_key)
            )
            episode = ConjunctionEpisode(
                contexts=tuple(contexts), keys=tuple(keys), values=tuple(values),
                active_contexts=tuple(active_contexts), active_keys=tuple(active_keys),
                active_values=tuple(sorted(value_slots)),
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


def _latin_valid(episode: ConjunctionEpisode) -> bool:
    expected_values = set(episode.active_values)
    pairs = list(zip(episode.contexts, episode.keys))
    if len(pairs) != len(set(pairs)):
        return False
    for context in episode.active_contexts:
        row = {
            value for seen_context, value in zip(episode.contexts, episode.values)
            if seen_context == context
        }
        if row != expected_values:
            return False
    for key in episode.active_keys:
        column = {
            value for seen_key, value in zip(episode.keys, episode.values)
            if seen_key == key
        }
        if column != expected_values:
            return False
    return True


def dataset_audit(episodes: list[ConjunctionEpisode],
                  spec: dict = CONJUNCTION_SPEC) -> dict:
    def counts(values, categories: int) -> dict[str, int]:
        counter = Counter(values)
        return {str(index): counter[index] for index in range(categories)}

    triples = Counter(
        (row.query_context, row.query_key, row.target) for row in episodes
    )
    fingerprints = [row.fingerprint() for row in episodes]
    return {
        "episodes": len(episodes),
        "unique_fingerprints": len(set(fingerprints)),
        "target_counts": counts((row.target for row in episodes), spec["values"]),
        "query_context_counts": counts(
            (row.query_context for row in episodes), spec["contexts"]
        ),
        "query_key_counts": counts((row.query_key for row in episodes), spec["keys"]),
        "query_triple_counts": {
            f"{context}:{key}:{value}": triples[(context, key, value)]
            for context in range(spec["contexts"])
            for key in range(spec["keys"])
            for value in range(spec["values"])
        },
        "latin_valid_episodes": sum(_latin_valid(row) for row in episodes),
        "minimum_unique_pairs": min(
            len(set(zip(row.contexts, row.keys))) for row in episodes
        ),
        "maximum_unique_pairs": max(
            len(set(zip(row.contexts, row.keys))) for row in episodes
        ),
        "fingerprint_set_sha256": hashlib.sha256(
            "\n".join(sorted(fingerprints)).encode()
        ).hexdigest(),
    }


def trace_episode(episode: ConjunctionEpisode, trial_seed: int,
                  spec: dict = CONJUNCTION_SPEC) -> dict:
    c, encoder = _new_engine(trial_seed, EPISODE_SPEC)
    contexts, keys, values, cell_counts = [], [], [], []
    for context, key, value in zip(episode.contexts, episode.keys, episode.values):
        context_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][context],
            EPISODE_SPEC["sense_steps"], spec,
        )
        key_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["key_words"][key],
            EPISODE_SPEC["sense_steps"], spec,
        )
        value_state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["value_words"][value],
            EPISODE_SPEC["sense_steps"], spec,
        )
        contexts.append(context_state)
        keys.append(key_state)
        values.append(value_state)
        cell_counts.extend((context_state.shape[0], key_state.shape[0], value_state.shape[0]))
    for distractor in episode.distractors:
        state = _sense_separation_token(
            c, encoder, EPISODE_SPEC["distractor_words"][distractor],
            EPISODE_SPEC["distractor_sense_steps"], spec,
        )
        cell_counts.append(state.shape[0])
    query_rng = torch.get_rng_state().clone()
    state_before = c.get_phase_states().clone()
    for _ in range(spec["pre_query_updates"]):
        c.step(dynamics_ablation=spec["pre_query_dynamics_ablation"])
        cell_counts.append(c.n_cells)
    state_after = c.get_phase_states().clone()
    torch.set_rng_state(query_rng)
    query_context = _sense_separation_token(
        c, encoder, EPISODE_SPEC["distractor_words"][episode.query_context],
        EPISODE_SPEC["sense_steps"], spec,
    )
    query_key = _sense_separation_token(
        c, encoder, EPISODE_SPEC["key_words"][episode.query_key],
        EPISODE_SPEC["sense_steps"], spec,
    )
    cell_counts.extend((query_context.shape[0], query_key.shape[0]))
    return {
        "contexts": contexts, "keys": keys, "values": values,
        "query_context": query_context, "query": query_key,
        "cell_counts": cell_counts,
        "update_audit": {
            "requested_updates": spec["pre_query_updates"],
            "performed_updates": spec["pre_query_updates"],
            "disabled": list(spec["pre_query_dynamics_ablation"]),
            "state_before_sha256": hashlib.sha256(
                state_before.contiguous().numpy().tobytes()
            ).hexdigest(),
            "state_after_sha256": hashlib.sha256(
                state_after.contiguous().numpy().tobytes()
            ).hexdigest(),
            "query_rng_sha256": hashlib.sha256(query_rng.numpy().tobytes()).hexdigest(),
        },
    }


def _memory_outcome(memory: VectorMemory, query, stored_values,
                    prototypes: torch.Tensor):
    retrieved = memory.retrieve(query, top_k=1)[0]
    if memory.key_transform is not None and hasattr(memory.key_transform, "outputs"):
        prepared_query = memory.key_transform.outputs[-1]
    else:
        prepared_query = memory._prepare_key(query, for_query=True)
    similarities = torch.stack([
        F.cosine_similarity(prepared_query, address, dim=0) for address in memory.keys
    ])
    # Match VectorMemory.retrieve's canonical top-k rule, including exact ties.
    selected = int(similarities.topk(1).indices[0])
    return (
        _decode(retrieved, prototypes), selected,
        bool(torch.equal(retrieved, memory.values[selected])),
        float(similarities.max() - similarities.min()),
    )


def _integrated_prediction(trace: dict, prototypes: torch.Tensor,
                           context_projector, key_projector,
                           spec: dict = CONJUNCTION_SPEC, *,
                           mask_context: bool = False, mask_key: bool = False):
    transform = CompositeStateTransform(
        context_projector, key_projector, spec,
        mask_context=mask_context, mask_key=mask_key,
    )
    memory = VectorMemory(
        capacity=spec["events_per_episode"], dim=spec["state_dim"],
        key_transform=transform,
    )
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], trace["values"]
    ):
        memory.store((context_state, key_state), value_state)
    outcome = _memory_outcome(
        memory, (trace["query_context"], trace["query"]), trace["values"], prototypes
    )
    return outcome, {
        "calls": transform.calls,
        "minimum_components": min(transform.component_counts),
        "maximum_components": max(transform.component_counts),
        "minimum_address_width": min(transform.address_widths),
        "maximum_address_width": max(transform.address_widths),
        "stored_keys": len(memory.keys),
        "retrievals": 1,
    }


def _external_prediction(trace: dict, prototypes: torch.Tensor,
                         context_projector, key_projector,
                         spec: dict = CONJUNCTION_SPEC):
    memory = VectorMemory(capacity=spec["events_per_episode"], dim=spec["state_dim"])
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], trace["values"]
    ):
        memory.store(
            _composite(context_projector, key_projector, context_state, key_state, spec),
            value_state,
        )
    query = _composite(
        context_projector, key_projector, trace["query_context"], trace["query"], spec
    )
    return _memory_outcome(memory, query, trace["values"], prototypes)


def _exact_addresses(episode: ConjunctionEpisode, *, mask_context: bool = False,
                     mask_key: bool = False, spec: dict = CONJUNCTION_SPEC):
    def address(context: int, key: int) -> torch.Tensor:
        context_code = torch.zeros(spec["contexts"]) if mask_context else F.one_hot(
            torch.tensor(context), spec["contexts"]
        ).float()
        key_code = torch.zeros(spec["keys"]) if mask_key else F.one_hot(
            torch.tensor(key), spec["keys"]
        ).float()
        return torch.cat((context_code, key_code))

    return (
        [address(context, key) for context, key in zip(episode.contexts, episode.keys)],
        address(episode.query_context, episode.query_key),
    )


def _direct_prediction(addresses, values, query, prototypes, *, stored_values=None):
    stored_values = values if stored_values is None else stored_values
    memory = VectorMemory(capacity=len(addresses), dim=addresses[0].numel())
    for address, value in zip(addresses, stored_values):
        memory.store(address, value)
    return _memory_outcome(memory, query, stored_values, prototypes)


def _wrong_value_states(episode: ConjunctionEpisode, trace: dict) -> list[torch.Tensor]:
    representatives = {}
    for label, state in zip(episode.values, trace["values"]):
        representatives.setdefault(label, state)
    ordered = list(episode.active_values)
    successor = {value: ordered[(index + 1) % len(ordered)] for index, value in enumerate(ordered)}
    return [representatives[successor[label]] for label in episode.values]


def run_evaluation(prototype_seed: int, engine_seed: int,
                   episodes: list[ConjunctionEpisode], source: dict,
                   spec: dict = CONJUNCTION_SPEC) -> dict:
    context_projector = _load_context_projector(source["context_checkpoint"], spec)
    key_projector = _load_key_projector(source["canonical_checkpoint"], spec)
    before_context = {
        name: value.detach().clone() for name, value in context_projector.state_dict().items()
    }
    before_key = {
        name: value.detach().clone() for name, value in key_projector.state_dict().items()
    }
    prototype_receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    integrated_names = (
        "integrated_conjunction_normal", "integrated_context_masked",
        "integrated_key_masked", "integrated_conjunction_recovered",
    )
    call_audits = {
        name: {"calls": [], "components": [], "widths": [], "stores": [], "retrievals": []}
        for name in integrated_names
    }
    episode_seeds, cell_counts, before_digests, after_digests, rng_digests = [], [], [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, spec)
        cell_counts.extend(trace["cell_counts"])
        before_digests.append(trace["update_audit"]["state_before_sha256"])
        after_digests.append(trace["update_audit"]["state_after_sha256"])
        rng_digests.append(trace["update_audit"]["query_rng_sha256"])
        normal, normal_audit = _integrated_prediction(
            trace, prototypes, context_projector, key_projector, spec
        )
        context_masked, context_audit = _integrated_prediction(
            trace, prototypes, context_projector, key_projector, spec, mask_context=True
        )
        key_masked, key_audit = _integrated_prediction(
            trace, prototypes, context_projector, key_projector, spec, mask_key=True
        )
        recovered, recovered_audit = _integrated_prediction(
            trace, prototypes, context_projector, key_projector, spec
        )
        exact, exact_query = _exact_addresses(episode, spec=spec)
        context_only, context_only_query = _exact_addresses(
            episode, mask_key=True, spec=spec
        )
        key_only, key_only_query = _exact_addresses(
            episode, mask_context=True, spec=spec
        )
        outcomes = {
            "integrated_conjunction_normal": normal,
            "external_conjunction_reference": _external_prediction(
                trace, prototypes, context_projector, key_projector, spec
            ),
            "integrated_context_masked": context_masked,
            "integrated_key_masked": key_masked,
            "exact_context_key_control": _direct_prediction(
                exact, trace["values"], exact_query, prototypes
            ),
            "exact_context_only_control": _direct_prediction(
                context_only, trace["values"], context_only_query, prototypes
            ),
            "exact_key_only_control": _direct_prediction(
                key_only, trace["values"], key_only_query, prototypes
            ),
            "exact_context_key_partner_swap": _direct_prediction(
                exact, trace["values"], exact_query, prototypes,
                stored_values=_wrong_value_states(episode, trace),
            ),
            "integrated_conjunction_recovered": recovered,
        }
        for name, audit in (
            ("integrated_conjunction_normal", normal_audit),
            ("integrated_context_masked", context_audit),
            ("integrated_key_masked", key_audit),
            ("integrated_conjunction_recovered", recovered_audit),
        ):
            target = call_audits[name]
            target["calls"].append(audit["calls"])
            target["components"].extend((audit["minimum_components"], audit["maximum_components"]))
            target["widths"].extend((audit["minimum_address_width"], audit["maximum_address_width"]))
            target["stores"].append(audit["stored_keys"])
            target["retrievals"].append(audit["retrievals"])
        content = _decode(trace["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        if (index + 1) % 128 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed}] "
                f"evaluated {index + 1}/{len(episodes)} episodes", flush=True,
            )
    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    normal_records = records["integrated_conjunction_normal"]
    reference_records = records["external_conjunction_reference"]
    arms["integrated_conjunction_normal"]["reference_prediction_match"] = float(
        normal_records["predictions"] == reference_records["predictions"]
    )
    arms["integrated_conjunction_normal"]["reference_selection_match"] = float(
        normal_records["selections"] == reference_records["selections"]
    )
    arms["integrated_conjunction_recovered"]["prediction_match"] = float(
        records["integrated_conjunction_recovered"]["predictions"]
        == normal_records["predictions"]
    )
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "memory_path_audit": {
            name: {
                "minimum_calls": min(row["calls"]), "maximum_calls": max(row["calls"]),
                "minimum_components": min(row["components"]),
                "maximum_components": max(row["components"]),
                "minimum_address_width": min(row["widths"]),
                "maximum_address_width": max(row["widths"]),
                "minimum_stores": min(row["stores"]), "maximum_stores": max(row["stores"]),
                "minimum_retrievals": min(row["retrievals"]),
                "maximum_retrievals": max(row["retrievals"]),
            }
            for name, row in call_audits.items()
        },
        "integration_audit": {
            "component_weight": spec["component_weight"],
            "component_address_dim": spec["component_address_dim"],
            "composite_address_dim": spec["composite_address_dim"],
            "context_projector_frozen": not any(
                parameter.requires_grad for parameter in context_projector.parameters()
            ),
            "context_projector_unchanged": all(
                torch.equal(before_context[name], context_projector.state_dict()[name])
                for name in before_context
            ),
            "key_projector_frozen": not any(
                parameter.requires_grad for parameter in key_projector.parameters()
            ),
            "key_projector_unchanged": all(
                torch.equal(before_key[name], key_projector.state_dict()[name])
                for name in before_key
            ),
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts), "maximum_cells": max(cell_counts),
        },
        "update_audit": {
            "requested_updates": spec["pre_query_updates"],
            "performed_updates_minimum": spec["pre_query_updates"],
            "performed_updates_maximum": spec["pre_query_updates"],
            "disabled": list(spec["pre_query_dynamics_ablation"]),
            "state_before_sha256": hashlib.sha256("\n".join(before_digests).encode()).hexdigest(),
            "state_after_sha256": hashlib.sha256("\n".join(after_digests).encode()).hexdigest(),
            "query_rng_sha256": hashlib.sha256("\n".join(rng_digests).encode()).hexdigest(),
        },
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/conjunction_results.json")
    parser.add_argument("--verdict", default="measurement/conjunction_verdict.json")
    args = parser.parse_args()
    spec = CONJUNCTION_SPEC
    _, source = _source_receipt(spec)
    episodes = build_episodes(spec)
    payload = {
        "experiment": spec["experiment"], "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_context2": source,
        "dataset_audit": dataset_audit(episodes, spec),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": [
            {
                "name": evaluation_name(row),
                **run_evaluation(
                    row["prototype_seed"], row["engine_seed"], episodes, source, spec
                ),
            }
            for row in spec["evaluation_combinations"]
        ],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.conjunction_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
