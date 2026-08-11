#!/usr/bin/env python3
"""CUE-HISTORY-1: isolate event-order and distractor-history effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from conjunction import ConjunctionEpisode, _atomic_json, build_episodes, dataset_audit
from cue_context import _classification, _collect_pairs, _load_source_context, _mask_rows
from measurement.cue_align_gate import adjudicate as adjudicate_align
from measurement.cue_align_registry import CUE_ALIGN_SPEC, spec_sha256 as align_spec_sha256
from measurement.completion_registry import COMPLETION_SPEC, cue_mask_indices
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_context_registry import CUE_CONTEXT_SPEC
from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256
from measurement.projector_registry import evaluation_name


def _source_receipt(spec: dict = CUE_HISTORY_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    source_sha = align_spec_sha256(CUE_ALIGN_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CUE_ALIGN_SPEC
        or results.get("spec_sha256") != source_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_align(results) != verdict
    ):
        raise RuntimeError("registered CUE-ALIGN-1 source changed")
    return results, {
        "results": {
            "path": str(results_path),
            "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        },
        "verdict": {
            "path": str(verdict_path),
            "sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
        },
        "source_spec_sha256": source_sha,
    }


def _reverse_events(episode: ConjunctionEpisode) -> ConjunctionEpisode:
    count = len(episode.contexts)
    return ConjunctionEpisode(
        contexts=tuple(reversed(episode.contexts)),
        keys=tuple(reversed(episode.keys)),
        values=tuple(reversed(episode.values)),
        active_contexts=episode.active_contexts,
        active_keys=episode.active_keys,
        active_values=episode.active_values,
        distractors=episode.distractors,
        query_position=count - 1 - episode.query_position,
    )


def _swap_distractors(episodes: list[ConjunctionEpisode]) -> list[ConjunctionEpisode]:
    by_context: dict[int, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        by_context[episode.query_context].append(index)
    donors = {}
    for indices in by_context.values():
        if len(indices) < 2:
            raise ValueError("distractor history swap needs at least two episodes per context")
        donors.update({index: indices[(offset + 1) % len(indices)]
                       for offset, index in enumerate(indices)})
    return [
        ConjunctionEpisode(
            contexts=episode.contexts, keys=episode.keys, values=episode.values,
            active_contexts=episode.active_contexts, active_keys=episode.active_keys,
            active_values=episode.active_values,
            distractors=episodes[donors[index]].distractors,
            query_position=episode.query_position,
        )
        for index, episode in enumerate(episodes)
    ]


def _history_variants(episodes: list[ConjunctionEpisode]) -> dict[str, list[ConjunctionEpisode]]:
    reversed_events = [_reverse_events(row) for row in episodes]
    swapped = _swap_distractors(episodes)
    return {
        "original": episodes,
        "original_repeat": episodes,
        "event_reversed": reversed_events,
        "distractor_swapped": swapped,
        "both_changed": [_reverse_events(row) for row in swapped],
    }


def _event_counter(episode: ConjunctionEpisode) -> Counter:
    return Counter(zip(episode.contexts, episode.keys, episode.values))


def _distractor_roster(episodes: list[ConjunctionEpisode]) -> dict[str, list[tuple[int, ...]]]:
    return {
        str(label): sorted(row.distractors for row in episodes if row.query_context == label)
        for label in sorted({row.query_context for row in episodes})
    }


def _history_audit(original: list[ConjunctionEpisode], variants: dict) -> dict:
    original_roster = _distractor_roster(original)
    rows = {}
    for name, episodes in variants.items():
        rows[name] = {
            "episodes": len(episodes),
            "query_identity_preserved": sum(
                (row.query_context, row.query_key, row.target)
                == (base.query_context, base.query_key, base.target)
                for base, row in zip(original, episodes)
            ),
            "event_multiset_preserved": sum(
                _event_counter(base) == _event_counter(row)
                for base, row in zip(original, episodes)
            ),
            "event_order_changed": sum(
                (base.contexts, base.keys, base.values)
                != (row.contexts, row.keys, row.values)
                for base, row in zip(original, episodes)
            ),
            "distractor_history_changed": sum(
                base.distractors != row.distractors for base, row in zip(original, episodes)
            ),
            "distractor_roster_preserved_by_context": (
                _distractor_roster(episodes) == original_roster
            ),
        }
    return rows


def _predictions(projector, states: torch.Tensor) -> torch.Tensor:
    addresses = projector.address(states).detach()
    prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    return (addresses @ prototypes.T).argmax(1)


def _comparison(base_states: torch.Tensor, states: torch.Tensor,
                labels: torch.Tensor, base_predictions: torch.Tensor,
                predictions: torch.Tensor) -> dict:
    baseline_correct = base_predictions == labels
    changed_correct = predictions == labels
    errors = (~baseline_correct).sum().item()
    correct = baseline_correct.sum().item()
    return {
        "prediction_agreement": float((predictions == base_predictions).float().mean()),
        "prediction_disagreement": float((predictions != base_predictions).float().mean()),
        "baseline_errors_corrected_fraction": (
            float(((~baseline_correct) & changed_correct).sum().item() / errors) if errors else 0.0
        ),
        "baseline_correct_regression_fraction": (
            float((baseline_correct & (~changed_correct)).sum().item() / correct)
            if correct else 0.0
        ),
        "accuracy_gain": float(changed_correct.float().mean() - baseline_correct.float().mean()),
        "state_cosine_similarity": float(F.cosine_similarity(base_states, states, dim=1).mean()),
        "state_mse": float(F.mse_loss(states, base_states)),
    }


def _evaluate(projector, pairs_by_history: dict, masks: list[tuple[int, ...]]) -> dict:
    prepared = {}
    for history, pairs in pairs_by_history.items():
        prepared[history] = {
            "query_full": pairs["query"],
            "query_quarter_missing": _mask_rows(pairs["query"], masks),
        }
    labels = pairs_by_history["original"]["labels"]
    base_predictions = {
        condition: _predictions(projector, states)
        for condition, states in prepared["original"].items()
    }
    result = {}
    for history, conditions in prepared.items():
        result[history] = {"conditions": {}}
        for condition, states in conditions.items():
            predictions = _predictions(projector, states)
            result[history]["conditions"][condition] = {
                "metric": _classification(projector, states, labels),
                "comparison_to_original": _comparison(
                    prepared["original"][condition], states, labels,
                    base_predictions[condition], predictions,
                ),
            }
    return result


def _digest_pairs(pairs: dict) -> dict:
    return {
        name: hashlib.sha256(pairs[name].contiguous().numpy().tobytes()).hexdigest()
        for name in ("storage", "query", "labels")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/cue_history_results.json")
    parser.add_argument("--verdict", default="measurement/cue_history_verdict.json")
    args = parser.parse_args()
    spec = CUE_HISTORY_SPEC
    align_results, align_source = _source_receipt(spec)
    projector = _load_source_context(align_results["source"]["upstream"], CUE_CONTEXT_SPEC)

    episodes = build_episodes(CONJUNCTION2_SPEC)
    variants = _history_variants(episodes)
    masks = [
        cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
        for index in range(len(episodes))
    ]
    history_audit = _history_audit(episodes, variants)
    by_engine = {}
    for engine_seed in sorted({row["engine_seed"] for row in spec["evaluation_combinations"]}):
        pairs = {}
        for history in spec["histories"]:
            print(f"[engine {engine_seed}] running {history}", flush=True)
            pairs[history] = _collect_pairs(
                variants[history], [engine_seed], spec["episode_seed_base"], spec
            )
        by_engine[engine_seed] = {
            "pair_audits": {name: row["audit"] for name, row in pairs.items()},
            "pair_digests": {name: _digest_pairs(row) for name, row in pairs.items()},
            "repeat_exact": {
                name: torch.equal(pairs["original"][name], pairs["original_repeat"][name])
                for name in ("storage", "query", "labels")
            },
            "histories": _evaluate(projector, pairs, masks),
        }

    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = by_engine[identity["engine_seed"]]
        reference = next(item for item in align_results["evaluations"]
                         if item["name"] == evaluation_name(identity))
        original = row["histories"]["original"]["conditions"]
        evaluations.append({
            "name": evaluation_name(identity), **identity, **row,
            "source_reference_audit": {
                condition: original[condition]["metric"]
                == reference["models"]["source"]["conditions"][condition]
                for condition in spec["conditions"]
            },
        })

    payload = {
        "experiment": spec["experiment"], "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": align_source,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "history_audit": history_audit,
        "mask_audit": {
            "states": len(masks), "removed_per_state": len(masks[0]),
            "unique_masks": len(set(masks)),
            "sha256": hashlib.sha256(
                "\n".join(",".join(map(str, row)) for row in masks).encode()
            ).hexdigest(),
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": spec["device"]},
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.cue_history_gate import adjudicate
    verdict = adjudicate(payload, source_results=align_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
