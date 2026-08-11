#!/usr/bin/env python3
"""QUERY-REFRESH-1: vary only query-context sensing duration."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from conjunction import _atomic_json, build_episodes, dataset_audit
from cue_context import _classification, _collect_pairs, _load_source_context, _mask_rows
from cue_history import _comparison, _digest_pairs, _history_audit, _reverse_events
from measurement.completion_registry import COMPLETION_SPEC, cue_mask_indices
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_context_registry import CUE_CONTEXT_SPEC
from measurement.cue_history_gate import adjudicate as adjudicate_history
from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256 as history_spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _source_receipt(spec: dict = QUERY_REFRESH_SPEC) -> tuple[dict, dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    source_sha = history_spec_sha256(CUE_HISTORY_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CUE_HISTORY_SPEC
        or results.get("spec_sha256") != source_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_history(results) != verdict
    ):
        raise RuntimeError("registered CUE-HISTORY-1 source changed")
    align_results = json.loads(Path(results["source"]["results"]["path"]).read_text())
    return results, align_results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": source_sha,
    }


def _candidate_spec(query_steps: int, spec: dict = QUERY_REFRESH_SPEC) -> dict:
    if query_steps not in spec["query_context_steps"]:
        raise ValueError("query-context steps are outside the registered candidates")
    return {**spec, "query_context_sense_steps": query_steps}


def _pair_audit(pairs: dict, query_steps: int, spec: dict = QUERY_REFRESH_SPEC) -> dict:
    audit = dict(pairs["audit"])
    count = audit["pairs"]
    storage_calls = count * spec["events_per_episode"] * spec["settled_context_steps"]
    query_calls = count * query_steps
    audit.update({
        "storage_context_step_calls": storage_calls,
        "query_context_step_calls": query_calls,
        "query_context_sense_steps": query_steps,
    })
    if audit["context_step_calls"] != storage_calls + query_calls:
        raise RuntimeError("storage and query context call accounting diverged")
    return audit


def _predictions(projector, states: torch.Tensor) -> torch.Tensor:
    addresses = projector.address(states).detach()
    prototypes = F.normalize(projector.prototypes.detach(), dim=-1)
    return (addresses @ prototypes.T).argmax(1)


def _condition(projector, states: torch.Tensor, labels: torch.Tensor,
               baseline_states: torch.Tensor, baseline_predictions: torch.Tensor) -> dict:
    predictions = _predictions(projector, states)
    return {
        "metric": _classification(projector, states, labels),
        "comparison_to_baseline_steps": _comparison(
            baseline_states, states, labels, baseline_predictions, predictions
        ),
    }


def _evaluate(projector, pairs_by_step: dict, masks: list[tuple[int, ...]],
              spec: dict = QUERY_REFRESH_SPEC) -> dict:
    baseline_steps = spec["baseline_query_context_steps"]
    prepared = {}
    for steps, histories in pairs_by_step.items():
        prepared[steps] = {}
        for history, pairs in histories.items():
            prepared[steps][history] = {
                "query_full": pairs["query"],
                "query_quarter_missing": _mask_rows(pairs["query"], masks),
            }
    labels = pairs_by_step[baseline_steps]["original"]["labels"]
    baseline_predictions = {
        history: {
            condition: _predictions(projector, states)
            for condition, states in prepared[baseline_steps][history].items()
        }
        for history in spec["histories"]
    }
    candidates = {}
    for steps in spec["query_context_steps"]:
        histories = {}
        for history in spec["histories"]:
            conditions = {
                condition: _condition(
                    projector, states, labels,
                    prepared[baseline_steps][history][condition],
                    baseline_predictions[history][condition],
                )
                for condition, states in prepared[steps][history].items()
            }
            histories[history] = {"conditions": conditions}
        history_comparison = {}
        for condition in spec["conditions"]:
            original_states = prepared[steps]["original"][condition]
            reversed_states = prepared[steps]["event_reversed"][condition]
            history_comparison[condition] = _comparison(
                original_states, reversed_states, labels,
                _predictions(projector, original_states),
                _predictions(projector, reversed_states),
            )
        candidates[str(steps)] = {
            "query_context_steps": steps,
            "histories": histories,
            "history_comparison": history_comparison,
        }
    return candidates


def _source_audit(identity: dict, pairs_by_step: dict, candidates: dict,
                  source_results: dict, spec: dict = QUERY_REFRESH_SPEC) -> dict:
    source = next(row for row in source_results["evaluations"]
                  if row["name"] == evaluation_name(identity))
    baseline = candidates[str(spec["baseline_query_context_steps"])]
    audits = {}
    for history in spec["histories"]:
        audits[history] = {
            "pair_digest_match": (
                _digest_pairs(pairs_by_step[spec["baseline_query_context_steps"]][history])
                == source["pair_digests"][history]
            ),
            "query_full_metric_match": (
                baseline["histories"][history]["conditions"]["query_full"]["metric"]
                == source["histories"][history]["conditions"]["query_full"]["metric"]
            ),
            "query_quarter_missing_metric_match": (
                baseline["histories"][history]["conditions"][
                    "query_quarter_missing"
                ]["metric"]
                == source["histories"][history]["conditions"][
                    "query_quarter_missing"
                ]["metric"]
            ),
        }
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/query_refresh_results.json")
    parser.add_argument("--verdict", default="measurement/query_refresh_verdict.json")
    args = parser.parse_args()
    spec = QUERY_REFRESH_SPEC
    source_results, align_results, source = _source_receipt(spec)
    projector = _load_source_context(align_results["source"]["upstream"], CUE_CONTEXT_SPEC)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    variants = {
        "original": episodes,
        "event_reversed": [_reverse_events(row) for row in episodes],
    }
    masks = [
        cue_mask_indices(index, "context", spec["missing_fraction"], COMPLETION_SPEC)
        for index in range(len(episodes))
    ]
    by_engine = {}
    engine_seeds = sorted({row["engine_seed"] for row in spec["evaluation_combinations"]})
    for engine_seed in engine_seeds:
        pairs_by_step = {}
        for steps in spec["query_context_steps"]:
            pairs_by_step[steps] = {}
            for history in spec["histories"]:
                print(f"[engine {engine_seed} query-steps {steps}] running {history}", flush=True)
                pairs_by_step[steps][history] = _collect_pairs(
                    variants[history], [engine_seed], spec["episode_seed_base"],
                    _candidate_spec(steps, spec),
                )
        candidates = _evaluate(projector, pairs_by_step, masks, spec)
        by_engine[engine_seed] = {
            "pair_audits": {
                str(steps): {
                    history: _pair_audit(pairs, steps, spec)
                    for history, pairs in histories.items()
                }
                for steps, histories in pairs_by_step.items()
            },
            "pair_digests": {
                str(steps): {
                    history: _digest_pairs(pairs)
                    for history, pairs in histories.items()
                }
                for steps, histories in pairs_by_step.items()
            },
            "candidates": candidates,
            "pairs_by_step": pairs_by_step,
        }

    evaluations = []
    for identity in spec["evaluation_combinations"]:
        engine = by_engine[identity["engine_seed"]]
        evaluations.append({
            "name": evaluation_name(identity), **identity,
            "pair_audits": engine["pair_audits"],
            "pair_digests": engine["pair_digests"],
            "candidates": engine["candidates"],
            "source_reference_audit": _source_audit(
                identity, engine["pairs_by_step"], engine["candidates"], source_results, spec
            ),
        })

    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "history_audit": _history_audit(episodes, variants),
        "mask_audit": {
            "states": len(masks),
            "removed_per_state": len(masks[0]),
            "unique_masks": len(set(masks)),
            "sha256": hashlib.sha256(
                "\n".join(",".join(map(str, row)) for row in masks).encode()
            ).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.query_refresh_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
