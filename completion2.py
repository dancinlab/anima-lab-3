#!/usr/bin/env python3
"""COMPLETION-2: measure the repaired shared-memory partial-cue boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from address_center2 import _receipt
from address_margin import _center, _join, _outcome
from completion import _integrated_outcome
from conjunction import _atomic_json, build_episodes, dataset_audit, trace_episode
from conjunction2 import _load_value_projector
from context_settle2 import _runtime_spec
from graft_behavior import sha256_file
from key_refresh2 import _source_receipt as validate_key_refresh_source
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.key_refresh2_gate import adjudicate as adjudicate_key_refresh2
from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC
from measurement.completion2_registry import (
    COMPLETION2_SPEC, mask_plan_audit, spec_sha256,
)
from measurement.projector_registry import evaluation_name
from query_refresh2 import _load_robust_components
from separation import _arm_metrics


def _source_receipt(spec: dict = COMPLETION2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    key_results, _, validated = validate_key_refresh_source(KEY_REFRESH2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != KEY_REFRESH2_SPEC
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_key_refresh2(results, source_results=key_results) != verdict
        or results.get("source") != validated
    ):
        raise RuntimeError("registered KEY-REFRESH-2 source changed")
    upstream = results["source"]["upstream"]
    receipts = [
        upstream["robust_checkpoint"], upstream["component_checkpoint"],
        upstream["value_checkpoint"], *upstream["prototype_checkpoints"].values(),
    ]
    if any(
        not Path(row["path"]).is_file()
        or sha256_file(Path(row["path"])) != row["sha256"]
        for row in receipts
    ):
        raise RuntimeError("registered COMPLETION-2 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": results["spec_sha256"],
        "upstream": upstream,
    }


def _records_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(
        {"predictions": row["predictions"], "selections": row["selections"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source: dict,
                   source_results: dict, spec: dict = COMPLETION2_SPEC) -> dict:
    upstream = source["upstream"]
    context_projector, key_projector = _load_robust_components(upstream, spec)
    value_projector = _load_value_projector(upstream["value_checkpoint"], spec)
    models = {"context": context_projector, "key": key_projector, "value": value_projector}
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    prototypes = F.normalize(value_projector.prototypes.detach(), dim=-1)
    runtime = _runtime_spec(spec["settled_context_steps"], spec)
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    call_audits = {name: [] for name in spec["conditions"]}
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    episode_seeds, cell_counts, sense_audits = [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        trace = trace_episode(episode, trial_seed, runtime)
        cell_counts.extend(trace["cell_counts"])
        sense_audits.append(trace["sense_audit"])
        outcomes = {}
        for name, (context_missing, key_missing) in spec["conditions"].items():
            outcome, audit = _integrated_outcome(
                trace, index, context_projector, key_projector, value_projector,
                prototypes, spec, context_missing=context_missing,
                key_missing=key_missing,
            )
            outcomes[name] = outcome
            call_audits[name].append(audit)

        oracle_rows = [
            _join(_center(context_projector, context_label),
                  _center(key_projector, key_label), spec)
            for context_label, key_label in zip(episode.contexts, episode.keys)
        ]
        query_oracle = _join(
            _center(context_projector, episode.query_context),
            _center(key_projector, episode.query_key), spec,
        )
        exact, _ = _outcome(
            oracle_rows, query_oracle, trace["values"], prototypes, value_projector
        )
        partner, _ = _outcome(
            oracle_rows, query_oracle,
            trace["values"][1:] + trace["values"][:1], prototypes, value_projector,
        )
        outcomes["exact_context_key_control"] = exact
        outcomes["exact_context_key_partner_swap"] = partner
        content = int(value_projector(
            trace["values"][episode.query_position].mean(0).unsqueeze(0)
        ).argmax(1)[0])
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
    record_digests = {name: _records_digest(row) for name, row in records.items()}
    source_row = next(
        row for row in source_results["evaluations"]
        if row["name"] == evaluation_name({
            "prototype_seed": prototype_seed, "engine_seed": engine_seed,
        })
    )["conditions"][spec["source_condition"]]
    reference_names = (
        "full_cue", "context_quarter_missing", "key_quarter_missing",
        "both_quarter_missing", "exact_context_key_control",
        "exact_context_key_partner_swap",
    )
    event_queries = len(episodes) * (spec["events_per_episode"] + 1)
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "record_digests": record_digests,
        "reference_audit": {
            name: {
                "metric_match": arms[name] == source_row["arms"][name],
                "record_match": record_digests[name] == source_row["record_digests"][name],
            }
            for name in reference_names
        },
        "frozen_audit": {
            name: all(
                torch.equal(before[name][key], model.state_dict()[key])
                for key in before[name]
            )
            for name, model in models.items()
        },
        "memory_path_audit": {
            name: {
                "minimum_calls": min(row["transform_calls"] for row in rows),
                "maximum_calls": max(row["transform_calls"] for row in rows),
                "minimum_components": min(row["minimum_components"] for row in rows),
                "maximum_components": max(row["maximum_components"] for row in rows),
                "minimum_address_width": min(row["minimum_address_width"] for row in rows),
                "maximum_address_width": max(row["maximum_address_width"] for row in rows),
                "value_calls": sum(row["value_calls"] for row in rows),
                "stores": sum(row["stores"] for row in rows),
                "retrievals": sum(row["retrievals"] for row in rows),
            }
            for name, rows in call_audits.items()
        },
        "state_audit": {
            "episodes": len(episodes),
            "unique_episode_seeds": len(set(episode_seeds)),
            "episode_seed_sha256": hashlib.sha256(
                "\n".join(map(str, episode_seeds)).encode()
            ).hexdigest(),
            "minimum_cells": min(cell_counts),
            "maximum_cells": max(cell_counts),
            "context_step_calls": sum(row["context_step_calls"] for row in sense_audits),
            "key_step_calls": sum(row["key_step_calls"] for row in sense_audits),
            "value_step_calls": sum(row["value_step_calls"] for row in sense_audits),
            "distractor_step_calls": sum(row["distractor_step_calls"] for row in sense_audits),
            "expected_context_states": event_queries,
            "expected_key_states": event_queries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/completion2_results.json")
    parser.add_argument("--verdict", default="measurement/completion2_verdict.json")
    args = parser.parse_args()
    spec = COMPLETION2_SPEC
    source_results, source = _source_receipt(spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    evaluations = [
        {"name": evaluation_name(row), **run_evaluation(
            row["prototype_seed"], row["engine_seed"], episodes, source,
            source_results, spec,
        )}
        for row in spec["evaluation_combinations"]
    ]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "mask_plan_audit": mask_plan_audit(spec),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.completion2_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
