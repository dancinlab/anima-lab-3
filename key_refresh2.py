#!/usr/bin/env python3
"""KEY-REFRESH-2: integrate four-step query-key sensing into shared memory."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import platform
from pathlib import Path

import torch

from address_center2 import _receipt
from conjunction import _atomic_json, build_episodes, dataset_audit
from cue_mechanism import run_evaluation
from key_refresh import _source_receipt as key_refresh_source_receipt
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.key_refresh2_registry import KEY_REFRESH2_SPEC, spec_sha256
from measurement.key_refresh_registry import KEY_REFRESH_SPEC
from measurement.projector_registry import evaluation_name
from query_refresh2 import _load_robust_components


def _source_receipt(spec: dict = KEY_REFRESH2_SPEC) -> tuple[dict, dict, dict]:
    upstream_results, robust_results, inherited = key_refresh_source_receipt(
        KEY_REFRESH_SPEC
    )
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    key_results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    from measurement.key_refresh_gate import adjudicate
    if (
        key_results.get("experiment") != spec["source_experiment"]
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("minimum_sustained_recovery_steps")
        != spec["query_key_sense_steps"]
        or adjudicate(
            key_results, source_results=upstream_results,
            robust_results=robust_results,
        ) != verdict
    ):
        raise RuntimeError("registered KEY-REFRESH-1 source changed")
    return key_results, robust_results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": key_results["spec_sha256"],
        "upstream": deepcopy(key_results["source"]),
    }


def _runtime_spec(query_key_steps: int, spec: dict = KEY_REFRESH2_SPEC) -> dict:
    if query_key_steps not in set(spec["runtime_conditions"].values()):
        raise ValueError("query key steps are outside the registered conditions")
    return {
        **spec,
        "query_context_sense_steps": spec["query_context_sense_steps"],
        "query_key_sense_steps": query_key_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/key_refresh2_results.json")
    parser.add_argument("--verdict", default="measurement/key_refresh2_verdict.json")
    args = parser.parse_args()
    spec = KEY_REFRESH2_SPEC
    source_results, robust_results, source = _source_receipt(spec)
    runtime_source = source["upstream"]
    context_projector, key_projector = _load_robust_components(runtime_source, spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    representative_prototype = min(
        row["prototype_seed"] for row in spec["evaluation_combinations"]
    )
    unique_runs = {}
    for engine_seed in sorted({
        row["engine_seed"] for row in spec["evaluation_combinations"]
    }):
        for steps in sorted(set(spec["runtime_conditions"].values())):
            print(f"[engine {engine_seed}] running integrated query key {steps}", flush=True)
            unique_runs[(engine_seed, steps)] = run_evaluation(
                representative_prototype, engine_seed, episodes, runtime_source,
                robust_results, _runtime_spec(steps, spec),
                context_projector_override=context_projector,
                key_projector_override=key_projector,
                include_trace_digests=True,
            )
    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = {"name": evaluation_name(identity), **identity, "conditions": {}}
        for condition, steps in spec["runtime_conditions"].items():
            result = deepcopy(unique_runs[(identity["engine_seed"], steps)])
            result["prototype_seed"] = identity["prototype_seed"]
            row["conditions"][condition] = result
        evaluations.append(row)
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": source,
        "dataset_audit": dataset_audit(episodes, CONJUNCTION2_SPEC),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "execution_audit": {
            "unique_engine_step_runs": len(unique_runs),
            "logical_condition_evaluations": (
                len(spec["evaluation_combinations"]) * len(spec["runtime_conditions"])
            ),
            "representative_prototype_seed": representative_prototype,
            "stable_value_path_is_prototype_independent": True,
            "condition_aliases": {
                "disabled_3": "baseline_3", "recovered_4": "integrated_4",
            },
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.key_refresh2_gate import adjudicate
    verdict = adjudicate(payload, source_results=source_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
