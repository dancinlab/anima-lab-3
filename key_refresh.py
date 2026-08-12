#!/usr/bin/env python3
"""KEY-REFRESH-1: vary only query-key processing time in shared memory."""
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
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.key_refresh_registry import KEY_REFRESH_SPEC, spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.query_refresh2_gate import adjudicate as adjudicate_source
from measurement.query_refresh2_registry import (
    QUERY_REFRESH2_SPEC, spec_sha256 as source_spec_sha256,
)
from query_refresh2 import _load_robust_components, _source_receipt as upstream_receipt


def _source_receipt(spec: dict = KEY_REFRESH_SPEC) -> tuple[dict, dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = source_spec_sha256(QUERY_REFRESH2_SPEC)
    robust_results, upstream = upstream_receipt(QUERY_REFRESH2_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != QUERY_REFRESH2_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or adjudicate_source(results, robust_results=robust_results) != verdict
        or results.get("source") != upstream
    ):
        raise RuntimeError("registered QUERY-REFRESH-2 source changed")
    return results, robust_results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        **deepcopy(upstream),
    }


def _runtime_spec(query_key_steps: int, spec: dict = KEY_REFRESH_SPEC) -> dict:
    if query_key_steps not in spec["query_key_steps"]:
        raise ValueError("query key steps are outside the registered candidates")
    return {
        **spec,
        "query_context_sense_steps": spec["query_context_sense_steps"],
        "query_key_sense_steps": query_key_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/key_refresh_results.json")
    parser.add_argument("--verdict", default="measurement/key_refresh_verdict.json")
    args = parser.parse_args()
    spec = KEY_REFRESH_SPEC
    source_results, robust_results, source = _source_receipt(spec)
    context_projector, key_projector = _load_robust_components(source, spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    representative_prototype = min(
        row["prototype_seed"] for row in spec["evaluation_combinations"]
    )
    unique_runs = {}
    for engine_seed in sorted({
        row["engine_seed"] for row in spec["evaluation_combinations"]
    }):
        for steps in spec["query_key_steps"]:
            print(f"[engine {engine_seed}] running query key steps {steps}", flush=True)
            unique_runs[(engine_seed, steps)] = run_evaluation(
                representative_prototype, engine_seed, episodes, source,
                robust_results, _runtime_spec(steps, spec),
                context_projector_override=context_projector,
                key_projector_override=key_projector,
                include_trace_digests=True,
            )
    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = {"name": evaluation_name(identity), **identity, "candidates": {}}
        for steps in spec["query_key_steps"]:
            result = deepcopy(unique_runs[(identity["engine_seed"], steps)])
            result["prototype_seed"] = identity["prototype_seed"]
            row["candidates"][str(steps)] = result
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
            "logical_candidate_evaluations": (
                len(spec["evaluation_combinations"]) * len(spec["query_key_steps"])
            ),
            "representative_prototype_seed": representative_prototype,
            "stable_value_path_is_prototype_independent": True,
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.key_refresh_gate import adjudicate
    verdict = adjudicate(
        payload, source_results=source_results, robust_results=robust_results
    )
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
