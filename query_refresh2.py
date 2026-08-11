#!/usr/bin/env python3
"""QUERY-REFRESH-2: route eight-step query context through shared memory."""
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
from context_settle2 import _load_components
from graft_behavior import sha256_file
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.cue_robust_gate import adjudicate as adjudicate_robust
from measurement.cue_robust_registry import CUE_ROBUST_SPEC, spec_sha256 as robust_spec_sha256
from measurement.projector_registry import evaluation_name
from measurement.query_refresh2_registry import QUERY_REFRESH2_SPEC, spec_sha256
from measurement.query_refresh_gate import adjudicate as adjudicate_refresh
from measurement.query_refresh_registry import QUERY_REFRESH_SPEC, spec_sha256 as refresh_spec_sha256


def _source_receipt(spec: dict = QUERY_REFRESH2_SPEC) -> tuple[dict, dict]:
    refresh_results_path = Path(spec["source_refresh_results"])
    refresh_verdict_path = Path(spec["source_refresh_verdict_path"])
    refresh_results = json.loads(refresh_results_path.read_text())
    refresh_verdict = json.loads(refresh_verdict_path.read_text())
    refresh_sha = refresh_spec_sha256(QUERY_REFRESH_SPEC)
    if (
        refresh_results.get("experiment") != spec["source_refresh_experiment"]
        or refresh_results.get("spec") != QUERY_REFRESH_SPEC
        or refresh_results.get("spec_sha256") != refresh_sha
        or refresh_verdict.get("verdict") != spec["source_refresh_verdict"]
        or refresh_verdict.get("minimum_sustained_recovery_steps") != 8
        or adjudicate_refresh(refresh_results) != refresh_verdict
    ):
        raise RuntimeError("registered QUERY-REFRESH-1 source changed")

    robust_results_path = Path(spec["source_robust_results"])
    robust_verdict_path = Path(spec["source_robust_verdict_path"])
    robust_results = json.loads(robust_results_path.read_text())
    robust_verdict = json.loads(robust_verdict_path.read_text())
    robust_sha = robust_spec_sha256(CUE_ROBUST_SPEC)
    if (
        robust_results.get("experiment") != spec["source_robust_experiment"]
        or robust_results.get("spec") != CUE_ROBUST_SPEC
        or robust_results.get("spec_sha256") != robust_sha
        or robust_verdict.get("verdict") != spec["source_robust_verdict"]
        or adjudicate_robust(robust_results) != robust_verdict
    ):
        raise RuntimeError("registered CUE-ROBUST-1 source changed")
    inherited = robust_results["source"]
    receipts = [
        robust_results["checkpoint"], inherited["component_checkpoint"],
        inherited["value_checkpoint"], *inherited["prototype_checkpoints"].values(),
    ]
    if any(
        not Path(row["path"]).is_file()
        or sha256_file(Path(row["path"])) != row["sha256"]
        for row in receipts
    ):
        raise RuntimeError("registered QUERY-REFRESH-2 checkpoint changed")
    return robust_results, {
        "refresh_results": _receipt(refresh_results_path),
        "refresh_verdict": _receipt(refresh_verdict_path),
        "refresh_spec_sha256": refresh_sha,
        "robust_results": _receipt(robust_results_path),
        "robust_verdict": _receipt(robust_verdict_path),
        "robust_spec_sha256": robust_sha,
        "robust_checkpoint": dict(robust_results["checkpoint"]),
        "component_checkpoint": dict(inherited["component_checkpoint"]),
        "value_checkpoint": dict(inherited["value_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in inherited["prototype_checkpoints"].items()
        },
    }


def _load_robust_components(source: dict, spec: dict = QUERY_REFRESH2_SPEC):
    context, key = _load_components(source["component_checkpoint"], spec)
    checkpoint = torch.load(
        source["robust_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("experiment") != CUE_ROBUST_SPEC["experiment"]
        or checkpoint.get("spec_sha256") != robust_spec_sha256(CUE_ROBUST_SPEC)
        or checkpoint.get("deterministic") is not True
    ):
        raise RuntimeError("registered robust component checkpoint identity changed")
    context.load_state_dict(checkpoint["context_projector"])
    key.load_state_dict(checkpoint["key_projector"])
    context.eval().requires_grad_(False)
    key.eval().requires_grad_(False)
    return context, key


def _runtime_spec(query_steps: int, spec: dict = QUERY_REFRESH2_SPEC) -> dict:
    if query_steps not in set(spec["runtime_conditions"].values()):
        raise ValueError("query context steps are outside the registered conditions")
    return {**spec, "query_context_sense_steps": query_steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/query_refresh2_results.json")
    parser.add_argument("--verdict", default="measurement/query_refresh2_verdict.json")
    args = parser.parse_args()
    spec = QUERY_REFRESH2_SPEC
    robust_results, source = _source_receipt(spec)
    context_projector, key_projector = _load_robust_components(source, spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    representative_prototype = min(
        row["prototype_seed"] for row in spec["evaluation_combinations"]
    )
    unique_runs = {}
    for engine_seed in sorted({
        row["engine_seed"] for row in spec["evaluation_combinations"]
    }):
        for query_steps in sorted(set(spec["runtime_conditions"].values())):
            print(f"[engine {engine_seed}] running query steps {query_steps}", flush=True)
            unique_runs[(engine_seed, query_steps)] = run_evaluation(
                representative_prototype, engine_seed, episodes,
                source, robust_results, _runtime_spec(query_steps, spec),
                context_projector_override=context_projector,
                key_projector_override=key_projector,
                include_trace_digests=True,
            )
    evaluations = []
    for identity in spec["evaluation_combinations"]:
        row = {"name": evaluation_name(identity), **identity, "conditions": {}}
        for condition, query_steps in spec["runtime_conditions"].items():
            result = deepcopy(unique_runs[(identity["engine_seed"], query_steps)])
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
                "disabled_6": "baseline_6", "recovered_8": "refreshed_8",
            },
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.query_refresh2_gate import adjudicate
    verdict = adjudicate(payload, robust_results=robust_results)
    _atomic_json(Path(args.verdict), verdict)
    print(f'[{verdict["verdict"]}] {verdict["reason"]}')


if __name__ == "__main__":
    main()
