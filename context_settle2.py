#!/usr/bin/env python3
"""CONTEXT-SETTLE-2: integrate six-step context sensing into conjunction memory."""
from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path

import torch

from conjunction import _atomic_json, build_episodes, dataset_audit
from conjunction2 import _source_receipt as conjunction2_source_receipt, run_evaluation
from key_stability import StableKeyProjector
from measurement.component2_gate import adjudicate as adjudicate_component2
from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component2_spec_sha256
from measurement.conjunction2_registry import CONJUNCTION2_SPEC
from measurement.context_settle2_registry import CONTEXT_SETTLE2_SPEC, spec_sha256
from measurement.context_settle_gate import adjudicate as adjudicate_context_settle
from measurement.context_settle_registry import CONTEXT_SETTLE_SPEC, spec_sha256 as settle_spec_sha256
from measurement.projector_registry import evaluation_name
from graft_behavior import sha256_file


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = CONTEXT_SETTLE2_SPEC) -> dict:
    settle_results_path = Path(spec["source_settle_results"])
    settle_verdict_path = Path(spec["source_settle_verdict_path"])
    settle_results = json.loads(settle_results_path.read_text())
    settle_verdict = json.loads(settle_verdict_path.read_text())
    settle_sha = settle_spec_sha256(CONTEXT_SETTLE_SPEC)
    if (
        settle_results.get("experiment") != spec["source_settle_experiment"]
        or settle_results.get("spec") != CONTEXT_SETTLE_SPEC
        or settle_results.get("spec_sha256") != settle_sha
        or settle_verdict.get("verdict") != spec["source_settle_verdict"]
        or settle_verdict.get("minimum_settling_steps") != spec["settled_context_steps"]
        or adjudicate_context_settle(settle_results) != settle_verdict
    ):
        raise RuntimeError("registered CONTEXT-SETTLE-1 source changed")

    component_results_path = Path(spec["source_component_results"])
    component_verdict_path = Path(spec["source_component_verdict_path"])
    component_results = json.loads(component_results_path.read_text())
    component_verdict = json.loads(component_verdict_path.read_text())
    component_sha = component2_spec_sha256(COMPONENT2_SPEC)
    if (
        component_results.get("experiment") != spec["source_component_experiment"]
        or component_results.get("spec") != COMPONENT2_SPEC
        or component_results.get("spec_sha256") != component_sha
        or component_verdict.get("verdict") != spec["source_component_verdict"]
        or adjudicate_component2(component_results) != component_verdict
        or settle_results.get("source_component2", {}).get("checkpoint")
        != component_results.get("checkpoint")
    ):
        raise RuntimeError("registered COMPONENT-2 source changed")
    checkpoint = component_results["checkpoint"]
    if not Path(checkpoint["path"]).is_file() or sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise RuntimeError("registered component checkpoint changed")
    return {
        "settle_results": _receipt(settle_results_path),
        "settle_verdict": _receipt(settle_verdict_path),
        "settle_spec_sha256": settle_sha,
        "component_results": _receipt(component_results_path),
        "component_verdict": _receipt(component_verdict_path),
        "component_spec_sha256": component_sha,
        "component_checkpoint": dict(checkpoint),
        "conjunction_source": conjunction2_source_receipt(CONJUNCTION2_SPEC),
    }


def _load_components(receipt: dict, spec: dict = CONTEXT_SETTLE2_SPEC):
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != COMPONENT2_SPEC["experiment"]
        or checkpoint.get("spec_sha256") != component2_spec_sha256(COMPONENT2_SPEC)
        or checkpoint.get("deterministic") is not True
    ):
        raise RuntimeError("frozen component checkpoint identity changed")
    context = StableKeyProjector(
        spec["state_dim"], spec["component_address_dim"], spec["contexts"],
        spec["temperature"], spec["bias"],
    )
    key = StableKeyProjector(
        spec["state_dim"], spec["component_address_dim"], spec["keys"],
        spec["temperature"], spec["bias"],
    )
    context.load_state_dict(checkpoint["context_projector"])
    key.load_state_dict(checkpoint["key_projector"])
    context.eval().requires_grad_(False)
    key.eval().requires_grad_(False)
    return context, key


def _runtime_spec(context_steps: int, spec: dict = CONTEXT_SETTLE2_SPEC) -> dict:
    runtime = deepcopy(CONJUNCTION2_SPEC)
    runtime.update({
        "context_sense_steps": context_steps,
        "key_sense_steps": spec["key_sense_steps"],
        "value_sense_steps": spec["value_sense_steps"],
        "distractor_sense_steps": spec["distractor_sense_steps"],
    })
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/context_settle2_results.json")
    parser.add_argument("--verdict", default="measurement/context_settle2_verdict.json")
    args = parser.parse_args()
    spec = CONTEXT_SETTLE2_SPEC
    source = _source_receipt(spec)
    context_projector, key_projector = _load_components(source["component_checkpoint"], spec)
    episodes = build_episodes(CONJUNCTION2_SPEC)
    conditions = {
        "baseline_3": _runtime_spec(spec["baseline_context_steps"], spec),
        "settled_6": _runtime_spec(spec["settled_context_steps"], spec),
    }
    evaluations = []
    for combination in spec["evaluation_combinations"]:
        row = {"name": evaluation_name(combination), **combination, "conditions": {}}
        for condition, runtime_spec in conditions.items():
            print(f"[{row['name']}] running {condition}", flush=True)
            row["conditions"][condition] = run_evaluation(
                combination["prototype_seed"], combination["engine_seed"],
                episodes, source["conjunction_source"], runtime_spec,
                context_projector_override=context_projector,
                key_projector_override=key_projector,
            )
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
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.context_settle2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
