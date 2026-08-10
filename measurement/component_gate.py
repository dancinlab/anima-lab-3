#!/usr/bin/env python3
"""Fail-closed adjudication for COMPONENT-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.capacity_gate import _balanced, _finite_tree, _valid_receipt
    from measurement.component_registry import COMPONENT_SPEC, spec_sha256
    from measurement.conjunction2_gate import adjudicate as adjudicate_source
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256 as source_spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _balanced, _finite_tree, _valid_receipt
    from measurement.component_registry import COMPONENT_SPEC, spec_sha256
    from measurement.conjunction2_gate import adjudicate as adjudicate_source
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC, spec_sha256 as source_spec_sha256


def _shape(value, classes):
    return (len(value["per_key_recall"]) == classes
            and len(value["confusion_matrix"]) == classes
            and all(len(row) == classes for row in value["confusion_matrix"]))


def adjudicate(payload, spec=COMPONENT_SPEC):
    def invalid(reason):
        return {"experiment": payload.get("experiment", spec["experiment"]),
                "verdict": "AC0_INVALID", "reason": reason,
                "spec_sha256": spec_sha256(spec)}
    try:
        if payload["experiment"] != spec["experiment"] or payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("experiment or registered spec changed")
        if not _finite_tree(payload): return invalid("result contains a non-finite number")
        source = payload["source_conjunction2"]
        if any(not _valid_receipt(source[name]) for name in ("results", "verdict", "context_checkpoint", "canonical_checkpoint")):
            return invalid("source file changed")
        results = json.loads(Path(source["results"]["path"]).read_text())
        verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        sha = source_spec_sha256(CONJUNCTION2_SPEC)
        if (results.get("experiment") != spec["source_experiment"] or results.get("spec") != CONJUNCTION2_SPEC
            or results.get("spec_sha256") != sha or verdict.get("verdict") != spec["source_verdict"]
            or verdict.get("spec_sha256") != sha or adjudicate_source(results) != verdict
            or source["source_spec_sha256"] != sha): return invalid("source identity changed")
        audit = payload["dataset_audit"]; total = spec["eval_episodes"]
        if (audit["episodes"] != total or audit["unique_fingerprints"] != total
            or audit["latin_valid_episodes"] != total or not _balanced(audit["target_counts"], spec["values"], total)):
            return invalid("dataset balance changed")
        engines = {row["engine_seed"]: row for row in payload["engines"]}
        if set(engines) != set(spec["engine_seeds"]) or len(engines) != len(payload["engines"]):
            return invalid("engine roster changed")
        judged = {}
        for seed, row in engines.items():
            if row["frozen_audit"] != {"context": True, "key": True}: return invalid(f"engine {seed} projector changed")
            state = row["state_audit"]
            if (state["episodes"] != total or state["unique_episode_seeds"] != total
                or len(state["episode_seed_sha256"]) != 64
                or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]):
                return invalid(f"engine {seed} state stream changed")
            positions = {item["position"]: item for item in row["positions"]}
            if set(positions) != set(spec["positions"]) or len(positions) != len(row["positions"]):
                return invalid(f"engine {seed} position roster changed")
            rows = {}
            for position in spec["positions"]:
                item = positions[position]
                if item["position_label"] != position + 1 or not _shape(item["context"], spec["contexts"]) or not _shape(item["key"], spec["keys"]):
                    return invalid(f"engine {seed} position {position + 1} metrics changed")
                rows[str(position)] = {
                    "context_accuracy": item["context"]["accuracy"],
                    "context_minimum_recall": min(item["context"]["per_key_recall"]),
                    "key_accuracy": item["key"]["accuracy"],
                    "key_minimum_recall": min(item["key"]["per_key_recall"]),
                    "context_passed": item["context"]["accuracy"] >= spec["thresholds"]["classification_accuracy"] and min(item["context"]["per_key_recall"]) >= spec["thresholds"]["minimum_class_recall"],
                    "key_passed": item["key"]["accuracy"] >= spec["thresholds"]["classification_accuracy"] and min(item["key"]["per_key_recall"]) >= spec["thresholds"]["minimum_class_recall"],
                }
            judged[str(seed)] = rows
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))
    context_loss = any(not item["context_passed"] for rows in judged.values() for item in rows.values())
    key_loss = any(not item["key_passed"] for rows in judged.values() for item in rows.values())
    if context_loss and key_loss: verdict, reason = "AC3_BOTH_COMPONENTS_LOSS", "both frozen address components lost serial-position stability"
    elif context_loss: verdict, reason = "AC1_CONTEXT_LOSS", "only the frozen context component lost serial-position stability"
    elif key_loss: verdict, reason = "AC2_KEY_LOSS", "only the frozen key component lost serial-position stability"
    else: verdict, reason = "AC4_COMPONENTS_STABLE", "both components stayed stable; collision lies in composition"
    return {"experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "engines": judged}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("results", nargs="?", default="measurement/component_results.json"); parser.add_argument("--output", default="measurement/component_verdict.json"); args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text()); verdict = adjudicate(payload)
    path = Path(args.output); temporary = path.with_name(path.name + ".tmp"); temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"); os.replace(temporary, path)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__": main()
