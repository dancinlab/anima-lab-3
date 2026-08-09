#!/usr/bin/env python3
"""Fail-closed adjudicator for VALIDITY-1 action-path decomposition."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.validity_registry import VALIDITY_SPEC, experiment, spec_sha256
except ModuleNotFoundError:
    from validity_registry import VALIDITY_SPEC, experiment, spec_sha256


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _probe_pass(metrics: dict, threshold: float, shuffled_max: float) -> bool:
    accuracy = float(metrics["accuracy"])
    shuffled = float(metrics["shuffled_label_accuracy"])
    return accuracy >= threshold and shuffled <= shuffled_max


def _all_actions_selected(metrics: dict, spec: dict) -> bool:
    matrix = metrics["confusion_matrix"]
    count = len(spec["actions"])
    if len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError("probe confusion matrix shape is invalid")
    return all(sum(matrix[row][column] for row in range(count)) > 0 for column in range(count))


def adjudicate(payload: dict) -> dict:
    try:
        spec = experiment(payload.get("experiment"))
    except (TypeError, ValueError):
        return {"verdict": "V0_INVALID", "reason": "result names an unregistered experiment"}
    if payload.get("spec") != spec or payload.get("spec_sha256") != spec_sha256(spec):
        return {"verdict": "V0_INVALID", "reason": "result spec does not match the registered SSOT"}
    if not _finite_tree(payload):
        return {"verdict": "V0_INVALID", "reason": "result contains a non-finite measurement"}
    source = payload.get("source", {})
    if (source.get("experiment") != spec["source_experiment"]
            or source.get("verdict") != spec["source_verdict"]
            or source.get("results_sha256") != spec["source_results_sha256"]
            or source.get("verdict_sha256") != spec["source_verdict_sha256"]
            or source.get("reproduced") is not True):
        return {"verdict": "V0_INVALID", "reason": "registered RELATION-1 source is missing or changed"}
    if payload.get("dataset_audit") != source.get("dataset_audit"):
        return {"verdict": "V0_INVALID", "reason": "recreated dataset audit differs from RELATION-1"}
    tokens = payload.get("action_tokens", {})
    if (tokens.get("unique_single_tokens") is not True
            or set(tokens.get("token_ids", {})) != set(spec["actions"])):
        return {"verdict": "V0_INVALID", "reason": "registered A-E actions are not unique single tokens"}
    rows = {row.get("seed"): row for row in payload.get("seeds", [])}
    if set(rows) != set(spec["seeds"]):
        return {"verdict": "V0_INVALID", "reason": "registered seed pair is incomplete"}

    if "invalid_results" in spec:
        invalid = payload.get("invalid_run", {})
        if (invalid.get("results_sha256") != spec["invalid_results_sha256"]
                or invalid.get("verdict_sha256") != spec["invalid_verdict_sha256"]
                or not isinstance(invalid.get("results"), dict)
                or not isinstance(invalid.get("verdict"), dict)
                or adjudicate(invalid["results"]) != invalid["verdict"]
                or invalid["verdict"].get("verdict") != "V0_INVALID"):
            return {"verdict": "V0_INVALID", "reason": "registered invalid first run is missing or changed"}
        if payload.get("model_revision") != spec["model_revision"]:
            return {"verdict": "V0_INVALID", "reason": "pinned language-model revision changed"}

    bars = spec["thresholds"]
    shuffled_max = bars["shuffled_label_max_accuracy"]
    judged = {}
    exact_language_replays = 0
    try:
        for seed in spec["seeds"]:
            row = rows[seed]
            if set(row["arms"]) != set(spec["arms"]):
                raise ValueError(f"seed {seed} arm set is incomplete")
            receipts = row["checkpoints"]
            if set(receipts) != set(spec["arms"]):
                raise ValueError(f"seed {seed} checkpoint set is incomplete")
            for arm in spec["arms"]:
                if receipts[arm]["sha256"] != spec["checkpoint_sha256"][str(seed)][arm]:
                    raise ValueError(f"seed {seed} {arm} checkpoint hash changed")
            sensory = row["sensory"]
            if set(sensory) != {"quantum", "memory"}:
                raise ValueError(f"seed {seed} sensory source set is incomplete")
            for source_name in sensory:
                if set(sensory[source_name]) != {"module_a", "module_b"}:
                    raise ValueError(f"seed {seed} {source_name} cue probes are incomplete")
                for metrics in sensory[source_name].values():
                    _probe_pass(metrics, bars["sensory_accuracy"], shuffled_max)
            judged[str(seed)] = {}
            for arm, metrics in row["arms"].items():
                if set(metrics) != {"relation", "direct_action", "normalization", "language"}:
                    raise ValueError(f"seed {seed} {arm} stage set is incomplete")
                if set(metrics["direct_action"]) != set(spec["normalization_modes"]):
                    raise ValueError(f"seed {seed} {arm} normalization modes are incomplete")
                language = metrics["language"]
                accuracy = float(language["accuracy"])
                source_accuracy = float(language["source_accuracy"])
                exact = accuracy == source_accuracy
                exact_language_replays += int(exact)
                replay = spec.get("language_replay")
                if replay is None:
                    replay_ok = language["source_accuracy_exact"] is True and exact
                else:
                    same_side = ((accuracy >= bars["language_accuracy"])
                                 == (source_accuracy >= bars["language_accuracy"]))
                    replay_ok = (
                        bool(language["source_accuracy_exact"]) is exact
                        and abs(accuracy - source_accuracy)
                        <= replay["maximum_accuracy_delta"] + 1e-8
                        and (same_side or not replay["require_same_threshold_side"])
                    )
                if not replay_ok:
                    raise ValueError(f"seed {seed} {arm} language result did not reproduce")
                if set(language["selection_counts"]) != set(spec["actions"]):
                    raise ValueError(f"seed {seed} {arm} action selection count is incomplete")
                judged[str(seed)][arm] = {
                    "relation_pass": _probe_pass(
                        metrics["relation"], bars["relation_accuracy"], shuffled_max
                    ),
                    "train_style_pass": _probe_pass(
                        metrics["direct_action"]["train_style"],
                        bars["direct_action_accuracy"], shuffled_max,
                    ),
                    "runtime_style_pass": _probe_pass(
                        metrics["direct_action"]["runtime_style"],
                        bars["direct_action_accuracy"], shuffled_max,
                    ),
                    "language_pass": float(language["accuracy"]) >= bars["language_accuracy"],
                    **metrics,
                }
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "V0_INVALID", "reason": str(exc)}

    replay = spec.get("language_replay")
    if replay is not None and exact_language_replays < replay["minimum_exact_arms"]:
        return {"verdict": "V0_INVALID", "reason": "too few language arms reproduced exactly"}

    all_probes = []
    for seed in spec["seeds"]:
        all_probes.extend(rows[seed]["sensory"][source][label]
                          for source in ("quantum", "memory")
                          for label in ("module_a", "module_b"))
        for arm in spec["arms"]:
            all_probes.append(rows[seed]["arms"][arm]["relation"])
            all_probes.extend(rows[seed]["arms"][arm]["direct_action"].values())
    if any(float(metrics["shuffled_label_accuracy"]) > shuffled_max for metrics in all_probes):
        return {"verdict": "V0_INVALID", "reason": "a shuffled-label control exceeded its ceiling"}

    control = spec["validation_arm"]
    sensory_ok = all(
        _probe_pass(rows[seed]["sensory"]["memory"][label], bars["sensory_accuracy"], shuffled_max)
        for seed in spec["seeds"] for label in ("module_a", "module_b")
    )
    relation_ok = all(judged[str(seed)][control]["relation_pass"] for seed in spec["seeds"])
    train_style_ok = all(judged[str(seed)][control]["train_style_pass"] for seed in spec["seeds"])
    runtime_style_ok = all(judged[str(seed)][control]["runtime_style_pass"] for seed in spec["seeds"])
    try:
        direct_actions_available = all(
            _all_actions_selected(rows[seed]["arms"][control]["direct_action"][mode], spec)
            for seed in spec["seeds"] for mode in spec["normalization_modes"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "V0_INVALID", "reason": str(exc)}
    language_ok = all(judged[str(seed)][control]["language_pass"] for seed in spec["seeds"])
    if not direct_actions_available:
        return {"verdict": "V0_INVALID", "reason": "the direct action board cannot select every action"}
    if not sensory_ok:
        verdict, reason = "V1_SENSE_LOSS", "the GRU input no longer preserves both registered cues"
    elif train_style_ok and not runtime_style_ok:
        verdict, reason = "V4_PROTOCOL_LOSS", (
            "the direct action readout passes with training-batch centering but fails with runtime centering"
        )
    elif not relation_ok:
        verdict, reason = "V2_RELATION_LOSS", "the GRU relation state does not preserve the registered answer"
    elif not runtime_style_ok:
        verdict, reason = "V2_RELATION_LOSS", (
            "the answer is readable before but not after the registered direct action code"
        )
    elif not language_ok:
        verdict, reason = "V3_LANGUAGE_LOSS", (
            "the frozen direct action code preserves the answer but the language action output loses it"
        )
    else:
        verdict, reason = "V5_PATH_VALID", "every registered action-path stage passes in both seeds"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "seeds": judged,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    args = parser.parse_args()
    verdict = adjudicate(json.loads(Path(args.results).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
