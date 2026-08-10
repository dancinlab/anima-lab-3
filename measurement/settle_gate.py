#!/usr/bin/env python3
"""Fail-closed adjudication for SETTLE-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.decay_gate import (
        _finite_tree, _metric_shape, _valid_receipt,
        _verify_projector, _verify_prototypes,
    )
    from measurement.recovery_gate import (
        _balanced, _close, _geometry_shape, _passes, _pooled_matches_replicates,
    )
    from measurement.reset_registry import RESET_SPEC, spec_sha256 as reset_spec_sha256
    from measurement.settle_registry import SETTLE_SPEC, spec_sha256
except ModuleNotFoundError:
    from decay_gate import (
        _finite_tree, _metric_shape, _valid_receipt,
        _verify_projector, _verify_prototypes,
    )
    from recovery_gate import (
        _balanced, _close, _geometry_shape, _passes, _pooled_matches_replicates,
    )
    from reset_registry import RESET_SPEC, spec_sha256 as reset_spec_sha256
    from settle_registry import SETTLE_SPEC, spec_sha256


def _classify(passed: dict[str, bool]) -> tuple[str, str]:
    count = sum(passed.values())
    if count == len(passed):
        return (
            "ST1_AUTONOMOUS_SETTLING_CAUSAL",
            "autonomous evolution beat the frozen state and passed the memory path in both seeds",
        )
    if count == 1:
        return "ST2_SEED_CONDITIONAL_SETTLING", "only one seed met every settling criterion"
    return "ST3_NO_AUTONOMOUS_SETTLING", "neither seed met every settling criterion"


def _pair_valid(value: dict, total: int) -> bool:
    fields = (
        "both_correct", "both_wrong", "autonomous_only_correct", "frozen_only_correct",
    )
    return (
        value["episodes"] == total
        and all(isinstance(value[name], int) and value[name] >= 0 for name in fields)
        and sum(value[name] for name in fields) == total
        and 0.0 <= value["exact_two_sided_p"] <= 1.0
        and _close(
            value["net_accuracy_delta"],
            (value["autonomous_only_correct"] - value["frozen_only_correct"]) / total,
        )
    )


def adjudicate(payload: dict, spec: dict = SETTLE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "ST0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_reset"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("RESET-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = reset_spec_sha256(RESET_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != RESET_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("RESET-1 source identity or verdict changed")

        total = spec["episodes_per_replicate"]
        combined = total * len(spec["replicates"])
        audit = payload["dataset_audit"]
        audits = {row["replicate"]: row for row in audit["replicates"]}
        if set(audits) != set(spec["replicates"]) or len(audits) != len(audit["replicates"]):
            return invalid("dataset replicate roster changed")
        for replicate, row in audits.items():
            if (
                row["episodes"] != total or row["unique_fingerprints"] != total
                or len(row["fingerprint_set_sha256"]) != 64
                or not _balanced(row["target_counts"], spec["values"], total)
                or not _balanced(row["query_position_counts"], spec["queryable_events"], total)
                or not _balanced(row["query_key_counts"], spec["keys"], total)
                or not _balanced(row["query_context_counts"], spec["contexts"], total)
            ):
                return invalid(f"replicate {replicate} dataset balance or uniqueness changed")
        expected_pairs = {
            f"{left}:{right}" for index, left in enumerate(spec["replicates"])
            for right in spec["replicates"][index + 1:]
        }
        if (
            set(audit["cross_replicate_overlap"]) != expected_pairs
            or any(audit["cross_replicate_overlap"].values())
            or audit["combined_unique_fingerprints"] != combined
            or len(audit["source_overlap"]) != len(RESET_SPEC["replicates"]) * len(spec["replicates"])
            or any(audit["source_overlap"].values())
        ):
            return invalid("new independent datasets overlap")

        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")
        thresholds = spec["thresholds"]
        seed_passed, summaries = {}, {}
        for seed, row in rows.items():
            expected_projector = source_results["source_recovery"]["checkpoints"][str(seed)]
            expected_prototypes = source_results["source_recovery"]["prototype_checkpoints"][str(seed)]
            if (
                row["source_checkpoint"] != expected_projector
                or row["prototype_checkpoint"] != expected_prototypes
                or source["checkpoints"][str(seed)] != expected_projector
                or source["prototype_checkpoints"][str(seed)] != expected_prototypes
                or not _verify_projector(row["source_checkpoint"], seed, spec)
                or not _verify_prototypes(row["prototype_checkpoint"], spec)
                or not row["projector_frozen"] or not row["projector_unchanged"]
            ):
                return invalid(f"seed {seed} frozen source changed")

            modes = {item["mode"]: item["updates"] for item in row["modes"]}
            if set(modes) != set(spec["update_modes"]) or len(modes) != len(row["modes"]):
                return invalid(f"seed {seed} mode roster changed")
            by_mode, audit_signatures = {}, {}
            for mode, values in modes.items():
                updates = {item["update_steps"]: item for item in values}
                if set(updates) != set(spec["update_steps"]) or len(updates) != len(values):
                    return invalid(f"seed {seed} mode {mode} update roster changed")
                by_mode[mode] = updates
                for count, item in updates.items():
                    if len(item["replicates"]) != len(spec["replicates"]):
                        return invalid(f"seed {seed} mode {mode} updates {count} replicate count changed")
                    if not _pooled_matches_replicates(item["pooled"], item["replicates"], spec):
                        return invalid(f"seed {seed} mode {mode} updates {count} pooled metrics changed")
                    for value in [*item["replicates"], {
                        "arms": item["pooled"]["arms"], "geometry": item["pooled"]["geometry"]
                    }]:
                        episode_count = total if "replicate" in value else combined
                        if not _geometry_shape(value["geometry"], episode_count):
                            return invalid(f"seed {seed} mode {mode} updates {count} geometry incomplete")
                        for name, metrics in value["arms"].items():
                            if not _metric_shape(metrics, spec["values"]):
                                return invalid(f"seed {seed} mode {mode} updates {count} {name} metrics incomplete")
                    for value in item["replicates"]:
                        replicate = value["replicate"]
                        state = value["state_audit"]
                        update = value["update_audit"]
                        expected_updates = count if mode == "autonomous" else 0
                        unchanged = total if mode == "frozen" or count == 0 else 0
                        if (
                            state["episodes"] != total or state["unique_episode_seeds"] != total
                            or update["requested_updates"] != count
                            or update["performed_updates_minimum"] != expected_updates
                            or update["performed_updates_maximum"] != expected_updates
                            or update["sensory_inputs_minimum"] != 0
                            or update["sensory_inputs_maximum"] != 0
                            or update["unchanged_state_count"] != unchanged
                            or not all(len(update[name]) == 64 for name in (
                                "state_before_sha256", "state_after_sha256", "query_rng_sha256"
                            ))
                        ):
                            return invalid(f"seed {seed} mode {mode} updates {count} state audit changed")
                        audit_signatures.setdefault((count, replicate), []).append((
                            state["episode_seed_sha256"], update["state_before_sha256"],
                            update["query_rng_sha256"],
                        ))
                        if mode == "frozen" and update["state_after_sha256"] != update["state_before_sha256"]:
                            return invalid(f"seed {seed} frozen state changed")
                        if mode == "autonomous" and count == 0 and update["state_after_sha256"] != update["state_before_sha256"]:
                            return invalid(f"seed {seed} zero-update autonomous state changed")
                        integration = value["integration_audit"]
                        for name, expected_calls in spec["expected_transform_calls"].items():
                            call = integration["stable_transform_calls"][name]
                            if call != {
                                "episodes": total, "total": total * expected_calls,
                                "minimum": expected_calls, "maximum": expected_calls,
                            }:
                                return invalid(f"seed {seed} mode {mode} {name} path changed")
                        exact = value["arms"]["exact_three_candidates"]
                        if (
                            exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                            or exact["accuracy"] < thresholds["exact_final_accuracy"]
                            or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                            or value["arms"]["exact_three_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                            or value["arms"]["exact_three_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                        ):
                            return invalid(f"seed {seed} mode {mode} updates {count} control failed")
                    if not _passes(item["pooled"]["arms"]["stable_two_candidates"], thresholds):
                        return invalid(f"seed {seed} mode {mode} updates {count} two-candidate path failed")
            if any(len(set(values)) != 1 for values in audit_signatures.values()):
                return invalid(f"seed {seed} paired starts or question RNG differ")

            frozen_zero = by_mode["frozen"][0]
            frozen_signature = {
                "pooled": frozen_zero["pooled"],
                "replicates": [{"arms": value["arms"], "geometry": value["geometry"]} for value in frozen_zero["replicates"]],
            }
            for count in spec["update_steps"]:
                frozen = by_mode["frozen"][count]
                signature = {
                    "pooled": frozen["pooled"],
                    "replicates": [{"arms": value["arms"], "geometry": value["geometry"]} for value in frozen["replicates"]],
                }
                if signature != frozen_signature:
                    return invalid(f"seed {seed} frozen predictions changed at {count}")
            if by_mode["autonomous"][0]["pooled"] != frozen_zero["pooled"]:
                return invalid(f"seed {seed} zero-update modes differ")

            paired = {item["update_steps"]: item for item in row["paired"]}
            if set(paired) != set(spec["update_steps"]) or len(paired) != len(row["paired"]):
                return invalid(f"seed {seed} paired roster changed")
            for count, value in paired.items():
                active_metrics = by_mode["autonomous"][count]["pooled"]["arms"]["stable_three_candidates"]
                frozen_metrics = by_mode["frozen"][count]["pooled"]["arms"]["stable_three_candidates"]
                if (
                    not _pair_valid(value["final"], combined)
                    or not _pair_valid(value["selection"], combined)
                    or not _close(value["final"]["net_accuracy_delta"], active_metrics["accuracy"] - frozen_metrics["accuracy"])
                    or not _close(value["selection"]["net_accuracy_delta"], active_metrics["selection_accuracy"] - frozen_metrics["selection_accuracy"])
                    or {item["replicate"] for item in value["replicates"]} != set(spec["replicates"])
                    or any(not _pair_valid(item["final"], total) or not _pair_valid(item["selection"], total) for item in value["replicates"])
                ):
                    return invalid(f"seed {seed} updates {count} paired audit changed")
                if count == 0 and (
                    value["final"]["autonomous_only_correct"] or value["final"]["frozen_only_correct"]
                    or value["selection"]["autonomous_only_correct"] or value["selection"]["frozen_only_correct"]
                ):
                    return invalid(f"seed {seed} zero-update paired predictions differ")

            endpoint = paired[8]
            active = by_mode["autonomous"][8]["pooled"]
            frozen = by_mode["frozen"][8]["pooled"]
            improving = sum(
                item["final"]["net_accuracy_delta"] > 0
                and item["selection"]["net_accuracy_delta"] > 0
                for item in endpoint["replicates"]
            )
            passed = (
                _passes(active["arms"]["stable_three_candidates"], thresholds)
                and endpoint["final"]["net_accuracy_delta"] > 0
                and endpoint["selection"]["net_accuracy_delta"] > 0
                and endpoint["final"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                and endpoint["selection"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                and improving >= thresholds["minimum_improving_replicates"]
                and active["geometry"]["target_minus_strongest_wrong_mean"]
                > frozen["geometry"]["target_minus_strongest_wrong_mean"]
            )
            seed_passed[str(seed)] = passed
            summaries[str(seed)] = {
                "passed": passed,
                "autonomous_final_accuracy": active["arms"]["stable_three_candidates"]["accuracy"],
                "frozen_final_accuracy": frozen["arms"]["stable_three_candidates"]["accuracy"],
                "autonomous_selection_accuracy": active["arms"]["stable_three_candidates"]["selection_accuracy"],
                "frozen_selection_accuracy": frozen["arms"]["stable_three_candidates"]["selection_accuracy"],
                "final_exact_two_sided_p": endpoint["final"]["exact_two_sided_p"],
                "selection_exact_two_sided_p": endpoint["selection"]["exact_two_sided_p"],
                "improving_replicates": improving,
                "margin_delta": active["geometry"]["target_minus_strongest_wrong_mean"]
                - frozen["geometry"]["target_minus_strongest_wrong_mean"],
            }
        verdict, reason = _classify(seed_passed)
        return {
            "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
            "spec_sha256": spec_sha256(spec), "seeds": summaries,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/settle_results.json")
    parser.add_argument("--output", default="measurement/settle_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    temporary = Path(args.output).with_name(Path(args.output).name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, Path(args.output))
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
