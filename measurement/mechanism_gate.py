#!/usr/bin/env python3
"""Fail-closed adjudication for MECHANISM-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from measurement.decay_gate import (
    _finite_tree,
    _metric_shape,
    _valid_receipt,
    _verify_projector,
    _verify_prototypes,
)
from measurement.mechanism_registry import MECHANISM_SPEC, spec_sha256
from measurement.recovery_gate import (
    _balanced,
    _close,
    _geometry_shape,
    _passes,
    _pooled_matches_replicates,
)
from measurement.settle_gate import _pair_valid
from measurement.settle_registry import SETTLE_SPEC, spec_sha256 as settle_spec_sha256


def _classify(necessary: dict[str, list[str]]) -> tuple[str, str]:
    sets = [set(values) for values in necessary.values()]
    if all(not values for values in sets):
        return (
            "MC3_NO_SINGLE_COMPONENT_NECESSARY",
            "no single registered component ablation broke settling in both seeds",
        )
    if all(values == sets[0] for values in sets[1:]):
        if len(sets[0]) == 1:
            return (
                "MC1_SINGLE_COMPONENT_NECESSARY",
                "exactly one registered component was necessary in both seeds",
            )
        return (
            "MC2_DISTRIBUTED_COMPONENTS_NECESSARY",
            "multiple registered components were necessary in both seeds",
        )
    return (
        "MC4_SEED_CONDITIONAL_COMPONENT",
        "the necessary component roster differed between seeds",
    )


def adjudicate(payload: dict, spec: dict = MECHANISM_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "MC0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_settle"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("SETTLE-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = settle_spec_sha256(SETTLE_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != SETTLE_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("SETTLE-1 source identity or verdict changed")

        total = spec["episodes_per_replicate"]
        combined = total * len(spec["replicates"])
        audit = payload["dataset_audit"]
        audits = {row["replicate"]: row for row in audit["replicates"]}
        if set(audits) != set(spec["replicates"]) or len(audits) != len(audit["replicates"]):
            return invalid("dataset replicate roster changed")
        for replicate, row in audits.items():
            if (
                row["episodes"] != total
                or row["unique_fingerprints"] != total
                or len(row["fingerprint_set_sha256"]) != 64
                or not _balanced(row["target_counts"], spec["values"], total)
                or not _balanced(row["query_position_counts"], spec["queryable_events"], total)
                or not _balanced(row["query_key_counts"], spec["keys"], total)
                or not _balanced(row["query_context_counts"], spec["contexts"], total)
            ):
                return invalid(f"replicate {replicate} dataset balance or uniqueness changed")
        expected_pairs = {
            f"{left}:{right}"
            for index, left in enumerate(spec["replicates"])
            for right in spec["replicates"][index + 1:]
        }
        if (
            set(audit["cross_replicate_overlap"]) != expected_pairs
            or any(audit["cross_replicate_overlap"].values())
            or audit["combined_unique_fingerprints"] != combined
            or len(audit["source_overlap"]) != len(SETTLE_SPEC["replicates"]) * len(spec["replicates"])
            or any(audit["source_overlap"].values())
        ):
            return invalid("new independent datasets overlap")

        expected_interventions = {row["name"]: row for row in spec["interventions"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")
        thresholds = spec["thresholds"]
        necessary: dict[str, list[str]] = {}
        summaries = {}
        for seed, row in rows.items():
            expected_projector = source_results["source_reset"]["checkpoints"][str(seed)]
            expected_prototypes = source_results["source_reset"]["prototype_checkpoints"][str(seed)]
            if (
                row["source_checkpoint"] != expected_projector
                or row["prototype_checkpoint"] != expected_prototypes
                or source["checkpoints"][str(seed)] != expected_projector
                or source["prototype_checkpoints"][str(seed)] != expected_prototypes
                or not _verify_projector(row["source_checkpoint"], seed, spec)
                or not _verify_prototypes(row["prototype_checkpoint"], spec)
                or not row["projector_frozen"]
                or not row["projector_unchanged"]
            ):
                return invalid(f"seed {seed} frozen source changed")

            arms = {item["name"]: item for item in row["interventions"]}
            if set(arms) != set(expected_interventions) or len(arms) != len(row["interventions"]):
                return invalid(f"seed {seed} intervention roster changed")
            audit_signatures = {}
            for name, item in arms.items():
                registered = expected_interventions[name]
                if item["mode"] != registered["mode"] or item["disabled"] != registered["disabled"]:
                    return invalid(f"seed {seed} intervention {name} changed")
                if len(item["replicates"]) != len(spec["replicates"]):
                    return invalid(f"seed {seed} intervention {name} replicate count changed")
                if not _pooled_matches_replicates(item["pooled"], item["replicates"], spec):
                    return invalid(f"seed {seed} intervention {name} pooled metrics changed")
                for value in [
                    *item["replicates"],
                    {"arms": item["pooled"]["arms"], "geometry": item["pooled"]["geometry"]},
                ]:
                    episode_count = total if "replicate" in value else combined
                    if not _geometry_shape(value["geometry"], episode_count):
                        return invalid(f"seed {seed} intervention {name} geometry incomplete")
                    if any(not _metric_shape(metrics, spec["values"]) for metrics in value["arms"].values()):
                        return invalid(f"seed {seed} intervention {name} metrics incomplete")
                for value in item["replicates"]:
                    replicate = value["replicate"]
                    state = value["state_audit"]
                    update = value["update_audit"]
                    frozen = registered["mode"] == "frozen"
                    expected_updates = 0 if frozen else spec["update_steps"][-1]
                    if (
                        state["episodes"] != total
                        or state["unique_episode_seeds"] != total
                        or update["requested_updates"] != spec["update_steps"][-1]
                        or update["performed_updates_minimum"] != expected_updates
                        or update["performed_updates_maximum"] != expected_updates
                        or update["sensory_inputs_minimum"] != 0
                        or update["sensory_inputs_maximum"] != 0
                        or not all(len(update[field]) == 64 for field in (
                            "state_before_sha256", "state_after_sha256", "query_rng_sha256",
                        ))
                    ):
                        return invalid(f"seed {seed} intervention {name} state audit changed")
                    if frozen and (
                        update["unchanged_state_count"] != total
                        or update["state_after_sha256"] != update["state_before_sha256"]
                    ):
                        return invalid(f"seed {seed} frozen state changed")
                    audit_signatures.setdefault(replicate, []).append((
                        state["episode_seed_sha256"],
                        update["state_before_sha256"],
                        update["query_rng_sha256"],
                    ))
                    integration = value["integration_audit"]
                    for arm_name, expected_calls in spec["expected_transform_calls"].items():
                        call = integration["stable_transform_calls"][arm_name]
                        if call != {
                            "episodes": total,
                            "total": total * expected_calls,
                            "minimum": expected_calls,
                            "maximum": expected_calls,
                        }:
                            return invalid(f"seed {seed} intervention {name} memory path changed")
                    exact = value["arms"]["exact_three_candidates"]
                    if (
                        exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                        or exact["accuracy"] < thresholds["exact_final_accuracy"]
                        or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                        or value["arms"]["exact_three_partner_swap"]["accuracy"]
                        > thresholds["partner_swap_max_accuracy"]
                        or value["arms"]["exact_three_recovered"]["prediction_match"]
                        != thresholds["recovery_prediction_match"]
                    ):
                        return invalid(f"seed {seed} intervention {name} control failed")
                if not _passes(item["pooled"]["arms"]["stable_two_candidates"], thresholds):
                    return invalid(f"seed {seed} intervention {name} two-candidate path failed")
            if any(len(set(values)) != 1 for values in audit_signatures.values()):
                return invalid(f"seed {seed} paired starts or question RNG differ")

            intact = arms["intact"]["pooled"]
            if not _passes(intact["arms"]["stable_three_candidates"], thresholds):
                return invalid(f"seed {seed} intact settling positive control failed")
            comparisons = {item["name"]: item for item in row["comparisons"]}
            if set(comparisons) != set(arms).difference({"intact"}) or len(comparisons) != len(row["comparisons"]):
                return invalid(f"seed {seed} comparison roster changed")
            passed_components, component_rows = [], {}
            for name, comparison in comparisons.items():
                metrics = arms[name]["pooled"]["arms"]["stable_three_candidates"]
                if (
                    not _pair_valid(comparison["final"], combined)
                    or not _pair_valid(comparison["selection"], combined)
                    or not _close(
                        comparison["final"]["net_accuracy_delta"],
                        intact["arms"]["stable_three_candidates"]["accuracy"] - metrics["accuracy"],
                    )
                    or not _close(
                        comparison["selection"]["net_accuracy_delta"],
                        intact["arms"]["stable_three_candidates"]["selection_accuracy"]
                        - metrics["selection_accuracy"],
                    )
                    or {value["replicate"] for value in comparison["replicates"]} != set(spec["replicates"])
                    or any(
                        not _pair_valid(value["final"], total)
                        or not _pair_valid(value["selection"], total)
                        for value in comparison["replicates"]
                    )
                ):
                    return invalid(f"seed {seed} comparison {name} changed")
                worsening = sum(
                    value["final"]["net_accuracy_delta"] > 0
                    and value["selection"]["net_accuracy_delta"] > 0
                    for value in comparison["replicates"]
                )
                margin_drop = (
                    intact["geometry"]["target_minus_strongest_wrong_mean"]
                    - arms[name]["pooled"]["geometry"]["target_minus_strongest_wrong_mean"]
                )
                is_necessary = (
                    name != "frozen"
                    and comparison["final"]["net_accuracy_delta"] >= thresholds["minimum_accuracy_drop"]
                    and comparison["selection"]["net_accuracy_delta"] >= thresholds["minimum_accuracy_drop"]
                    and comparison["final"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                    and comparison["selection"]["exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                    and worsening >= thresholds["minimum_worsening_replicates"]
                    and margin_drop > 0
                )
                if is_necessary:
                    passed_components.append(expected_interventions[name]["disabled"][0])
                component_rows[name] = {
                    "necessary": is_necessary,
                    "final_accuracy": metrics["accuracy"],
                    "selection_accuracy": metrics["selection_accuracy"],
                    "final_drop": comparison["final"]["net_accuracy_delta"],
                    "selection_drop": comparison["selection"]["net_accuracy_delta"],
                    "final_exact_two_sided_p": comparison["final"]["exact_two_sided_p"],
                    "selection_exact_two_sided_p": comparison["selection"]["exact_two_sided_p"],
                    "worsening_replicates": worsening,
                    "margin_drop": margin_drop,
                }
            frozen = component_rows["frozen"]
            if not (
                frozen["final_drop"] > 0
                and frozen["selection_drop"] > 0
                and frozen["final_exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                and frozen["selection_exact_two_sided_p"] <= thresholds["paired_exact_p_maximum"]
                and frozen["worsening_replicates"] >= thresholds["minimum_worsening_replicates"]
                and frozen["margin_drop"] > 0
            ):
                return invalid(f"seed {seed} frozen causal control failed")
            necessary[str(seed)] = sorted(passed_components)
            summaries[str(seed)] = {
                "intact_final_accuracy": intact["arms"]["stable_three_candidates"]["accuracy"],
                "intact_selection_accuracy": intact["arms"]["stable_three_candidates"]["selection_accuracy"],
                "necessary_components": sorted(passed_components),
                "interventions": component_rows,
            }
        verdict, reason = _classify(necessary)
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "necessary_components": necessary,
            "seeds": summaries,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/mechanism_results.json")
    parser.add_argument("--output", default="measurement/mechanism_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    temporary = Path(args.output).with_name(Path(args.output).name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, Path(args.output))
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
