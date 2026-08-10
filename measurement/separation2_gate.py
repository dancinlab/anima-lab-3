#!/usr/bin/env python3
"""Fail-closed adjudication for SEPARATION-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from measurement.canonical2_gate import _checkpoint_valid, adjudicate as adjudicate_canonical2
    from measurement.canonical2_registry import CANONICAL2_SPEC, spec_sha256 as canonical2_spec_sha256
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
    from measurement.projector_registry import evaluation_name
    from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.canonical2_gate import _checkpoint_valid, adjudicate as adjudicate_canonical2
    from measurement.canonical2_registry import CANONICAL2_SPEC, spec_sha256 as canonical2_spec_sha256
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
    from measurement.projector_registry import evaluation_name
    from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256


def _balanced(counts: dict, categories: int, total: int) -> bool:
    expected = total // categories
    return counts == {str(index): expected for index in range(categories)}


def _passes(metrics: dict, spec: dict) -> bool:
    thresholds = spec["thresholds"]
    return (
        metrics["selection_accuracy"] >= thresholds["similar_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["similar_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["similar_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = SEPARATION2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "SP0_INVALID",
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

        source = payload["source_canonical2"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CANONICAL-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = canonical2_spec_sha256(CANONICAL2_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CANONICAL2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_canonical2(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("checkpoint") != source_results.get("checkpoint")
            or source.get("prototype_checkpoints")
            != source_results.get("source_canonical", {}).get("prototype_checkpoints")
            or not _checkpoint_valid(source["checkpoint"], source_results, CANONICAL2_SPEC)
        ):
            return invalid("registered CANONICAL-2 source identity changed")
        prototype_seeds = {row["prototype_seed"] for row in spec["evaluation_combinations"]}
        if set(source["prototype_checkpoints"]) != {str(seed) for seed in prototype_seeds}:
            return invalid("prototype checkpoint roster changed")
        for seed in prototype_seeds:
            if not _verify_prototypes(source["prototype_checkpoints"][str(seed)], spec):
                return invalid(f"prototype seed {seed} changed")

        audit = payload["dataset_audit"]
        total = spec["eval_episodes"]
        if (
            audit["episodes"] != total
            or audit["unique_fingerprints"] != total
            or len(audit["fingerprint_set_sha256"]) != 64
            or not _balanced(audit["target_counts"], spec["values"], total)
            or not _balanced(audit["query_position_counts"], spec["events_per_episode"], total)
            or not _balanced(audit["shared_key_counts"], spec["keys"], total)
            or not _balanced(audit["query_context_counts"], spec["contexts"], total)
        ):
            return invalid("registered episode balance or uniqueness changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        thresholds = spec["thresholds"]
        judged, signatures = {}, []
        for name, row in evaluations.items():
            expected = registered[name]
            if (
                row["prototype_seed"] != expected["prototype_seed"]
                or row["engine_seed"] != expected["engine_seed"]
                or row["prototype_checkpoint"]
                != source["prototype_checkpoints"][str(expected["prototype_seed"])]
            ):
                return invalid(f"evaluation {name} identity changed")
            state = row["state_audit"]
            update = row["update_audit"]
            integration = row["integration_audit"]
            calls = spec["expected_stable_transform_calls_per_episode"]
            expected_call_audit = {
                "episodes": total,
                "total": total * calls,
                "minimum": calls,
                "maximum": calls,
            }
            if (
                state["episodes"] != total
                or state["unique_episode_seeds"] != total
                or len(state["episode_seed_sha256"]) != 64
                or not spec["minimum_cells"] <= state["minimum_cells"]
                <= state["maximum_cells"] <= spec["maximum_cells"]
                or update["requested_updates"] != spec["settling_updates"]
                or update["performed_updates_minimum"] != spec["settling_updates"]
                or update["performed_updates_maximum"] != spec["settling_updates"]
                or update["disabled"] != spec["pre_query_dynamics_ablation"]
                or any(len(update[key]) != 64 for key in (
                    "state_before_sha256", "state_after_sha256", "query_rng_sha256"
                ))
                or integration["similar_transform_calls"] != expected_call_audit
                or integration["distinct_transform_calls"] != expected_call_audit
                or integration["expected_calls_per_episode"] != calls
                or integration["address_width_minimum"] != spec["address_dim"]
                or integration["address_width_maximum"] != spec["address_dim"]
                or integration["projector_frozen"] is not True
                or integration["projector_unchanged"] is not True
            ):
                return invalid(f"evaluation {name} execution changed")
            signatures.append((
                row["engine_seed"], state["episode_seed_sha256"],
                update["state_before_sha256"], update["state_after_sha256"],
                update["query_rng_sha256"],
            ))
            arms = row["arms"]
            if set(arms) != set(spec["arms"]):
                return invalid(f"evaluation {name} arm roster changed")
            for arm_name in spec["arms"]:
                if (
                    not _metric_shape(arms[arm_name], spec["values"])
                    or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                ):
                    return invalid(f"evaluation {name} {arm_name} metrics changed")
            distinct = arms["stable_distinct_key_control"]
            exact = arms["exact_context_key_control"]
            if (
                distinct["selection_accuracy"] < thresholds["distinct_selection_accuracy"]
                or distinct["accuracy"] < thresholds["distinct_final_accuracy"]
                or min(distinct["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or distinct["correct_content_accuracy"] < thresholds["content_readout_accuracy"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or arms["context_removed_control"]["accuracy"]
                > thresholds["context_removed_max_accuracy"]
                or arms["exact_context_key_partner_swap"]["accuracy"]
                > thresholds["partner_swap_max_accuracy"]
                or arms["exact_context_key_recovered"]["prediction_match"]
                != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"evaluation {name} positive, negative, or recovery control failed")
            stable = arms["stable_similar_normal"]
            raw = arms["raw_similar_normal"]
            judged[name] = {
                "prototype_seed": row["prototype_seed"],
                "engine_seed": row["engine_seed"],
                "stable_passed": _passes(stable, spec),
                "stable_selection_accuracy": stable["selection_accuracy"],
                "stable_final_accuracy": stable["accuracy"],
                "stable_minimum_value_recall": min(stable["per_value_recall"]),
                "stable_content_accuracy": stable["correct_content_accuracy"],
                "raw_passed": _passes(raw, spec),
                "raw_selection_accuracy": raw["selection_accuracy"],
                "raw_final_accuracy": raw["accuracy"],
                "distinct_selection_accuracy": distinct["selection_accuracy"],
                "distinct_final_accuracy": distinct["accuracy"],
                "exact_final_accuracy": exact["accuracy"],
                "context_removed_accuracy": arms["context_removed_control"]["accuracy"],
                "partner_swap_accuracy": arms["exact_context_key_partner_swap"]["accuracy"],
            }
        for engine_seed in {row["engine_seed"] for row in spec["evaluation_combinations"]}:
            paired = {signature[1:] for signature in signatures if signature[0] == engine_seed}
            if len(paired) != 1:
                return invalid(f"engine seed {engine_seed} did not keep paired state streams")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    rows = list(judged.values())
    selection_pass = all(
        row["stable_selection_accuracy"] >= spec["thresholds"]["similar_selection_accuracy"]
        for row in rows
    )
    stable_pass = all(row["stable_passed"] for row in rows)
    raw_pass = all(row["raw_passed"] for row in rows)
    if selection_pass and not stable_pass:
        verdict = "SP4_VALUE_READOUT_LOSS"
        reason = "canonical addresses selected the right episode, but value readout did not support balanced behavior"
    elif stable_pass:
        verdict = "SP1_SIMILAR_EPISODES_SEPARATED_NOT_UNIQUE"
        reason = "the canonical memory path separated four same-key episodes in every evaluation"
    elif raw_pass:
        verdict = "SP2_CANONICAL_KEY_COLLISION"
        reason = "raw states separated the episodes, but canonical key fitting removed their context"
    else:
        verdict = "SP3_CONTEXT_NOT_IN_KEY_STATE"
        reason = "validated controls passed, but neither canonical nor raw key-time states retrieved same-key contexts"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "evaluations": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/separation2_results.json")
    parser.add_argument("--output", default="measurement/separation2_verdict.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    verdict = adjudicate(payload)
    path = Path(args.output)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
