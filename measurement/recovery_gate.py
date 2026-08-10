#!/usr/bin/env python3
"""Fail-closed adjudication for RECOVERY-1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from measurement.decay_gate import (
        _finite_tree,
        _metric_shape,
        _valid_receipt,
        _verify_projector,
        _verify_prototypes,
    )
    from measurement.decay_registry import DECAY_SPEC, spec_sha256 as decay_spec_sha256
    from measurement.recovery_registry import RECOVERY_SPEC, spec_sha256
except ModuleNotFoundError:
    from decay_gate import (
        _finite_tree,
        _metric_shape,
        _valid_receipt,
        _verify_projector,
        _verify_prototypes,
    )
    from decay_registry import DECAY_SPEC, spec_sha256 as decay_spec_sha256
    from recovery_registry import RECOVERY_SPEC, spec_sha256


def _balanced(counts: dict, categories: int, total: int) -> bool:
    expected = total // categories
    return counts == {str(index): expected for index in range(categories)}


def _passes(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["selection_accuracy"] >= thresholds["stable_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["stable_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["stable_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _geometry_shape(value: dict, total: int) -> bool:
    confusion = value["selection_position_confusion"]
    return (
        value["episodes"] == total
        and len(value["target_rank_counts"]) == 3
        and sum(value["target_rank_counts"]) == total
        and len(confusion) == 3
        and all(len(row) == 3 for row in confusion)
        and sum(sum(row) for row in confusion) == total
        and all(name in value for name in (
            "target_similarity_mean", "strongest_wrong_similarity_mean",
            "third_candidate_similarity_mean", "target_minus_strongest_wrong_mean",
            "target_minus_third_candidate_mean",
        ))
    )


def _pooled_matches_replicates(pooled: dict, replicates: list[dict], spec: dict) -> bool:
    count = len(replicates)
    for arm in spec["arms"]:
        combined = pooled["arms"][arm]
        rows = [row["arms"][arm] for row in replicates]
        for field in (
            "accuracy", "selection_accuracy", "correct_content_accuracy",
            "retrieval_api_match", "key_margin_mean",
        ):
            if not _close(combined[field], sum(row[field] for row in rows) / count):
                return False
        expected_confusion = [
            [sum(row["confusion_matrix"][i][j] for row in rows) for j in range(spec["values"])]
            for i in range(spec["values"])
        ]
        if combined["confusion_matrix"] != expected_confusion:
            return False
    geometry = pooled["geometry"]
    rows = [row["geometry"] for row in replicates]
    if geometry["target_rank_counts"] != [
        sum(row["target_rank_counts"][rank] for row in rows) for rank in range(3)
    ]:
        return False
    expected_selection = [
        [sum(row["selection_position_confusion"][i][j] for row in rows) for j in range(3)]
        for i in range(3)
    ]
    if geometry["selection_position_confusion"] != expected_selection:
        return False
    for field in (
        "target_similarity_mean", "strongest_wrong_similarity_mean",
        "third_candidate_similarity_mean", "target_minus_strongest_wrong_mean",
        "target_minus_third_candidate_mean",
    ):
        if not _close(geometry[field], sum(row[field] for row in rows) / count):
            return False
    return True


def adjudicate(payload: dict, spec: dict = RECOVERY_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "RC0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_decay"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("DECAY-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = decay_spec_sha256(DECAY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != DECAY_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("DECAY-1 source identity or verdict changed")

        total = spec["episodes_per_replicate"]
        audits = {row["replicate"]: row for row in payload["dataset_audit"]["replicates"]}
        if set(audits) != set(spec["replicates"]) or len(audits) != len(payload["dataset_audit"]["replicates"]):
            return invalid("dataset replicate roster changed")
        for replicate, audit in audits.items():
            if (
                audit["episodes"] != total
                or audit["unique_fingerprints"] != total
                or len(audit["fingerprint_set_sha256"]) != 64
                or not _balanced(audit["target_counts"], spec["values"], total)
                or not _balanced(audit["query_position_counts"], spec["queryable_events"], total)
                or not _balanced(audit["query_key_counts"], spec["keys"], total)
                or not _balanced(audit["query_context_counts"], spec["contexts"], total)
            ):
                return invalid(f"replicate {replicate} dataset balance or uniqueness changed")
        expected_pairs = {
            f"{left}:{right}"
            for index, left in enumerate(spec["replicates"])
            for right in spec["replicates"][index + 1:]
        }
        overlaps = payload["dataset_audit"]["cross_replicate_overlap"]
        if (
            set(overlaps) != expected_pairs
            or any(overlaps.values())
            or payload["dataset_audit"]["combined_unique_fingerprints"]
            != total * len(spec["replicates"])
        ):
            return invalid("independent dataset replicates overlap")

        source_rows = {row["seed"]: row for row in source_results["seeds"]}
        rows = {row["seed"]: row for row in payload["seeds"]}
        if set(source_rows) != set(spec["seeds"]) or set(rows) != set(spec["seeds"]) or len(rows) != len(payload["seeds"]):
            return invalid("registered seeds are missing or duplicated")

        thresholds = spec["thresholds"]
        judged = {str(seed): {} for seed in spec["seeds"]}
        replicate_direction = {str(seed): {} for seed in spec["seeds"]}
        for seed in spec["seeds"]:
            row = rows[seed]
            source_row = source_rows[seed]
            projector_receipt = row["source_checkpoint"]
            prototype_receipt = row["prototype_checkpoint"]
            if (
                projector_receipt != source["checkpoints"].get(str(seed))
                or projector_receipt != source_row["source_checkpoint"]
                or prototype_receipt != source["prototype_checkpoints"].get(str(seed))
                or prototype_receipt != source_row["prototype_checkpoint"]
                or not _verify_projector(projector_receipt, seed, spec)
                or not _verify_prototypes(prototype_receipt, spec)
            ):
                return invalid(f"seed {seed} source checkpoint changed")
            if not row["projector_frozen"] or not row["projector_unchanged"]:
                return invalid(f"seed {seed} stable projector changed during evaluation")

            delays = {item["distractor_steps"]: item for item in row["delays"]}
            if set(delays) != set(spec["distractor_steps"]) or len(delays) != len(row["delays"]):
                return invalid(f"seed {seed} delay roster changed")
            seed_digests = {replicate: set() for replicate in spec["replicates"]}
            for delay in spec["distractor_steps"]:
                item = delays[delay]
                replicate_rows = {value["replicate"]: value for value in item["replicates"]}
                if set(replicate_rows) != set(spec["replicates"]) or len(replicate_rows) != len(item["replicates"]):
                    return invalid(f"seed {seed} delay {delay} replicate roster changed")
                if set(item["pooled"]["arms"]) != set(spec["arms"]):
                    return invalid(f"seed {seed} delay {delay} pooled arm roster changed")
                if not _geometry_shape(item["pooled"]["geometry"], total * len(spec["replicates"])):
                    return invalid(f"seed {seed} delay {delay} pooled geometry is incomplete")
                if not _pooled_matches_replicates(item["pooled"], list(replicate_rows.values()), spec):
                    return invalid(f"seed {seed} delay {delay} pooled result does not match replicates")
                for replicate in spec["replicates"]:
                    value = replicate_rows[replicate]
                    if set(value["arms"]) != set(spec["arms"]):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} arm roster changed")
                    state = value["state_audit"]
                    seed_digests[replicate].add(state["episode_seed_sha256"])
                    if (
                        state["episodes"] != total
                        or state["unique_episode_seeds"] != total
                        or len(state["episode_seed_sha256"]) != 64
                        or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
                    ):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} state stream changed")
                    integration = value["integration_audit"]
                    calls = integration["stable_transform_calls"]
                    if set(calls) != set(spec["stable_arms"]):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} transform roster changed")
                    for name, expected_calls in spec["expected_transform_calls"].items():
                        call = calls[name]
                        if (
                            call["episodes"] != total
                            or call["total"] != total * expected_calls
                            or call["minimum"] != expected_calls
                            or call["maximum"] != expected_calls
                        ):
                            return invalid(f"seed {seed} replicate {replicate} delay {delay} {name} transform path changed")
                    if integration["address_width_minimum"] != spec["address_dim"] or integration["address_width_maximum"] != spec["address_dim"]:
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} address width changed")
                    geometry = value["geometry"]
                    if not _geometry_shape(geometry, total):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} geometry is incomplete")
                    stable = value["arms"]["stable_three_candidates"]
                    selection_correct = sum(geometry["selection_position_confusion"][index][index] for index in range(3))
                    if geometry["target_rank_counts"][0] != selection_correct or not _close(stable["selection_accuracy"], selection_correct / total):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} address ranks disagree")
                    for name, metrics in value["arms"].items():
                        if not _metric_shape(metrics, spec["values"]):
                            return invalid(f"seed {seed} replicate {replicate} delay {delay} {name} metrics are incomplete")
                        if metrics["retrieval_api_match"] != thresholds["retrieval_api_match"]:
                            return invalid(f"seed {seed} replicate {replicate} delay {delay} memory API mismatch")
                    exact = value["arms"]["exact_three_candidates"]
                    if (
                        exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                        or exact["accuracy"] < thresholds["exact_final_accuracy"]
                        or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                        or value["arms"]["exact_three_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                        or value["arms"]["exact_three_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                    ):
                        return invalid(f"seed {seed} replicate {replicate} delay {delay} control failed")
                pooled = item["pooled"]["arms"]
                if not _passes(pooled["stable_two_candidates"], thresholds):
                    return invalid(f"seed {seed} delay {delay} two-candidate positive path failed")
                main = pooled["stable_three_candidates"]
                judged[str(seed)][str(delay)] = {
                    "passed": _passes(main, thresholds),
                    "selection_accuracy": main["selection_accuracy"],
                    "final_accuracy": main["accuracy"],
                    "minimum_value_recall": min(main["per_value_recall"]),
                    "target_rank_counts": item["pooled"]["geometry"]["target_rank_counts"],
                    "target_minus_strongest_wrong_mean": item["pooled"]["geometry"]["target_minus_strongest_wrong_mean"],
                    "target_minus_third_candidate_mean": item["pooled"]["geometry"]["target_minus_third_candidate_mean"],
                }
            if any(len(values) != 1 for values in seed_digests.values()) or len({next(iter(values)) for values in seed_digests.values()}) != len(spec["replicates"]):
                return invalid(f"seed {seed} delays or replicates did not use independent fixed initial states")
            start = delays[0]
            end = delays[8]
            for replicate in spec["replicates"]:
                start_metrics = {row["replicate"]: row for row in start["replicates"]}[replicate]["arms"]["stable_three_candidates"]
                end_metrics = {row["replicate"]: row for row in end["replicates"]}[replicate]["arms"]["stable_three_candidates"]
                replicate_direction[str(seed)][str(replicate)] = {
                    "final_improved": end_metrics["accuracy"] > start_metrics["accuracy"],
                    "selection_improved": end_metrics["selection_accuracy"] > start_metrics["selection_accuracy"],
                    "final_delta": end_metrics["accuracy"] - start_metrics["accuracy"],
                    "selection_delta": end_metrics["selection_accuracy"] - start_metrics["selection_accuracy"],
                }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))

    summaries = {}
    ordered_all = True
    base_recovery_all = True
    delay_loss = False
    for seed in spec["seeds"]:
        curve = judged[str(seed)]
        start, end = curve["0"], curve["8"]
        final_delta = end["final_accuracy"] - start["final_accuracy"]
        selection_delta = end["selection_accuracy"] - start["selection_accuracy"]
        passes = [curve[str(delay)]["passed"] for delay in spec["distractor_steps"]]
        first_pass = next((delay for delay, passed in zip(spec["distractor_steps"], passes) if passed), None)
        monotonic_after_pass = first_pass is not None and all(
            curve[str(delay)]["passed"] for delay in spec["distractor_steps"] if delay >= first_pass
        )
        directions = replicate_direction[str(seed)]
        all_replicates_improved = all(
            value["final_improved"] and value["selection_improved"]
            for value in directions.values()
        )
        base_recovery = (
            not start["passed"] and end["passed"]
            and final_delta >= thresholds["minimum_recovery_delta"]
            and selection_delta >= thresholds["minimum_recovery_delta"]
        )
        ordered = base_recovery and monotonic_after_pass and all_replicates_improved
        ordered_all &= ordered
        base_recovery_all &= base_recovery
        delay_loss |= start["passed"] and any(not value for value in passes[1:]) and not end["passed"]
        summaries[str(seed)] = {
            "first_pass_delay": first_pass,
            "final_accuracy_delta_0_to_8": final_delta,
            "selection_accuracy_delta_0_to_8": selection_delta,
            "monotonic_after_first_pass": monotonic_after_pass,
            "all_replicates_improved": all_replicates_improved,
            "ordered_recovery": ordered,
        }

    if delay_loss:
        verdict = "RC4_DELAY_LOSS"
        reason = "a seed passed without delay and then failed without recovering by delay eight"
    elif ordered_all:
        verdict = "RC1_ORDERED_RECOVERY_REPRODUCED"
        reason = "both seeds and every independent replicate recovered in one direction without a later pooled failure"
    elif base_recovery_all:
        verdict = "RC2_RECOVERY_REPRODUCED_MIXED"
        reason = "both seeds recovered by delay eight but an intermediate boundary or replicate direction was mixed"
    else:
        verdict = "RC3_RECOVERY_NOT_REPRODUCED"
        reason = "the registered failure-to-recovery shape did not repeat in both seeds"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "seeds": summaries,
        "curve": judged, "replicate_direction": replicate_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/recovery_results.json")
    parser.add_argument("--output", default="measurement/recovery_verdict.json")
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
