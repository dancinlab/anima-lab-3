#!/usr/bin/env python3
"""Fail-closed adjudication for RESET-1."""
from __future__ import annotations

import argparse
import json
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
    from measurement.recovery_gate import (
        _balanced,
        _close,
        _geometry_shape,
        _passes,
        _pooled_matches_replicates,
    )
    from measurement.recovery_registry import RECOVERY_SPEC, spec_sha256 as recovery_spec_sha256
    from measurement.reset_registry import RESET_SPEC, spec_sha256
except ModuleNotFoundError:
    from decay_gate import (
        _finite_tree,
        _metric_shape,
        _valid_receipt,
        _verify_projector,
        _verify_prototypes,
    )
    from recovery_gate import (
        _balanced,
        _close,
        _geometry_shape,
        _passes,
        _pooled_matches_replicates,
    )
    from recovery_registry import RECOVERY_SPEC, spec_sha256 as recovery_spec_sha256
    from reset_registry import RESET_SPEC, spec_sha256


def _baseline_signature(item: dict) -> dict:
    return {
        "pooled": item["pooled"],
        "replicates": [
            {
                "replicate": row["replicate"],
                "arms": row["arms"],
                "geometry": row["geometry"],
                "integration_audit": row["integration_audit"],
                "state_audit": row["state_audit"],
            }
            for row in item["replicates"]
        ],
    }


def _classify(recovery: dict, spec: dict = RESET_SPEC) -> tuple[str, str]:
    by_mode = {
        mode: [recovery[mode][str(seed)] for seed in spec["seeds"]]
        for mode in spec["update_modes"]
    }
    all_true = {mode: all(values) for mode, values in by_mode.items()}
    all_false = {mode: not any(values) for mode, values in by_mode.items()}
    if all(all_true.values()):
        return (
            "RS1_AUTONOMOUS_SETTLING",
            "all modes recovered in both seeds, so external sensory input was not required",
        )
    if all_true["varied_sensory"] and all_true["repeated_sensory"] and all_false["autonomous"]:
        return (
            "RS2_SENSORY_FORCING_RESET",
            "both sensory modes recovered in both seeds while autonomous updates did not",
        )
    if all_true["varied_sensory"] and all_false["repeated_sensory"] and all_false["autonomous"]:
        return "RS3_VARIED_INPUT_RESET", "only varied sensory input recovered in both seeds"
    if all(all_false.values()):
        return "RS4_NO_REGISTERED_RECOVERY", "no registered mode recovered in either seed"
    return (
        "RS5_MIXED_MECHANISM",
        "mode or seed recovery did not match a registered single-mechanism pattern",
    )


def adjudicate(payload: dict, spec: dict = RESET_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "RS0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")

        source = payload["source_recovery"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("RECOVERY-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = recovery_spec_sha256(RECOVERY_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != RECOVERY_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or source.get("source_verdict") != spec["source_verdict"]
            or source.get("source_spec_sha256") != source_sha
        ):
            return invalid("RECOVERY-1 source identity or verdict changed")

        total = spec["episodes_per_replicate"]
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
            f"{left}:{right}" for index, left in enumerate(spec["replicates"])
            for right in spec["replicates"][index + 1:]
        }
        if (
            set(audit["cross_replicate_overlap"]) != expected_pairs
            or any(audit["cross_replicate_overlap"].values())
            or audit["combined_unique_fingerprints"] != total * len(spec["replicates"])
            or audit["varied_input_distinct_count_minimum"] != max(spec["update_steps"])
            or audit["varied_input_distinct_count_maximum"] != max(spec["update_steps"])
            or audit["repeated_neutral_word"] != spec["repeated_neutral_word"]
        ):
            return invalid("registered reset input dataset changed")

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

            modes = {item["mode"]: item for item in row["modes"]}
            if set(modes) != set(spec["update_modes"]) or len(modes) != len(row["modes"]):
                return invalid(f"seed {seed} update mode roster changed")
            baselines = []
            seed_digests = {replicate: set() for replicate in spec["replicates"]}
            for mode in spec["update_modes"]:
                updates = {item["update_steps"]: item for item in modes[mode]["updates"]}
                if set(updates) != set(spec["update_steps"]) or len(updates) != len(modes[mode]["updates"]):
                    return invalid(f"seed {seed} mode {mode} update roster changed")
                baselines.append(_baseline_signature(updates[0]))
                judged[str(seed)][mode] = {}
                for count in spec["update_steps"]:
                    item = updates[count]
                    replicate_rows = {value["replicate"]: value for value in item["replicates"]}
                    if set(replicate_rows) != set(spec["replicates"]) or len(replicate_rows) != len(item["replicates"]):
                        return invalid(f"seed {seed} mode {mode} updates {count} replicate roster changed")
                    if set(item["pooled"]["arms"]) != set(spec["arms"]):
                        return invalid(f"seed {seed} mode {mode} updates {count} pooled arm roster changed")
                    if not _geometry_shape(item["pooled"]["geometry"], total * len(spec["replicates"])):
                        return invalid(f"seed {seed} mode {mode} updates {count} pooled geometry incomplete")
                    if not _pooled_matches_replicates(item["pooled"], list(replicate_rows.values()), spec):
                        return invalid(f"seed {seed} mode {mode} updates {count} pooled result mismatch")
                    for replicate in spec["replicates"]:
                        value = replicate_rows[replicate]
                        if set(value["arms"]) != set(spec["arms"]):
                            return invalid(f"seed {seed} mode {mode} replicate {replicate} arm roster changed")
                        state = value["state_audit"]
                        seed_digests[replicate].add(state["episode_seed_sha256"])
                        if (
                            state["episodes"] != total
                            or state["unique_episode_seeds"] != total
                            or len(state["episode_seed_sha256"]) != 64
                            or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
                        ):
                            return invalid(f"seed {seed} mode {mode} replicate {replicate} state stream changed")
                        update = value["update_audit"]
                        sensory = 0 if mode == "autonomous" else count
                        distinct = 0 if mode == "autonomous" or count == 0 else (count if mode == "varied_sensory" else 1)
                        if (
                            update["requested_updates"] != count
                            or update["performed_updates_minimum"] != count
                            or update["performed_updates_maximum"] != count
                            or update["sensory_inputs_minimum"] != sensory
                            or update["sensory_inputs_maximum"] != sensory
                            or update["distinct_sensory_inputs_minimum"] != distinct
                            or update["distinct_sensory_inputs_maximum"] != distinct
                        ):
                            return invalid(f"seed {seed} mode {mode} updates {count} update audit changed")
                        integration = value["integration_audit"]
                        calls = integration["stable_transform_calls"]
                        if set(calls) != set(spec["stable_arms"]):
                            return invalid(f"seed {seed} mode {mode} transform roster changed")
                        for name, expected_calls in spec["expected_transform_calls"].items():
                            call = calls[name]
                            if (
                                call["episodes"] != total or call["total"] != total * expected_calls
                                or call["minimum"] != expected_calls or call["maximum"] != expected_calls
                            ):
                                return invalid(f"seed {seed} mode {mode} {name} transform path changed")
                        if integration["address_width_minimum"] != spec["address_dim"] or integration["address_width_maximum"] != spec["address_dim"]:
                            return invalid(f"seed {seed} mode {mode} address width changed")
                        geometry = value["geometry"]
                        if not _geometry_shape(geometry, total):
                            return invalid(f"seed {seed} mode {mode} geometry incomplete")
                        stable = value["arms"]["stable_three_candidates"]
                        correct = sum(geometry["selection_position_confusion"][index][index] for index in range(3))
                        if geometry["target_rank_counts"][0] != correct or not _close(stable["selection_accuracy"], correct / total):
                            return invalid(f"seed {seed} mode {mode} address ranks disagree")
                        for name, metrics in value["arms"].items():
                            if not _metric_shape(metrics, spec["values"]):
                                return invalid(f"seed {seed} mode {mode} {name} metrics incomplete")
                            if metrics["retrieval_api_match"] != thresholds["retrieval_api_match"]:
                                return invalid(f"seed {seed} mode {mode} memory API mismatch")
                        exact = value["arms"]["exact_three_candidates"]
                        if (
                            exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                            or exact["accuracy"] < thresholds["exact_final_accuracy"]
                            or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                            or value["arms"]["exact_three_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                            or value["arms"]["exact_three_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                        ):
                            return invalid(f"seed {seed} mode {mode} updates {count} control failed")
                    pooled = item["pooled"]["arms"]
                    if not _passes(pooled["stable_two_candidates"], thresholds):
                        return invalid(f"seed {seed} mode {mode} updates {count} two-candidate path failed")
                    main = pooled["stable_three_candidates"]
                    judged[str(seed)][mode][str(count)] = {
                        "passed": _passes(main, thresholds),
                        "selection_accuracy": main["selection_accuracy"],
                        "final_accuracy": main["accuracy"],
                        "minimum_value_recall": min(main["per_value_recall"]),
                        "target_minus_strongest_wrong_mean": item["pooled"]["geometry"]["target_minus_strongest_wrong_mean"],
                    }
                start, end = updates[0], updates[8]
                replicate_direction[str(seed)][mode] = {}
                for replicate in spec["replicates"]:
                    start_metrics = {value["replicate"]: value for value in start["replicates"]}[replicate]["arms"]["stable_three_candidates"]
                    end_metrics = {value["replicate"]: value for value in end["replicates"]}[replicate]["arms"]["stable_three_candidates"]
                    replicate_direction[str(seed)][mode][str(replicate)] = {
                        "final_improved": end_metrics["accuracy"] > start_metrics["accuracy"],
                        "selection_improved": end_metrics["selection_accuracy"] > start_metrics["selection_accuracy"],
                        "final_delta": end_metrics["accuracy"] - start_metrics["accuracy"],
                        "selection_delta": end_metrics["selection_accuracy"] - start_metrics["selection_accuracy"],
                    }
            if any(value != baselines[0] for value in baselines[1:]):
                return invalid(f"seed {seed} zero-update modes did not start identically")
            if any(len(values) != 1 for values in seed_digests.values()) or len({next(iter(values)) for values in seed_digests.values()}) != len(spec["replicates"]):
                return invalid(f"seed {seed} modes or updates did not use fixed independent initial states")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))

    recovery = {mode: {} for mode in spec["update_modes"]}
    summaries = {str(seed): {} for seed in spec["seeds"]}
    for seed in spec["seeds"]:
        for mode in spec["update_modes"]:
            curve = judged[str(seed)][mode]
            start, end = curve["0"], curve["8"]
            final_delta = end["final_accuracy"] - start["final_accuracy"]
            selection_delta = end["selection_accuracy"] - start["selection_accuracy"]
            directions = replicate_direction[str(seed)][mode]
            all_replicates_improved = all(
                value["final_improved"] and value["selection_improved"]
                for value in directions.values()
            )
            recovered = (
                not start["passed"] and end["passed"]
                and final_delta >= thresholds["minimum_recovery_delta"]
                and selection_delta >= thresholds["minimum_recovery_delta"]
                and all_replicates_improved
            )
            recovery[mode][str(seed)] = recovered
            summaries[str(seed)][mode] = {
                "recovered": recovered,
                "final_accuracy_delta_0_to_8": final_delta,
                "selection_accuracy_delta_0_to_8": selection_delta,
                "all_replicates_improved": all_replicates_improved,
            }

    verdict, reason = _classify(recovery, spec)
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "seeds": summaries,
        "curve": judged, "replicate_direction": replicate_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/reset_results.json")
    parser.add_argument("--output", default="measurement/reset_verdict.json")
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
