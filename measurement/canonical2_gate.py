#!/usr/bin/env python3
"""Fail-closed adjudication for CANONICAL-2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from key_stability import DEFAULT_STABLE_KEY_FIT_METHOD
from measurement.canonical2_registry import CANONICAL2_SPEC, spec_sha256
from measurement.canonical_gate import adjudicate as adjudicate_canonical
from measurement.canonical_registry import CANONICAL_SPEC, spec_sha256 as canonical_spec_sha256
from measurement.capacity2_gate import _passes
from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt, _verify_prototypes
from measurement.projector_registry import evaluation_name
from measurement.seedmap_registry import SEEDMAP_SPEC


def _checkpoint_valid(receipt: dict, payload: dict, spec: dict) -> bool:
    path = Path(receipt["path"])
    if not path.is_file() or sha256_file(path) != receipt["sha256"]:
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "projection.weight": (spec["address_dim"], spec["input_dim"]),
        "projection.bias": (spec["address_dim"],),
        "prototypes": (spec["keys"], spec["address_dim"]),
    }
    state = checkpoint.get("projector", {})
    return (
        checkpoint.get("experiment") == spec["experiment"]
        and checkpoint.get("spec_sha256") == spec_sha256(spec)
        and checkpoint.get("fit_method") == spec["fit_method"]
        and checkpoint.get("calibration_seeds") == spec["calibration_seeds"]
        and checkpoint.get("model_class") == spec["model_class"]
        and checkpoint.get("source_audits") == payload["source_audits"]
        and checkpoint.get("fit_audit") == payload["fit_audit"]
        and checkpoint.get("canonical1_pooled_match") == payload["canonical1_pooled_match"]
        and set(state) == set(expected)
        and all(tuple(state[name].shape) == shape and torch.isfinite(state[name]).all() for name, shape in expected.items())
    )


def adjudicate(payload: dict, spec: dict = CANONICAL2_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CI0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if payload["experiment"] != spec["experiment"]:
            return invalid("experiment identity changed")
        if payload["spec"] != spec or payload["spec_sha256"] != spec_sha256(spec):
            return invalid("registered spec changed")
        if spec["fit_method"] != DEFAULT_STABLE_KEY_FIT_METHOD:
            return invalid("public default fit method changed")
        if not _finite_tree(payload):
            return invalid("result contains a non-finite number")
        source = payload["source_canonical"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("CANONICAL-1 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        expected_sha = canonical_spec_sha256(CANONICAL_SPEC)
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != CANONICAL_SPEC
            or source_results.get("spec_sha256") != expected_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != expected_sha
            or adjudicate_canonical(source_results) != source_verdict
            or source["source_spec_sha256"] != expected_sha
        ):
            return invalid("registered CANONICAL-1 source identity changed")
        pooled = next(row for row in source_results["canonical_projectors"] if row["name"] == "pooled")
        if source["pooled_checkpoint"] != pooled["checkpoint"]:
            return invalid("CANONICAL-1 pooled checkpoint changed")
        if source["prototype_checkpoints"] != source_results["source_training"]["prototype_checkpoints"]:
            return invalid("prototype checkpoint roster changed")
        for seed in spec["calibration_seeds"]:
            if not _verify_prototypes(source["prototype_checkpoints"][str(seed)], SEEDMAP_SPEC):
                return invalid(f"prototype seed {seed} changed")
        if payload["source_audits"] != pooled["source_audits"]:
            return invalid("pooled calibration stream changed")
        fit = payload["fit_audit"]
        if (
            fit["method"] != CANONICAL_SPEC["method"]
            or fit["examples"] != spec["calibration_episodes"] * len(spec["calibration_seeds"]) * 3
            or fit["input_dim"] != spec["input_dim"]
            or fit["address_dim"] != spec["address_dim"]
            or fit["keys"] != spec["keys"]
            or fit["weight_regularization"] != spec["weight_decay"]
            or fit["bias_regularized"] is not False
            or fit["design_rank"] <= 0
            or len(fit["label_sha256"]) != 64
            or payload["canonical1_pooled_match"] is not True
            or not payload["projector_frozen"]
            or not payload["projector_unchanged"]
            or not _checkpoint_valid(payload["checkpoint"], payload, spec)
        ):
            return invalid("integrated canonical fit or checkpoint changed")
        legacy = payload["legacy_compatibility"]
        if (
            legacy["checkpoint_loaded"] is not True
            or legacy["default_matches_explicit"] is not True
            or not _valid_receipt(legacy["checkpoint"])
        ):
            return invalid("legacy call or checkpoint compatibility failed")

        audits = payload["dataset_audit"]
        if set(audits) != {str(count) for count in spec["event_counts"]}:
            return invalid("event-count dataset roster changed")
        if audits[str(spec["event_counts"][-1])] != source_results["capacity_dataset_audit"]:
            return invalid("event-four source episodes changed")
        for count in spec["event_counts"]:
            audit = audits[str(count)]
            if (
                audit["episodes"] != spec["eval_episodes"]
                or audit["unique_fingerprints"] != spec["eval_episodes"]
                or len(audit["fingerprint_set_sha256"]) != 64
                or sum(audit["target_counts"].values()) != spec["eval_episodes"]
                or sum(audit["query_position_counts"].values()) != spec["eval_episodes"]
            ):
                return invalid(f"event count {count} dataset changed")

        expected_evals = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        count_rows = {row["event_count"]: row for row in payload["counts"]}
        if set(count_rows) != set(spec["event_counts"]) or len(count_rows) != len(payload["counts"]):
            return invalid("event-count result roster changed")
        source_event4 = {row["name"]: row["result"] for row in pooled["evaluations"]}
        thresholds = spec["thresholds"]
        count_passes, summaries, block_rows = {}, {}, {}
        for count, row in count_rows.items():
            evaluations = {item["name"]: item for item in row["evaluations"]}
            if set(evaluations) != set(expected_evals) or len(evaluations) != len(row["evaluations"]):
                return invalid(f"event count {count} evaluation roster changed")
            passed_rows, eval_summary, signatures = [], {}, []
            for name, item in evaluations.items():
                registered = expected_evals[name]
                if item["prototype_seed"] != registered["prototype_seed"] or item["engine_seed"] != registered["engine_seed"]:
                    return invalid(f"event count {count} evaluation {name} identity changed")
                result = item["result"]
                total = spec["eval_episodes"]
                state, update, integration = result["state_audit"], result["update_audit"], result["integration_audit"]
                calls = count + 1
                if (
                    result["event_count"] != count
                    or set(result["arms"]) != set(spec["arms"])
                    or state["episodes"] != total
                    or state["unique_episode_seeds"] != total
                    or len(state["episode_seed_sha256"]) != 64
                    or not spec["minimum_cells"] <= state["minimum_cells"] <= state["maximum_cells"] <= spec["maximum_cells"]
                    or update["requested_updates"] != spec["settling_updates"]
                    or update["performed_updates_minimum"] != spec["settling_updates"]
                    or update["performed_updates_maximum"] != spec["settling_updates"]
                    or update["disabled"] != []
                    or integration["stable_transform_calls"] != {
                        "episodes": total, "total": total * calls, "minimum": calls, "maximum": calls,
                    }
                    or integration["address_width_minimum"] != spec["address_dim"]
                    or integration["address_width_maximum"] != spec["address_dim"]
                ):
                    return invalid(f"event count {count} evaluation {name} execution changed")
                signatures.append((item["engine_seed"], state["episode_seed_sha256"], update["state_before_sha256"], update["query_rng_sha256"]))
                arms = result["arms"]
                if any(
                    not _metric_shape(arms[arm], spec["values"])
                    or arms[arm]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                    for arm in spec["arms"]
                ):
                    return invalid(f"event count {count} evaluation {name} metrics changed")
                exact = arms["exact_key_control"]
                if (
                    exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                    or exact["accuracy"] < thresholds["exact_final_accuracy"]
                    or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                    or arms["exact_key_partner_swap"]["accuracy"] > thresholds["partner_swap_max_accuracy"]
                    or arms["exact_key_recovered"]["prediction_match"] != thresholds["recovery_prediction_match"]
                ):
                    return invalid(f"event count {count} evaluation {name} control failed")
                if count == spec["event_counts"][-1] and result != source_event4[name]:
                    return invalid(f"event count {count} evaluation {name} did not replay CANONICAL-1")
                stable = arms["stable_distinct_normal"]
                raw = arms["raw_distinct_control"]
                passed = _passes(stable, thresholds)
                passed_rows.append(passed)
                eval_summary[name] = {
                    "passed": passed,
                    "selection_accuracy": stable["selection_accuracy"],
                    "content_accuracy": stable["correct_content_accuracy"],
                    "final_accuracy": stable["accuracy"],
                    "minimum_value_recall": min(stable["per_value_recall"]),
                }
                if count == spec["event_counts"][-1]:
                    block_rows[name] = {
                        "selection_drop": stable["selection_accuracy"] - raw["selection_accuracy"],
                        "final_drop": stable["accuracy"] - raw["accuracy"],
                    }
            for engine_seed in spec["calibration_seeds"]:
                if len({value[1:] for value in signatures if value[0] == engine_seed}) != 1:
                    return invalid(f"event count {count} changed paired engine state")
            count_passes[str(count)] = all(passed_rows)
            summaries[str(count)] = eval_summary
        block_causal = all(
            row["selection_drop"] >= spec["minimum_event4_block_delta"]
            and row["final_drop"] >= spec["minimum_event4_block_delta"]
            for row in block_rows.values()
        )
        ordered = [count_passes[str(count)] for count in spec["event_counts"]]
        if all(ordered) and block_causal:
            verdict, reason = "CI1_CANONICAL_DEFAULT_INTEGRATED", "the canonical default passed every count and its event-four block control"
        elif all(ordered):
            verdict, reason = "CI3_TRANSFORM_NOT_CAUSAL", "all counts passed but the event-four transform block was too small"
        elif any(ordered) and not any(ordered[index] and not all(ordered[:index + 1]) for index in range(len(ordered))):
            verdict, reason = "CI2_PARTIAL_COUNT_BOUNDARY", "only a consecutive prefix of event counts passed"
        else:
            verdict, reason = "CI4_INTEGRATION_REGRESSION", "the integrated default did not preserve the registered address behavior"
        return {
            "experiment": spec["experiment"],
            "verdict": verdict,
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
            "count_passes": count_passes,
            "counts": summaries,
            "event4_block": block_rows,
            "block_causal": block_causal,
            "legacy_compatibility": legacy,
        }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        return invalid(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/canonical2_results.json")
    parser.add_argument("--output", default="measurement/canonical2_verdict.json")
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
