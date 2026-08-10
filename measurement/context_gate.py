#!/usr/bin/env python3
"""Fail-closed adjudication for CONTEXT-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt
    from measurement.context_registry import CONTEXT_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.separation2_gate import adjudicate as adjudicate_separation2
    from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256 as separation2_spec_sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _finite_tree, _metric_shape, _valid_receipt
    from measurement.context_registry import CONTEXT_SPEC, spec_sha256
    from measurement.projector_registry import evaluation_name
    from measurement.separation2_gate import adjudicate as adjudicate_separation2
    from measurement.separation2_registry import SEPARATION2_SPEC, spec_sha256 as separation2_spec_sha256


def _balanced(counts: dict, categories: int, total: int) -> bool:
    expected = total // categories
    return counts == {str(index): expected for index in range(categories)}


def _dataset_valid(audit: dict, total: int, spec: dict) -> bool:
    return (
        audit["episodes"] == total
        and audit["unique_fingerprints"] == total
        and len(audit["fingerprint_set_sha256"]) == 64
        and _balanced(audit["target_counts"], spec["values"], total)
        and _balanced(audit["query_position_counts"], spec["events_per_episode"], total)
        and _balanced(audit["shared_key_counts"], spec["keys"], total)
        and _balanced(audit["query_context_counts"], spec["contexts"], total)
    )


def _context_metrics_valid(metrics: dict, spec: dict) -> bool:
    categories = spec["contexts"]
    return (
        len(metrics["per_key_recall"]) == categories
        and len(metrics["confusion_matrix"]) == categories
        and all(len(row) == categories for row in metrics["confusion_matrix"])
    )


def _context_checkpoint_valid(receipt: dict, payload: dict, spec: dict) -> bool:
    if not _valid_receipt(receipt):
        return False
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    state = checkpoint.get("projector", {})
    expected = {
        "projection.weight": (spec["component_address_dim"], spec["state_dim"]),
        "projection.bias": (spec["component_address_dim"],),
        "prototypes": (spec["contexts"], spec["component_address_dim"]),
    }
    return (
        checkpoint.get("experiment") == spec["experiment"]
        and checkpoint.get("spec_sha256") == spec_sha256(spec)
        and checkpoint.get("fit_method") == spec["fit_method"]
        and checkpoint.get("model_class") == spec["model_class"]
        and checkpoint.get("fit_audit") == payload["fit_audit"]
        and checkpoint.get("calibration_state_audit") == payload["calibration_state_audit"]
        and set(state) == set(expected)
        and all(
            tuple(state[name].shape) == shape and torch.isfinite(state[name]).all()
            for name, shape in expected.items()
        )
    )


def _normal_pass(metrics: dict, spec: dict) -> bool:
    thresholds = spec["thresholds"]
    return (
        metrics["selection_accuracy"] >= thresholds["normal_selection_accuracy"]
        and metrics["accuracy"] >= thresholds["normal_final_accuracy"]
        and min(metrics["per_value_recall"]) >= thresholds["normal_minimum_value_recall"]
        and metrics["correct_content_accuracy"] >= thresholds["content_readout_accuracy"]
    )


def adjudicate(payload: dict, spec: dict = CONTEXT_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CX0_INVALID",
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

        source = payload["source_separation2"]
        if not _valid_receipt(source["results"]) or not _valid_receipt(source["verdict"]):
            return invalid("SEPARATION-2 source file changed")
        source_results = json.loads(Path(source["results"]["path"]).read_text())
        source_verdict = json.loads(Path(source["verdict"]["path"]).read_text())
        source_sha = separation2_spec_sha256(SEPARATION2_SPEC)
        canonical = source_results.get("source_canonical2", {})
        if (
            source_results.get("experiment") != spec["source_experiment"]
            or source_results.get("spec") != SEPARATION2_SPEC
            or source_results.get("spec_sha256") != source_sha
            or source_verdict.get("verdict") != spec["source_verdict"]
            or source_verdict.get("spec_sha256") != source_sha
            or adjudicate_separation2(source_results) != source_verdict
            or source.get("source_spec_sha256") != source_sha
            or source.get("canonical_checkpoint") != canonical.get("checkpoint")
            or source.get("prototype_checkpoints") != canonical.get("prototype_checkpoints")
        ):
            return invalid("registered SEPARATION-2 source identity changed")

        if not _dataset_valid(
            payload["calibration_dataset_audit"], spec["calibration_episodes"], spec
        ):
            return invalid("calibration dataset balance or uniqueness changed")
        if not _dataset_valid(
            payload["evaluation_dataset_audit"], spec["eval_episodes"], spec
        ):
            return invalid("evaluation dataset balance or uniqueness changed")
        if payload["calibration_evaluation_overlap"] != 0:
            return invalid("calibration and evaluation episodes overlap")

        calibration = payload["calibration_state_audit"]
        expected_states = (
            spec["calibration_episodes"] * spec["states_per_episode"]
            * len(spec["calibration_engine_seeds"])
        )
        if (
            calibration["states"] != expected_states
            or sum(calibration["context_counts"].values()) != expected_states
            or set(calibration["context_counts"]) != {
                str(index) for index in range(spec["contexts"])
            }
            or min(calibration["context_counts"].values()) <= 0
            or not spec["minimum_cells"] <= calibration["minimum_cells"]
            <= calibration["maximum_cells"] <= spec["maximum_cells"]
            or set(calibration["engine_seeds"]) != {
                str(seed) for seed in spec["calibration_engine_seeds"]
            }
        ):
            return invalid("calibration state roster changed")
        for seed in spec["calibration_engine_seeds"]:
            row = calibration["engine_seeds"][str(seed)]
            if (
                row["episodes"] != spec["calibration_episodes"]
                or row["unique_episode_seeds"] != spec["calibration_episodes"]
                or len(row["episode_seed_sha256"]) != 64
            ):
                return invalid(f"calibration engine seed {seed} changed")
        fit = payload["fit_audit"]
        if (
            fit["method"] != "ridge_fixed_orthogonal_targets"
            or fit["examples"] != expected_states
            or fit["input_dim"] != spec["state_dim"]
            or fit["address_dim"] != spec["component_address_dim"]
            or fit["keys"] != spec["contexts"]
            or fit["weight_regularization"] != spec["weight_decay"]
            or fit["bias_regularized"] is not False
            or len(fit["label_sha256"]) != 64
            or not _context_checkpoint_valid(payload["context_checkpoint"], payload, spec)
        ):
            return invalid("context projector fit or checkpoint changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        if set(evaluations) != set(registered) or len(evaluations) != len(payload["evaluations"]):
            return invalid("registered evaluation roster changed")
        judged, signatures = {}, []
        thresholds = spec["thresholds"]
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
            if (
                state["episodes"] != spec["eval_episodes"]
                or state["unique_episode_seeds"] != spec["eval_episodes"]
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
                or integration["component_weight"] != spec["component_weight"]
                or integration["component_address_dim"] != spec["component_address_dim"]
                or integration["composite_address_dim"] != spec["composite_address_dim"]
                or integration["context_projector_frozen"] is not True
                or integration["context_projector_unchanged"] is not True
                or integration["key_projector_frozen"] is not True
                or integration["key_projector_unchanged"] is not True
            ):
                return invalid(f"evaluation {name} execution changed")
            signatures.append((
                row["engine_seed"], state["episode_seed_sha256"],
                update["state_before_sha256"], update["state_after_sha256"],
                update["query_rng_sha256"],
            ))
            context = row["context_classification"]
            if not _context_metrics_valid(context, spec):
                return invalid(f"evaluation {name} context metrics changed")
            arms = row["arms"]
            if set(arms) != set(spec["arms"]):
                return invalid(f"evaluation {name} arm roster changed")
            for arm_name in spec["arms"]:
                if (
                    not _metric_shape(arms[arm_name], spec["values"])
                    or arms[arm_name]["retrieval_api_match"] != thresholds["retrieval_api_match"]
                ):
                    return invalid(f"evaluation {name} {arm_name} metrics changed")
            distinct = arms["composite_distinct_key_control"]
            exact = arms["exact_context_key_control"]
            if (
                distinct["selection_accuracy"] < thresholds["distinct_selection_accuracy"]
                or distinct["accuracy"] < thresholds["distinct_final_accuracy"]
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or arms["context_masked_control"]["accuracy"]
                > thresholds["context_masked_max_accuracy"]
                or arms["exact_context_key_partner_swap"]["accuracy"]
                > thresholds["partner_swap_max_accuracy"]
                or arms["composite_context_key_recovered"]["prediction_match"]
                != thresholds["recovery_prediction_match"]
            ):
                return invalid(f"evaluation {name} positive, negative, or recovery control failed")
            normal = arms["composite_context_key_normal"]
            context_passed = (
                context["accuracy"] >= thresholds["context_classification_accuracy"]
                and min(context["per_key_recall"]) >= thresholds["context_minimum_recall"]
            )
            judged[name] = {
                "prototype_seed": row["prototype_seed"],
                "engine_seed": row["engine_seed"],
                "context_passed": context_passed,
                "context_accuracy": context["accuracy"],
                "context_minimum_recall": min(context["per_key_recall"]),
                "normal_passed": _normal_pass(normal, spec),
                "normal_selection_accuracy": normal["selection_accuracy"],
                "normal_final_accuracy": normal["accuracy"],
                "normal_minimum_value_recall": min(normal["per_value_recall"]),
                "normal_content_accuracy": normal["correct_content_accuracy"],
                "context_masked_accuracy": arms["context_masked_control"]["accuracy"],
                "key_masked_accuracy": arms["key_masked_control"]["accuracy"],
                "distinct_final_accuracy": distinct["accuracy"],
                "exact_final_accuracy": exact["accuracy"],
                "partner_swap_accuracy": arms["exact_context_key_partner_swap"]["accuracy"],
            }
        for engine_seed in {row["engine_seed"] for row in spec["evaluation_combinations"]}:
            paired = {signature[1:] for signature in signatures if signature[0] == engine_seed}
            if len(paired) != 1:
                return invalid(f"engine seed {engine_seed} did not keep paired state streams")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    rows = list(judged.values())
    context_pass = all(row["context_passed"] for row in rows)
    selection_pass = all(
        row["normal_selection_accuracy"] >= spec["thresholds"]["normal_selection_accuracy"]
        for row in rows
    )
    normal_pass = all(row["normal_passed"] for row in rows)
    if not context_pass:
        verdict = "CX2_CONTEXT_CODE_LOSS"
        reason = "independent context states were not classified reliably enough for composition"
    elif selection_pass and not normal_pass:
        verdict = "CX4_VALUE_READOUT_LOSS"
        reason = "composite addresses selected the episode, but value readout did not support balanced behavior"
    elif normal_pass:
        verdict = "CX1_CONTEXT_KEY_VALID_NOT_UNIQUE"
        reason = "canonical context and key components retrieved four same-key episodes in every evaluation"
    else:
        verdict = "CX3_COMPOSITION_LOSS"
        reason = "context codes were valid, but the fixed equal-weight composite address did not retrieve every episode"
    return {
        "experiment": spec["experiment"],
        "verdict": verdict,
        "reason": reason,
        "spec_sha256": spec_sha256(spec),
        "evaluations": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/context_results.json")
    parser.add_argument("--output", default="measurement/context_verdict.json")
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
