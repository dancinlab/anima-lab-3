#!/usr/bin/env python3
"""Fail-closed adjudication for CUE-ROBUST-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from measurement.capacity_gate import _metric_shape
    from measurement.component2_gate import adjudicate as adjudicate_component
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component_spec_sha256
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC
    from measurement.cue_mechanism_gate import (
        _classification_shape,
        _distance_shape,
        _finite,
        _receipt_valid,
        adjudicate as adjudicate_cue,
    )
    from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256 as cue_spec_sha256
    from measurement.cue_robust_registry import (
        CUE_ROBUST_SPEC,
        spec_sha256,
        training_examples_per_component,
        training_mask_plan_audit,
    )
    from measurement.projector_registry import evaluation_name
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.capacity_gate import _metric_shape
    from measurement.component2_gate import adjudicate as adjudicate_component
    from measurement.component2_registry import COMPONENT2_SPEC, spec_sha256 as component_spec_sha256
    from measurement.completion_registry import COMPLETION_SPEC, mask_plan_audit
    from measurement.conjunction2_registry import CONJUNCTION2_SPEC
    from measurement.cue_mechanism_gate import (
        _classification_shape,
        _distance_shape,
        _finite,
        _receipt_valid,
        adjudicate as adjudicate_cue,
    )
    from measurement.cue_mechanism_registry import CUE_MECHANISM_SPEC, spec_sha256 as cue_spec_sha256
    from measurement.cue_robust_registry import (
        CUE_ROBUST_SPEC,
        spec_sha256,
        training_examples_per_component,
        training_mask_plan_audit,
    )
    from measurement.projector_registry import evaluation_name


def adjudicate(payload: dict, spec: dict = CUE_ROBUST_SPEC,
               *, cue_results: dict | None = None) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "CR0_INVALID", "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    try:
        if (
            payload["experiment"] != spec["experiment"]
            or payload["spec"] != spec
            or payload["spec_sha256"] != spec_sha256(spec)
            or not _finite(payload)
        ):
            return invalid("experiment, registered spec, digest, or finite-value check failed")

        source = payload["source"]
        for name in (
            "cue_results", "cue_verdict", "component_results", "component_verdict",
            "component_checkpoint", "value_checkpoint",
        ):
            if not _receipt_valid(source[name]):
                return invalid(f"registered source {name} changed")
        if any(not _receipt_valid(row) for row in source["prototype_checkpoints"].values()):
            return invalid("registered prototype checkpoint changed")
        if not _receipt_valid(payload["checkpoint"]):
            return invalid("robust component checkpoint changed")

        if cue_results is None:
            cue_results = json.loads(Path(source["cue_results"]["path"]).read_text())
        cue_verdict = json.loads(Path(source["cue_verdict"]["path"]).read_text())
        cue_sha = cue_spec_sha256(CUE_MECHANISM_SPEC)
        if (
            cue_results.get("experiment") != spec["source_experiment"]
            or cue_results.get("spec") != CUE_MECHANISM_SPEC
            or cue_results.get("spec_sha256") != cue_sha
            or cue_verdict.get("verdict") != spec["source_verdict"]
            or adjudicate_cue(cue_results) != cue_verdict
            or source["cue_spec_sha256"] != cue_sha
        ):
            return invalid("registered CUE-MECHANISM-1 identity changed")

        component_results = json.loads(Path(source["component_results"]["path"]).read_text())
        component_verdict = json.loads(Path(source["component_verdict"]["path"]).read_text())
        component_sha = component_spec_sha256(COMPONENT2_SPEC)
        if (
            component_results.get("experiment") != COMPONENT2_SPEC["experiment"]
            or component_results.get("spec") != COMPONENT2_SPEC
            or component_results.get("spec_sha256") != component_sha
            or component_verdict.get("verdict") != spec["source_component_verdict"]
            or adjudicate_component(component_results) != component_verdict
            or source["component_spec_sha256"] != component_sha
            or source["component_checkpoint"] != component_results["checkpoint"]
            or source["component_checkpoint"] != cue_results["source"]["component_checkpoint"]
            or source["value_checkpoint"] != cue_results["source"]["value_checkpoint"]
            or source["prototype_checkpoints"] != cue_results["source"]["prototype_checkpoints"]
        ):
            return invalid("registered COMPONENT-2 or inherited checkpoint identity changed")

        checkpoint = torch.load(
            payload["checkpoint"]["path"], map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("experiment") != spec["experiment"]
            or checkpoint.get("spec_sha256") != spec_sha256(spec)
            or checkpoint.get("deterministic") is not True
            or checkpoint.get("context_fit") != payload["context_fit"]
            or checkpoint.get("key_fit") != payload["key_fit"]
        ):
            return invalid("robust checkpoint identity or fit audit changed")

        if (
            payload["calibration_dataset_audit"]
            != component_results["calibration_dataset_audit"]
            or payload["calibration_state_audit"]
            != component_results["calibration_state_audit"]
            or payload["calibration_evaluation_overlap"] != 0
            or payload["training_mask_plan_audit"] != training_mask_plan_audit(spec)
            or payload["evaluation_mask_plan_audit"] != mask_plan_audit(COMPLETION_SPEC)
        ):
            return invalid("calibration source, overlap, or registered mask plan changed")
        expected_full = training_examples_per_component(spec)
        expected_labels = {
            str(label): expected_full * 2 // spec["contexts"]
            for label in range(spec["contexts"])
        }
        for component in spec["training_mask_components"]:
            fit = payload[f"{component}_fit"]
            if (
                fit["full_examples"] != expected_full
                or fit["masked_examples"] != expected_full
                or fit["total_examples"] != expected_full * 2
                or fit["label_counts"] != expected_labels
                or fit["full_refit_matches_source"] is not True
                or fit["deterministic"] is not True
                or fit["full_refit"]["method"] != "ridge_fixed_orthogonal_targets"
                or fit["robust_fit"]["method"] != "ridge_fixed_orthogonal_targets"
                or fit["fake_fit"]["method"] != "ridge_fixed_orthogonal_targets"
                or fit["robust_fit"]["examples"] != expected_full * 2
                or fit["fake_fit"]["examples"] != expected_full * 2
            ):
                return invalid(f"{component} damage-only fit isolation changed")
            overlap = payload["mask_overlap_audit"][component]
            mask_audit = payload["training_mask_plan_audit"][component]
            eval_audit = payload["evaluation_mask_plan_audit"][
                f"{component}:{spec['training_missing_fraction']:.2f}"
            ]
            if overlap != {
                "training_unique_masks": mask_audit["unique_masks"],
                "evaluation_unique_masks": eval_audit["unique_masks"],
                "exact_overlap": 0,
            }:
                return invalid(f"{component} training and evaluation masks overlap")

        audit = payload["dataset_audit"]
        if (
            audit["episodes"] != spec["eval_episodes"]
            or audit["unique_fingerprints"] != spec["eval_episodes"]
            or audit["latin_valid_episodes"] != spec["eval_episodes"]
            or audit["minimum_unique_pairs"] != spec["events_per_episode"]
            or audit["maximum_unique_pairs"] != spec["events_per_episode"]
        ):
            return invalid("registered evaluation dataset changed")

        registered = {evaluation_name(row): row for row in spec["evaluation_combinations"]}
        evaluations = {row["name"]: row for row in payload["evaluations"]}
        baselines = {row["name"]: row for row in cue_results["evaluations"]}
        if (
            set(evaluations) != set(registered)
            or set(baselines) != set(registered)
            or len(evaluations) != len(payload["evaluations"])
        ):
            return invalid("registered evaluation roster changed")

        thresholds = spec["thresholds"]
        full_pass = True
        context_pass = True
        key_pass = True
        retrieval_pass = True
        judged = {}
        expected_path = {
            "minimum_calls": spec["transform_calls_per_episode"],
            "maximum_calls": spec["transform_calls_per_episode"],
            "minimum_components": spec["components_per_key"],
            "maximum_components": spec["components_per_key"],
            "minimum_address_width": spec["composite_address_dim"],
            "maximum_address_width": spec["composite_address_dim"],
            "value_calls": spec["eval_episodes"] * spec["stores_per_episode"],
            "stores": spec["eval_episodes"] * spec["stores_per_episode"],
            "retrievals": spec["eval_episodes"] * spec["retrievals_per_episode"],
        }
        for name, row in evaluations.items():
            identity = registered[name]
            baseline = baselines[name]
            if (
                row["prototype_seed"] != identity["prototype_seed"]
                or row["engine_seed"] != identity["engine_seed"]
                or set(row["arms"]) != set(spec["arms"])
                or any(not _metric_shape(arm, spec["values"]) for arm in row["arms"].values())
                or row["frozen_audit"] != {"context": True, "key": True, "value": True}
                or row["state_audit"] != baseline["state_audit"]
                or set(row["memory_path_audit"]) != set(spec["conditions"])
                or any(value != expected_path for value in row["memory_path_audit"].values())
                or set(row["component_metrics"]) != set(spec["conditions"])
                or set(row["distance_metrics"]) != set(spec["conditions"])
                or set(row["fake_component_metrics"]) != set(spec["conditions"])
                or row["reference_audit"] != {
                    "full_metric_match": row["arms"]["full_cue"] == baseline["arms"]["full_cue"],
                    "both_quarter_metric_match": (
                        row["arms"]["both_quarter_missing"]
                        == baseline["arms"]["both_quarter_missing"]
                    ),
                }
                or row["restoration_audit"] != {
                    "context_restored_to_full": True, "key_restored_to_full": True,
                }
            ):
                return invalid(f"evaluation {name} identity, path, source, or restore audit changed")
            for condition in spec["conditions"]:
                for roster in (row["component_metrics"], row["fake_component_metrics"]):
                    if (
                        set(roster[condition]) != {"context", "key"}
                        or not _classification_shape(roster[condition]["context"], spec["contexts"])
                        or not _classification_shape(roster[condition]["key"], spec["keys"])
                    ):
                        return invalid(f"evaluation {name} component diagnostic shape changed")
                if not _distance_shape(row["distance_metrics"][condition]):
                    return invalid(f"evaluation {name} distance diagnostic shape changed")

            arms = row["arms"]
            full = arms["full_cue"]
            context_arm = arms["context_quarter_missing"]
            key_arm = arms["key_quarter_missing"]
            both_arm = arms["both_quarter_missing"]
            exact = arms["exact_context_key_control"]
            partner = arms["exact_context_key_partner_swap"]
            if (
                any(arm["retrieval_api_match"] != 1.0 for arm in arms.values())
                or exact["selection_accuracy"] < thresholds["exact_selection_accuracy"]
                or exact["accuracy"] < thresholds["exact_final_accuracy"]
                or min(exact["per_value_recall"]) < thresholds["exact_minimum_value_recall"]
                or partner["accuracy"] > thresholds["partner_swap_max_accuracy"]
            ):
                return invalid(f"evaluation {name} API, exact-address, or partner control failed")

            full_context = row["component_metrics"]["full_cue"]["context"]
            full_key = row["component_metrics"]["full_cue"]["key"]
            baseline_full_context = baseline["component_metrics"]["full_cue"]["context"]
            baseline_full_key = baseline["component_metrics"]["full_cue"]["key"]
            current_full = (
                full_context["accuracy"] >= thresholds["full_category_accuracy"]
                and full_context["minimum_class_recall"]
                >= thresholds["full_minimum_category_recall"]
                and full_key["accuracy"] >= thresholds["full_category_accuracy"]
                and full_key["minimum_class_recall"]
                >= thresholds["full_minimum_category_recall"]
                and full_context["accuracy"]
                >= baseline_full_context["accuracy"]
                - thresholds["maximum_full_category_regression"]
                and full_key["accuracy"]
                >= baseline_full_key["accuracy"]
                - thresholds["maximum_full_category_regression"]
                and full["selection_accuracy"] >= thresholds["full_selection_accuracy"]
                and full["accuracy"] >= thresholds["full_final_accuracy"]
                and min(full["per_value_recall"]) >= thresholds["full_minimum_value_recall"]
            )
            context_category = row["component_metrics"]["context_quarter_missing"]["context"]
            key_category = row["component_metrics"]["key_quarter_missing"]["key"]
            current_context = (
                context_category["accuracy"] >= thresholds["partial_category_accuracy"]
                and context_category["minimum_class_recall"]
                >= thresholds["partial_minimum_category_recall"]
            )
            current_key = (
                key_category["accuracy"] >= thresholds["partial_category_accuracy"]
                and key_category["minimum_class_recall"]
                >= thresholds["partial_minimum_category_recall"]
            )
            current_retrieval = (
                context_arm["selection_accuracy"]
                >= thresholds["single_quarter_selection_accuracy"]
                and context_arm["accuracy"] >= thresholds["single_quarter_final_accuracy"]
                and key_arm["selection_accuracy"]
                >= thresholds["single_quarter_selection_accuracy"]
                and key_arm["accuracy"] >= thresholds["single_quarter_final_accuracy"]
                and both_arm["selection_accuracy"]
                >= thresholds["both_quarter_selection_accuracy"]
                and both_arm["accuracy"] >= thresholds["both_quarter_final_accuracy"]
            )
            fake_context = row["fake_component_metrics"]["context_quarter_missing"]["context"]
            fake_key = row["fake_component_metrics"]["key_quarter_missing"]["key"]
            if (
                fake_context["accuracy"] > thresholds["fake_category_max_accuracy"]
                or fake_key["accuracy"] > thresholds["fake_category_max_accuracy"]
            ):
                return invalid(f"evaluation {name} shuffled-label control failed")
            full_pass &= current_full
            context_pass &= current_context
            key_pass &= current_key
            retrieval_pass &= current_retrieval
            judged[name] = {
                "full_context_category_accuracy": full_context["accuracy"],
                "full_key_category_accuracy": full_key["accuracy"],
                "context_partial_category_accuracy": context_category["accuracy"],
                "key_partial_category_accuracy": key_category["accuracy"],
                "full_selection_accuracy": full["selection_accuracy"],
                "full_final_accuracy": full["accuracy"],
                "context_partial_selection_accuracy": context_arm["selection_accuracy"],
                "context_partial_final_accuracy": context_arm["accuracy"],
                "key_partial_selection_accuracy": key_arm["selection_accuracy"],
                "key_partial_final_accuracy": key_arm["accuracy"],
                "both_partial_selection_accuracy": both_arm["selection_accuracy"],
                "both_partial_final_accuracy": both_arm["accuracy"],
                "fake_context_accuracy": fake_context["accuracy"],
                "fake_key_accuracy": fake_key["accuracy"],
            }
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError) as exc:
        return invalid(str(exc))

    if not full_pass:
        verdict = "CR6_FULL_CUE_REGRESSION"
        reason = "damage augmentation regressed the registered full-cue path"
    elif not context_pass and not key_pass:
        verdict = "CR4_DUAL_CATEGORY_NOT_RECOVERED"
        reason = "neither partial component category recovered"
    elif not context_pass:
        verdict = "CR2_CONTEXT_CATEGORY_NOT_RECOVERED"
        reason = "the partial context category did not recover"
    elif not key_pass:
        verdict = "CR3_KEY_CATEGORY_NOT_RECOVERED"
        reason = "the partial key category did not recover"
    elif not retrieval_pass:
        verdict = "CR5_RETRIEVAL_NOT_RECOVERED"
        reason = "component categories recovered but the shared partial-cue memory path did not"
    else:
        verdict = "CR1_ROBUST_CUE_PATH_VALID_NOT_UNIQUE"
        reason = "the unchanged standard readout recovered all registered unseen partial cues"
    return {
        "experiment": spec["experiment"], "verdict": verdict, "reason": reason,
        "spec_sha256": spec_sha256(spec), "evaluations": judged,
        "checkpoint": payload["checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="measurement/cue_robust_results.json")
    parser.add_argument("--output", default="measurement/cue_robust_verdict.json")
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
