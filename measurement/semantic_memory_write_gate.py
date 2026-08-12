#!/usr/bin/env python3
"""Fail-closed adjudication for GATE-CONTROL-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from measurement.semantic_memory_write_registry import SEMANTIC_MEMORY_WRITE_SPEC, spec_sha256
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measurement.semantic_memory_write_registry import SEMANTIC_MEMORY_WRITE_SPEC, spec_sha256


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _checkpoint_valid(receipt: dict, encoder_spec: dict) -> bool:
    path = Path(receipt.get("path", ""))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("sha256"):
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_keys = {
        "format", "method", "model_id", "revision", "feature_dim",
        "weight", "bias", "threshold",
    }
    weight = payload.get("weight", [])
    numeric = lambda value: (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return (
        set(payload) == expected_keys
        and payload.get("format") == "semantic_dialogue_memory_gate_control_v1"
        and payload.get("method") == "canonical_ridge"
        and payload.get("model_id") == encoder_spec["model_id"]
        and payload.get("revision") == encoder_spec["revision"]
        and payload.get("feature_dim") == encoder_spec["feature_dim"]
        and len(weight) == encoder_spec["feature_dim"]
        and all(numeric(value) for value in weight)
        and numeric(payload.get("bias"))
        and numeric(payload.get("threshold"))
    )


def adjudicate(payload: dict, spec: dict = SEMANTIC_MEMORY_WRITE_SPEC) -> dict:
    def invalid(reason: str) -> dict:
        return {
            "experiment": payload.get("experiment", spec["experiment"]),
            "verdict": "GC0_INVALID",
            "reason": reason,
            "spec_sha256": spec_sha256(spec),
        }

    if (
        payload.get("experiment") != spec["experiment"]
        or payload.get("spec") != spec
        or payload.get("spec_sha256") != spec_sha256(spec)
        or not _finite(payload)
    ):
        return invalid("experiment, registered spec, digest, or finite-value check failed")
    runtime = payload.get("runtime", {})
    if (
        not str(runtime.get("python", "")).startswith(spec["runtime"]["python"] + ".")
        or str(runtime.get("torch", "")).split("+")[0] != spec["runtime"]["torch"]
        or runtime.get("transformers") != spec["runtime"]["transformers"]
        or runtime.get("device") != spec["encoder"]["device"]
    ):
        return invalid("registered runtime changed")

    data_spec = spec["gate1_spec"]
    rows = payload.get("seeds", [])
    if [row.get("seed") for row in rows] != data_spec["seeds"]:
        return invalid("registered seed roster changed")
    thresholds = spec["thresholds"]
    summaries = []
    passed = True
    for row in rows:
        seed = row.get("seed")
        audit = row.get("dataset_audit", {})
        if (
            audit.get("calibration_rows") != data_spec["calibration_rows"]
            or audit.get("calibration_unique") != data_spec["calibration_rows"]
            or audit.get("calibration_positive") != data_spec["calibration_rows"] // 2
            or audit.get("calibration_negative") != data_spec["calibration_rows"] // 2
            or audit.get("evaluation_episodes") != data_spec["evaluation_episodes"]
            or audit.get("evaluation_candidates")
            != data_spec["evaluation_episodes"] * data_spec["candidates_per_episode"]
            or audit.get("evaluation_unique") != audit.get("evaluation_candidates")
            or audit.get("overlap") != 0
            or set(audit.get("fact_counts", {})) != set(data_spec["fact_kinds"])
            or any(count != data_spec["evaluation_episodes"] // len(data_spec["fact_kinds"])
                   for count in audit.get("fact_counts", {}).values())
            or any(len(audit.get(name, "")) != 64
                   for name in ("calibration_sha256", "evaluation_sha256"))
        ):
            return invalid(f"seed {seed} dataset audit changed")

        encoder = row.get("encoder_audit", {})
        encoder_spec = spec["encoder"]
        if (
            encoder.get("model_id") != encoder_spec["model_id"]
            or encoder.get("requested_revision") != encoder_spec["revision"]
            or encoder.get("loaded_revision") != encoder_spec["revision"]
            or encoder.get("embedding_dim") != encoder_spec["embedding_dim"]
            or encoder.get("pooling") != encoder_spec["pooling"]
            or encoder.get("normalize") != encoder_spec["normalize"]
            or encoder.get("max_length") != encoder_spec["max_length"]
        ):
            return invalid(f"seed {seed} encoder audit changed")
        embedding = row.get("embedding_audit", {})
        expected_rows = {
            "calibration": data_spec["calibration_rows"],
            "evaluation": data_spec["evaluation_episodes"] * data_spec["candidates_per_episode"],
        }
        for split, count in expected_rows.items():
            split_audit = embedding.get(split, {})
            if (
                split_audit.get("rows") != count
                or split_audit.get("feature_dim") != encoder_spec["feature_dim"]
                or not 0.999999 <= split_audit.get("sentence_norm_min", 0) <= 1.000001
                or not 0.999999 <= split_audit.get("sentence_norm_max", 0) <= 1.000001
                or len(split_audit.get("features_sha256", "")) != 64
            ):
                return invalid(f"seed {seed} {split} embedding audit changed")

        for name in ("fit_audit", "shuffled_fit_audit"):
            fit = row.get(name, {})
            if (
                fit.get("method") != spec["fit_method"]
                or fit.get("examples") != data_spec["calibration_rows"]
                or fit.get("positives") != data_spec["calibration_rows"] // 2
                or fit.get("negatives") != data_spec["calibration_rows"] // 2
                or fit.get("feature_dim") != encoder_spec["feature_dim"]
                or fit.get("ridge") != spec["ridge"]
            ):
                return invalid(f"seed {seed} {name} changed")
        checkpoints = row.get("checkpoints", {})
        if set(checkpoints) != {"semantic", "shuffled"} or any(
            not _checkpoint_valid(receipt, encoder_spec) for receipt in checkpoints.values()
        ):
            return invalid(f"seed {seed} checkpoint changed")

        arms = row.get("arms", {})
        if set(arms) != set(spec["arms"]):
            return invalid(f"seed {seed} arm roster changed")
        expected_metric_keys = {
            "important_storage_rate", "distractor_storage_rate", "search_size_ratio",
            "recall_at_3", "stored", "per_kind_recall", "records_sha256",
        }
        for name, metrics in arms.items():
            if (
                set(metrics) != expected_metric_keys
                or set(metrics.get("per_kind_recall", {})) != set(data_spec["fact_kinds"])
                or len(metrics.get("records_sha256", "")) != 64
                or not 0 <= metrics.get("stored", -1)
                <= data_spec["evaluation_episodes"] * data_spec["candidates_per_episode"]
            ):
                return invalid(f"seed {seed} arm {name} shape changed")

        semantic = arms["semantic_gate"]
        all_rows = arms["store_all"]
        oracle = arms["oracle_gate"]
        random_arm = arms["matched_random"]
        shuffled = arms["shuffled_gate"]
        none = arms["no_memory"]
        if (
            oracle["important_storage_rate"] < thresholds["oracle_important_storage_rate"]
            or oracle["recall_at_3"] < thresholds["oracle_recall_at_3"]
            or all_rows["recall_at_3"] < thresholds["store_all_recall_at_3"]
            or none["recall_at_3"] > thresholds["no_memory_max_recall_at_3"]
            or semantic["recall_at_3"] - random_arm["recall_at_3"]
            < thresholds["minimum_fake_recall_gap"]
            or semantic["recall_at_3"] - shuffled["recall_at_3"]
            < thresholds["minimum_fake_recall_gap"]
        ):
            return invalid(f"seed {seed} positive, fake, or no-memory control failed")
        seed_pass = (
            semantic["important_storage_rate"] >= thresholds["important_storage_rate"]
            and semantic["recall_at_3"] >= thresholds["recall_at_3"]
            and semantic["distractor_storage_rate"]
            <= thresholds["maximum_distractor_storage_rate"]
            and semantic["search_size_ratio"] <= thresholds["maximum_search_size_ratio"]
            and all_rows["recall_at_3"] - semantic["recall_at_3"]
            <= thresholds["maximum_recall_drop_from_all"]
        )
        passed = passed and seed_pass
        summaries.append({
            "seed": seed,
            "passed": seed_pass,
            "semantic": semantic,
            "store_all_recall_at_3": all_rows["recall_at_3"],
            "matched_random_recall_at_3": random_arm["recall_at_3"],
            "shuffled_recall_at_3": shuffled["recall_at_3"],
        })
    return {
        "experiment": spec["experiment"],
        "verdict": "GC1_SEMANTIC_CONTROL_VALID" if passed else "GC2_SEMANTIC_CONTROL_LOSS",
        "reason": (
            "the frozen standard semantic representation supported controlled write selection"
            if passed else "the semantic positive control missed a registered selection or recall threshold"
        ),
        "spec_sha256": spec_sha256(spec),
        "seeds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/semantic_memory_write_verdict.json"),
    )
    args = parser.parse_args()
    verdict = adjudicate(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
