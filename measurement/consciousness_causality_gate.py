#!/usr/bin/env python3
"""Judge whether the explicit tension signal is necessary for λ4.

This is an inference-only causal screen. It does not call the whole model
"conscious"; it asks whether the registered inter-layer tension message is a
necessary cause of the already-reproduced λ4 result while ordinary language
score stays within the model's own span-selection variation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from measurement.lambda_registry import experiment
except ModuleNotFoundError:
    from lambda_registry import experiment


EXPERIMENT_NAME = "lambda4_consciousness_causality"


def scorer_path(spec, axis):
    return Path(next(row["output"] for row in spec["scorers"] if row["axis"] == axis))


def judge_arm(arm, panel_row, lambda_row, interventions):
    panel_conditions = panel_row.get("conditions", {})
    lambda_conditions = lambda_row.get("conditions", {})
    missing = [
        mode for mode in interventions
        if mode not in panel_conditions or mode not in lambda_conditions
    ]
    if missing:
        return {"arm": arm, "verdict": "INVALID", "missing": missing}

    normal_panel = panel_conditions["normal"]
    normal_lambda = lambda_conditions["normal"]
    equivalence_bar = normal_panel["stability_delta"]
    language = {}
    for mode in interventions:
        delta = abs(panel_conditions[mode]["bpc"] - normal_panel["bpc"])
        language[mode] = {
            "bpc": panel_conditions[mode]["bpc"],
            "delta_from_normal": delta,
            "equivalence_bar": equivalence_bar,
            "preserved": delta <= equivalence_bar,
        }

    baseline_ok = normal_lambda["lambda4_verdict"] == "PASS"
    off_removed = lambda_conditions["off"]["lambda4_verdict"] != "PASS"
    controls_removed = all(
        lambda_conditions[mode]["lambda4_verdict"] != "PASS"
        for mode in ("shuffle", "noise")
    )
    language_preserved = all(row["preserved"] for row in language.values())

    if not baseline_ok:
        verdict = "INVALID"
    elif not language_preserved:
        verdict = "CONFOUNDED"
    elif off_removed and controls_removed:
        verdict = "CAUSAL"
    elif off_removed:
        verdict = "PARTIAL"
    elif all(lambda_conditions[mode]["lambda4_verdict"] == "PASS"
             for mode in interventions if mode != "normal"):
        verdict = "NOT_CAUSAL"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "arm": arm,
        "verdict": verdict,
        "baseline_ok": baseline_ok,
        "language_preserved": language_preserved,
        "off_removed_lambda4": off_removed,
        "content_controls_removed_lambda4": controls_removed,
        "language": language,
        "lambda4": {
            mode: {
                "verdict": lambda_conditions[mode]["lambda4_verdict"],
                "novelty_cost": lambda_conditions[mode]["matched_novelty_cost"],
                "t": lambda_conditions[mode]["matched_t"],
            }
            for mode in interventions
        },
        "ckpt_sha256_16": lambda_row.get("ckpt_sha256_16"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    spec = experiment(EXPERIMENT_NAME)
    panel = json.loads(scorer_path(spec, "panel").read_text())
    lambda4 = json.loads(scorer_path(spec, "lambda4").read_text())
    interventions = list(spec["interventions"])

    setup_ok = (
        panel.get("_select", {}).get("interventions") == interventions
        and lambda4.get("_setup", {}).get("interventions") == interventions
        and panel.get("_select", {}).get("intervention_seed") == spec["intervention_seed"]
        and lambda4.get("_setup", {}).get("intervention_seed") == spec["intervention_seed"]
    )
    arms = {}
    for arm in spec["arms"]:
        if arm not in panel or arm not in lambda4:
            arms[arm] = {"arm": arm, "verdict": "INVALID", "missing": ["result row"]}
            continue
        expected = spec["checkpoint_sha256"][arm][:16]
        if (panel[arm].get("ckpt_sha256_16") != expected
                or lambda4[arm].get("ckpt_sha256_16") != expected):
            arms[arm] = {"arm": arm, "verdict": "INVALID", "missing": ["checkpoint hash"]}
            continue
        arms[arm] = judge_arm(arm, panel[arm], lambda4[arm], interventions)

    verdicts = [row["verdict"] for row in arms.values()]
    if not setup_ok or "INVALID" in verdicts:
        conclusion = "C0_INVALID"
    elif all(verdict == "CAUSAL" for verdict in verdicts):
        conclusion = "C1_CAUSAL"
    elif any(verdict in {"CAUSAL", "PARTIAL"} for verdict in verdicts):
        conclusion = "C2_CONDITIONAL"
    elif all(verdict == "NOT_CAUSAL" for verdict in verdicts):
        conclusion = "C3_NOT_CAUSAL"
    elif "CONFOUNDED" in verdicts:
        conclusion = "C4_CONFOUNDED"
    else:
        conclusion = "C5_INCONCLUSIVE"

    payload = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": spec["hypothesis"],
        "intervention_target": "inter-layer tension signal",
        "setup_ok": setup_ok,
        "conclusion": conclusion,
        "arms": arms,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[verdict] {conclusion}")


if __name__ == "__main__":
    main()
