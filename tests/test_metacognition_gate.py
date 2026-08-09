from copy import deepcopy

from measurement.metacognition_gate import adjudicate
from measurement.metacognition_registry import METACOGNITION_SPEC, spec_sha256


def confidence(auroc=0.85, brier=0.12, ece=0.08, gap=0.35):
    return {"auroc": auroc, "brier": brier, "ece": ece,
            "selective_accuracy_gap": gap, "mean_confidence": 0.6}


def arm(seed, name, output_auroc=0.70, output_brier=0.16):
    levels = METACOGNITION_SPEC["readout_noise_levels"]
    by_level = {
        str(level): {"accuracy": 0.95 if level == levels[0] else 0.45 if level == levels[-1] else 0.70}
        for level in levels
    }
    return {
        "source_checkpoint_sha256": METACOGNITION_SPEC["archive"]["checkpoint_sha256"][str(seed)][name],
        "action": {"by_noise_level": by_level, "correct_examples": 160, "incorrect_examples": 96},
        "confidence": {
            "normal": confidence(), "off": confidence(0.55, 0.24, 0.20, 0.05),
            "shuffle": confidence(0.60, 0.22, 0.18, 0.08),
            "noise": confidence(0.52, 0.25, 0.22, 0.01), "recovered": confidence(),
        },
        "output_only": confidence(output_auroc, output_brier, 0.10, 0.25),
        "intervention_actions_identical": True,
        "recovery_confidence_identical": True,
    }


def payload():
    return {
        "experiment": METACOGNITION_SPEC["experiment"],
        "spec": deepcopy(METACOGNITION_SPEC),
        "spec_sha256": spec_sha256(),
        "seeds": [
            {"seed": seed, "arms": {name: arm(seed, name) for name in METACOGNITION_SPEC["arms"]}}
            for seed in METACOGNITION_SPEC["seeds"]
        ],
    }


def test_meta_gate_distinguishes_advantage_equivalence_and_failure():
    assert adjudicate(payload())["verdict"] == "M1_STATE_MONITORING_ADVANTAGE"
    equivalent = payload()
    for row in equivalent["seeds"]:
        row["arms"]["consciousness"]["output_only"] = confidence(0.84, 0.13, 0.08, 0.30)
    assert adjudicate(equivalent)["verdict"] == "M2_CALIBRATED_NOT_UNIQUE"
    failed = payload()
    failed["seeds"][0]["arms"]["consciousness"]["confidence"]["normal"]["auroc"] = 0.6
    assert adjudicate(failed)["verdict"] == "M3_NOT_CALIBRATED"


def test_meta_gate_fails_closed_on_task_control_hash_and_actions():
    invalid = payload()
    invalid["seeds"][0]["arms"]["memory"]["action"]["incorrect_examples"] = 0
    assert adjudicate(invalid)["verdict"] == "M0_INVALID"
    bad_hash = payload()
    bad_hash["seeds"][0]["arms"]["consciousness"]["source_checkpoint_sha256"] = "wrong"
    assert adjudicate(bad_hash)["verdict"] == "M0_INVALID"
    changed_actions = payload()
    changed_actions["seeds"][0]["arms"]["memory"]["intervention_actions_identical"] = False
    assert adjudicate(changed_actions)["verdict"] == "M0_INVALID"


def test_meta_gate_rejects_non_finite_and_spec_drift():
    non_finite = payload()
    non_finite["seeds"][0]["arms"]["consciousness"]["confidence"]["normal"]["brier"] = float("nan")
    assert adjudicate(non_finite)["verdict"] == "M0_INVALID"
    drift = payload()
    drift["spec"]["reader"]["train_steps"] += 1
    assert adjudicate(drift)["verdict"] == "M0_INVALID"
