from measurement.consciousness_causality_gate import judge_arm


MODES = ("normal", "off", "shuffle", "noise")


def panel_row(delta=0.01, bar=0.02):
    return {
        "conditions": {
            mode: {
                "bpc": 2.0 if mode == "normal" else 2.0 + delta,
                "stability_delta": bar,
            }
            for mode in MODES
        }
    }


def lambda_row(non_normal="NULL"):
    return {
        "conditions": {
            mode: {
                "lambda4_verdict": "PASS" if mode == "normal" else non_normal,
                "matched_novelty_cost": -0.004 if mode == "normal" else 0.0,
                "matched_t": -8.0 if mode == "normal" else 0.0,
            }
            for mode in MODES
        }
    }


def test_gate_requires_signal_loss_without_language_damage():
    row = judge_arm("seed", panel_row(), lambda_row(), MODES)
    assert row["verdict"] == "CAUSAL"
    assert row["language_preserved"] is True
    assert row["off_removed_lambda4"] is True


def test_gate_marks_language_damage_as_confound():
    row = judge_arm("seed", panel_row(delta=0.03, bar=0.02), lambda_row(), MODES)
    assert row["verdict"] == "CONFOUNDED"


def test_gate_rejects_unchanged_lambda4_as_noncausal():
    row = judge_arm("seed", panel_row(), lambda_row(non_normal="PASS"), MODES)
    assert row["verdict"] == "NOT_CAUSAL"
