import json
import sys

from measurement import gate


def test_coverage_status_checks_both_directions(monkeypatch):
    monkeypatch.setattr(gate, "ARMS", {"measured": (), "pending": ()})

    missing, uncovered = gate.coverage_status({"measured": {}, "extra": {}})

    assert missing == ["pending"]
    assert uncovered == ["extra"]


def test_main_serializes_registered_arm_without_result_as_pending(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "scores.json"
    output = tmp_path / "verdicts.json"
    source.write_text(
        json.dumps(
            {
                "_select": {"keep_rate": 0.5},
                "measured": {"bpc": 1.0, "ckpt_step": 1},
            }
        )
    )
    monkeypatch.setattr(
        gate,
        "ARMS",
        {
            "measured": ("corpus", "measured arm"),
            "pending": ("corpus", "pending arm"),
        },
    )
    monkeypatch.setattr(
        gate,
        "FLOORS",
        {"corpus": {"unigram": 3.0, "bigram": 2.0, "corpus": "corpus.txt"}},
    )
    monkeypatch.setattr(gate, "CONTEXT_JSON", str(tmp_path / "missing-context.json"))
    monkeypatch.setattr(gate, "PANEL_JSONS", (str(tmp_path / "missing-panel.json"),))
    monkeypatch.setattr(sys, "argv", ["gate.py", str(output), str(source)])

    gate.main()

    payload = json.loads(output.read_text())
    assert payload["_gate"]["registered_count"] == 2
    assert payload["_gate"]["measured_count"] == 1
    assert payload["_gate"]["pending"] == ["pending"]
    assert payload["arms"]["pending"]["tier"] == "PENDING"
    assert payload["arms"]["pending"]["verdict"] == "PENDING"
    assert "[validity] NOT YET -- 1 registered-but-pending" in capsys.readouterr().out


def test_attach_ladder_requires_every_control_and_preserves_null():
    row = {"regime": "natural", "verdict": "PASS", "controls_collapsed": True}
    g_row = {
        "G0": True,
        "G2": True,
        "_controls": {
            "_control_positive": {"valid": True},
            "_control_before_backbone": {"valid": True},
            "_control_retrieval": {"valid": True},
        },
    }
    passed = gate.attach_ladder(row, g_row, {"lambda4_verdict": "PASS"})
    unresolved = gate.attach_ladder(row, g_row, {"lambda4_verdict": "NULL"})
    assert passed["ladder_verdict"] == "PASS"
    assert unresolved["ladder_verdict"] == "NULL"


def test_attach_ladder_fails_closed_when_axis_receipt_is_missing():
    row = {"regime": "natural", "verdict": "PASS", "controls_collapsed": True}
    assert gate.attach_ladder(row)["ladder_verdict"] == "PENDING"
