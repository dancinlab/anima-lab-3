import copy
import json
from pathlib import Path

from gate_write_control1 import (
    SYNTHETIC_PATTERN,
    build_balanced_calibration,
    build_balanced_evaluation,
    dataset_audit,
)
from measurement.balanced_natural_write_gate import adjudicate
from measurement.balanced_natural_write_registry import (
    BALANCED_NATURAL_WRITE_SPEC,
    spec_sha256,
)


def test_balanced_natural_write_is_preregistered_and_pinned():
    spec = BALANCED_NATURAL_WRITE_SPEC
    assert spec["preregistration_commit"] != "__PREREGISTRATION_COMMIT__"
    assert spec["replicates"] == ["daily", "work"]
    assert spec["thresholds"]["minimum_per_template_storage_rate"] == 0.90
    assert len(spec_sha256()) == 64


def test_natural_data_balances_every_template_without_synthetic_identifiers():
    spec = BALANCED_NATURAL_WRITE_SPEC
    calibration = build_balanced_calibration(1337, spec)
    evaluations = {
        name: build_balanced_evaluation(1337, name, spec)
        for name in spec["replicates"]
    }
    audit = dataset_audit(calibration, evaluations, spec)
    assert len(calibration) == spec["calibration_rows"]
    assert audit["calibration_unique"] == spec["calibration_rows"]
    assert audit["calibration_positive"] == audit["calibration_negative"]
    assert len(set(audit["calibration_template_counts"].values())) == 1
    assert all(len(set(rows.values())) == 1 for rows in audit["evaluation_fact_template_counts"].values())
    assert not any(audit["calibration_evaluation_overlap"].values())
    assert not any(audit["cross_replicate_overlap"].values())
    assert audit["synthetic_token_count"] == 0
    assert all(
        not SYNTHETIC_PATTERN.search(row["text"])
        for row in calibration
    )


def test_fact_positions_are_balanced_in_every_natural_replicate():
    spec = BALANCED_NATURAL_WRITE_SPEC
    for seed in spec["seeds"]:
        for replicate in spec["replicates"]:
            episodes = build_balanced_evaluation(seed, replicate, spec)
            counts = {
                position: sum(row["fact_position"] == position for row in episodes)
                for position in spec["fact_positions"]
            }
            assert len(set(counts.values())) == 1


def test_adjudicator_fails_closed_on_registration_change():
    changed = copy.deepcopy(BALANCED_NATURAL_WRITE_SPEC)
    changed["thresholds"]["minimum_important_storage_rate"] = 0.5
    payload = {
        "experiment": changed["experiment"],
        "spec": changed,
        "spec_sha256": spec_sha256(changed),
        "runtime": changed["runtime"],
        "seeds": [],
    }
    assert adjudicate(payload)["verdict"] == "GWC0_INVALID"


def test_recorded_result_replays_registered_verdict():
    path = Path("measurement/balanced_natural_write_results.json")
    if not path.exists():
        return
    verdict = adjudicate(json.loads(path.read_text()))
    assert verdict["verdict"] in {
        "GWC1_BALANCED_NATURAL_WRITE_VALID", "GWC2_WRITE_SELECTION_LOSS",
    }
