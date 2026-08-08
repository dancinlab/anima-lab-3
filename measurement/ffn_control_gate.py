#!/usr/bin/env python3
"""Adjudicate the pre-registered standard-FFN structural control."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from measurement.lambda_registry import ARMS, experiment


EXPERIMENT = "ffn_structural_control"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def judge() -> dict:
    spec = experiment(EXPERIMENT)
    gate = load("measurement/gate_verdicts.json")
    lambda4 = load("measurement/lambda4_ffn_control_results.json")
    reference = load("measurement/lambda4_literary_results.json")
    gate_rows = gate["arms"]

    reference_hashes = {}
    reference_ok = True
    for arm in spec["reference_arms"]:
        checkpoint = ROOT / ARMS[arm]["checkpoint"]
        expected = spec["reference_checkpoint_sha256"][arm]
        actual = sha256(checkpoint) if checkpoint.is_file() else None
        reference_hashes[arm] = actual
        reference_ok = reference_ok and actual == expected
        reference_ok = reference_ok and reference.get(arm, {}).get("lambda4_verdict") == "PASS"

    arms = {}
    language_ok = True
    passes = 0
    for arm in spec["arms"]:
        gate_row = gate_rows.get(arm, {})
        lambda_row = lambda4.get(arm, {})
        arm_language_ok = all(gate_row.get(key) is True for key in (
            "lambda0_1", "lambda2", "lambda3"
        ))
        verdict = lambda_row.get("lambda4_verdict")
        passes += verdict == "PASS"
        language_ok = language_ok and arm_language_ok
        arms[arm] = {
            "language_ok": arm_language_ok,
            "bpc": gate_row.get("bpc"),
            "lambda4_verdict": verdict,
            "lambda4_cost": lambda_row.get("matched_novelty_cost"),
            "lambda4_t": lambda_row.get("matched_t"),
            "checkpoint_sha256": sha256(ROOT / ARMS[arm]["checkpoint"]),
        }

    if not reference_ok:
        verdict = "F0"
        reason = "reference checkpoint or its two-seed lambda4 baseline did not reproduce"
    elif not language_ok:
        verdict = "F4"
        reason = "standard FFN failed at least one lambda0-lambda3 language control"
    elif passes == len(spec["arms"]):
        verdict = "F1"
        reason = "lambda4 reproduced in both standard-FFN seeds; PureFieldFFN is not required"
    elif passes == 1:
        verdict = "F2"
        reason = "lambda4 reproduced in only one standard-FFN seed; structure effect is conditional"
    else:
        verdict = "F3"
        reason = "language controls held but lambda4 reproduced in neither standard-FFN seed"

    return {
        "experiment": EXPERIMENT,
        "hypothesis": spec["hypothesis"],
        "verdict": verdict,
        "reason": reason,
        "adjudicated_at": datetime.now(timezone.utc).isoformat(),
        "reference_ok": reference_ok,
        "reference_checkpoint_sha256": reference_hashes,
        "language_ok": language_ok,
        "lambda4_passes": passes,
        "arms": arms,
    }


def main() -> None:
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = judge()
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{payload['verdict']}] {payload['reason']}")
    print(f"[json] {output}")


if __name__ == "__main__":
    main()
