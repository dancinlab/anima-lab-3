#!/usr/bin/env python3
"""Run a pre-registered λ experiment through training, scoring, and receipts."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from measurement.lambda_registry import ARMS, FAMILIES, RESULT_SETS, experiment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def cli_args(values: dict) -> list[str]:
    result = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend((flag, str(value)))
    return result


def paths_for(name: str) -> tuple[dict, dict, Path, Path]:
    exp = experiment(name)
    family = FAMILIES[exp["family"]]
    corpus = ROOT / family["corpus"]
    fresh = ROOT / family["fresh"]
    return exp, family, corpus, fresh


def validate(name: str, score_only: bool = False) -> dict:
    exp, family, corpus, fresh = paths_for(name)
    checks = {
        "experiment": name,
        "hypothesis": exp["hypothesis"],
        "host": socket.gethostname(),
        "family": exp["family"],
        "arms": list(exp["arms"]),
    }
    if "trainer_args" in exp:
        checks["effective_batch_size"] = (
            exp["trainer_args"]["batch_size"]
            * exp["trainer_args"]["grad_accum_steps"]
        )
    for path, expected, label in (
        (corpus, exp["corpus_sha256"], "corpus"),
        (fresh, exp["fresh_sha256"], "fresh"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing registered {label}: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"registered {label} checksum mismatch: {actual} != {expected}"
            )
        checks[f"{label}_sha256"] = actual
    for arm_name in exp["arms"]:
        spec = ARMS[arm_name]
        if spec["family"] != exp["family"]:
            raise SystemExit(f"arm {arm_name} is registered to another family")
        checkpoint = ROOT / spec["checkpoint"]
        completed = ROOT / "logs" / f"{name}_{arm_name}.done.json"
        if score_only and not checkpoint.is_file():
            raise SystemExit(f"score-only requested but checkpoint is missing: {checkpoint}")
        if (not score_only and not exp.get("measurement_only")
                and checkpoint.parent.exists() and not completed.is_file()):
            raise SystemExit(
                f"refusing ambiguous checkpoint directory without completion receipt: "
                f"{checkpoint.parent}"
            )
        expected_checkpoint = exp.get("checkpoint_sha256", {}).get(arm_name)
        if checkpoint.is_file() and expected_checkpoint:
            actual_checkpoint = sha256(checkpoint)
            if actual_checkpoint != expected_checkpoint:
                raise SystemExit(
                    f"registered checkpoint checksum mismatch for {arm_name}: "
                    f"{actual_checkpoint} != {expected_checkpoint}"
                )
    if exp.get("trainer"):
        checks["trainer_sha256"] = sha256(ROOT / exp["trainer"])
    return checks


def train_arm(name: str, exp: dict, arm_name: str) -> dict:
    spec = ARMS[arm_name]
    checkpoint = ROOT / spec["checkpoint"]
    checkpoint_dir = checkpoint.parent
    done_path = ROOT / "logs" / f"{name}_{arm_name}.done.json"
    if done_path.is_file() and checkpoint.is_file():
        receipt = json.loads(done_path.read_text())
        if sha256(checkpoint) != receipt.get("checkpoint_sha256"):
            raise SystemExit(f"completed checkpoint hash drifted: {checkpoint}")
        print(f"[skip] {arm_name}: verified completed receipt", flush=True)
        return receipt

    family = FAMILIES[exp["family"]]
    command = [
        sys.executable,
        "-u",
        str(ROOT / exp["trainer"]),
        *cli_args(exp["trainer_args"]),
        "--seed",
        str(exp["seeds"][arm_name]),
        "--data",
        str(ROOT / family["corpus"]),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    log_path = ROOT / "logs" / f"{arm_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] {arm_name} -> {log_path}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise SystemExit(f"trainer failed for {arm_name}: exit {completed.returncode}")
    best = ROOT / spec["checkpoint"]
    final = checkpoint_dir / "final.pt"
    if not best.is_file() or not final.is_file():
        raise SystemExit(f"trainer exited without best/final checkpoint for {arm_name}")
    match = re.search(r"\[model\] ConsciousLM: ([\d,]+) params", log_path.read_text())
    measured_params = int(match.group(1).replace(",", "")) if match else None
    if measured_params != exp["expected_params"]:
        raise SystemExit(
            f"parameter count mismatch for {arm_name}: {measured_params} "
            f"!= {exp['expected_params']}"
        )
    receipt = {
        "experiment": name,
        "arm": arm_name,
        "seed": exp["seeds"][arm_name],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "params": measured_params,
        "checkpoint": str(best.relative_to(ROOT)),
        "checkpoint_size": best.stat().st_size,
        "checkpoint_sha256": sha256(best),
        "trainer_args": exp["trainer_args"],
    }
    atomic_json(done_path, receipt)
    return receipt


def score(name: str, exp: dict) -> dict:
    if exp.get("scorers"):
        return score_registered(name, exp)
    result_set = RESULT_SETS[exp["results"]]
    env = {**os.environ, "LAMBDA_FAMILY": exp["family"]}
    receipts = {}
    for axis, script in (
        ("panel", "panel.py"),
        ("g_gates", "g_gates.py"),
        ("lambda4", "lambda4.py"),
    ):
        output = ROOT / result_set[axis]
        log_path = ROOT / "logs" / f"{name}_{axis}.log"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "measurement" / script),
            str(output),
            *exp["arms"],
        ]
        print(f"[score] {axis} -> {output}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode or not output.is_file():
            raise SystemExit(f"{axis} scorer failed: exit {completed.returncode}")
        receipts[axis] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
        }

    gate_log = ROOT / "logs" / f"{name}_gate.log"
    with gate_log.open("w") as log:
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "measurement" / "gate.py")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise SystemExit(f"gate failed: exit {completed.returncode}")
    receipts["gate"] = {
        "path": "measurement/gate_verdicts.json",
        "sha256": sha256(ROOT / "measurement/gate_verdicts.json"),
    }
    preflight_log = ROOT / "logs" / f"{name}_preflight.log"
    with preflight_log.open("w") as log:
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "measurement" / "preflight.py")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise SystemExit(f"preflight failed: exit {completed.returncode}")
    receipts["preflight"] = {
        "path": str(preflight_log.relative_to(ROOT)),
        "sha256": sha256(preflight_log),
    }
    for scorer in exp.get("post_scorers", ()):
        output = ROOT / scorer["output"]
        log_path = ROOT / "logs" / f"{name}_{scorer['axis']}.log"
        command = [sys.executable, "-u", str(ROOT / scorer["script"]), str(output)]
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode or not output.is_file():
            raise SystemExit(f"{scorer['axis']} scorer failed: exit {completed.returncode}")
        receipts[scorer["axis"]] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
        }
    return receipts


def score_registered(name: str, exp: dict) -> dict:
    """Run an experiment-defined scorer roster without a one-off shell path."""
    env = {
        **os.environ,
        "ANIMA_LAB_ROOT": str(ROOT),
        "LAMBDA_FAMILY": exp["family"],
    }
    interventions = list(exp.get("interventions", ()))
    if interventions:
        env["CONSCIOUSNESS_INTERVENTION_SEED"] = str(exp["intervention_seed"])
    receipts = {}
    for scorer in exp["scorers"]:
        output = ROOT / scorer["output"]
        log_path = ROOT / "logs" / f"{name}_{scorer['axis']}.log"
        command = [sys.executable, "-u", str(ROOT / scorer["script"]), str(output)]
        if scorer["axis"] != "verdict":
            command.extend(exp["arms"])
            if interventions:
                command.extend(("--interventions", *interventions))
        print(f"[score] {scorer['axis']} -> {output}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode or not output.is_file():
            raise SystemExit(
                f"{scorer['axis']} scorer failed: exit {completed.returncode}"
            )
        receipts[scorer["axis"]] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
        }
    return receipts


def run(args: argparse.Namespace) -> None:
    checks = validate(args.experiment, score_only=args.score_only)
    if args.preflight:
        print(json.dumps(checks, indent=2))
        return

    lock_path = ROOT / "logs" / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another research job holds logs/gpu.lock") from exc
        exp = experiment(args.experiment)
        training = []
        if not args.score_only and not exp.get("measurement_only"):
            training = [train_arm(args.experiment, exp, arm) for arm in exp["arms"]]
        scoring = {} if args.train_only else score(args.experiment, exp)
        receipt = {
            **checks,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "training": training,
            "scoring": scoring,
        }
        atomic_json(ROOT / "logs" / f"{args.experiment}.receipt.json", receipt)
        print(f"[done] {args.experiment}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), args.experiment]
        if args.train_only:
            command.append("--train-only")
        elif args.score_only:
            command.append("--score-only")
        launch_log = ROOT / "logs" / f"{args.experiment}_launch.log"
        launch_log.parent.mkdir(parents=True, exist_ok=True)
        with launch_log.open("a") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"launched pid={process.pid} log={launch_log}")
        return
    run(args)


if __name__ == "__main__":
    main()
