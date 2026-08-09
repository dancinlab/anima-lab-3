#!/usr/bin/env python3
"""Map both SYNERGY-1 clues through the exact trained single-pass bridge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from measurement.workspace_registry import WORKSPACE_INFORMATION_SPEC, spec_sha256
from state_survival import probe_channel
from synergy import SynergyActionChannel, build_examples
from measurement.synergy_registry import experiment as synergy_experiment


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _load_source() -> tuple[dict, dict]:
    spec = WORKSPACE_INFORMATION_SPEC
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if results.get("experiment") != spec["source_experiment"]:
        raise ValueError("SYNERGY-1 results do not match the registered source experiment")
    if verdict.get("experiment") != spec["source_experiment"]:
        raise ValueError("SYNERGY-1 verdict does not match the registered source experiment")
    return results, verdict


def _checkpoint(seed: int, source_results: dict) -> tuple[dict, dict]:
    rows = {row["seed"]: row for row in source_results["seeds"]}
    receipt = rows[seed]["checkpoints"]["quantum_pair"]
    path = Path(WORKSPACE_INFORMATION_SPEC["checkpoint_dir"]) / f"seed_{seed}_quantum_pair.pt"
    digest = sha256_file(path)
    if digest != receipt["sha256"]:
        raise ValueError(f"seed {seed} checkpoint SHA-256 does not match SYNERGY-1 receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("seed") != seed or payload.get("arm") != "quantum_pair":
        raise ValueError(f"seed {seed} checkpoint identity is invalid")
    return payload, {"path": str(path), "sha256": digest}


@torch.no_grad()
def _features(channel: SynergyActionChannel, examples: list, source_spec: dict) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in WORKSPACE_INFORMATION_SPEC["channels"]}
    for example in examples:
        pair = example.quantum
        joined = torch.cat(pair, dim=0)
        trace = channel.action.bridge.trace(joined, seq_len=1)
        rows["raw_pair"].append(joined.reshape(-1).float())
        rows["bridge_cells"].append(trace["cells"].reshape(-1).float())
        rows["bridge_pooled"].append(trace["pooled"].reshape(-1).float())
        rows["bridge_gate"].append(trace["gate"].reshape(-1).float())
        code = channel.inference_codes([pair]).squeeze(0)
        rows["normalized_code"].append(code.reshape(-1).float())
    return {name: torch.stack(values) for name, values in rows.items()}


def run_seed(seed: int, source_results: dict) -> dict:
    map_spec = WORKSPACE_INFORMATION_SPEC
    source_spec = synergy_experiment(map_spec["source_experiment"])
    payload, receipt = _checkpoint(seed, source_results)
    state = payload["channel"]
    try:
        d_model = int(state["action.projector.weight"].shape[0])
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("SYNERGY-1 channel checkpoint has no valid decoder projector") from exc
    channel = SynergyActionChannel(
        "quantum_pair", source_spec["state_dim"], d_model, source_spec
    ).eval()
    channel.load_state_dict(state, strict=True)
    train_examples = build_examples(seed, "train", source_spec)
    eval_examples = build_examples(seed, "eval", source_spec)
    train_features = _features(channel, train_examples, source_spec)
    eval_features = _features(channel, eval_examples, source_spec)
    train_labels = {
        "module_a": torch.tensor([row.module_a for row in train_examples], dtype=torch.long),
        "module_b": torch.tensor([row.module_b for row in train_examples], dtype=torch.long),
    }
    eval_labels = {
        "module_a": torch.tensor([row.module_a for row in eval_examples], dtype=torch.long),
        "module_b": torch.tensor([row.module_b for row in eval_examples], dtype=torch.long),
    }
    channels = {}
    for stage_index, stage in enumerate(map_spec["channels"]):
        channels[stage] = {}
        for label_index, label in enumerate(map_spec["labels"]):
            probe_spec = {
                "probe_ridge": map_spec["probe_ridge"],
                "situations": map_spec[f"{label}_cues"],
                "label_control": map_spec["label_control"],
            }
            channels[stage][label] = probe_channel(
                train_features[stage], train_labels[label],
                eval_features[stage], eval_labels[label],
                seed + stage_index * 10_007 + label_index * 1_000_003,
                spec=probe_spec,
            )
        print(
            f"[seed {seed}] {stage} "
            + " ".join(f"{label}={channels[stage][label]['accuracy']:.3f}"
                       for label in map_spec["labels"]),
            flush=True,
        )
    return {"seed": seed, "checkpoint": receipt, "channels": channels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/workspace_information_results.json")
    parser.add_argument("--verdict", default="measurement/workspace_information_verdict.json")
    args = parser.parse_args()
    source_results, source_verdict = _load_source()
    spec = WORKSPACE_INFORMATION_SPEC
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source": {
            "experiment": source_results["experiment"],
            "verdict": source_verdict.get("verdict"),
            "results_path": spec["source_results"],
            "results_sha256": sha256_file(Path(spec["source_results"])),
            "verdict_path": spec["source_verdict_path"],
            "verdict_sha256": sha256_file(Path(spec["source_verdict_path"])),
        },
        "seeds": [run_seed(seed, source_results) for seed in spec["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.workspace_information_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
