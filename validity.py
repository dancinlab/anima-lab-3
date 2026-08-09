#!/usr/bin/env python3
"""VALIDITY-1: locate the first failure in the frozen RELATION-1 action path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from graft_behavior import sha256_file
from measurement.relation_gate import adjudicate as adjudicate_relation
from measurement.relation_registry import RELATION_ROLE_REPAIR_SPEC, spec_sha256 as relation_spec_sha256
from measurement.validity_registry import VALIDITY_SPEC, spec_sha256
from state_survival import _ridge_predict, probe_channel
from synergy import (
    SynergyActionChannel,
    _action_token_ids,
    _pairs,
    _prompt_tokens,
    audit_examples,
    build_examples,
)
from trinity import HFDecoder


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _load_source() -> tuple[dict, dict, dict]:
    spec = VALIDITY_SPEC
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    if sha256_file(results_path) != spec["source_results_sha256"]:
        raise ValueError("RELATION-1 source result SHA-256 changed")
    if sha256_file(verdict_path) != spec["source_verdict_sha256"]:
        raise ValueError("RELATION-1 source verdict SHA-256 changed")
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    reproduced = adjudicate_relation(results)
    if reproduced != verdict or verdict.get("verdict") != spec["source_verdict"]:
        raise ValueError("RELATION-1 source verdict does not reproduce")
    if results.get("spec_sha256") != relation_spec_sha256(RELATION_ROLE_REPAIR_SPEC):
        raise ValueError("RELATION-1 source spec changed")
    source = {
        "experiment": results.get("experiment"),
        "verdict": verdict.get("verdict"),
        "results_path": spec["source_results"],
        "results_sha256": spec["source_results_sha256"],
        "verdict_path": spec["source_verdict_path"],
        "verdict_sha256": spec["source_verdict_sha256"],
        "dataset_audit": results.get("dataset_audit"),
        "reproduced": True,
    }
    return results, verdict, source


def _confusion(labels: torch.Tensor, predictions: torch.Tensor, classes: int) -> list[list[int]]:
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    for expected, actual in zip(labels.tolist(), predictions.tolist()):
        matrix[expected, actual] += 1
    return matrix.tolist()


def _probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor,
           eval_y: torch.Tensor, seed: int) -> dict:
    probe_spec = {
        "probe_ridge": VALIDITY_SPEC["probe_ridge"],
        "situations": VALIDITY_SPEC["actions"],
        "label_control": VALIDITY_SPEC["label_control"],
    }
    metrics = probe_channel(train_x, train_y, eval_x, eval_y, seed, spec=probe_spec)
    logits = _ridge_predict(
        train_x, train_y, eval_x, probe_spec["probe_ridge"], len(VALIDITY_SPEC["actions"])
    )
    predictions = logits.argmax(-1)
    matrix = _confusion(eval_y, predictions, len(VALIDITY_SPEC["actions"]))
    metrics["confusion_matrix"] = matrix
    metrics["per_class_recall"] = [
        matrix[index][index] / max(sum(matrix[index]), 1) for index in range(len(matrix))
    ]
    return metrics


def _source_pairs(examples: list, arm: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _pairs(examples, arm)


@torch.no_grad()
def _relation_features(channel: SynergyActionChannel,
                       pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    rows = []
    for pair in pairs:
        if channel.recurrent is not None:
            sequence = torch.stack((pair[0].mean(0), pair[1].mean(0))).unsqueeze(0)
            rows.append(channel.recurrent(sequence)[0][:, -1].squeeze(0))
            continue
        bridge = channel.action.bridge
        if not hasattr(bridge, "trace_modules"):
            rows.append(torch.cat(pair, dim=0).reshape(-1))
            continue
        trace = bridge.trace_modules(pair, seq_len=1)
        relation = trace.get("relation_context")
        if relation is None:
            relation = trace["workspace_timeline"][:, -1]
        rows.append(relation.reshape(-1))
    return torch.stack([row.detach().float().cpu() for row in rows])


@torch.no_grad()
def _raw_codes(channel: SynergyActionChannel,
               pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    rows = []
    for pair in pairs:
        bridge_states = channel._bridge_states([pair])
        rows.append(channel.action.raw_code(bridge_states[0]).squeeze(0))
    return torch.stack([row.detach().float().cpu() for row in rows])


def _normalize(raw: torch.Tensor, mean: torch.Tensor, rho: float) -> torch.Tensor:
    centered = raw - mean
    return rho * centered / centered.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def _normalized_codes(channel: SynergyActionChannel, raw: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "runtime_style":
        mean = channel.action.population_mean.detach().float().cpu()
        return _normalize(raw, mean, channel.action.gate_rho)
    if mode != "train_style":
        raise ValueError(f"unknown normalization mode: {mode}")
    rows = []
    batch_size = RELATION_ROLE_REPAIR_SPEC["batch_size"]
    for start in range(0, len(raw), batch_size):
        batch = raw[start:start + batch_size]
        rows.append(_normalize(batch, batch.mean(0, keepdim=True), channel.action.gate_rho))
    return torch.cat(rows)


def _sensory_probes(train: list, evaluate: list, seed: int) -> dict:
    train_labels = {
        "module_a": torch.tensor([row.module_a for row in train]),
        "module_b": torch.tensor([row.module_b for row in train]),
    }
    eval_labels = {
        "module_a": torch.tensor([row.module_a for row in evaluate]),
        "module_b": torch.tensor([row.module_b for row in evaluate]),
    }
    result = {}
    for source_index, source in enumerate(("quantum", "memory")):
        result[source] = {}
        for module_index, label in enumerate(("module_a", "module_b")):
            train_x = torch.stack([
                getattr(row, source)[module_index].reshape(-1).float() for row in train
            ])
            eval_x = torch.stack([
                getattr(row, source)[module_index].reshape(-1).float() for row in evaluate
            ])
            result[source][label] = _probe(
                train_x, train_labels[label], eval_x, eval_labels[label],
                seed + source_index * 1_000_003 + module_index * 10_007,
            )
    return result


@torch.no_grad()
def _language_metrics(decoder: HFDecoder, channel: SynergyActionChannel, examples: list,
                      arm: str, action_ids: list[int], expected_accuracy: float) -> dict:
    channel.to(decoder.device).eval()
    prompt = _prompt_tokens(decoder, RELATION_ROLE_REPAIR_SPEC)
    actions = torch.tensor(action_ids, device=decoder.device)
    pairs = _source_pairs(examples, arm)
    rows = []
    batch_size = RELATION_ROLE_REPAIR_SPEC["batch_size"]
    for start in range(0, len(pairs), batch_size):
        batch = [tuple(part.to(decoder.device) for part in pair)
                 for pair in pairs[start:start + batch_size]]
        codes = channel.inference_codes(batch)
        tokens = prompt.expand(len(batch), -1)
        gate = codes.unsqueeze(1).expand(-1, prompt.shape[1], -1)
        logits = decoder(tokens, gate, gate_projector=channel.projector)[:, -1, :].float()
        rows.append(logits.index_select(-1, actions).cpu())
    logits = torch.cat(rows)
    labels = torch.tensor([row.target for row in examples])
    predictions = logits.argmax(-1)
    accuracy = float((predictions == labels).float().mean())
    matrix = _confusion(labels, predictions, len(action_ids))
    counts = torch.bincount(predictions, minlength=len(action_ids)).tolist()
    return {
        "accuracy": accuracy,
        "source_accuracy": expected_accuracy,
        "source_accuracy_exact": accuracy == expected_accuracy,
        "selection_counts": {
            action: counts[index] for index, action in enumerate(VALIDITY_SPEC["actions"])
        },
        "confusion_matrix": matrix,
        "per_class_recall": [
            matrix[index][index] / max(sum(matrix[index]), 1) for index in range(len(matrix))
        ],
    }


def _checkpoint(seed: int, arm: str, source_results: dict) -> tuple[dict, dict]:
    expected = VALIDITY_SPEC["checkpoint_sha256"][str(seed)][arm]
    source_rows = {row["seed"]: row for row in source_results["seeds"]}
    if source_rows[seed]["checkpoints"][arm]["sha256"] != expected:
        raise ValueError(f"seed {seed} {arm} source checkpoint receipt changed")
    path = Path(VALIDITY_SPEC["checkpoint_dir"]) / f"seed_{seed}_{arm}.pt"
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"seed {seed} {arm} checkpoint SHA-256 does not match")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("seed") != seed or payload.get("arm") != arm
            or payload.get("spec_sha256") != relation_spec_sha256(RELATION_ROLE_REPAIR_SPEC)):
        raise ValueError(f"seed {seed} {arm} checkpoint identity is invalid")
    return payload, {"path": str(path), "sha256": digest}


def run_seed(decoder: HFDecoder, seed: int, source_results: dict,
             action_ids: list[int]) -> dict:
    spec = RELATION_ROLE_REPAIR_SPEC
    train = build_examples(seed, "train", spec)
    evaluate = build_examples(seed, "eval", spec)
    sensory = _sensory_probes(train, evaluate, seed)
    labels_train = torch.tensor([row.target for row in train])
    labels_eval = torch.tensor([row.target for row in evaluate])
    source_rows = {row["seed"]: row for row in source_results["seeds"]}
    arms = {}
    checkpoints = {}
    for arm_index, arm in enumerate(VALIDITY_SPEC["arms"]):
        saved, receipt = _checkpoint(seed, arm, source_results)
        state = saved["channel"]
        d_model = int(state["action.projector.weight"].shape[0])
        channel = SynergyActionChannel(arm, spec["state_dim"], d_model, spec).eval()
        channel.load_state_dict(state, strict=True)
        train_pairs = _source_pairs(train, arm)
        eval_pairs = _source_pairs(evaluate, arm)
        relation_train = _relation_features(channel, train_pairs)
        relation_eval = _relation_features(channel, eval_pairs)
        raw_train = _raw_codes(channel, train_pairs)
        raw_eval = _raw_codes(channel, eval_pairs)
        direct = {}
        normalized = {}
        for mode_index, mode in enumerate(VALIDITY_SPEC["normalization_modes"]):
            normalized[mode] = (
                _normalized_codes(channel, raw_train, mode),
                _normalized_codes(channel, raw_eval, mode),
            )
            direct[mode] = _probe(
                normalized[mode][0], labels_train, normalized[mode][1], labels_eval,
                seed + arm_index * 100_003 + mode_index * 1_000_003,
            )
        drift = normalized["train_style"][1] - normalized["runtime_style"][1]
        expected_accuracy = source_rows[seed]["arms"][arm]["conditions"]["normal"]["accuracy"]
        language = _language_metrics(decoder, channel, evaluate, arm, action_ids, expected_accuracy)
        arms[arm] = {
            "relation": _probe(
                relation_train, labels_train, relation_eval, labels_eval,
                seed + arm_index * 1_000_003 + 9_000_001,
            ),
            "direct_action": direct,
            "normalization": {
                "eval_rms_difference": float(drift.pow(2).mean().sqrt()),
                "population_mean_ready": bool(channel.action.mean_ready),
            },
            "language": language,
        }
        checkpoints[arm] = receipt
        print(
            f"[seed {seed}:{arm}] relation={arms[arm]['relation']['accuracy']:.3f} "
            f"train={direct['train_style']['accuracy']:.3f} "
            f"runtime={direct['runtime_style']['accuracy']:.3f} "
            f"language={language['accuracy']:.3f}",
            flush=True,
        )
        del channel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"seed": seed, "sensory": sensory, "arms": arms, "checkpoints": checkpoints}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/validity_results.json")
    parser.add_argument("--verdict", default="measurement/validity_verdict.json")
    parser.add_argument("--model", default=VALIDITY_SPEC["model"])
    args = parser.parse_args()
    source_results, _, source = _load_source()
    spec = RELATION_ROLE_REPAIR_SPEC
    audits = {
        split: audit_examples(build_examples(spec["seeds"][0], split, spec))
        for split in ("train", "eval")
    }
    if audits != source_results.get("dataset_audit"):
        raise ValueError("recreated RELATION-1 dataset audit changed")
    decoder = HFDecoder(
        args.model, lora=False, freeze_base=True,
        gate_strength=spec["gate_strength"], gate_rms_max=spec["gate_rms_max"],
    )
    decoder.model.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    action_ids = _action_token_ids(decoder, spec)
    payload = {
        "experiment": VALIDITY_SPEC["experiment"],
        "spec": VALIDITY_SPEC,
        "spec_sha256": spec_sha256(VALIDITY_SPEC),
        "model": args.model,
        "source": source,
        "dataset_audit": audits,
        "action_tokens": {
            "token_ids": dict(zip(VALIDITY_SPEC["actions"], action_ids)),
            "unique_single_tokens": len(set(action_ids)) == len(action_ids),
        },
        "seeds": [run_seed(decoder, seed, source_results, action_ids)
                  for seed in VALIDITY_SPEC["seeds"]],
    }
    _atomic_json(Path(args.output), payload)
    from measurement.validity_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
