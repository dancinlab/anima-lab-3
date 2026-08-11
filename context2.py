#!/usr/bin/env python3
"""CONTEXT-2: route raw context+key states through the common memory API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from context import (
    _component_address, _composite, _composite_prediction, _load_key_projector,
)
from episode import _decode
from graft_behavior import sha256_file
from key_stability import StableKeyProjector
from measurement.context2_registry import CONTEXT2_SPEC, spec_sha256
from measurement.context_gate import adjudicate as adjudicate_context
from measurement.context_registry import CONTEXT_SPEC, spec_sha256 as context_spec_sha256
from measurement.projector_registry import evaluation_name
from separation import (
    _arm_metrics,
    _direct_prediction,
    _exact_addresses,
    build_episodes,
    dataset_audit,
    trace_similar_episode,
)
from trinity import VectorMemory


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _receipt(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _source_receipt(spec: dict = CONTEXT2_SPEC) -> tuple[dict, dict]:
    results_path = Path(spec["source_results"])
    verdict_path = Path(spec["source_verdict_path"])
    results = json.loads(results_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    expected_sha = context_spec_sha256(CONTEXT_SPEC)
    if (
        results.get("experiment") != spec["source_experiment"]
        or results.get("spec") != CONTEXT_SPEC
        or results.get("spec_sha256") != expected_sha
        or verdict.get("verdict") != spec["source_verdict"]
        or verdict.get("spec_sha256") != expected_sha
        or adjudicate_context(results) != verdict
    ):
        raise RuntimeError("registered CONTEXT-1 source changed")
    source = results["source_separation2"]
    receipts = (
        results["context_checkpoint"], source["canonical_checkpoint"],
        *source["prototype_checkpoints"].values(),
    )
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError("registered CONTEXT-1 checkpoint changed")
    return results, {
        "results": _receipt(results_path),
        "verdict": _receipt(verdict_path),
        "source_spec_sha256": expected_sha,
        "context_checkpoint": dict(results["context_checkpoint"]),
        "canonical_checkpoint": dict(source["canonical_checkpoint"]),
        "prototype_checkpoints": {
            key: dict(value) for key, value in source["prototype_checkpoints"].items()
        },
    }


def _load_context_projector(receipt: dict,
                            spec: dict = CONTEXT2_SPEC) -> StableKeyProjector:
    checkpoint = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != CONTEXT_SPEC["experiment"]
        or checkpoint.get("spec_sha256") != context_spec_sha256(CONTEXT_SPEC)
        or checkpoint.get("fit_method") != spec["fit_method"]
        or checkpoint.get("model_class") != spec["model_class"]
    ):
        raise RuntimeError("CONTEXT-1 context projector identity changed")
    model = StableKeyProjector(
        spec["state_dim"], spec["component_address_dim"], spec["contexts"],
        CONTEXT_SPEC["temperature"], CONTEXT_SPEC["bias"],
    )
    model.load_state_dict(checkpoint["projector"])
    model.eval()
    model.requires_grad_(False)
    return model


class CompositeStateTransform:
    """Validated experiment adapter for the registered context+key projectors."""

    def __init__(self, context_projector, key_projector, spec: dict = CONTEXT2_SPEC,
                 *, mask_context: bool = False, mask_key: bool = False,
                 center_context: bool = False):
        self.context_projector = context_projector
        self.key_projector = key_projector
        self.spec = spec
        self.mask_context = mask_context
        self.mask_key = mask_key
        self.center_context = center_context
        self.calls = 0
        self.component_counts: list[int] = []
        self.address_widths: list[int] = []
        self.outputs: list[torch.Tensor] = []

    def __call__(self, components) -> torch.Tensor:
        if not isinstance(components, (tuple, list)):
            raise TypeError("composite address transform requires a component sequence")
        if len(components) != self.spec["components_per_key"]:
            raise ValueError("composite address transform received the wrong component count")
        for component in components:
            if (
                not isinstance(component, torch.Tensor)
                or component.dim() != 2
                or component.shape[1] != self.spec["state_dim"]
                or not self.spec["minimum_cells"] <= component.shape[0]
                <= self.spec["maximum_cells"]
                or not torch.isfinite(component).all()
            ):
                raise ValueError("composite address component changed shape or became non-finite")
        if self.center_context:
            context_state = components[0].mean(0).unsqueeze(0)
            context_label = int(self.context_projector(context_state).argmax(1)[0])
            context = F.normalize(
                self.context_projector.prototypes.detach(), dim=-1
            )[context_label] * self.spec["component_weight"]
            key = _component_address(
                self.key_projector, components[1]
            ) * self.spec["component_weight"]
            if self.mask_context:
                context = torch.zeros_like(context)
            if self.mask_key:
                key = torch.zeros_like(key)
            address = torch.cat((context, key))
            if (
                address.numel() != self.spec["composite_address_dim"]
                or not torch.isfinite(address).all()
            ):
                raise RuntimeError(
                    "composite memory address changed shape or became non-finite"
                )
        else:
            address = _composite(
                self.context_projector, self.key_projector, components[0], components[1],
                self.spec, mask_context=self.mask_context, mask_key=self.mask_key,
            )
        self.calls += 1
        self.component_counts.append(len(components))
        self.address_widths.append(address.numel())
        self.outputs.append(address.detach().clone())
        return address


def _integrated_prediction(trace: dict, prototypes: torch.Tensor,
                           context_projector: StableKeyProjector,
                           key_projector: StableKeyProjector,
                           spec: dict = CONTEXT2_SPEC, *, mask_context: bool = False,
                           rotate: bool = False):
    transform = CompositeStateTransform(
        context_projector, key_projector, spec, mask_context=mask_context
    )
    memory = VectorMemory(
        capacity=spec["events_per_episode"], dim=spec["state_dim"],
        key_transform=transform,
    )
    stored_values = trace["values"][1:] + trace["values"][:1] if rotate else trace["values"]
    for context_state, key_state, value_state in zip(
        trace["contexts"], trace["keys"], stored_values
    ):
        memory.store((context_state, key_state), value_state)
    retrieved = memory.retrieve((trace["query_context"], trace["query"]), top_k=1)[0]
    query = transform.outputs[-1]
    similarities = torch.stack([
        F.cosine_similarity(query, address, dim=0) for address in memory.keys
    ])
    selected = int(similarities.argmax())
    outcome = (
        _decode(retrieved, prototypes), selected,
        bool(torch.equal(retrieved, stored_values[selected].mean(0))),
        float(similarities.max() - similarities.min()),
    )
    audit = {
        "calls": transform.calls,
        "minimum_components": min(transform.component_counts),
        "maximum_components": max(transform.component_counts),
        "minimum_address_width": min(transform.address_widths),
        "maximum_address_width": max(transform.address_widths),
        "stored_keys": len(memory.keys),
        "retrievals": 1,
    }
    return outcome, audit


def run_evaluation(prototype_seed: int, engine_seed: int, episodes, source_results: dict,
                   source: dict, spec: dict = CONTEXT2_SPEC) -> dict:
    context_projector = _load_context_projector(source["context_checkpoint"], spec)
    key_projector = _load_key_projector(source["canonical_checkpoint"], spec)
    before_context = {
        name: value.detach().clone() for name, value in context_projector.state_dict().items()
    }
    before_key = {
        name: value.detach().clone() for name, value in key_projector.state_dict().items()
    }
    prototype_receipt = source["prototype_checkpoints"][str(prototype_seed)]
    checkpoint = torch.load(prototype_receipt["path"], map_location="cpu", weights_only=True)
    prototypes = checkpoint["prototypes"]["quantum"]
    expected = torch.tensor([episode.target for episode in episodes])
    positions = [episode.query_position for episode in episodes]
    records = {
        name: {"predictions": [], "selections": [], "contents": [], "api": [], "margins": []}
        for name in spec["arms"]
    }
    call_audits = {
        name: {"calls": [], "components": [], "widths": [], "stores": [], "retrievals": []}
        for name in (
            "integrated_composite_normal", "integrated_context_masked",
            "integrated_composite_recovered",
        )
    }
    episode_seeds, cell_counts = [], []
    before_digests, after_digests, query_rng_digests = [], [], []
    base = spec["episode_seed_base"] + engine_seed * spec["seed_stride"]
    for index, episode in enumerate(episodes):
        trial_seed = base + index
        episode_seeds.append(trial_seed)
        similar = trace_similar_episode(episode, trial_seed, distinct=False, spec=spec)
        paired_distinct = trace_similar_episode(episode, trial_seed, distinct=True, spec=spec)
        for trace in (similar, paired_distinct):
            cell_counts.extend(trace["cell_counts"])
            before_digests.append(trace["update_audit"]["state_before_sha256"])
            after_digests.append(trace["update_audit"]["state_after_sha256"])
            query_rng_digests.append(trace["update_audit"]["query_rng_sha256"])

        normal, normal_audit = _integrated_prediction(
            similar, prototypes, context_projector, key_projector, spec
        )
        reference = _composite_prediction(
            similar, prototypes, context_projector, key_projector, spec
        )
        masked, masked_audit = _integrated_prediction(
            similar, prototypes, context_projector, key_projector, spec, mask_context=True
        )
        recovered, recovered_audit = _integrated_prediction(
            similar, prototypes, context_projector, key_projector, spec
        )
        exact, exact_query = _exact_addresses(episode, spec=spec)
        outcomes = {
            "integrated_composite_normal": normal,
            "external_composite_reference": reference,
            "integrated_context_masked": masked,
            "exact_context_key_control": _direct_prediction(
                exact, similar["values"], exact_query, prototypes
            ),
            "exact_context_key_partner_swap": _direct_prediction(
                exact, similar["values"], exact_query, prototypes, rotate=True
            ),
            "integrated_composite_recovered": recovered,
        }
        for name, audit in (
            ("integrated_composite_normal", normal_audit),
            ("integrated_context_masked", masked_audit),
            ("integrated_composite_recovered", recovered_audit),
        ):
            target = call_audits[name]
            target["calls"].append(audit["calls"])
            target["components"].extend((audit["minimum_components"], audit["maximum_components"]))
            target["widths"].extend((audit["minimum_address_width"], audit["maximum_address_width"]))
            target["stores"].append(audit["stored_keys"])
            target["retrievals"].append(audit["retrievals"])
        content = _decode(similar["values"][episode.query_position], prototypes)
        for name, outcome in outcomes.items():
            row = records[name]
            row["predictions"].append(outcome[0])
            row["selections"].append(outcome[1])
            row["contents"].append(content)
            row["api"].append(outcome[2])
            row["margins"].append(outcome[3])
        if (index + 1) % 256 == 0:
            print(
                f"[prototype {prototype_seed} engine {engine_seed}] "
                f"evaluated {index + 1}/{len(episodes)} episodes", flush=True,
            )

    arms = {
        name: _arm_metrics(
            expected, row["predictions"], row["selections"], positions,
            row["contents"], row["api"], row["margins"], spec,
        )
        for name, row in records.items()
    }
    normal_records = records["integrated_composite_normal"]
    reference_records = records["external_composite_reference"]
    arms["integrated_composite_normal"]["reference_prediction_match"] = float(
        normal_records["predictions"] == reference_records["predictions"]
    )
    arms["integrated_composite_normal"]["reference_selection_match"] = float(
        normal_records["selections"] == reference_records["selections"]
    )
    arms["integrated_composite_recovered"]["prediction_match"] = float(
        records["integrated_composite_recovered"]["predictions"]
        == normal_records["predictions"]
    )
    source_evaluation = {
        row["name"]: row for row in source_results["evaluations"]
    }[evaluation_name({"prototype_seed": prototype_seed, "engine_seed": engine_seed})]
    state_audit = {
        "episodes": len(episodes),
        "unique_episode_seeds": len(set(episode_seeds)),
        "episode_seed_sha256": hashlib.sha256(
            "\n".join(map(str, episode_seeds)).encode()
        ).hexdigest(),
        "minimum_cells": min(cell_counts),
        "maximum_cells": max(cell_counts),
    }
    update_audit = {
        "requested_updates": spec["settling_updates"],
        "performed_updates_minimum": spec["settling_updates"],
        "performed_updates_maximum": spec["settling_updates"],
        "disabled": list(spec["pre_query_dynamics_ablation"]),
        "state_before_sha256": hashlib.sha256("\n".join(before_digests).encode()).hexdigest(),
        "state_after_sha256": hashlib.sha256("\n".join(after_digests).encode()).hexdigest(),
        "query_rng_sha256": hashlib.sha256("\n".join(query_rng_digests).encode()).hexdigest(),
    }
    return {
        "prototype_seed": prototype_seed,
        "engine_seed": engine_seed,
        "arms": arms,
        "memory_path_audit": {
            name: {
                "minimum_calls": min(row["calls"]),
                "maximum_calls": max(row["calls"]),
                "minimum_components": min(row["components"]),
                "maximum_components": max(row["components"]),
                "minimum_address_width": min(row["widths"]),
                "maximum_address_width": max(row["widths"]),
                "minimum_stores": min(row["stores"]),
                "maximum_stores": max(row["stores"]),
                "minimum_retrievals": min(row["retrievals"]),
                "maximum_retrievals": max(row["retrievals"]),
            }
            for name, row in call_audits.items()
        },
        "integration_audit": {
            "component_weight": spec["component_weight"],
            "component_address_dim": spec["component_address_dim"],
            "composite_address_dim": spec["composite_address_dim"],
            "context_projector_frozen": not any(
                parameter.requires_grad for parameter in context_projector.parameters()
            ),
            "context_projector_unchanged": all(
                torch.equal(before_context[name], context_projector.state_dict()[name])
                for name in before_context
            ),
            "key_projector_frozen": not any(
                parameter.requires_grad for parameter in key_projector.parameters()
            ),
            "key_projector_unchanged": all(
                torch.equal(before_key[name], key_projector.state_dict()[name])
                for name in before_key
            ),
            "source_normal_metrics_match": (
                arms["external_composite_reference"]
                == source_evaluation["arms"]["composite_context_key_normal"]
            ),
            "source_state_audit_match": state_audit == source_evaluation["state_audit"],
            "source_update_audit_match": update_audit == source_evaluation["update_audit"],
        },
        "state_audit": state_audit,
        "update_audit": update_audit,
        "prototype_checkpoint": prototype_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="measurement/context2_results.json")
    parser.add_argument("--verdict", default="measurement/context2_verdict.json")
    args = parser.parse_args()
    spec = CONTEXT2_SPEC
    source_results, source = _source_receipt(spec)
    episodes = build_episodes(spec)
    evaluations = [
        {
            "name": evaluation_name(row),
            **run_evaluation(
                row["prototype_seed"], row["engine_seed"], episodes,
                source_results, source, spec,
            ),
        }
        for row in spec["evaluation_combinations"]
    ]
    payload = {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "source_context1": source,
        "evaluation_dataset_audit": dataset_audit(episodes, spec),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": spec["device"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(Path(args.output), payload)
    from measurement.context2_gate import adjudicate
    verdict = adjudicate(payload)
    _atomic_json(Path(args.verdict), verdict)
    print(f"[{verdict['verdict']}] {verdict['reason']}")


if __name__ == "__main__":
    main()
