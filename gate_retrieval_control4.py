#!/usr/bin/env python3
"""GATE-RETRIEVAL-CONTROL-4: order-independent within-pool score swap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from gate_control1 import FrozenSentenceEncoder
from gate_retrieval_control2 import _content_rankings, _metrics
from gate_retrieval_control3 import _seed_measurements
from measurement.balanced_retrieval_control_registry import BALANCED_RETRIEVAL_CONTROL_SPEC
from measurement.content_swap_retrieval_control_registry import (
    CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC,
    spec_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def swap_pool_scores(scores: torch.Tensor, pools: list[list[int]]) -> torch.Tensor:
    if scores.dim() != 2 or len(pools) != scores.shape[0]:
        raise ValueError("score rows must match every address pool")
    swapped = scores.clone()
    for row_index, pool in enumerate(pools):
        if len(pool) != 2 or len(set(pool)) != 2 or any(
            index < 0 or index >= scores.shape[1] for index in pool
        ):
            raise ValueError("every registered address pool must contain two unique candidates")
        left, right = pool
        swapped[row_index, left] = scores[row_index, right]
        swapped[row_index, right] = scores[row_index, left]
    return swapped


def _digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _digest_pools(pools: list[list[int]]) -> str:
    body = json.dumps(pools, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _swap_audit(normal: torch.Tensor, swapped: torch.Tensor, restored: torch.Tensor,
                pools: list[list[int]]) -> dict:
    outside_unchanged = True
    pair_exchange_exact = True
    for row_index, pool in enumerate(pools):
        selected = set(pool)
        outside_unchanged &= all(
            torch.equal(normal[row_index, index], swapped[row_index, index])
            for index in range(normal.shape[1]) if index not in selected
        )
        left, right = pool
        pair_exchange_exact &= bool(
            torch.equal(swapped[row_index, left], normal[row_index, right])
            and torch.equal(swapped[row_index, right], normal[row_index, left])
        )
    multisets_preserved = bool(torch.equal(
        torch.sort(normal, dim=1).values,
        torch.sort(swapped, dim=1).values,
    ))
    return {
        "pool_rows": len(pools),
        "pool_size": 2,
        "swapped_pairs": len(pools),
        "uses_labels": False,
        "uses_episode_order": False,
        "outside_pool_unchanged": bool(outside_unchanged),
        "pair_exchange_exact": bool(pair_exchange_exact),
        "score_multisets_preserved": multisets_preserved,
        "restored_scores_exact": bool(torch.equal(normal, restored)),
        "normal_pool_sha256": _digest_pools(pools),
        "normal_scores_sha256": _digest_tensor(normal),
        "swapped_scores_sha256": _digest_tensor(swapped),
        "restored_scores_sha256": _digest_tensor(restored),
    }


def run_seed(seed: int, encoder: FrozenSentenceEncoder,
             spec: dict = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC) -> dict:
    baseline, state = _seed_measurements(seed, encoder, BALANCED_RETRIEVAL_CONTROL_SPEC)
    episodes = state["episodes"]
    normal_scores = state["content_scores"]
    pools = state["normal_pools"]
    swapped_scores = swap_pool_scores(normal_scores, pools)
    restored_scores = swap_pool_scores(swapped_scores, pools)
    swapped_rankings = _content_rankings(pools, swapped_scores)
    restored_rankings = _content_rankings(pools, restored_scores)
    arms = {
        "within_pool_content_swap": _metrics(
            episodes, swapped_rankings, swapped_scores,
            fact_kinds=spec["fact_kinds"], fact_positions=spec["fact_positions"],
        ),
        "restored_content": _metrics(
            episodes, restored_rankings, restored_scores,
            fact_kinds=spec["fact_kinds"], fact_positions=spec["fact_positions"],
        ),
    }
    audit = _swap_audit(normal_scores, swapped_scores, restored_scores, pools)
    audit["restored_rankings_exact"] = (
        arms["restored_content"]["rankings_sha256"]
        == baseline["arms"]["balanced_split"]["rankings_sha256"]
    )
    return {
        "seed": seed,
        "baseline": baseline,
        "swap_audit": audit,
        "arms": arms,
    }


def run(spec: dict = CONTENT_SWAP_RETRIEVAL_CONTROL_SPEC) -> dict:
    torch.use_deterministic_algorithms(True)
    encoder = FrozenSentenceEncoder(spec["encoder"])
    return {
        "experiment": spec["experiment"],
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": spec["encoder"]["device"],
        },
        "seeds": [run_seed(seed, encoder, spec) for seed in spec["seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/content_swap_retrieval_control_results.json"),
    )
    args = parser.parse_args()
    _atomic_json(args.output, run())


if __name__ == "__main__":
    main()
