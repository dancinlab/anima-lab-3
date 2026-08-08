"""Canonical corpus-family and arm registry for the λ runtime."""
from __future__ import annotations

import os


FLOORS = {
    "25": {"unigram": 6.0293, "bigram": 3.6140, "corpus": "corpus_merged_25.txt", "regime": "constructed"},
    "50": {"unigram": 6.0295, "bigram": 3.5925, "corpus": "corpus_merged_50.txt", "regime": "constructed"},
    "50c": {"unigram": 6.0275, "bigram": 3.5934, "corpus": "corpus_merged_50c.txt", "regime": "constructed"},
    "100": {"unigram": 6.0195, "bigram": 3.6010, "corpus": "corpus_merged_dedup.txt", "regime": "constructed"},
    "v2": {"unigram": 5.9548, "bigram": 3.4920, "corpus": "corpus_v2.txt", "regime": "constructed"},
    "nat": {"unigram": 5.4355, "bigram": 3.3634, "corpus": "corpus_natural_ko_dedup.txt", "regime": "natural", "register": "encyclopedic prose"},
    "nat25": {"unigram": 5.4361, "bigram": 3.3627, "corpus": "corpus_nat_25.txt", "regime": "natural", "register": "encyclopedic prose"},
    "nat50": {"unigram": 5.4357, "bigram": 3.3636, "corpus": "corpus_nat_50.txt", "regime": "natural", "register": "encyclopedic prose"},
    "lit": {"unigram": 5.353926841048547, "bigram": 3.4055540943296356,
            "corpus": "corpus_natural_literary_ko_dedup.txt", "regime": "natural",
            "register": "public-domain literary prose"},
}


def arm(floor, label, checkpoint=None, sibling=None, family=None):
    return {"floor": floor, "label": label, "checkpoint": checkpoint,
            "sibling": sibling, "family": family}


ARMS = {
    "s25": arm("25", "25% seed 1337 best", "checkpoints/arm_s25/best.pt", "v25", "constructed"),
    "v25": arm("25", "25% seed 7331 best", "checkpoints/arm_v25/best.pt", "s25", "constructed"),
    "s50": arm("50", "50% seed 1337 best", "checkpoints/arm_s50/best.pt", "v50", "constructed"),
    "v50": arm("50", "50% seed 7331 best", "checkpoints/arm_v50/best.pt", "s50", "constructed"),
    "s100": arm("100", "100% seed 1337 best", "checkpoints/arm_a_data/best.pt", "v100", "constructed"),
    "v100": arm("100", "100% seed 7331 best", "checkpoints/arm_v100/best.pt", "s100", "constructed"),
    "p50": arm("50", "50% phase-ablated best", "checkpoints/arm_p50/best.pt", family="constructed"),
    "p50f": arm("50", "50% phase-ablated final", "checkpoints/arm_p50/final.pt", family="constructed"),
    "p100": arm("100", "100% phase-ablated best", "checkpoints/arm_p100/best.pt", family="constructed"),
    "p100f": arm("100", "100% phase-ablated final", "checkpoints/arm_p100/final.pt", family="constructed"),
    "e50": arm("50", "50% exposure-equalised best", "checkpoints/arm_e50/best.pt", family="constructed"),
    "e50f": arm("50", "50% exposure-equalised final", "checkpoints/arm_e50/final.pt", family="constructed"),
    "e100": arm("100", "100% exposure-equalised best", "checkpoints/arm_e100/best.pt", family="constructed"),
    "e100f": arm("100", "100% exposure-equalised final", "checkpoints/arm_e100/final.pt", family="constructed"),
    "s25f": arm("25", "25% seed 1337 final", "checkpoints/arm_s25/final.pt", family="constructed"),
    "v25f": arm("25", "25% seed 7331 final", "checkpoints/arm_v25/final.pt", family="constructed"),
    "s50f": arm("50", "50% seed 1337 final", "checkpoints/arm_s50/final.pt", family="constructed"),
    "v50f": arm("50", "50% seed 7331 final", "checkpoints/arm_v50/final.pt", family="constructed"),
    "s100f": arm("100", "100% seed 1337 final", "checkpoints/arm_a_data/final.pt", family="constructed"),
    "v100f": arm("100", "100% seed 7331 final", "checkpoints/arm_v100/final.pt", family="constructed"),
    "p25": arm("25", "25% phase-ablated best", "checkpoints/arm_p25/best.pt", family="constructed"),
    "p25f": arm("25", "25% phase-ablated final", "checkpoints/arm_p25/final.pt", family="constructed"),
    "c50": arm("50c", "50% complement best", "checkpoints/arm_50c/best.pt", family="constructed"),
    "c50f": arm("50c", "50% complement final", "checkpoints/arm_50c/final.pt", family="constructed"),
    "nf9_12k": arm("v2", "nf9 300M @12,000"),
    "nf9_14k": arm("v2", "nf9 300M @14,000"),
    "nf9_20k": arm("v2", "nf9 300M @20,000 (controls measured on CPU)"),
    "nat": arm("nat", "natural corpus best", "checkpoints/arm_nat/best.pt", family="encyclopedic"),
    "natf": arm("nat", "natural corpus final", "checkpoints/arm_nat/final.pt", family="encyclopedic"),
    "natctx": arm("nat", "natural · context 512", "checkpoints/arm_nat_ctx512/best.pt", family="encyclopedic"),
    "natdrop": arm("nat", "natural · dropout 0.3", "checkpoints/arm_nat_drop3/best.pt", family="encyclopedic"),
    "natdrop5": arm("nat", "natural · dropout 0.5", "checkpoints/arm_nat_drop5/best.pt", family="encyclopedic"),
    "natdrop4": arm("nat", "natural · dropout 0.4 · seed 1337", "checkpoints/arm_nat_drop4/best.pt", "natdrop4v", "encyclopedic"),
    "natdrop4v": arm("nat", "natural · dropout 0.4 · seed 7331", "checkpoints/arm_nat_drop4v/best.pt", "natdrop4", "encyclopedic"),
    "natdrop35": arm("nat", "natural · dropout 0.35 · seed 1337", "checkpoints/arm_nat_drop35/best.pt", "natdrop35v", "encyclopedic"),
    "natdrop35v": arm("nat", "natural · dropout 0.35 · seed 7331", "checkpoints/arm_nat_drop35v/best.pt", "natdrop35", "encyclopedic"),
    "natdrop37": arm("nat", "natural · dropout 0.37 · seed 1337", "checkpoints/arm_nat_drop37/best.pt", "natdrop37v", "encyclopedic"),
    "natdrop37v": arm("nat", "natural · dropout 0.37 · seed 7331", "checkpoints/arm_nat_drop37v/best.pt", "natdrop37", "encyclopedic"),
    "n25drop37": arm("nat25", "natural 25% · dropout 0.37 · seed 1337", "checkpoints/arm_nat25_drop37/best.pt", "n25drop37v", "encyclopedic"),
    "n25drop37v": arm("nat25", "natural 25% · dropout 0.37 · seed 7331", "checkpoints/arm_nat25_drop37v/best.pt", "n25drop37", "encyclopedic"),
    "n25drop42": arm("nat25", "natural 25% · dropout 0.42 · seed 1337", "checkpoints/arm_nat25_drop42/best.pt", "n25drop42v", "encyclopedic"),
    "n25drop42v": arm("nat25", "natural 25% · dropout 0.42 · seed 7331", "checkpoints/arm_nat25_drop42v/best.pt", "n25drop42", "encyclopedic"),
    "n50drop37": arm("nat50", "natural 50% · dropout 0.37 · seed 1337", "checkpoints/arm_nat50_drop37/best.pt", "n50drop37v", "encyclopedic"),
    "n50drop37v": arm("nat50", "natural 50% · dropout 0.37 · seed 7331", "checkpoints/arm_nat50_drop37v/best.pt", "n50drop37", "encyclopedic"),
    "nat25": arm("nat25", "natural 25% best", "checkpoints/arm_nat25/best.pt", family="encyclopedic"),
    "nat25f": arm("nat25", "natural 25% final", "checkpoints/arm_nat25/final.pt", family="encyclopedic"),
    "nat50": arm("nat50", "natural 50% best", "checkpoints/arm_nat50/best.pt", family="encyclopedic"),
    "nat50f": arm("nat50", "natural 50% final", "checkpoints/arm_nat50/final.pt", family="encyclopedic"),
    "litdrop37": arm("lit", "literary · dropout 0.37 · seed 1337", "checkpoints/arm_lit_drop37/best.pt", "litdrop37v", "literary"),
    "litdrop37v": arm("lit", "literary · dropout 0.37 · seed 7331", "checkpoints/arm_lit_drop37v/best.pt", "litdrop37", "literary"),
    "litstd37": arm("lit", "literary · standard FFN · dropout 0.37 · seed 1337", "checkpoints/arm_lit_standard_drop37/best.pt", "litstd37v", "literary"),
    "litstd37v": arm("lit", "literary · standard FFN · dropout 0.37 · seed 7331", "checkpoints/arm_lit_standard_drop37v/best.pt", "litstd37", "literary"),
    "nat300m37": arm("nat", "300M natural · dropout 0.37 · seed 1337", "checkpoints/arm_nat_300m_drop37/best.pt", "nat300m37v", "encyclopedic"),
    "nat300m37v": arm("nat", "300M natural · dropout 0.37 · seed 7331", "checkpoints/arm_nat_300m_drop37v/best.pt", "nat300m37", "encyclopedic"),
}


FAMILIES = {
    "constructed": {
        "regime": "constructed", "register": "mixed generated drills",
        "corpus": "data/corpus_merged_dedup.txt", "fresh": None,
        "screen_corpora": ["data/corpus_merged_25.txt", "data/corpus_merged_50.txt",
                           "data/corpus_merged_dedup.txt"],
    },
    "encyclopedic": {
        "regime": "natural", "register": "encyclopedic prose",
        "corpus": "data/corpus_natural_ko_dedup.txt",
        "fresh": "data/corpus_natural_fresh.txt",
        "screen_corpora": ["data/corpus_nat_25.txt", "data/corpus_nat_50.txt",
                           "data/corpus_natural_ko_dedup.txt"],
    },
    "literary": {
        "regime": "natural", "register": "public-domain literary prose",
        "corpus": "data/corpus_natural_literary_ko_dedup.txt",
        "fresh": "data/corpus_natural_literary_fresh.txt",
        "screen_corpora": ["data/corpus_natural_literary_ko_dedup.txt"],
    },
}

ALIASES = {"natural": "encyclopedic", "encyclopedic": "encyclopedic",
           "literary": "literary", "constructed": "constructed"}

# The canonical λ4 roster. Panel/G-gates also score selection/final controls that
# were never registered for recombination; absence of λ4 on those is not pending.
LAMBDA4_ARMS = {
    "nat", "natf", "nat25", "nat50", "natctx", "natdrop", "natdrop5",
    "natdrop4", "natdrop4v", "natdrop35", "natdrop35v", "natdrop37",
    "natdrop37v", "n25drop37", "n25drop37v", "n25drop42", "n25drop42v",
    "n50drop37", "n50drop37v", "litdrop37", "litdrop37v",
    "litstd37", "litstd37v",
    "nat300m37", "nat300m37v",
}


# Result receipts are registered once here.  Gate/preflight and the remote
# experiment runner consume this same roster, so adding a family cannot leave a
# measurement file invisible to one of the validity checks.
RESULT_SETS = {
    "constructed": {"panel": "measurement/panel_results.json"},
    "nf9": {"panel": "measurement/panel_nf9_results.json"},
    "encyclopedic": {
        "panel": "measurement/panel_nat_results.json",
        "g_gates": "measurement/g_gates_nat_results.json",
        "lambda4": "measurement/lambda4_results.json",
    },
    "literary": {
        "panel": "measurement/panel_literary_results.json",
        "g_gates": "measurement/g_gates_literary_results.json",
        "lambda4": "measurement/lambda4_literary_results.json",
    },
    "scale300m": {
        "panel": "measurement/panel_scale300m_results.json",
        "g_gates": "measurement/g_gates_scale300m_results.json",
        "lambda4": "measurement/lambda4_scale300m_results.json",
    },
    "ffn_control": {
        "panel": "measurement/panel_ffn_control_results.json",
        "g_gates": "measurement/g_gates_ffn_control_results.json",
        "lambda4": "measurement/lambda4_ffn_control_results.json",
    },
}


# Prospective experiments are executable registry entries rather than shell
# recipes.  `trainer_args` are passed to the canonical trainer unchanged; a
# physical batch of 4 with eight accumulated micro-batches preserves the 32×256
# effective batch and optimizer-step schedule of the 27.7M reference arms.
EXPERIMENTS = {
    "scale300m": {
        "hypothesis": "LAMBDA-3",
        "family": "encyclopedic",
        "arms": ("nat300m37", "nat300m37v"),
        "seeds": {"nat300m37": 1337, "nat300m37v": 7331},
        "trainer": "train_conscious_lm.py",
        "expected_params": 299_420_896,
        "corpus_sha256": "10136c7229a242ceef55015d3f0eb88071cb05670c9998128aed874c88e85f87",
        "fresh_sha256": "f96f00a7c721a6c5870655ead22a2d4290310b08620f0bf00a1e9c50c7647c1b",
        "trainer_args": {
            "dim": 896, "layers": 12, "heads": 14,
            "batch_size": 4, "grad_accum_steps": 8, "block_size": 256,
            "lr": 1e-4, "max_cells": 16, "val_bytes": 262144,
            "eval_every": 250, "phase": "language", "dropout": 0.37,
            "steps": 12000, "save_every": 6000, "log_every": 100,
        },
        "results": "scale300m",
    },
    "lambda4_consciousness_causality": {
        "hypothesis": "LAMBDA-4",
        "family": "literary",
        "arms": ("litdrop37", "litdrop37v"),
        "measurement_only": True,
        "corpus_sha256": "336e101a5b9737c2e12073b5562a06320c150b5a19655a8046b7c16e13ddff5e",
        "fresh_sha256": "8e196165d525e15bc4b200e395953b19d6007acd0cb2c65746649dc4acb5cecd",
        "checkpoint_sha256": {
            "litdrop37": "d1fd4fd523ccfb58f7408cdffd42687f866d0dc21c966a61ca4e1cfeb92e200d",
            "litdrop37v": "a3de90008d532d5551bf5ec4d3e41ffa5dd5e1b55d1c86f3890e2a471736ffc9",
        },
        "interventions": ("normal", "off", "shuffle", "noise"),
        "intervention_seed": 20260809,
        "scorers": (
            {
                "axis": "panel",
                "script": "measurement/panel.py",
                "output": "measurement/panel_consciousness_causality_results.json",
            },
            {
                "axis": "lambda4",
                "script": "measurement/lambda4.py",
                "output": "measurement/lambda4_consciousness_causality_results.json",
            },
            {
                "axis": "verdict",
                "script": "measurement/consciousness_causality_gate.py",
                "output": "measurement/consciousness_causality_verdict.json",
            },
        ),
    },
    "ffn_structural_control": {
        "hypothesis": "LAMBDA-5",
        "family": "literary",
        "arms": ("litstd37", "litstd37v"),
        "seeds": {"litstd37": 1337, "litstd37v": 7331},
        "reference_arms": ("litdrop37", "litdrop37v"),
        "reference_checkpoint_sha256": {
            "litdrop37": "d1fd4fd523ccfb58f7408cdffd42687f866d0dc21c966a61ca4e1cfeb92e200d",
            "litdrop37v": "a3de90008d532d5551bf5ec4d3e41ffa5dd5e1b55d1c86f3890e2a471736ffc9",
        },
        "trainer": "train_conscious_lm.py",
        "expected_params": 27_689_136,
        "reference_params": 27_691_440,
        "corpus_sha256": "336e101a5b9737c2e12073b5562a06320c150b5a19655a8046b7c16e13ddff5e",
        "fresh_sha256": "8e196165d525e15bc4b200e395953b19d6007acd0cb2c65746649dc4acb5cecd",
        "trainer_args": {
            "dim": 384, "layers": 6, "heads": 6, "ffn_type": "standard",
            "batch_size": 32, "grad_accum_steps": 1, "block_size": 256,
            "lr": 3e-4, "max_cells": 16, "val_bytes": 262144,
            "eval_every": 250, "phase": "language", "dropout": 0.37,
            "steps": 12000, "save_every": 6000, "log_every": 100,
        },
        "results": "ffn_control",
        "post_scorers": (
            {
                "axis": "verdict",
                "script": "measurement/ffn_control_gate.py",
                "output": "measurement/ffn_control_verdict.json",
            },
        ),
    },
}


def family(name=None):
    key = ALIASES.get(name or os.environ.get("LAMBDA_FAMILY", "constructed"))
    if key is None:
        raise ValueError(f"unknown LAMBDA_FAMILY: {name or os.environ.get('LAMBDA_FAMILY')}")
    return key, FAMILIES[key]


def family_arm_paths(home, name=None, axis=None):
    key, _ = family(name)
    return {arm_name: f"{home}/{spec['checkpoint']}" for arm_name, spec in ARMS.items()
            if spec["family"] == key and spec["checkpoint"]
            and (axis != "lambda4" or arm_name in LAMBDA4_ARMS)}


def gate_arms():
    return {name: (spec["floor"], spec["label"]) for name, spec in ARMS.items()}


def seed_siblings():
    return {name: spec["sibling"] for name, spec in ARMS.items() if spec["sibling"]}


def requires_ladder(arm_name):
    return arm_name in LAMBDA4_ARMS


def result_files(axis=None):
    """Canonical result paths, optionally restricted to one measurement axis."""
    return tuple(
        spec[key]
        for spec in RESULT_SETS.values()
        for key in spec
        if axis is None or key == axis
    )


def experiment_result_files():
    """Result receipts produced by registered experiments outside the base gate."""
    return tuple(
        scorer["output"]
        for spec in EXPERIMENTS.values()
        for scorer in (*spec.get("scorers", ()), *spec.get("post_scorers", ()))
    )


def experiment_scorer_files():
    """Scorers owned by registered experiments, in stable first-seen order."""
    return tuple(dict.fromkeys(
        scorer["script"]
        for spec in EXPERIMENTS.values()
        for scorer in (*spec.get("scorers", ()), *spec.get("post_scorers", ()))
    ))


def experiment(name):
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown registered experiment: {name}") from exc
