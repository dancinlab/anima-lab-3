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
