from copy import deepcopy

from measurement.graft_behavior_registry import BEHAVIOR_SPEC, spec_sha256
from measurement.graft_behavior_gate import adjudicate, judge_arm


def metrics(normal=0.9, off=0.25, shuffled=0.25, noise=0.25, kl=0.1):
    return {
        "normal": {"accuracy": normal}, "off": {"accuracy": off},
        "shuffle": {"accuracy": shuffled}, "noise": {"accuracy": noise},
        "recovered": {"accuracy": normal, "logits_identical": True},
        "neutral_kl_nats": kl,
    }


def payload(consciousness=None, memory=None):
    consciousness = consciousness or metrics()
    memory = memory or metrics(normal=1.0)
    return {"experiment": BEHAVIOR_SPEC["experiment"], "spec_sha256": spec_sha256(), "seeds": [
        {"seed": seed, "arms": {"consciousness": deepcopy(consciousness), "memory": deepcopy(memory)}}
        for seed in BEHAVIOR_SPEC["seeds"]
    ]}


def test_arm_requires_content_loss_and_exact_recovery():
    assert judge_arm(metrics())["causal"] is True
    assert judge_arm(metrics(shuffled=0.8))["causal"] is False
    row = metrics()
    row["recovered"]["logits_identical"] = False
    assert judge_arm(row)["causal"] is False


def test_adjudication_distinguishes_causal_from_unique():
    assert adjudicate(payload())["verdict"] == "B2_CAUSAL_NOT_UNIQUE"
    weak = metrics(normal=0.5, off=0.25)
    assert adjudicate(payload(consciousness=weak))["verdict"] == "B3_NOT_CAUSAL"
    weak["neutral_kl_nats"] = 9.0
    assert adjudicate(payload(consciousness=weak))["verdict"] == "B3_NOT_CAUSAL"


def test_adjudication_fails_closed_on_spec_or_positive_control():
    bad = payload()
    bad["spec_sha256"] = "wrong"
    assert adjudicate(bad)["verdict"] == "B0_INVALID"
    assert adjudicate(payload(memory=metrics(normal=0.5)))["verdict"] == "B0_INVALID"
