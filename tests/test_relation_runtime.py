from measurement.relation_registry import RELATION_SPEC
from measurement.synergy_gate import _expected_audit
from synergy import SynergyActionChannel, _target_index


def test_relation_target_is_balanced_per_role_and_role_sensitive():
    spec = RELATION_SPEC
    count = len(spec["actions"])
    for module_a in range(count):
        assert {_target_index(module_a, module_b, spec) for module_b in range(count)} == set(range(count))
    for module_b in range(count):
        assert {_target_index(module_a, module_b, spec) for module_a in range(count)} == set(range(count))
    swapped_same = sum(
        _target_index(a, b, spec) == _target_index(b, a, spec)
        for a in range(count) for b in range(count)
    )
    assert swapped_same / (count * count) == 0.20
    assert _expected_audit(spec, "train")["pair_count"] == 25


def test_relation_arms_reuse_registered_workspace_bridge():
    relation = SynergyActionChannel("quantum_relation", 96, 64, RELATION_SPEC)
    baseline = SynergyActionChannel("quantum_workspace_2", 96, 64, RELATION_SPEC)
    assert relation.action.bridge.bind_roles is True
    assert relation.action.bridge.rounds == 1
    assert baseline.action.bridge.bind_roles is False
    assert baseline.action.bridge.rounds == 2
