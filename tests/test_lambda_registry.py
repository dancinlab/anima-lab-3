from measurement import lambda_registry as registry


def test_family_alias_and_arm_rosters_share_one_registry():
    key, family = registry.family("natural")
    assert key == "encyclopedic"
    assert family["register"] == "encyclopedic prose"
    assert set(registry.family_arm_paths("/runtime", "literary")) == {
        "litdrop37", "litdrop37v"
    }
    assert registry.gate_arms()["litdrop37"][0] == "lit"
    assert registry.seed_siblings()["litdrop37"] == "litdrop37v"
    assert "nat25f" not in registry.family_arm_paths("/runtime", "natural", axis="lambda4")
    assert registry.requires_ladder("litdrop37")


def test_every_scored_arm_references_a_registered_floor():
    assert all(spec["floor"] in registry.FLOORS for spec in registry.ARMS.values())
    assert all(spec["checkpoint"] for spec in registry.ARMS.values() if spec["family"])
