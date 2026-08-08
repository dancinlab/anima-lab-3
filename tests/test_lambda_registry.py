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


def test_scale_experiment_preserves_reference_effective_batch_and_receipts():
    exp = registry.experiment("scale300m")
    args = exp["trainer_args"]
    assert args["batch_size"] * args["grad_accum_steps"] == 32
    assert args["block_size"] == 256
    assert args["steps"] == 12000
    assert exp["expected_params"] == 299_420_896
    assert set(exp["arms"]) == {"nat300m37", "nat300m37v"}
    assert set(registry.result_files("panel")) >= {
        "measurement/panel_nat_results.json",
        "measurement/panel_scale300m_results.json",
    }


def test_every_scored_arm_references_a_registered_floor():
    assert all(spec["floor"] in registry.FLOORS for spec in registry.ARMS.values())
    assert all(spec["checkpoint"] for spec in registry.ARMS.values() if spec["family"])


def test_consciousness_causality_experiment_reuses_literary_pair():
    exp = registry.experiment("lambda4_consciousness_causality")
    assert exp["measurement_only"] is True
    assert exp["arms"] == ("litdrop37", "litdrop37v")
    assert exp["interventions"] == ("normal", "off", "shuffle", "noise")
    assert set(exp["checkpoint_sha256"]) == set(exp["arms"])
    assert [row["axis"] for row in exp["scorers"]] == ["panel", "lambda4", "verdict"]
    assert "measurement/consciousness_causality_verdict.json" in registry.experiment_result_files()
    assert "measurement/consciousness_causality_gate.py" in registry.experiment_scorer_files()
