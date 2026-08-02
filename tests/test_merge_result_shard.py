import pytest

from scripts.merge_result_shard import merge


def test_merge_replaces_named_arms_and_preserves_receipt():
    canonical = {"_setup": {"seed": 7}, "old": {"score": 1}}
    shard = {"_setup": {"seed": 7}, "new": {"score": 2}}

    merged, before, added = merge(canonical, shard, ["new"])

    assert merged == {
        "_setup": {"seed": 7},
        "old": {"score": 1},
        "new": {"score": 2},
    }
    assert (before, added) == (1, 1)


def test_merge_rejects_different_experimental_receipt():
    with pytest.raises(ValueError, match="experimental receipt mismatch"):
        merge(
            {"_setup": {"seed": 7}, "old": {}},
            {"_setup": {"seed": 8}, "new": {}},
        )


def test_merge_rejects_incomplete_requested_roster():
    with pytest.raises(ValueError, match="shard roster mismatch"):
        merge({"_setup": {}}, {"_setup": {}, "one": {}}, ["one", "two"])
