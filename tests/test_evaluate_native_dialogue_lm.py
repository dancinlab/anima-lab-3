import json

from evaluate_native_dialogue_lm import (
    PANEL_PATH,
    jaccard,
    prompt_echo_ratio,
    repeated_trigram_ratio,
    semantic_pass,
)


def test_semantic_groups_and_forbidden_terms():
    turn = {"required_groups": [["red"], ["key"]], "forbidden_terms": ["blue key"]}
    assert semantic_pass("It contains a red key.", turn)
    assert not semantic_pass("It contains a blue key.", turn)


def test_repeated_trigram_ratio_detects_loop():
    assert repeated_trigram_ratio("the concept of the concept of the concept of") > 0.35
    assert repeated_trigram_ratio("TTTTTTTTTTTT") > 0.35
    assert repeated_trigram_ratio("ice melts into water in sunlight") == 0.0


def test_echo_and_cross_response_similarity_controls():
    prompt = "What happens when ice is left in sunlight?"
    assert prompt_echo_ratio(prompt, prompt) == 1.0
    assert prompt_echo_ratio(prompt, "Ice melts because sunlight adds heat.") < 0.9
    assert jaccard("same repeated answer", "same repeated answer") == 1.0
    assert jaccard("ice melts", "take a slow breath") == 0.0


def test_frozen_panel_has_seven_turns_per_language():
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    assert panel["source_sha256"] == "10b18a4c6fb6c4df1beb042529f1e485c5de7dfbbe480ed25771d6a1316e41ee"
    for language in ("en", "ko"):
        assert sum(len(item["turns"]) for item in panel["items"] if item["lang"] == language) == 7
