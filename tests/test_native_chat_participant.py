from scripts.run_native_chat_participant import normalized_history


def test_normalized_history_accepts_only_broker_dialogue_roles():
    rows = [
        {"kind": "user", "text": "질문"},
        {"kind": "anima", "text": "답변"},
        {"kind": "system", "text": "무시"},
        {"kind": "user", "text": "   "},
        "invalid",
    ]
    assert normalized_history(rows, 20) == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답변"},
    ]


def test_normalized_history_applies_broker_window_before_filtering():
    rows = [
        {"kind": "user", "text": "old"},
        {"kind": "system", "text": "ignored"},
        {"kind": "user", "text": "new"},
    ]
    assert normalized_history(rows, 2) == [{"role": "user", "content": "new"}]
