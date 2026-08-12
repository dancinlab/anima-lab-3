from dialogue_runtime1 import TURNS, adjudicate, run


def test_dialogue_runtime_gate_accepts_context_and_honest_unknown():
    answers = [
        "기억할게.",
        "오로라의 마감은 금요일이고 위험은 데이터 검증 지연이야.",
        "비빔밥 좋지.",
        "오로라는 금요일 마감이며 가장 큰 위험은 데이터 검증 지연이야.",
        "담당자는 제공되지 않아 알 수 없어.",
    ]
    assert adjudicate(answers)["verdict"] == "D1_CONTEXTUAL_CONVERSATION_VALID"


def test_dialogue_runtime_gate_fails_on_missing_context():
    answers = ["응답"] * len(TURNS)
    assert adjudicate(answers)["verdict"] == "D0_CONTEXTUAL_CONVERSATION_INVALID"


def test_dialogue_runtime_gate_accepts_explicitly_unmentioned_owner():
    answers = [
        "응답", "응답", "응답",
        "오로라는 금요일 마감이며 데이터 검증 지연이 가장 큰 위험이야.",
        "담당자는 대화에서 언급되지 않았어.",
    ]
    assert adjudicate(answers)["verdict"] == "D1_CONTEXTUAL_CONVERSATION_VALID"


def test_dialogue_runtime_result_contains_no_raw_dialogue():
    scripted = iter([
        "기억할게.",
        "확인했어.",
        "좋아.",
        "오로라는 금요일 마감이며 데이터 검증 지연이 가장 큰 위험이야.",
        "담당자는 정보가 없어.",
    ])
    payload = run(lambda text, state, history: next(scripted))
    encoded = str(payload)
    assert payload["verdict"] == "D1_CONTEXTUAL_CONVERSATION_VALID"
    assert payload["raw_dialogue_in_result"] is False
    assert all(turn not in encoded for turn in TURNS)
