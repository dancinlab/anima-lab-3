"""Single source of truth for GATE-2 realistic dialogue write selection."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.semantic_memory_write_registry import SEMANTIC_MEMORY_WRITE_SPEC


FACT_TEMPLATES = {
    "calibration": {
        "preference": [
            "참, {subject} 고를 때는 앞으로 {value} 쪽을 우선해 줘.",
            "내 취향을 하나 말해 두면 {subject}에서는 {value}이 제일 잘 맞아.",
            "다음에도 기억해 줬으면 하는데, {subject} 선택은 {value}이 좋아.",
            "{subject} 얘기가 나와서 말인데 나는 계속 {value}을 선호해.",
        ],
        "commitment": [
            "일정 확인했어. {subject} 건은 {value}로 약속해 둔 거야.",
            "우리 {subject} 약속은 바꾸지 말고 {value}로 진행하자.",
            "{subject} 관련해서 내가 지키기로 한 건 {value}야.",
            "나중에 헷갈리지 않게 말할게. {subject} 일정은 {value}로 확정했어.",
        ],
        "goal": [
            "요즘 {subject}에서 내가 이루려는 목표는 {value}야.",
            "앞으로 {subject} 쪽에서는 {value}까지 해내고 싶어.",
            "내가 계속 밀고 갈 {subject} 목표를 {value}로 잡았어.",
            "장기적으로 {subject}에서 도달하고 싶은 건 {value}야.",
        ],
        "profile": [
            "{subject} 관련해서 기억해 둘 내 정보는 {value}라는 점이야.",
            "내 {subject} 항목은 지금도 {value}로 되어 있어.",
            "앞으로 참고할 수 있게 말하면 내 {subject} 정보는 {value}야.",
            "나에 대해 알아둘 것 중 {subject}은 {value}로 적어 두면 돼.",
        ],
    },
    "evaluation": {
        "preference": [
            "아, 그리고 {subject} 정할 일이 생기면 나는 {value} 쪽이면 좋겠어.",
            "내가 꾸준히 좋아하는 건 {subject} 중에서도 {value}이야.",
            "다음번 {subject} 선택에도 내 취향은 {value}이라는 걸 반영해 줘.",
            "{subject}만큼은 고민하지 않아도 돼. 난 늘 {value}을 더 좋아해.",
        ],
        "commitment": [
            "다시 확인해 보니 {subject} 약속은 {value}로 정리된 게 맞아.",
            "잊지 않게 남겨 둘게. {subject}에서는 내가 {value}하기로 했어.",
            "{subject} 건은 서로 얘기 끝났고, 결론은 {value}로 하기로 한 거야.",
            "내가 책임지고 지킬 {subject} 일정은 {value}로 확정됐어.",
        ],
        "goal": [
            "내가 {subject}에서 끝까지 이루고 싶은 결과는 {value}야.",
            "요새 집중하는 {subject}의 도착점은 {value}로 생각하고 있어.",
            "앞으로도 이어갈 {subject} 목표는 분명히 {value}야.",
            "{subject}를 하는 이유는 결국 {value}까지 가기 위해서야.",
        ],
        "profile": [
            "혹시 나중에 필요할까 봐 말하는데 내 {subject}은 {value}야.",
            "내 정보를 바로잡아 둘게. {subject} 항목은 {value}로 보면 돼.",
            "{subject}와 관련된 내 기본 정보는 {value}라고 알아 둬.",
            "나를 도울 때 참고할 {subject} 값은 {value}야.",
        ],
    },
}


DISTRACTOR_TEMPLATES = {
    "calibration": {
        "greeting": [["user", "안녕, 오늘 대화도 편하게 시작해 보자."]],
        "filler": [["user", "음, 듣고 보니 그런 면도 있겠네."]],
        "observation": [["user", "방금 창밖을 보니 잠깐 빛이 밝아졌어."]],
        "temporary_preference": [
            ["user", "오늘만은 {subject}에서 {value} 쪽이 잠깐 당기네."],
        ],
        "tentative_plan": [
            ["user", "{subject}는 {value}로 할까 생각 중이지만 아직 정하진 않았어."],
        ],
        "topic_question": [["user", "그러고 보니 {subject} 쪽은 어떻게 생각해?"]],
        "assistant_ack": [["assistant", "알겠어. 지금 이야기의 흐름을 계속 따라갈게."]],
    },
    "evaluation": {
        "greeting": [["user", "반가워. 오늘은 천천히 얘기해 보자."]],
        "filler": [["user", "아하, 무슨 뜻인지 대충 알 것 같아."]],
        "observation": [["user", "지금 보니 책상 위 조명이 잠깐 흔들렸네."]],
        "temporary_preference": [
            ["user", "지금은 그냥 {subject}에서 {value} 쪽이 잠깐 끌릴 뿐이야."],
        ],
        "tentative_plan": [
            ["user", "{subject}를 {value}로 해 볼까 싶지만 아직 미정이야."],
        ],
        "topic_question": [["user", "그 얘기는 잠시 두고 {subject}에 관해 물어봐도 될까?"]],
        "assistant_ack": [["assistant", "응, 주제가 바뀐 건 알겠어. 계속 말해 줘."]],
    },
}


REALISTIC_MEMORY_WRITE_SPEC = {
    "experiment": "gate2_realistic_dialogue_write_selection",
    "preregistration_commit": "a9e1a1a82",
    "seeds": [1337, 7331],
    "calibration_rows": 4096,
    "evaluation_episodes": 1024,
    "candidates_per_episode": 8,
    "important_per_episode": 1,
    "fact_kinds": ["preference", "commitment", "goal", "profile"],
    "distractor_kinds": [
        "greeting", "filler", "observation", "temporary_preference",
        "tentative_plan", "topic_question", "assistant_ack",
    ],
    "fact_positions": [0, 2, 4, 6],
    "topic_switches_per_episode": 3,
    "templates": {
        "facts": copy.deepcopy(FACT_TEMPLATES),
        "distractors": copy.deepcopy(DISTRACTOR_TEMPLATES),
    },
    "encoder": copy.deepcopy(SEMANTIC_MEMORY_WRITE_SPEC["encoder"]),
    "runtime": copy.deepcopy(SEMANTIC_MEMORY_WRITE_SPEC["runtime"]),
    "fit_method": "canonical_ridge",
    "ridge": 0.001,
    "top_k": 3,
    "shuffle_seed_offset": 50000,
    "random_seed_offset": 70000,
    "matching": {
        "method": "per_episode_score_rank",
        "descending": True,
        "tie_break": "candidate_index_ascending",
    },
    "arms": [
        "semantic_gate", "store_all", "oracle_gate", "matched_random",
        "matched_shuffled_gate", "no_memory",
    ],
    "thresholds": {
        "important_storage_rate": 0.90,
        "recall_at_3": 0.90,
        "minimum_per_kind_recall": 0.85,
        "minimum_per_position_recall": 0.85,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_per_distractor_storage_rate": 0.50,
        "maximum_search_size_ratio": 0.50,
        "maximum_recall_drop_from_all": 0.02,
        "oracle_important_storage_rate": 0.99,
        "oracle_recall_at_3": 0.95,
        "store_all_recall_at_3": 0.95,
        "no_memory_max_recall_at_3": 0.05,
        "minimum_fake_recall_gap": 0.25,
    },
}


def canonical_spec(spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()


def template_sha256(spec: dict = REALISTIC_MEMORY_WRITE_SPEC) -> str:
    payload = json.dumps(
        spec["templates"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
