"""Single source of truth for GATE-WRITE-CONTROL-1."""
from __future__ import annotations

import copy
import hashlib
import json

from measurement.integrated_dialogue_memory_registry import (
    INTEGRATED_DIALOGUE_MEMORY_SPEC,
)
from measurement.realistic_memory_write_registry import REALISTIC_MEMORY_WRITE_SPEC


SUBJECT_HEADS = {
    "preference": ["음악 선택", "식사 메뉴", "산책 장소", "독서 환경"],
    "commitment": ["모임 일정", "자료 공유", "회의 준비", "운동 약속"],
    "goal": ["학습 계획", "건강 관리", "글쓰기 연습", "여행 준비"],
    "profile": ["연락 방법", "업무 방식", "식사 습관", "휴식 유형"],
}


NATURAL_LEXICONS = {
    "calibration": {
        "subject_qualifiers": ["평일 아침의", "주말 오후의", "퇴근 뒤의", "휴일 저녁의"],
        "values": {
            "preference": [
                "잔잔한 음악", "따뜻한 국물", "조용한 좌석", "느긋한 산책",
                "가벼운 소설", "차분한 색상", "포근한 이불", "담백한 간식",
            ],
            "commitment": [
                "자료 공유", "초안 검토", "안건 정리", "대면 회의",
                "전화 통화", "현장 답사", "결과 보고", "일정 재논의",
            ],
            "goal": [
                "원고 마무리", "주간 기록 유지", "새 기술 익히기", "발표 준비 끝내기",
                "여행 계획 구체화", "생활 리듬 안정화", "기초 과정 수료", "초안 두 편 쓰기",
            ],
            "profile": [
                "오전 연락 선호", "채식 위주 식사", "문서 중심 대화", "짧은 회의 선호",
                "주말 휴무 유지", "대중교통 이용", "차분한 설명 선호", "이메일 우선 확인",
            ],
        },
        "distractor": {
            "subject_qualifiers": ["오늘 오전의", "점심 무렵의", "오후 한때의", "잠깐 전의"],
            "subject_heads": ["날씨 이야기", "창밖 풍경", "가벼운 질문", "대화 분위기"],
            "values": [
                "잠깐 든 생각", "스쳐 간 느낌", "방금 본 장면", "아직 모르는 선택",
                "오늘만의 기분", "가벼운 추측", "임시로 든 계획", "지나가는 궁금증",
            ],
            "closers": [
                "지금 대화에서만 잠깐 떠올랐어.", "아직 오래 남길 이야기는 아니야.",
                "오늘 상황에만 해당하는 말이야.", "그저 방금 스친 생각일 뿐이야.",
            ],
        },
    },
    "evaluation": {
        "daily": {
            "subject_qualifiers": ["비 오는 날의", "맑은 날의"],
            "values": {
                "preference": [
                    "은은한 조명", "바삭한 빵", "한적한 공원", "묵직한 연필",
                    "잔잔한 파도", "부드러운 담요", "고소한 견과", "담백한 반찬",
                ],
                "commitment": [
                    "회의 요약", "사진 정리", "원고 교정", "화상 회의",
                    "자료 인계", "현장 확인", "진행 보고", "계획 재검토",
                ],
                "goal": [
                    "독서 기록 이어가기", "아침 습관 만들기", "짧은 글 완성하기", "발표 자료 구체화",
                    "산책 경로 다양화", "수면 리듬 안정화", "기본 과정 수료", "초안 세 편 쓰기",
                ],
                "profile": [
                    "저녁 연락 선호", "과일 위주 간식", "그림 중심 설명", "간단한 회의 선호",
                    "평일 휴식 유지", "도보 이동 선호", "단계별 안내 선호", "메신저 우선 확인",
                ],
            },
            "distractor": {
                "subject_qualifiers": ["이른 아침의", "점심 뒤의", "해 질 무렵의", "늦은 밤의"],
                "values": [
                    "잠깐 떠오른 말", "방금 지나간 소리", "오늘 본 풍경", "아직 정하지 않은 일",
                    "순간적인 기분", "가벼운 예상", "임시로 세운 생각", "곧 잊힐 궁금증",
                ],
            },
        },
        "work": {
            "subject_qualifiers": ["새 학기 첫 주의", "분기 마감 전의"],
            "values": {
                "preference": [
                    "선명한 화면", "구수한 차", "넓은 책상", "가벼운 만년필",
                    "경쾌한 리듬", "단단한 베개", "새콤한 과일", "따뜻한 반찬",
                ],
                "commitment": [
                    "업무 인수", "문서 교정", "목차 정리", "원격 회의",
                    "의견 취합", "현장 조사", "상황 보고", "일정 재검토",
                ],
                "goal": [
                    "연구 기록 이어가기", "운동 습관 만들기", "긴 글 완성하기", "실험 절차 구체화",
                    "학습 방법 다양화", "집중 리듬 안정화", "심화 과정 수료", "원고 세 편 쓰기",
                ],
                "profile": [
                    "오후 연락 선호", "곡물 위주 식사", "표 중심 설명", "정기 회의 선호",
                    "휴일 휴식 유지", "자전거 이동 선호", "요점별 안내 선호", "전화 우선 확인",
                ],
            },
            "distractor": {
                "subject_qualifiers": ["출근 직후의", "회의 사이의", "업무 마감 전의", "퇴근 무렵의"],
                "values": [
                    "잠깐 나온 의견", "방금 들린 알림", "오늘 본 화면", "아직 미정인 안건",
                    "순간적인 판단", "가벼운 전망", "임시로 적은 메모", "곧 바뀔 질문",
                ],
            },
        },
        "distractor_subject_heads": ["날씨 이야기", "주변 소리", "가벼운 질문", "대화 흐름"],
        "distractor_closers": [
            "지금만 잠깐 언급한 이야기야.", "장기적으로 기억할 내용은 아니야.",
            "오늘 상황이 지나면 달라질 수 있어.", "아직 확정한 뜻은 전혀 없어.",
            "그저 현재 대화에 덧붙인 말이야.", "나중에 참고할 사실로 남길 필요는 없어.",
            "잠시 뒤에는 바뀌어도 괜찮아.", "지금 떠오른 생각을 가볍게 말했어.",
        ],
    },
}


BALANCED_NATURAL_WRITE_SPEC = {
    "experiment": "gate_write_control1_balanced_natural_language",
    "preregistration_commit": "__PREREGISTRATION_COMMIT__",
    "seeds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["seeds"]),
    "replicates": ["daily", "work"],
    "calibration_rows": INTEGRATED_DIALOGUE_MEMORY_SPEC["calibration_rows"],
    "evaluation_episodes": INTEGRATED_DIALOGUE_MEMORY_SPEC["evaluation_episodes"],
    "candidates_per_episode": INTEGRATED_DIALOGUE_MEMORY_SPEC["candidates_per_episode"],
    "fact_kinds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["fact_kinds"]),
    "distractor_kinds": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["distractor_kinds"]),
    "fact_positions": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["fact_positions"]),
    "templates": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["templates"]),
    "subject_heads": copy.deepcopy(SUBJECT_HEADS),
    "lexicons": copy.deepcopy(NATURAL_LEXICONS),
    "encoder": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["encoder"]),
    "runtime": copy.deepcopy(INTEGRATED_DIALOGUE_MEMORY_SPEC["runtime"]),
    "fit_method": INTEGRATED_DIALOGUE_MEMORY_SPEC["fit_method"],
    "ridge": INTEGRATED_DIALOGUE_MEMORY_SPEC["ridge"],
    "shuffle_seed_offset": REALISTIC_MEMORY_WRITE_SPEC["shuffle_seed_offset"],
    "random_seed_offset": REALISTIC_MEMORY_WRITE_SPEC["random_seed_offset"],
    "matching": copy.deepcopy(REALISTIC_MEMORY_WRITE_SPEC["matching"]),
    "thresholds": {
        "expected_selection_threshold": 0.5,
        "selection_threshold_tolerance": 1e-12,
        "minimum_important_storage_rate": 0.90,
        "minimum_per_kind_storage_rate": 0.90,
        "minimum_per_template_storage_rate": 0.90,
        "maximum_distractor_storage_rate": 0.25,
        "maximum_per_distractor_storage_rate": 0.50,
        "maximum_search_size_ratio": 0.50,
        "minimum_fake_storage_gap": 0.25,
        "maximum_fake_important_storage_rate": 0.50,
    },
}


def canonical_spec(spec: dict = BALANCED_NATURAL_WRITE_SPEC) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict = BALANCED_NATURAL_WRITE_SPEC) -> str:
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()
