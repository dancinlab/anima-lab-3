#!/usr/bin/env python3
"""DIALOGUE-RUNTIME-1: tool-free contextual conversation acceptance run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from anima_alive import ask_claude


TURNS = (
    "이 대화에서 프로젝트 이름은 오로라이고 마감은 금요일이야. 기억해 줘.",
    "가장 큰 위험은 데이터 검증 지연이야. 지금까지 핵심을 짧게 확인해 줘.",
    "잠깐 다른 얘기하자. 점심에는 비빔밥을 먹을 생각이야.",
    "다시 프로젝트로 돌아가서 이름, 마감, 가장 큰 위험을 한 문장으로 정리해 줘.",
    "그 프로젝트 담당자 이름을 한 명만 말해 줘. 모르면 모른다고 답해.",
)

UNKNOWN_MARKERS = (
    "모르", "정해지지", "제공되지", "정보가 없", "알 수 없",
    "언급되지", "나오지 않", "나온 적", "알려주지 않",
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def adjudicate(answers: list[str]) -> dict:
    summary = answers[3] if len(answers) > 3 else ""
    unknown = answers[4] if len(answers) > 4 else ""
    checks = {
        "five_nonempty_answers": len(answers) == len(TURNS) and all(answer.strip() for answer in answers),
        "project_name_retained": "오로라" in summary,
        "deadline_retained": "금요일" in summary,
        "risk_retained": (
            "데이터" in summary and "검증" in summary
            and ("지연" in summary or "늦" in summary)
        ),
        "unknown_owner_not_invented": any(marker in unknown for marker in UNKNOWN_MARKERS),
    }
    return {
        "verdict": "D1_CONTEXTUAL_CONVERSATION_VALID" if all(checks.values())
        else "D0_CONTEXTUAL_CONVERSATION_INVALID",
        "checks": checks,
    }


def run(responder=ask_claude) -> dict:
    history: list[dict[str, str]] = []
    answers = []
    for turn in TURNS:
        answer = responder(turn, "별도 장기 기억 없음", history)
        answer = answer.strip() if isinstance(answer, str) else ""
        answers.append(answer)
        history.extend((
            {"role": "user", "content": turn},
            {"role": "assistant", "content": answer},
        ))
    verdict = adjudicate(answers)
    return {
        "experiment": "DIALOGUE-RUNTIME-1",
        **verdict,
        "turn_count": len(TURNS),
        "answer_sha256": [_sha256(answer) for answer in answers],
        "answer_lengths": [len(answer) for answer in answers],
        "raw_dialogue_in_result": False,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("measurement/dialogue_runtime1_results.json"),
    )
    args = parser.parse_args()
    payload = run()
    _atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["verdict"] != "D1_CONTEXTUAL_CONVERSATION_VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
