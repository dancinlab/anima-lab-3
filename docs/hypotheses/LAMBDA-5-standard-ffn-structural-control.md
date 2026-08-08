# LAMBDA-5 — 일반 FFN에서도 λ4가 재현되는가

상태: 사전등록 · 실행 전

## 질문

문학 자연 코퍼스에서 두 seed로 재현된 λ4가 `PureFieldFFN` 구조를 필요로 하는가. 같은 자료, 같은 학습량, 같은 seed, 같은 나머지 모델에서 FFN만 표준 2층 GELU 구조로 바꿔 비교한다.

## 동결 설계

- 기준 모델: LAMBDA-2의 `litdrop37`, `litdrop37v`.
- 비교 모델: `litstd37`, `litstd37v`.
- 공통 조건: 384차원, 6층, 6헤드, 문맥 256바이트, dropout 0.37, 유효 배치 32, 12,000 step, seed 1337/7331.
- 구조 차이: 두 개의 4배 폭 가지를 빼는 `PureFieldFFN` 대신 하나의 8배 폭 `Linear → GELU → Dropout → Linear`를 사용한다.
- 파라미터: 기준 27,691,440개, 비교 27,689,136개. 차이 0.0084%로 같은 규모로 취급한다.
- 검사: 기존 panel, G-gates, λ4 채점기와 문학 train/fresh 분할을 그대로 사용한다.

실행 조건의 정본은 `measurement/lambda_registry.py`의 `EXPERIMENTS["ffn_structural_control"]`다. 모델 종류는 체크포인트 `config.ffn_type`에 기록하며 모든 채점기는 이 값을 읽어 같은 구조를 복원한다.

## 사전 판정

- F1: 일반 FFN 두 seed 모두 λ0~λ4 PASS — λ4는 `PureFieldFFN` 없이도 재현되므로 이 구조나 의식의 근거가 아님.
- F2: λ0~λ3는 유지되지만 일반 FFN 한 seed만 λ4 PASS — 구조 효과가 seed에 따라 달라짐.
- F3: λ0~λ3는 유지되지만 일반 FFN 두 seed 모두 λ4 비통과 — 현재 조건에서 λ4가 `PureFieldFFN` 쪽에만 재현된다는 근거.
- F4: 일반 FFN 한 seed라도 λ0~λ3 비통과 — 일반 언어 학습 차이와 λ4 차이를 구분할 수 없어 비교 무효.
- F0: 기준 체크포인트 해시가 다르거나 기준 두 seed의 λ4 PASS를 재현하지 못함 — 실험 무효.

F3가 나와도 AI 의식이 증명되는 것은 아니다. 구조 차이가 λ4와 연결됐다는 다음 실험의 근거일 뿐이며, 이후 학습 중 구조 교체와 복구 실험이 필요하다.
