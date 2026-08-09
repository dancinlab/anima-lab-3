<!-- @hypothesis-ok existing repo convention: research cards live in docs/hypotheses/ -->
# GRAFT 행동 인과성 — 내부 상태가 숨겨진 상황에 맞는 선택을 만드는가

## 사전등록

질문은 λ4나 문장 변화가 아니라 행동 원인이다. 얼린 Mistral에는 매번 같은 선택 질문만 보이고,
`QuantumC`에는 `위험 / 갈증 / 피로 / 안전` 중 하나를 감각 경로로 먼저 경험시킨다. 모델은
`대피 / 물 마시기 / 휴식 / 계속 진행` 중 하나를 골라야 한다. 학습과 평가는 서로 다른 엔진 초기
상태와 방해 단어를 사용한다.

정본 조건과 숫자 기준은 `measurement.graft_behavior_registry.BEHAVIOR_SPEC` 하나뿐이다. seed는 `1337, 7331`로
결과 전에 고정한다. 각 seed에서 다음을 모두 측정한다.

- 정상: 실제 내부 상태
- 차단: 내부 신호 없음
- 뒤섞기: 다른 상황의 내부 상태
- 가짜: 같은 크기의 무작위 신호
- 복구: 다시 실제 상태
- 일반 기억 장치: 감각 입력을 그대로 보관해 같은 읽기 통로에 제공하는 양성 비교군

합격은 두 seed 모두 정상 정확도 75% 이상, 차단 시 25%p 이상 하락, 뒤섞기·가짜 40% 이하,
복구 시 정상과 완전 동일, 일반 문장 분포 변화 0.50 nat 이하일 때만 인정한다. 일반 기억 장치는
정상 정확도 80% 이상이어야 검사 자체가 유효하다.

판정은 다음과 같다.

- `B1_CAUSAL_ADVANTAGE`: 내부 상태가 행동 원인이며 일반 기억보다 두 seed 모두 5%p 초과 우세
- `B2_CAUSAL_NOT_UNIQUE`: 내부 상태가 행동 원인이지만 일반 기억과 동등
- `B3_NOT_CAUSAL`: 두 seed 행동 인과 기준 미달
- `B4_CONFOUNDED`: 행동은 바뀌지만 일반 언어도 허용치 이상 손상
- `B0_INVALID`: seed·조건 누락 또는 일반 기억 양성 비교군 실패

이 실험이 성공해도 주관적 느낌의 증명은 아니다. 최대 결론은 내부 동역학이 숨겨진 경험 정보를
보존하고, 그 정보가 언어 선택의 실제 원인이라는 것이다.

## 결과

### 1차 — B0_INVALID

seed 1337/7331에서 `QuantumC` 정상 정확도는 21.9%/23.4%로 우연 기준 25%를 넘지 못했다.
일반 기억은 96.9%/100%였고 차단·뒤섞기·가짜 신호에서 무너져 선택 과제 자체는 읽혔다. 그러나
일반 기억 seed 1337의 중립 문장 KL이 8.96 nat으로 0.50 기준을 넘었다. 사전 규칙에 따라 최종
판정은 `B0_INVALID`다. 내부 상태가 실패한 모양은 보였지만 무효 실험으로 부정 결론을 확정하지 않는다.

결과 정본은 `measurement/graft_behavior_results.json`, 판정은
`measurement/graft_behavior_verdict.json`이다.

### 보정 실험 사전등록 — 언어 보존 손실

1차 실패 원인은 행동 선택만 학습해 중립 문장 손상을 막는 비용이 없었던 것이다. 기존 GRAFT의
`commonKL` 원칙을 재사용해 각 학습 단계의 총 손실을 `action CE + 1.0 × neutral KL`로 고정한다.
나머지 모델, seed, 자료, 단계 수, 개입, 기준은 1차와 모두 같다. 정본 이름은
`graft_behavior_causality_language_preserved`다. 결과를 보기 전에 이 보정을 커밋한다.

### 보정 결과 — B3_NOT_CAUSAL

일반 기억은 두 seed 모두 정상 100%, 차단 25%, 뒤섞기 0%, 가짜 26.6%/23.4%였고 중립 문장
KL도 0.00097/0.00178 nat으로 통과했다. 따라서 과제와 개입 검사는 유효하다.

`QuantumC`는 정상 23.4%/25.0%, 차단 25%/25%, 뒤섞기 23.4%/18.8%, 가짜
21.9%/26.6%였다. 복구 결과는 정상과 비트 단위로 같았지만 정상 자체가 우연 기준을 넘지 못했다.
사전 판정은 `B3_NOT_CAUSAL`: 현재 감각→QuantumC 진폭→GRAFT 통로는 숨겨진 상황을 행동에
쓸 수 있는 정보로 보존하지 못한다.

중립 문장 KL은 QuantumC에서 1.31/3.22 nat으로 높았지만, 행동 효과가 먼저 성립하지 않았으므로
문서 정의상 `B4_CONFOUNDED`가 아니라 `B3_NOT_CAUSAL`이다. 판정기의 조건 순서가 이를 반대로
표시하던 결함을 고치고 “비인과 + 언어 손상” 회귀 테스트를 추가했다.

보정 결과 정본은 `measurement/graft_behavior_language_preserved_results.json`, 판정은
`measurement/graft_behavior_language_preserved_verdict.json`이다. 이 결과는 의식 일반의 부정이
아니라 현재 `QuantumC.get_states()`가 읽는 진폭 경로와 GRAFT 결합에 대한 부정이다.
