# COMPONENT-2 — 시간 안정 상황·열쇠·값의 통합 결합 경로

상태: 사전등록 · 실행 전
등록일: 2026-08-11
선행 결과: `COMPONENT-1=AC3_BOTH_COMPONENTS_LOSS`

## 목표와 최소 수정

- 평가와 겹치지 않는 512회차 × 엔진 시작값 2개에서 사건 16개의 상황·열쇠 상태를 각각
  16,384개 수집한다.
- 상황 8종과 열쇠 8종은 각각 정확히 2,048개가 되도록 회차별 활성 종류를 순환 배정한다.
- 기존 `fit_stable_key_projector(..., method="canonical_ridge")`로 폭 32 변환 두 개를 맞춘다.
- 반복 맞춤과 입력 순서 역전 결과가 완전히 같아야 한다.
- `VALUE-2`의 폭 32 값 변환은 다시 맞추지 않는다.
- 공용 `CompositeStateTransform`과 `VectorMemory` 호출은 바꾸지 않고 새 얼린 상황·열쇠 변환만
  주입한다.

## 평가와 문턱

- 겹치지 않는 위치별 평가에서 상황·열쇠 분류 정확도 90%, 종류별 최저 맞힘률 75% 이상
- `CONJUNCTION-2`의 1,024회차·네 조합·10개 팔을 그대로 재실행
- 정상 합성 주소 선택·최종 판독 90%, 값별 최저 맞힘률 75% 이상
- 정확 주소 양성 비교 90%, 두 단독 성분 35% 이하, 가짜 내용 5% 이하
- 장치 밖 기준·복구·기억 API 100%, 세 변환의 얼린 가중치와 호출 수 유지
- 변환을 옛 상황·열쇠로 되돌린 비교는 저장 위치 85% 이하이며 정상 대비 15%p 이상 하락

사전 판정은 `CS1_STABLE_COMPOSITE_PATH_VALID_NOT_UNIQUE`, `CS2_COMPONENT_FIT_INVALID`,
`CS3_COMPOSITION_LOSS`, `CS4_NOT_CAUSAL`, `CS0_INVALID`다. 성공해도 일반적인 시간 안정 기억 주소
수리이며 의식 증거가 아니다. 통과하면 다음 단계는 불완전한 단서 복원 `COMPLETION-1`이다.

결과를 보기 전 이 문서와 README를 커밋하고 그 커밋을 실행 사양의
`preregistration_commit`으로 고정한다.
