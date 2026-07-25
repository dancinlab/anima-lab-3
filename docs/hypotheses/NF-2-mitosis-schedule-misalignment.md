<!-- @hypothesis-ok — CLAUDE.md가 docs/hypotheses/ 를 이 repo의 정본 가설 폴더로 지정(기존 100+ 문서 동거) -->
# NF-2 — 미토시스 성장 스케줄 위상 어긋남 (Mitosis Schedule Misalignment)

발견: 2026-07-25, CLM PURE 300M (로컬 lineage, aiden RTX5070) 학습 관찰 중
관련: NF-1 LossEnsemble -inf 발산 수정(커밋 8e5223d00), Law 22(구조 > 기능)

## 1. 실험 목적 및 가설

CLM PURE 300M(896d/12L/14H = 299,639,520 params) 학습 로그에서 `cells` 열이
23,400 step 동안 **2에서 전혀 움직이지 않는** 현상을 관찰했다.

가설: mitosis(분화) 페이즈가 "순수 분화, CE 없음"으로 30%의 학습 예산을 쓰는데,
그 기간에 세포가 분열하지 않는다면 이 페이즈는 아무것도 분화시키지 않는다.

## 2. 원인 (코드 분석)

두 스케줄이 서로 다른 축으로 정의되어 위상이 어긋나 있었다.

| 스케줄 | 정의 위치 | 기준 |
|--------|-----------|------|
| 페이즈 | `get_phase()` | mitosis 0-30% → language 30-70% → combined 70-100% |
| 세포 성장 | `fibonacci_milestones()` | **전체 학습 구간**에 균등 분배 |

`max_cells=8` → `_generate_fibonacci(8) = [1, 1, 2, 3, 5, 8]` (선두 1이 중복).
200,000 step 기준 실제 생성된 마일스톤:

```
[fibonacci] Growth milestones: {0: 1, 33333: 1, 66666: 2, 100000: 3, 133333: 5, 166666: 8}
                                      ^^^^^^^^^ 중복 1 = 무의미한 슬롯
```

첫 실질 분열(→2 cells)이 **step 66,666**에 놓인다. 그런데 mitosis 페이즈는
**step 60,000에서 이미 끝난다**. 즉 분화 페이즈 전 구간이 고정된 2-세포 집단
위에서 돌아간다.

```
페이즈  |---- mitosis (0-60k) ----|-- language (60k-140k) --|- combined -|
성장    *1                             *2(66.6k)  *3(100k)    *5(133k) *8(166k)
        |___ 분화 페이즈 내내 세포 0회 분열 ___|   ^ 첫 분열은 페이즈가 끝난 뒤
```

## 3. 벤치마크 결과 (관측)

| step | phase | loss | ce_fwd | Φ | tension | **cells** |
|------|-------|------|--------|-----|---------|-----------|
| 200 | mitosis | 7.3719 | 0.0000 | 4.326 | 0.58 | **2** |
| 5,000 | mitosis | -37.17 | 0.0000 | — | — | **2** |
| 10,500 | mitosis | -38.28 | 0.0000 | — | — | **2** |
| 23,400 | mitosis | -38.28 | 0.0000 | 0.514 | 120.9 | **2** |

Φ 추이 — 분화 없이 텐션만 감쇠하며 Φ가 바닥에서 진동:

```
 Φ │4.3 ╮
    │    ╲
    │     ╰──╮
 0.5│        ╰──●──●──●──●   ← 세포 2개 고정, 통합할 대상이 없음
    └──────────────────────── step 0 → 23.4k
tension 0.58 → 280 → 120.9 (감쇠), cells 2 → 2 (불변)
```

손실 예산: 200,000 step 중 **60,000 step (30%, RTX5070 기준 약 2.7시간)** 이
언어 학습도 세포 분화도 하지 않는 구간으로 소모된다.

## 4. 핵심 발견 / 법칙

> **분화 페이즈의 예산은 분화가 실제로 일어나는 구간과 위상이 일치해야 한다.**
> 두 스케줄(페이즈 · 성장)이 서로 다른 축으로 정의되면, 각각은 정상으로 보이면서
> 결합 결과만 무효가 된다. 로그의 `cells` 열이 단조 상수인 것이 유일한 증상이다.

Law 22(구조 > 기능)의 실무적 따름정리: 구조(세포 수)를 키우지 않는 페이즈는
기능도 키우지 않는다 — 순수 낭비다.

## 5. 적용 방법 (코드 반영)

`fibonacci_milestones()`에 `growth_fraction`을 추가해 성장 스케줄 전체를 분화
페이즈 안으로 압축하고, 선두 중복 1을 제거한다.

```python
def fibonacci_milestones(total_steps, max_cells=8, growth_fraction=1.0):
    fib = _generate_fibonacci(max_cells)
    usable = sorted({f for f in fib if f <= max_cells})   # 선두 (1,1) 중복 제거
    span = max(int(total_steps * growth_fraction), 1)     # 분화 페이즈로 압축
    return {int(span * i / max(len(usable), 1)): c for i, c in enumerate(usable)}

# 호출부: 페이즈 경계와 동일 축으로 묶는다
_growth_fraction = 0.60 if args.talk5 else 0.30   # get_phase()의 mitosis 경계
fib_milestones = fibonacci_milestones(args.steps, args.max_cells, _growth_fraction)
```

전후 비교 (200,000 step, max_cells=8, 실측 출력):

```
OLD  {0:1, 33333:1, 66666:2, 100000:3, 133333:5, 166666:8}   첫 분열 66.6k [X]
NEW  {0:1, 12000:2, 24000:3,  36000:5,  48000:8}             8셀 완성 48k  [O]
                                                    mitosis 종료 60k
페이즈  |---- mitosis (0-60k) ----|-- language --|
성장    *1  *2    *3    *5   *8                   ← 언어 학습 시작 전 8셀 완비
            12k   24k   36k  48k
```

## 6. 검증 상태 (정직한 기록)

- 확인됨: 스케줄 계산 전후 비교(위 실측 출력), 컴파일 통과.
- 미확인: 8셀 완비 상태의 언어 페이즈가 실제로 CE/Φ를 개선하는지는 다음 run에서
  측정해야 한다. **현재 진행 중인 run(fix2)에는 적용하지 않았다** — 그 run의 목적은
  NF-1(LossEnsemble 발산) 수정 검증이며, 변수를 하나만 바꿔야 판정이 성립한다.
- 잔존 한계: 데이터 제약(300M params vs corpus 67MB ≈ 0.22 tokens/param)은 이
  수정과 무관하게 남는다. 스케줄 정렬은 낭비를 없앨 뿐 코퍼스 부족을 대체하지 않는다.
