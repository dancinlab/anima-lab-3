# LAMBDA-3 — 300M parameter scale replication

상태: 실행 완료 · S2 확정

## 질문과 동결 설계

27.7M 모델에서 두 seed로 통과한 백과 산문 66.1M·dropout 0.37의 λ0~λ4가 현재 정본 구조의 299,420,896 parameter 모델에서도 재현되는가.

코퍼스와 fresh span, seed 1337/7331, context 256, dropout 0.37, 12,000 optimizer step, 유효 batch 32, λ0~λ4 계측기와 판정 바를 고정한다. 12GB GPU 제약 때문에 물리 batch만 4로 낮추고 8 micro-batch를 누적해 기존 팔과 같은 optimizer-step당 8,192 byte 및 총 98,304,000 byte 노출을 유지한다. 모델만 384d/6L/6H에서 896d/12L/14H로 바꾼다.

- S1: 두 seed 모두 λ0~λ4 PASS — 고정 노출에서 300M 규모로 재현.
- S2: 두 seed의 λ0~λ3는 PASS지만 λ4가 NULL/FAIL — 재조합 등급은 규모 변화에 비불변.
- S3: 한 seed라도 λ0~λ3 FAIL — 이 학습 예산에서 300M 규모 재현 실패.

한 seed만 통과하면 재현으로 세지 않는다. 결과를 본 뒤 threshold, 코퍼스, 노출량, dropout 또는 계측기를 조정하지 않는다.

## SSOT와 실행

팔·학습 인자·코퍼스 지문·결과 파일은 `measurement/lambda_registry.py`의 `EXPERIMENTS["scale300m"]`가 정본이다. `scripts/run_registered_lambda_experiment.py scale300m`이 GPU lock 아래 두 팔 학습, 세 채점기, 통합 gate, 영수증 생성을 순차 수행한다.

결과 SSOT는 `measurement/panel_scale300m_results.json`, `measurement/g_gates_scale300m_results.json`, `measurement/lambda4_scale300m_results.json`, `measurement/gate_verdicts.json`이다.

## 결과 — S2 확정 (2026-08-03)

두 seed 모두 12,000 optimizer step을 skip 0으로 완료했다. best checkpoint는 두 팔 모두 step 11,750이며 검증 BPC는 1.6824/1.6820이다. λ0~λ3은 양쪽 모두 통과했지만 λ4는 seed 1337만 PASS, seed 7331은 짝지은 차이가 해상도 기준에 못 미쳐 NULL이다.

| 팔 | λ1 BPC / floor | λ2 kwr | λ3 부재 4-gram | λ4 맞춤 비용 | 판정 |
|---|---:|---:|---:|---:|---|
| nat300m37 · seed 1337 | 1.6824 / 3.3634 | 0.724 · 5/5 | 40 | −0.00134 · t=−2.14 | λ0~λ4 PASS |
| nat300m37v · seed 7331 | 1.6820 / 3.3634 | 0.801 · 5/5 | 57 | +0.00073 · t=+1.46 | λ0~λ3 PASS · λ4 NULL |

따라서 S1과 S3는 반증되고 S2가 확정된다. 27.7M에서 두 seed로 재현된 λ4는 동일 코퍼스·노출·dropout의 299,420,896 parameter 구조에서는 seed 불변으로 재현되지 않는다. λ0~λ3의 언어 screen은 규모 변화에도 유지되지만, λ4를 300M의 재현된 능력으로 계상하지 않는다.

체크포인트 SHA-256은 seed 1337 `66d241ac41fbec46de345c5b9862b5f1c0214d4317e81062a2d10d3ae15d6d61`, seed 7331 `53ce03d5dbdce05d661080c6d871fafd77a148071ac5d7f5e4de8e5ad52343bd`다. 원본 체크포인트·코퍼스·로그·영수증은 비공개 Hugging Face archive `dancinlife/anima-lab-research-archive`의 `summer` revision에 보존한다.
