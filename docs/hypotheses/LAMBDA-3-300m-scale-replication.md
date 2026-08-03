# LAMBDA-3 — 300M parameter scale replication

상태: 사전등록 · 결과 미확인

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
