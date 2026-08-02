# LAMBDA-2 — λ 통과 창은 다른 자연 register에서도 재현되는가

## 사전등록

결과를 보기 전에 백과 산문과 분리된 공개 도메인 문학 산문에서 66.1M 학습 split을 만들고, LAMBDA-1의 통과점과 같은 27.7M 구조·dropout 0.37·seed 1337/7331·12,000 step을 고정한다. λ0~λ4 계측기와 판정 바는 바꾸지 않는다.

- R1: 두 seed 모두 λ0~λ4 PASS — 66.1M 통과 창은 두 자연 register에서 재현된다.
- R2: λ0~λ3은 두 seed가 통과하지만 λ4가 NULL/FAIL — 언어 screen은 재현되지만 재조합 하한은 register 의존적이다.
- R3: λ1, λ2, λ3 중 하나가 어느 seed에서든 실패 — 기존 통과 창 전체가 백과 register에 의존한다.
- R0: 코퍼스 regime·출처·중복·fresh 분리 영수증이 실패 — 모델 결과를 내지 않고 입력 실험을 무효화한다.

## 고정 입력

- 출처: Open Korean Historical Corpus 리비전 `2d16d39c774ef788069d63223d07e31e038c05df`의 `gongu.jsonl`; 각 원문은 공유마당 공개 도메인 레코드이고 집합 배포 조건은 CC BY-NC 4.0이다.
- 학습 코퍼스: `corpus_natural_literary_ko_dedup.txt`, SHA-256 `336e101a5b9737c2e12073b5562a06320c150b5a19655a8046b7c16e13ddff5e`, 총 72,885,097B / trainer train 66,060,288B.
- 완전 미노출 fresh: SHA-256 `8e196165d525e15bc4b200e395953b19d6007acd0cb2c65746649dc4acb5cecd`, 29,899,997B. 문서 ID 해시로 분리하며 학습 코퍼스와 전역 줄 중복을 제거한다.
- regime: 718,161줄 / 고유 718,161줄 / repetition 1.00x / top-10 0.00%.
- 자기 floor: unigram 5.3539268410 BPC / bigram 3.4055540943 BPC. 다른 코퍼스의 floor를 재사용하지 않는다.

원자료 SSOT는 `measurement/lambda_registry.py`와 실행 후 생성되는 세 literary 결과 JSON이다.
