# 연구 장비 산출물 보관 — 2026-08

두 학습 호스트를 처분하기 전 `anima-clm-pure`의 재현 필수 산출물을 비공개 Hugging Face 조직 저장소 `dancinlab/anima-lab-research-archive`에 보관했다. 호스트 이름을 revision으로 사용하며 각 revision의 `archive-manifest-sha256.txt`가 파일 집합의 정본이다.

| revision | 범위 | 파일 | 바이트 | manifest SHA-256 |
|---|---|---:|---:|---|
| `summer` | 등록 실험의 코드·체크포인트·코퍼스·로그·측정 결과 전체 | 386 | 86,217,939,368 | `d7e40e5ff8e4f5a9be931a1abcbce2c76720d7f78d779946fe9d54882d956fd6` |
| `aiden` | NF 계열 코드·데이터·로그와 마지막 재현용 `nf9_v3/best.pt`, `nf9/step_40000.pt` | 139 | 13,879,729,728 | `2305451be018438d82a12c948d72a90ae9da0ff260975f67b0d1178a12824d31` |
| `lambda5-ffn` | 임시 RTX 5090의 LAMBDA-5 일반 FFN 두 seed 체크포인트·로그·영수증·측정 결과 | 12 | 668,060,681 | `d0edec5306dd701bffc958bc1cc0d3b5fc1fdd322d0a47093e879ba9550d87cb` |
| `graft-behavior` | 임시 RTX 5090의 GRAFT 행동 인과 1차·언어 보존 보정 실험 체크포인트·로그·결과·판정 | 16 | 1,612,415,811 | `ef534b4df49dc588cfc78570a7949774729d854214d9c3097e94fc6f7171fc2c` |
| `graft-phase-behavior` | 임시 RTX 5090의 GRAFT 위상 행동 인과 무효 1차·양성 비교군 수리 실험 체크포인트·로그·결과·판정 | 16 | 1,612,427,203 | `a54b81643d5e8854972d49bc1f02a7fe28bb86fce378d2d92c309734e6a85e94` |
| `graft-bridge32-behavior` | 임시 RTX 5090의 폭 32 행동 인과 무효 1차·양성 비교군 수리 실험 체크포인트·로그·결과·판정 | 15 | 1,614,925,450 | `55e98a02e8f3f0eaa799b4f634ea3927c6c77c2fcf775b1d21af5354d0afa5a4` |
| `meta1-self-monitoring` | 임시 RTX 5090의 META-1 무효 2회·최종 결과·확신 판독기·로그 | 11 | 2,329,458 | `4b3e5b976f70462a1e5f6d96fa799c7e765c59f41b4ecab9417aa36f0db89ae7` |
| `synergy1-split-cue` | 임시 RTX 5090의 SYNERGY-1 무효·최종 실행 체크포인트 12개·로그·결과·판정 | 21 | 2,424,573,584 | `bc8bbbb91d028fb404384526f78d6e20bed2d36769cfe999e2e4b3710844311d` |
| `workspace1-recurrent` | 임시 RTX 5090의 WORKSPACE-1 무효·수리 본실험 체크포인트 32개·로그·결과·판정 | 40 | 6,464,830,683 | `e9728489b228f448680cc8a63dbe73f835b438f58a64ae359ba260f123b6672b` |
| `episode-control1-dynamic-relation` | 로컬 CPU CONTROL-1 결과·판정·표준 GRU 두 seed 체크포인트 | 5 | 353,592 | `0cbc1373f0afd01892b95a3c130b0fbcd95e5658737fd013212fb47fc5dd94c1` |
| `episode-control2-online-dynamic-relation` | 로컬 CPU CONTROL-2 온라인 자료 결과·판정·표준 GRU 두 seed 체크포인트 | 5 | 357,690 | `51e44840f19ab605a5e1243980f72854d7a6ef2fdaa16fcf3fd6f7bd19db95f8` |
| `episode-control3-keyed-attention` | 로컬 CPU CONTROL-3 온라인 자료 결과·판정·표준 주의집중 두 seed 체크포인트 | 5 | 362,920 | `8884dcde70235caae5c2b2e254bd221e4b8decc0fcc5033f46a6bbac71ec9b35` |
| `episode1-one-shot-relation` | 로컬 CPU EPISODE-1 결과·판정·두 seed 값 원형 체크포인트 | 5 | 79,196 | `a252762d60cd8833360a7a0a5aa9018a1bb5ef53dc7d92d8f3fa33d7c22d3815` |
| `key1-temporal-key-stabilization` | 로컬 CPU KEY-1 결과·판정·두 seed 안정 주소 체크포인트 | 5 | 105,112 | `53a3d64ad0fa92870613a2a26e084e47975723f3a82aaf8bba5ae43aeea1c154` |
| `episode2-integrated-stable-memory` | 로컬 CPU EPISODE-2 결과·판정·재사용한 두 seed 안정 주소 체크포인트 | 5 | 102,198 | `4971dc8c21323278a3bbc4673cb8544b667e35de88e8d0287b16802bfa848c23` |
| `separation1-similar-episode` | 로컬 CPU SEPARATION-1 무효 결과·판정·재사용한 두 seed 안정 주소 체크포인트 | 5 | 97,980 | `d43396cf6a15d9786351faeb55a370136ac82f6b8db6ba099d733383e3186e97` |
| `capacity1-stable-address-boundary` | 로컬 CPU CAPACITY-1 결과·판정·재사용한 두 seed 안정 주소 체크포인트 | 5 | 154,683 | `039b0ddc2e501822b398f07ec5a1e565dc9da59c59320114e3d087527bf97d43` |
| `decay1-memory-decomposition` | 로컬 CPU DECAY-1 최초 무효·수리 결과와 판정·재사용한 두 seed 안정 주소 체크포인트 | 7 | 391,037 | `7f7c93c8479343a31915f79242fbb57ebc7498370ae0eb05d2d2920d55db4bda` |
| `recovery1-dense-curve` | 로컬 CPU RECOVERY-1 최초 정밀도 무효본·수리 판정·독립 회복 곡선 | 5 | 2,340,685 | `fa9bd692b42d387795dd88041ccc46accbdae5696287393fe678ad6fa982a6fa` |
| `reset1-recovery-mechanism` | 로컬 CPU RESET-1 감각 변화·같은 감각·자체 갱신 비교 결과와 판정 | 3 | 1,829,632 | `8badfd1b91e4bda72b72c584fa523e0290ca98426f42ac7359c459d2e0beb750` |
| `settle1-autonomous-memory` | 로컬 CPU SETTLE-1 자체 갱신·완전 정지 짝비교 결과와 판정 | 3 | 2,228,034 | `57254cd3f4b4d8cfd978c1b4ad6004643520bfe75b0c242e2a62bc36f6dc5971` |
| `mechanism1-settling-components` | 로컬 CPU MECHANISM-1 계산 차단 결과·최종 및 최초 무효 판정 | 5 | 3,976,973 | `cad78274e8521a645e4c0ef724ce4ceec00394d5903f5596ff083bae49b70beb` |
| `capacity2-settled-boundary` | 로컬 CPU CAPACITY-2 기존·안정화·조절 차단 용량 결과와 판정 | 3 | 314,213 | `687c7e234a7d229d09ac5f5fe37681ad74005a49efc13b9fae1999e2cc353a44` |
| `seedmap1-capacity-factorial` | 로컬 CPU SEEDMAP-1 주소 변환·값 원형·엔진 시작값 전체 교차 결과와 판정 | 3 | 117,954 | `c90ba94171477489856e4f5bfda6f2ecd76ac1cee82078f11a14877fd241d8af` |
| `projector1-address-training-factorial` | 로컬 CPU PROJECTOR-1 보정 상태·학습 난수 교차 결과, 판정, 주소 변환 4개 | 8 | 330,288 | `1430804fd5ebde87ccf9fee8755365481e0ca1ade2fad5db0f29281825a2b0d9` |
| `training1-address-randomness` | 로컬 CPU TRAINING-1 주소 초기값·학습 순서 교차 결과, 판정, 주소 변환 4개 | 7 | 329,991 | `cd58cf0be32c7fba9985fc3c3309f9506beb2cff29f1501475a3cc650027fa66` |
| `canonical1-deterministic-address` | 로컬 CPU CANONICAL-1 결정형 주소 세 개의 결과·판정·체크포인트 | 6 | 249,265 | `3f827cbe6e2b52fed0ef67ef1c3e30b7542e68ca0ec91cec3dde3a276b3700ed` |

`aiden`의 이전 NF 중간 체크포인트는 결론·계측 결과·로그가 남고 마지막 재현 체크포인트로 대체되므로 보관 집합에서 제외했다. `.git`, Hugging Face 캐시, `__pycache__`, GPU 잠금 파일과 인증정보도 제외했다. 서버 원본은 삭제하지 않았다.

복원과 무결성 확인은 secret CLI의 HF 토큰을 일시 환경변수로만 전달한다.

```bash
HF_TOKEN="$(secret get huggingface.token)" hf download \
  dancinlab/anima-lab-research-archive \
  --revision summer --local-dir anima-clm-pure-summer
cd anima-clm-pure-summer
sha256sum -c archive-manifest-sha256.txt
```

LAMBDA-5 보관은 Hugging Face 커밋 `df22be2066be2dfe08bd20cfee0a19ffc8c09281`에 올렸고, 원격에서 다시 받은 manifest가 위 SHA-256과 일치함을 확인했다. 두 체크포인트의 Hugging Face LFS SHA-256도 연구 원장의 전체 해시와 일치한다.

GRAFT 행동 보관은 Hugging Face 커밋 `f11dfcf1ea479a6fe84691b3ac91f96450cb472d`에 올렸다. 원격
manifest 15개 항목을 원본에서 전수 확인했고, 8개 체크포인트의 Hugging Face 대용량 파일 SHA-256이
manifest와 모두 일치한다.

GRAFT 위상 행동 보관은 Hugging Face 커밋 `af8a61395a05afdcf949d4fbb2f351ed736c8675`에 올렸다.
원격 revision을 새 디렉토리로 다시 내려받아 manifest 15개 항목을 전수 확인했고, 다운로드한
manifest 자체의 SHA-256도 위 표와 일치한다.

GRAFT 폭 32 행동 보관은 Hugging Face 커밋 `0f10c4441701bdfafdfb797720d1d9be1b29091c`에 올렸다.
원격 revision을 새 디렉토리로 다시 내려받아 manifest 14개 항목을 전수 확인했고, 다운로드한
manifest 자체의 SHA-256도 위 표와 일치한다.

META-1 보관은 `meta1-self-monitoring` revision에 올렸다. 원격 revision을 새 디렉토리로 다시
내려받아 manifest 11개 항목을 전수 확인했고, 다운로드한 manifest 자체의 SHA-256도 위 표와
일치한다. 원본 행동 체크포인트는 중복 보관하지 않고 `graft-bridge32-behavior` revision의 등록
SHA-256 네 개를 참조한다.

SYNERGY-1 보관은 Hugging Face 커밋 `10a729cfe448826a4a1a801b73f5ad97f2001f67`에 올렸다. 원격
revision을 새 디렉토리로 다시 내려받아 manifest 21개 항목을 전수 확인했고, 다운로드한 manifest
자체의 SHA-256도 위 표와 일치한다. 1차 무효와 비교군 역할 수리 실행의 체크포인트를 모두 보존했다.

WORKSPACE-1 보관은 Hugging Face 커밋 `fbff86383ded770fdc45e22c728231d062ba7d53`에 올렸다. 원격
revision을 이 Mac의 새 임시 디렉토리로 다시 내려받아 manifest 40개 항목을 전수 확인했고,
다운로드한 manifest 자체의 SHA-256도 위 표와 일치한다. 팔 순서 때문에 무효였던 1차와 이름별
난수를 고친 최종 실행의 체크포인트를 모두 보존했다.

CONTROL-1 보관은 Hugging Face 커밋 `fd0deebe88dfd6be5dfff3b3a1d0d72077965fe2`에 올렸다. 원격
revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 체크포인트의 SHA-256을 전수 확인했고,
다운로드한 manifest 자체의 SHA-256도 위 표와 일치한다. GPU를 빌리지 않은 로컬 CPU 실행이다.

CONTROL-2 보관은 Hugging Face 커밋 `f508c5664b5dc681e92f24e48415557be7b1a65a`에 올렸다. 원격
revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 체크포인트의 SHA-256을 전수 확인했고,
다운로드한 manifest 자체의 SHA-256도 위 표와 일치한다. GPU를 빌리지 않은 로컬 CPU 실행이다.

CONTROL-3 보관은 Hugging Face 커밋 `c3d6c6eaebf94e3b7ff9474faa86b230da992079`에 올렸다. 원격
revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 체크포인트의 SHA-256을 전수 확인했고,
다운로드한 manifest 자체의 SHA-256도 위 표와 일치한다. GPU를 빌리지 않은 로컬 CPU 실행이다.

EPISODE-1 보관은 Hugging Face 커밋 `a41546afdf5bbab9ed414c7d963f746787928819`에 올렸다. 원격
revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 값 원형 체크포인트의 SHA-256을 전수
확인했고, 다운로드한 manifest 자체의 SHA-256도 위 표와 일치한다. GPU를 빌리지 않은 로컬 CPU
실행이다.

KEY-1 보관은 Hugging Face 커밋 `cb828c2157c01476027542266a651add1357c48e`에 올렸다. 원격
`key1-temporal-key-stabilization` revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 주소
변환 체크포인트의 SHA-256을 전수 확인했다. manifest SHA-256은
`53a3d64ad0fa92870613a2a26e084e47975723f3a82aaf8bba5ae43aeea1c154`다. GPU를 빌리지 않은 로컬
CPU 실행이다.

EPISODE-2 보관은 Hugging Face 커밋 `691eadfdde0200967efd27f185b0ed23b7261fd1`에 올렸다. 원격
`episode2-integrated-stable-memory` revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·두 주소
변환 체크포인트의 SHA-256을 전수 확인했고, 다운로드한 manifest 자체의 SHA-256도 위 표와
일치한다. 새 학습 없이 로컬 CPU에서 공용 기억 경로만 재검증했다.

SEPARATION-1 보관은 Hugging Face 커밋 `2e78fe240745d1974f57f1883bfa39de45a56ab4`에 올렸다. 원격
`separation1-similar-episode` revision을 새 임시 디렉토리로 다시 내려받아 무효 결과·판정·재사용한
두 주소 변환 체크포인트의 SHA-256을 전수 확인했고, manifest 자체의 SHA-256도 위 표와 일치한다.
새 학습 없이 로컬 CPU에서 실행했다.

CAPACITY-1 보관은 Hugging Face 커밋 `a696d6891b15956a327e71b3c0125e8c1bb1d3c5`에 올렸다. 원격
`capacity1-stable-address-boundary` revision을 새 임시 디렉토리로 다시 내려받아 결과·판정·재사용한
두 주소 변환 체크포인트의 SHA-256을 전수 확인했고, manifest 자체의 SHA-256도 위 표와 일치한다.
새 학습 없이 로컬 CPU에서 실행했다.

DECAY-1 보관은 Hugging Face 커밋 `0bba39ddf9121dcb85686bbdd57d0372af425d6a`에 올렸다. 원격
`decay1-memory-decomposition` revision을 새 임시 디렉토리로 다시 내려받아 시간 조건별 초기 난수가
달랐던 최초 무효본, 동일 초기 상태로 고친 수리 결과·판정과 두 주소 변환 체크포인트의 SHA-256을
전수 확인했다. manifest 자체의 SHA-256도 위 표와 일치하며 새 학습이나 GPU 대여는 없었다.

RECOVERY-1 보관은 Hugging Face 커밋 `c4e7579b00864c0c66f72fb16c5d5f4b8837ff16`에 올렸다. 원격
`recovery1-dense-curve` revision을 새 임시 디렉토리로 다시 내려받아 최초 정밀도 무효본과 최종
결과·판정의 SHA-256을 전수 확인했다. manifest 자체의 SHA-256도 위 표와 일치한다.

RESET-1 보관은 Hugging Face 커밋 `91ee5c87f740571505a2f68e8da8d415f1027131`에 올렸다. 원격
`reset1-recovery-mechanism` revision의 결과·판정·manifest 세 파일을 새 임시 디렉토리로 다시
내려받아 SHA-256을 전수 확인했다. 새 학습이나 GPU 대여는 없었다.

SETTLE-1 보관은 Hugging Face 커밋 `919c475208d0c3abbc83b81b8d3d9523015fea79`에 올렸다. 원격
`settle1-autonomous-memory` revision의 결과·판정·manifest 세 파일을 새 임시 디렉토리로 다시
내려받아 SHA-256을 전수 확인했다. 새 학습이나 GPU 대여는 없었다.

MECHANISM-1 보관은 Hugging Face 커밋 `242ee033f47f3147090502f596587b4d5610094b`에 올렸다. 원격
`mechanism1-settling-components` revision의 결과·최종 판정·최초 무효 판정·manifest 다섯 파일을
새 임시 디렉토리로 다시 내려받아 SHA-256을 전수 확인했다. 새 학습이나 GPU 대여는 없었다.

CAPACITY-2 보관은 Hugging Face 커밋 `e9c4ef4d6d33945ab0a52b3a13f23f74f201b909`에 올렸다. 원격
`capacity2-settled-boundary` revision의 결과·판정·manifest 세 파일을 새 임시 디렉토리로 다시
내려받아 SHA-256을 전수 확인했다. 새 학습이나 GPU 대여는 없었다.

SEEDMAP-1 보관은 Hugging Face 커밋 `0b453b36a32868c9ae1d1afd4abc341afa3df1e4`에 올렸다. 원격
`seedmap1-capacity-factorial` revision의 결과·판정·manifest 세 파일을 새 임시 디렉토리로 다시
내려받아 SHA-256을 전수 확인했다. 새 학습이나 GPU 대여는 없었다.

PROJECTOR-1 보관은 Hugging Face 커밋 `bfc8d3a26a0e3e31ce8fab99c26b3d7c6bb5525b`에 올렸다. 원격
`projector1-address-training-factorial` revision의 결과·최종 판정·최초 무효 판정·주소 변환 네 개와
manifest를 새 임시 디렉토리로 다시 내려받아 SHA-256을 전수 확인했다. 로컬 CPU에서만 실행했다.

TRAINING-1 보관은 Hugging Face 커밋 `d6aa48a87650717cc58f28a88c1b6f962f3189d7`에 올렸다. 원격
`training1-address-randomness` revision의 결과·판정·주소 변환 네 개와 manifest를 새 임시
디렉토리로 다시 내려받아 SHA-256을 전수 확인했다. 로컬 CPU에서만 실행했다.

CANONICAL-1 보관은 Hugging Face 커밋 `683ddde2c745d18f66795631e6dea4a73bb63e67`에 올렸다. 원격
`canonical1-deterministic-address` revision의 결과·판정·결정형 주소 세 개와 manifest를 새 임시
디렉토리로 다시 내려받아 SHA-256을 전수 확인했다. 로컬 CPU에서만 실행했다.
