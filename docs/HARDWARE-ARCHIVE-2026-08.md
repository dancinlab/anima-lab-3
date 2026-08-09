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
