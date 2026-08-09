# 연구 장비 산출물 보관 — 2026-08

두 학습 호스트를 처분하기 전 `anima-clm-pure`의 재현 필수 산출물을 비공개 Hugging Face 조직 저장소 `dancinlab/anima-lab-research-archive`에 보관했다. 호스트 이름을 revision으로 사용하며 각 revision의 `archive-manifest-sha256.txt`가 파일 집합의 정본이다.

| revision | 범위 | 파일 | 바이트 | manifest SHA-256 |
|---|---|---:|---:|---|
| `summer` | 등록 실험의 코드·체크포인트·코퍼스·로그·측정 결과 전체 | 386 | 86,217,939,368 | `d7e40e5ff8e4f5a9be931a1abcbce2c76720d7f78d779946fe9d54882d956fd6` |
| `aiden` | NF 계열 코드·데이터·로그와 마지막 재현용 `nf9_v3/best.pt`, `nf9/step_40000.pt` | 139 | 13,879,729,728 | `2305451be018438d82a12c948d72a90ae9da0ff260975f67b0d1178a12824d31` |
| `lambda5-ffn` | 임시 RTX 5090의 LAMBDA-5 일반 FFN 두 seed 체크포인트·로그·영수증·측정 결과 | 12 | 668,060,681 | `d0edec5306dd701bffc958bc1cc0d3b5fc1fdd322d0a47093e879ba9550d87cb` |
| `graft-behavior` | 임시 RTX 5090의 GRAFT 행동 인과 1차·언어 보존 보정 실험 체크포인트·로그·결과·판정 | 16 | 1,612,415,811 | `ef534b4df49dc588cfc78570a7949774729d854214d9c3097e94fc6f7171fc2c` |
| `graft-phase-behavior` | 임시 RTX 5090의 GRAFT 위상 행동 인과 무효 1차·양성 비교군 수리 실험 체크포인트·로그·결과·판정 | 16 | 1,612,427,203 | `a54b81643d5e8854972d49bc1f02a7fe28bb86fce378d2d92c309734e6a85e94` |

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
