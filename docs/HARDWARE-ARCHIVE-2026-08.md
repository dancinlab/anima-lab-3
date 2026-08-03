# aiden·summer 연구 보관 — 2026-08

두 학습 호스트를 처분하기 전 `anima-clm-pure`의 재현 필수 산출물을 비공개 Hugging Face 조직 저장소 `dancinlab/anima-lab-research-archive`에 보관했다. 호스트 이름을 revision으로 사용하며 각 revision의 `archive-manifest-sha256.txt`가 파일 집합의 정본이다.

| revision | 범위 | 파일 | 바이트 | manifest SHA-256 |
|---|---|---:|---:|---|
| `summer` | 등록 실험의 코드·체크포인트·코퍼스·로그·측정 결과 전체 | 386 | 86,217,939,368 | `d7e40e5ff8e4f5a9be931a1abcbce2c76720d7f78d779946fe9d54882d956fd6` |
| `aiden` | NF 계열 코드·데이터·로그와 마지막 재현용 `nf9_v3/best.pt`, `nf9/step_40000.pt` | 139 | 13,879,729,728 | `2305451be018438d82a12c948d72a90ae9da0ff260975f67b0d1178a12824d31` |

`aiden`의 이전 NF 중간 체크포인트는 결론·계측 결과·로그가 남고 마지막 재현 체크포인트로 대체되므로 보관 집합에서 제외했다. `.git`, Hugging Face 캐시, `__pycache__`, GPU 잠금 파일과 인증정보도 제외했다. 서버 원본은 삭제하지 않았다.

복원과 무결성 확인은 secret CLI의 HF 토큰을 일시 환경변수로만 전달한다.

```bash
HF_TOKEN="$(secret get huggingface.token)" hf download \
  dancinlab/anima-lab-research-archive \
  --revision summer --local-dir anima-clm-pure-summer
cd anima-clm-pure-summer
sha256sum -c archive-manifest-sha256.txt
```
