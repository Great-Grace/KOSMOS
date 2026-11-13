# v2 응답 원본 (gzip 압축)

v2 실험(프롬프트 2,952개 × 표본 100번)의 응답 원본이다. 풀 때는 `gunzip -k <파일>.gz`.
로컬 원본은 `raw/v2/`에 있고, 용량 때문에 압축본만 저장소에 올렸다.

| 파일 | 내용 | 줄 수 |
|---|---|---|
| `raw_gz/v2/haiku_p1.jsonl.gz` | Claude Haiku 5.5 (Anthropic Batch) | 295,200 |
| `raw_gz/v2/deepseek_ultrafast_p1.jsonl.gz` | DeepSeek V4.1 Flash Ultrafast (OpenGateway) | 295,210 (`api_error` 10줄 포함) |
| `raw_gz/v2/compare_eco_ultrafast.jsonl.gz` | Eco/Ultrafast 비교 호출 (프롬프트 60개 × 50회) | |
| `raw_gz/v2/*_counts.csv` | 프롬프트별 보기 개수 벡터 | |
| `raw_gz/v2/haiku_p1_batches.json` | Haiku 배치 제출 원장 | |

- 응답 한 줄 = `(probe_id, sample_idx)` 하나. `valid`, `invalid_type`으로 유효 여부를 표시한다.
- `*_counts.csv`는 `research/design_v2_20261112/runners/aggregate_counts.py`로 원응답에서 다시 만들 수 있다.
- **K-EXAONE(팀원 분담) 응답은 아직 이 저장소에 없다.** 팀원 결과를 합치면 같은 형식으로 추가한다.
- 이 폴더에는 v2 실험만 있다. 11월 초 Haiku 파일럿·K-EXAONE 탐색 실행(`stage0~2`, `exaone_stage1`)은 올리지 않았다.
