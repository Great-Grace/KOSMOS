# 응답 원본 (gzip 압축)

LLM 응답 원본 JSONL을 gzip으로 압축한 것. 풀 때는 `gunzip -k <파일>.gz`.
로컬 원본은 `raw/`에 있고, 용량 때문에 압축본만 저장소에 올렸다.

| 파일 | 내용 |
|---|---|
| `raw_gz/v2/haiku_p1.jsonl.gz` | v2 1단계, Claude Haiku 5.5 |
| `raw_gz/v2/deepseek_ultrafast_p1.jsonl.gz` | v2 1단계, DeepSeek V4.1 Flash Ultrafast |
| `raw_gz/v2/compare_eco_ultrafast.jsonl.gz` | Eco/Ultrafast 비교 호출 (프롬프트 60개 × 50회) |
| `raw_gz/exaone_stage1.jsonl.gz` | K-EXAONE 1단계 (`raw_gz/exaone_stage1.log`는 실행 로그) |
| `raw_gz/stage0·1·2.jsonl.gz` | 초기 단계 응답 |
| `raw_gz/v2/*_counts.csv`, `haiku_p1_batches.json` | 보기 개수 벡터 집계, Haiku 배치 원장 |

`*_counts.csv`는 `research/design_v2_20261112/runners/aggregate_counts.py`로 원응답에서 다시 만들 수 있다.
