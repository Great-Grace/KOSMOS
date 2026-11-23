# v2 응답 원본 (gzip 압축)

v2 실험의 응답 원본이다. 로컬 원본은 `raw/v2/`에 있고, 용량 때문에 압축본만 저장소에 올렸다.

## 실험 코드를 다시 돌리려면

`experiments/*/run.py`는 `raw/v2/<파일>.jsonl` 경로를 읽는다. 저장소 루트에서 아래 명령으로 풀어 두면 그대로 돌아간다.

```bash
mkdir -p raw/v2 && for f in results/raw_gz/v2/*.jsonl.gz; do gzip -dc "$f" > "raw/v2/$(basename "$f" .gz)"; done
```

90MB가 넘어 나눠 올린 파일(`*.jsonl.partNN.gz`, 2026-11-23부터)은 이어 붙여 푼다.

```bash
for f in results/raw_gz/v2/*.jsonl.part01.gz; do b=$(basename "$f" .part01.gz); cat results/raw_gz/v2/$b.part*.gz | gzip -dc > raw/v2/$b; done
```

새로 수집한 원응답은 `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<이름>.jsonl`로 묶어 올린다.

사람 분포와 세 모델의 k=100 개수 벡터는 `research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz`에 커밋돼 있어서 풀 필요가 없다.

## 본실험 (프롬프트 2,952개 × 표본 100번)

| 파일 | 내용 | 줄 수 |
|---|---|---|
| `raw_gz/v2/haiku_p1.jsonl.gz` | Claude Haiku 5.5 (Anthropic Batch) | 295,200 |
| `raw_gz/v2/deepseek_ultrafast_p1.jsonl.gz` | DeepSeek V4.1 Flash Ultrafast (OpenGateway) | 295,210 (`api_error` 10줄 포함) |
| `raw_gz/v2/compare_eco_ultrafast.jsonl.gz` | Eco/Ultrafast 비교 호출 (프롬프트 60개 × 50회) | 6,000 |
| `raw_gz/v2/*_counts.csv` | 프롬프트별 보기 개수 벡터 | |
| `raw_gz/v2/haiku_p1_batches.json` | Haiku 배치 제출 원장 | |

## 추론 효과·인칭·화계 실험 (2026-11-18~19, 표본 50번)

| 파일 | 내용 | 줄 수 | 쓰는 실험 |
|---|---|---|---|
| `raw_gz/v2/deepseek_eco_p1.jsonl.gz` | DeepSeek Eco, 추론 끔, 원문 문체 | 147,600 | reasoning-effect, register-effect, beta-subgroup |
| `raw_gz/v2/deepseek_eco_low_p1.jsonl.gz` | DeepSeek Eco, 추론 low, 원문 문체 | 147,600 | reasoning-effect, register-effect, beta-subgroup |
| `raw_gz/v2/deepseek_eco_register_p1.jsonl.gz` | DeepSeek Eco, 추론 끔, 새 문체 3종 | 49,200 | register-effect |
| `raw_gz/v2/deepseek_eco_low_register_p1.jsonl.gz` | DeepSeek Eco, 추론 low, 새 문체 3종 | 49,200 | register-effect |
| `raw_gz/v2/deepseek_eco_medium_p1.jsonl.gz` | DeepSeek Eco, 추론 medium, 원문 문체 (2026-11-23) | 147,642 (`api_error` 42줄 포함, 다시 받은 건은 뒤에 추가) | reasoning-medium |
| `raw_gz/v2/haiku_register_p1.jsonl.gz` | Haiku 5.5, 추론 끔, 새 문체 3종 | 49,200 | register-effect |
| `raw_gz/v2/haiku_register_p1_batches.json` | Haiku 배치 제출 원장 | | |
| `raw_gz/v2/*_smoke.jsonl.gz` | 본 실행 전 소량 점검 호출 | | k-compromise 등 |
| `raw_gz/v2/haiku_{low,medium}_p1_smoke_batches.json` | 점검 호출 배치 원장 | | |

- 응답 한 줄 = `(probe_id, sample_idx)` 하나. `valid`, `invalid_type`으로 유효 여부를 표시한다.
- `*_counts.csv`는 `research/design_v2_20261112/runners/aggregate_counts.py`로 원응답에서 다시 만들 수 있다.
- **K-EXAONE(팀원 분담) 응답 원본은 아직 이 저장소에 없다.** 개수 벡터만 `counts_k100.npz`에 들어 있다.
- 11월 초 Haiku 파일럿·K-EXAONE 탐색 실행(`stage0~2`, `exaone_stage1`)은 올리지 않았다.
