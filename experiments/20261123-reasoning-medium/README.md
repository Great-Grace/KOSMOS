# 20261123-reasoning-medium

| 항목 | 내용 |
|---|---|
| 담당 | 이지은 |
| 시작 | 2026-11-23 |
| 상태·결론 | `meta.json` (`status`: 진행/완료/중단, `conclusion`: 한 줄) |
| 데이터 | `research/design_v2_20261112/dataset/kosmos_long.csv.gz` (해시는 `out/run.json`), 추론 medium 원응답 `raw/v2/deepseek_eco_medium_p1.jsonl` (압축본 `results/raw_gz/v2/`) |
| 수집 | DeepSeek Eco, 추론 medium, 원문, 페르소나 없음·한국 성인·성별×연령 10셀, 246문항, k=50 (147,600건). 분담 테스크 `tasks/reasoning-medium.md` |

## 질문 (돌리기 전에 쓴다)

추론을 low에서 medium으로 올리면 (1) 사람 분포와의 거리, (2) 셀끼리 차이의 크기, (3) 그 차이가 사람 셀 차이와 같은 방향인지가 어떻게 바뀌는가.

## 가설과 판정 기준 (돌리기 전에 쓴다)

- 가설: 방향 예측은 두지 않는다(양쪽 모두 가능).
- 판정: 주 지표 4개 모두 Holm 보정 p < 0.05면 "있음", 아니면 "없음"으로 적는다.
- 쓰는 문항 분할: calibration 148문항(β 고르기) / validation 49문항(확인). test 49문항은 쓰지 않는다.

### 주 지표 4개 (medium − low, Holm 보정)

1. 보정 후 사람 TVD, 한국 성인 (validation 49문항)
2. 보정 후 사람 TVD, 10셀 pop_share 가중 (validation 49문항)
3. 셀 간 분산비 (모델/사람, 잡음 뺀 값, test 뺀 순서형·이진 문항)
4. 셀 편차 정렬 (가중 상관, 같은 문항)

- 구간: 같은 문항끼리 짝지은 문항 부트스트랩 95%, 2,000회. p = 2·min(비율(차이 ≤ 0), 비율(차이 ≥ 0)).
- β: 설정(끔/low/medium)마다 페르소나 없음 하나, 한국 성인과 10셀이 하나. calibration 148문항에서 TVD가 최소인 값, 격자 0.05~1.5(0.005 간격).
- 지표 1·2는 `experiments/20261118-reasoning-effect/run.py`, 지표 3·4는 `experiments/20261123-reasoning-explain/run.py`의 정의를 그대로 쓴다. 함수는 복사하고 출처를 주석으로 남긴다.

### 탐색 (보정 없음, 결과에 "탐색"으로 표기)

medium − 끔(같은 4지표), 기울기, 이동 분해(공통 이동/셀 고유 이동), 무효율, 평균 추론 토큰, 끔→low→medium 추세 표.

## 방법

`run.py`가 하는 일은 위 지표 계산이다. 데이터는 추론 끔 `raw/v2/deepseek_eco_p1.jsonl`, low `raw/v2/deepseek_eco_low_p1.jsonl`, medium(새 파일)이고, 정답은 `counts_k100.npz`의 H·NEFF다.

## 결과 (돌린 뒤 쓴다)

`out/`의 표에서 옮긴 숫자만 쓴다. 표 파일 이름을 같이 적는다.

## 결론

한 줄. 가설이 기각돼도 그대로 쓴다.

## 다음
