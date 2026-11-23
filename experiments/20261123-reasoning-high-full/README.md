# 20261123-reasoning-high-full (추론 high, test 제외 197문항 전체)

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-23 |
| 상태·결론 | `meta.json` |
| 데이터 | 추론 끔 `raw/v2/deepseek_eco_p1.jsonl`, low `raw/v2/deepseek_eco_low_p1.jsonl`, high = `raw/v2/deepseek_eco_high_high60_p1.jsonl`(기존 60문항) + `raw/v2/deepseek_eco_high_high_rest137_p1.jsonl`(신규 137문항). DeepSeek V4.1 Flash Eco, k=50, 페르소나 없음·한국 성인·성별×연령 10셀, 원문 문체. medium `raw/v2/deepseek_eco_medium_p1.jsonl`이 있으면 추세 표에 넣는다. 정답은 `counts_k100.npz`의 H·NEFF. 해시는 `out/run.json` |
| 문항 | test를 뺀 197문항(calibration 148, validation 49). 신규 수집분은 `subsets/high_rest137.csv`(`subsets/build_high_rest137.py`, high60에 없는 문항) |

## 질문 (돌리기 전에 쓴다)

`20261123-reasoning-high`(60문항)에서 low → high 차이는 분산비 +0.23(Holm p 0.857), 정렬 +0.032(95% −0.006 ~ +0.074, Holm p 0.196)였다. 정렬의 작은 상승이 문항을 197개로 늘리면 확인되는가, 분산비 "멈춤"은 유지되는가.

## 가설과 판정 기준 (돌리기 전에 쓴다)

방향 예측은 두지 않는다. test 49문항은 쓰지 않는다.

주의: 197문항 중 60문항은 이미 결과를 본 문항이다(위 실험). 그래서 이 실험은 독립 재현이 아니라 같은 질문의 정밀도를 높이는 확장이다. 결과 보고에 60문항 결과와 신규 137문항만의 결과를 함께 적는다(탐색).

대상은 197문항 중 순서형·이진 문항(`scale_type`이 ordinal_monotone 또는 binary, 188개)이다. 모델 분포는 원응답 비율이고 β를 적용하지 않는다.

**주 지표 2개** (Holm 보정, high − low, 문항 부트스트랩 2,000회, p = 2 × min(비율(≤0), 비율(≥0)), 최소 1/2,000)

| 지표 | 정의 |
|---|---|
| 셀 간 분산비 | Σ문항 V모델 / Σ문항 V사람. V = Σ셀 w(s셀 − s̄)²에서 표집 잡음 기댓값 Σ w(1−w)·분산/표본수를 뺀 값. 10셀 가중치 w는 `cells.csv`의 pop_share |
| 셀 편차 정렬 | 문항·셀을 모두 모은 가중 상관 Σ w·d모델·d사람 / √(Σ w·d모델² · Σ w·d사람²), d = s셀 − s̄ |

정의는 `experiments/20261123-reasoning-high/run.py`와 같다. 문항 선택만 197문항으로 바꾼다.

판정: Holm 보정 p < 0.05면 "있음"이다.

**탐색** (보정 없음, "탐색"으로 표기)
- 보정 후 사람 TVD. β는 설정마다 페르소나 없음 하나, 한국 성인과 10셀 하나이고, calibration 148문항에서 고른 뒤 validation 49문항에서 잰다.
- 기울기 Σ w·d모델·d사람 / Σ w·d사람².
- 이동 분해(low → high): 공통 이동과 셀 고유 이동의 제곱합.
- 신규 137문항만으로 다시 낸 주 지표(이미 본 60문항을 뺀 값).
- 무효율, 평균 추론 토큰, 길이 초과 비율.
- 끔 → low → (medium) → high 추세 표.

## 방법

`run.py` 하나다. `20261123-reasoning-high/run.py`를 복사해 문항을 197개로 바꾸고, high를 두 파일에서 합쳐 읽으며, 신규 137문항만의 주 지표를 추가했다.

수집: `run_deepseek_eco.py --reasoning high --items subsets/high_rest137.csv --k 50 --max-usd 4` (82,200건, 예상 약 $2.2).

출력:
- `out/primary.csv`(주 지표와 Holm), `out/explore.csv`
- `out/trend.csv`(설정별 지표), `out/quality.csv`
- `out/run.log`, `out/run.json`

## 결과 (돌린 뒤 쓴다)

## 결론

## 다음
