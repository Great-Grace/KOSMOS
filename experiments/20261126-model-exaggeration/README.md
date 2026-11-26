# 20261126-model-exaggeration (작업 C: 집단 차이 과장이 세 모델 공통인가)

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-26 |
| 상태·결론 | `meta.json` |
| 데이터 | `research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz` (세 모델 k=100 개수 텐서와 사람 정답지). 셀 가중치 `research/design_v2_20261112/cells.csv`, 척도 유형 `item_manifest_v2.csv`. 해시는 `out/run.json` |
| 모델 | K-EXAONE 2.5(추론 low 고정), DeepSeek V4.1 Flash Ultrafast(추론 끔), Haiku 5.5(추론 끔). 원문 문체, 성별×연령 10셀 |
| 문항 | test를 뺀 197문항(calibration 148, validation 49) 중 순서형·이진 문항(188개). test는 쓰지 않는다 |

## 질문 (돌리기 전에 쓴다)

`20261123-reasoning-explain`은 DeepSeek Eco(추론 끔)에서 셀 간 분산비 3.38, 셀 편차 정렬 0.479를 얻었다. 이 "집단 차이 과장"이 DeepSeek만의 성질인지, K-EXAONE·Haiku에도 있는지 본다. 보고서(`research/summary_20261123.md` 4절 작업 C)의 가장 큰 주장을 한 모델에서 세 모델로 넓힐 수 있는지가 걸려 있다.

돌리기 전에 아는 것: DeepSeek Eco 끔의 값(위)과, 11/15 test 49문항에서 본 셀 이동 TVD(사람 0.075, K-EXAONE 0.124, DeepSeek 0.209, Haiku 0.246). K-EXAONE·Haiku의 분산비·정렬은 아직 보지 않았다.

## 가설과 판정 기준 (돌리기 전에 쓴다)

방향 예측은 두지 않는다.

**주 지표** (모델마다, 문항 부트스트랩 2,000회, 95% 구간)

| 지표 | 정의 (`20261123-reasoning-high-full/run.py`와 같다) |
|---|---|
| 셀 간 분산비 | Σ문항 V모델 / Σ문항 V사람. V = Σ셀 w(s셀 − s̄)²에서 표집 잡음 기댓값을 뺀 값. w = `cells.csv`의 pop_share |
| 셀 편차 정렬 | 문항·셀을 모은 가중 상관 Σ w·d모델·d사람 / √(Σ w·d모델² · Σ w·d사람²) |

**판정**
- 모델별 과장: 분산비의 95% 구간 하한이 1보다 크면 그 모델은 "과장 있음".
- 세 모델 모두 "과장 있음"이면 보고서에 "집단 차이 과장은 세 모델 공통"으로 쓴다. 일부만이면 해당 모델로 한정한다.
- 정렬은 판정에 쓰지 않고 함께 보고한다(0보다 큰지, 구간).

**탐색** (보정 없음, "탐색"으로 표기)
- 기울기 Σ w·d모델·d사람 / Σ w·d사람².
- 모델 간 짝지은 차이(같은 부트스트랩 문항 표본): 분산비·정렬, 세 쌍.
- 같은 정의로 DeepSeek Ultrafast(k=100)와 DeepSeek Eco 끔(k=50, 기존 결과 3.38/0.479)이 비슷한지 확인한다(엔드포인트·k 차이 점검).

## 방법

`run.py` 하나다. 지표 함수(`stats`, `parts`, `agg`)는 `experiments/20261123-reasoning-high-full/run.py`에서 그대로 옮긴다. 원응답 대신 `counts_k100.npz`의 개수 텐서 C[모델, 페르소나, 문항, 보기]를 쓴다(셀별 표본 수는 개수의 합). 신규 API 호출 없음.

출력: `out/primary.csv`(모델별 주 지표와 판정), `out/explore.csv`(모델 간 차이, 기울기), `out/run.log`, `out/run.json`.

## 결과 (돌린 뒤 쓴다)

## 결론

## 다음
