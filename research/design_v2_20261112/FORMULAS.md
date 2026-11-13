# KOSMOS 파이프라인 수식 정리 (v2)

> **한 줄 요약**: v2 실험(프롬프트 2,952개를 모델마다 **k = 100번** 호출해 보기별 **개수 벡터**를 얻는 방식)의 분석 파이프라인을, 사람 정답지 → 개수 → 평활화·지수 보정 → 거리·순위 지표 → 동점 규칙·신뢰영역 순서로 코드와 1:1로 적었다.
>
> 대상은 2026-11-12에 DeepSeek V4.1 Flash Ultrafast(OpenGateway)와 Claude Haiku 5.5(Anthropic Batch)로 돌린 1단계 실험뿐이다. 11월 초 Haiku 파일럿이나 오픈웨이트 로짓 실험(v1)은 다루지 않는다.

| | |
|---|---|
| 기준 | `DESIGN.md`(확정 설계), `analysis/`의 노트북·스크립트 (`v2_calibration_test.ipynb`, `v2_metrics.ipynb`, `radius_ties.py`, `population_region.py`) |
| 수식 표기 | GitHub 수식 렌더링(`$…$`, `$$…$$`)을 쓴다. 안 보이면 코드블록의 일반 텍스트 식을 보면 된다. |
| 작성 | 2026-11-13 |
| 원칙 | 코드에 있는 식만 적었다. 코드와 다른 곳은 "⚠"로 표시했다. |

## 목차
0. [기호](#0-기호)
1. [실험이 고정한 입력](#1-실험이-고정한-입력)
2. [사람 정답지](#2-사람-정답지)
3. [모델 응답 → 개수 텐서](#3-모델-응답--개수-텐서)
4. [평활화와 지수 보정, 균등 혼합](#4-평활화와-지수-보정-균등-혼합)
5. [분포 거리 지표](#5-분포-거리-지표)
6. [순위 지표](#6-순위-지표)
7. [RADIUS 동점 규칙](#7-radius-동점-규칙)
8. [모분포 95% 신뢰영역](#8-모분포-95-신뢰영역)
9. [분할과 실행 순서](#9-분할과-실행-순서)
10. [호출 수와 비용](#10-호출-수와-비용)
11. [코드 대응표](#11-코드-대응표)

---

## 0. 기호

| 기호 | 뜻 |
|---|---|
| $m$ | 모델 (DeepSeek, Haiku, …) |
| $p$ | 페르소나 (c0, L0, L1의 10셀) |
| $i$ | 문항 ($I = 246$) |
| $j$ | 보기 (문항 $i$의 보기 수 $K_i$, 최대 $K = 7$) |
| $x_{ij}$ | 문항 $i$의 $j$번째 보기에 붙은 KGSS 응답 코드 (연속 정수가 아닐 수 있음) |
| $c_{mpij}$ | 모델이 보기 $j$를 고른 횟수 (유효 응답만) |
| $n_{mpi} = \sum_j c_{mpij}$ | 유효 응답 수 (목표 $k=100$, 거부·형식오류로 줄 수 있음) |
| $H_{pij}$ | 사람(KGSS 2023) 가중 분포 |
| $P_{mpij}$ | 모델 분포 (평활화·보정 후) |
| $s_c$ | L1 셀 $c$의 인구 비중 ($\sum_c s_c = 1$로 다시 정규화) |
| $\mathrm{mask}_{ij}$ | 보기 $j$가 문항 $i$에 실제로 있으면 1 |

---

## 1. 실험이 고정한 입력

실험 전에 `MANIFEST.json`의 SHA-256으로 고정한 것들이다. 이 값이 바뀌면 설계가 바뀐 것이다.

| 항목 | 값 | 파일 |
|---|---|---|
| 문항 | 246개 (KGSS 2023). 척도 유형: 서열형 211 · 이진형 23 · 명목형 11 · 비단조 빈도코드 1 | `item_manifest_v2.csv`, `items_246.csv` |
| 보기 | 문항당 $K_i$개, 최대 7개. 보기 $j$의 응답 코드 $x_{ij}$는 연속 정수가 아닐 수 있다 | `item_manifest_v2.csv`의 `codes` |
| 페르소나 | 12개: c0(없음), L0("한국 성인"), L1 10셀(성별 2 × 연령 5구간) | `cells.csv` |
| 프롬프트 | $246 \times 12 = 2{,}952$개 (1단계) | `prompts_v2.csv` |
| 표본 | 프롬프트마다 $k = 100$번. 표본 번호 $s = 0,\dots,99$ | |
| 문항 분할 | calibration 148 / validation 49 / test 49 (9절) | `item_manifest_v2.csv`의 `split_60_20_20` |

- 250문항에서 선택지에 결측 코드가 섞인 3개(`MEDIAPOL`, `HLPRT1`, `HUMNAT`)와 설문 원문이 빈 1개(`HATOPSEX`)를 빼 246문항이 됐다. 246문항 모두 프롬프트 선택지 코드와 KGSS 2023 실제 응답 코드가 일치하고, 유효 응답 비율 90% 미만 문항은 없다.
- 문항 주제·민감도 분류는 분석용 라벨일 뿐이고 프롬프트에는 들어가지 않는다.

**모델별 호출 설정** (같은 프롬프트, `DESIGN.md` 4절)

| | DeepSeek Ultrafast | Claude Haiku 5.5 |
|---|---|---|
| 추론 | 끔 (`reasoning_effort: "none"`) | 끔 (`thinking: disabled`) |
| temperature | 1.0 | 기본값(전송 안 함) |
| 최대 출력 | 16토큰 | 5토큰 |
| 결과 | 295,200건 | 295,200건 |

---

## 2. 사람 정답지

### 2.1 가중 분포
KGSS 2023(n = 1,230), 조사 가중치 `FINALWT` $w_n$. 문항 $i$의 유효 응답(선택지에 있는 코드, $w_n>0$)만 쓴다.

$$
H_{pij} = \frac{\sum_{n \in G_p,\ y_{ni} = x_{ij}} w_n}{\sum_{n \in G_p} w_n}
$$

$G_p$는 페르소나 $p$에 해당하는 응답자 집합이다.
- c0, L0: 전체 응답자
- L1 셀: 성별 × 연령대 5구간 (18–29, 30–39, 40–49, 50–59, 60+)에 해당하는 응답자

### 2.2 평균 점수와 표준오차
서열형·이진형 문항에서 쓰는 가중 평균 점수와, 유효 표본 크기로 구한 표준오차.

$$
\mu_{pi} = \frac{\sum_n w_n x_n}{\sum_n w_n},\qquad
SE_{pi} = \sqrt{\frac{\sum_n w_n (x_n - \mu_{pi})^2 / \sum_n w_n}{n_{\mathrm{eff},pi}}}
$$

### 2.3 유효 표본 크기 (Kish)
가중치가 고르지 않으면 실제 인원보다 정보가 적다. **페르소나 × 문항마다** 따로 계산한다.

$$
n_{\mathrm{eff}} = \frac{\left(\sum_n w_n\right)^2}{\sum_n w_n^2}
$$

- 모집단: 884~952 (실제 약 1,230)
- L1 셀: 32~250

### 2.4 L1 셀의 인구 비중과 대표 나이
- 인구 비중 $s_c$: 가중 인구에서 셀이 차지하는 비율 (`cells.csv`의 `pop_share`). 합이 1이 되도록 $s_c \leftarrow s_c / \sum s_c$로 다시 맞춘다.
- 대표 나이: 셀 안 가중 중앙값 (`age_wmedian`). 프롬프트의 "당신은 *N*세 …"에 쓴다. 26, 35, 45, 55, 66세.

### 2.5 L1 합성 분포 (`L1agg`)
10개 셀 결과를 모집단 하나로 합친다. 모델·사람 모두 같은 가중치를 쓴다. **모델 응답 자체에는 가중치를 쓰지 않는다**(셀별로 100번씩 똑같이 물었다).

$$
P^{\mathrm{L1agg}}_{mij} = \sum_{c=1}^{10} s_c\,P_{m,c,ij},\qquad
H^{\mathrm{L1agg}}_{ij} = \sum_{c=1}^{10} s_c\,H_{c,ij}
$$

> ⚠ 두 노트북의 정답지가 다르다.
> - `v2_calibration_test.ipynb`: `L1agg`의 정답지 = 위의 $H^{\mathrm{L1agg}}$ (사람 셀 분포의 가중합)
> - `v2_metrics.ipynb`: `L1agg`를 **모집단 분포 $H_{0}$** (c0와 같은 분포)와 비교
>
> 두 정답지는 완전히 같지 않다. 표를 읽을 때 어느 노트북 결과인지 확인할 것.

---

## 3. 모델 응답 → 개수 텐서

### 3.1 프롬프트와 표본
- 같은 프롬프트(`prompts_v2.csv`, 단계 1은 2,952개)를 모델마다 $k = 100$번씩 물어본다. 표본 번호 $s = 0,\dots,99$.
- 응답 한 줄 = `(probe_id, sample_idx)` 하나. 응답을 보기 번호로 파싱해 `valid`를 표시한다.

### 3.2 개수
$$
c_{mpij} = \sum_{s=0}^{99} \mathbf 1\left[\text{valid}_{mpis},\ \text{parsed}_{mpis} = x_{ij}\right]
$$

- `api_error` 줄은 버린다.
- 같은 `(probe_id, sample_idx)`가 겹치면 **먼저 나온 것**만 쓴다.
- 파싱 실패·거부는 유효 응답에서 빠진다. 그래서 $n_{mpi} < 100$일 수 있고, 평활화의 영향이 커진다.

**실제 유효 응답 수** (`raw/v2`, 프롬프트 2,952개 × 100번 = 295,200건)

| | 유효 | 무효 (`invalid_type`별) | 유효율 |
|---|---|---|---|
| DeepSeek Ultrafast | 295,158 | `length` 28 · `explanation` 8 · `out_of_range` 4 · `refusal` 2 | 99.99% |
| Haiku 5.5 | 290,643 | `explanation` 2,840 · `refusal` 1,717 | 98.5% |

- DeepSeek 파일은 295,210줄이다. 초과 10줄은 `api_error`이고, 위 규칙대로 버린다(같은 `(probe_id, sample_idx)`는 첫 줄만 센다).
- **파싱 규칙** (`runners/run_deepseek_eco.py`의 `parse_response`, DeepSeek 기준): 응답 전체가 숫자 하나이면, 그 숫자가 문항 보기 코드에 있을 때 유효이고 없으면 `out_of_range`다. 빈 응답이나 거부 표현("죄송", "저는 AI" 등)이 있으면 `refusal`, 그 밖의 문장은 `explanation`이다. 출력 상한(`finish_reason = length`)에 걸려 유효하지 않으면 `length`다. Haiku의 분류 코드는 확인하지 않았다.

### 3.3 원비율
$$
\hat r_{mpij} = \frac{c_{mpij}}{\max(n_{mpi},\,1)}
\quad\left(n_{mpi}=0\text{이면 균등 } 1/K_i\right)
$$

최빈값·순위·엔트로피 지표는 모두 이 **원비율**로 계산한다(지수 보정은 순서를 바꾸지 않으므로, 4.3절 참고).

---

## 4. 평활화와 지수 보정, 균등 혼합

### 4.1 평활화 + 지수 (주 분석)
개수에 $\alpha$를 더한 뒤 지수 $\beta$로 올리고 다시 정규화한다.

$$
P_{mpij}(\beta) = \frac{(c_{mpij} + \alpha)^{\beta}\,\mathrm{mask}_{ij}}{\sum_{j'} (c_{mpij'} + \alpha)^{\beta}\,\mathrm{mask}_{ij'}}
$$

| 설정 | $\alpha$ | 쓰임 |
|---|---|---|
| 라플라스 | 1 | **주 분석** |
| Jeffreys | 0.5 | 민감도 분석 |

- $\beta = 1$이면 평활화만 한 "보정 전" 분포다.
- $\beta = 1/T$이므로 **온도** $T$를 올리는 것($T>1$, $\beta<1$)은 분포를 평평하게 만든다. 모드 붕괴(한 보기로 쏠림)를 풀어 준다.

### 4.2 $\beta^*$ 고르기 (calibration 148문항)
그리드 $\beta \in \{0.05, 0.055, \dots, 1.5\}$ (간격 0.005)에서 calibration 문항의 평균 TVD가 최소인 값을 고른다.

$$
\beta^*_{m,\ell} = \arg\min_{\beta}\ \frac{1}{|\mathcal I_{cal}|}\sum_{i \in \mathcal I_{cal}} \mathrm{TVD}\left(P^{(\ell)}_{mi}(\beta),\ H^{(\ell)}_{i}\right)
$$

- $\ell \in \{\text{c0},\ \text{L0},\ \text{L1agg},\ \text{L1cell}\}$ 수준마다, 모델마다 따로 고른다.
- L1cell에서는 10개 셀이 $\beta$ 하나를 공유한다(셀 평균 TVD를 최소화).
- 고른 값은 **개수 텐서의 SHA-256**과 함께 `analysis/out_calib/frozen_beta.json`에 고정한다. 그 뒤 test 49문항을 한 번만 평가한다. 개수 텐서가 바뀌면 고정값 검사에서 멈춘다.

### 4.3 지수 보정은 순서를 바꾸지 않는다
$\beta > 0$이면 $(c+\alpha)^\beta$는 $c$에 대해 단조증가이므로, **한 분포 안의 보기 순서는 그대로**다.
- 그래서 최빈값과 한 분포 안의 보기 순위는 $\beta$와 무관하다. 이 지표들은 평활화 없는 원비율 $\hat r$로 계산한다(6, 7절).
- 집단 간 쌍별 순위(6.2절)는 셀마다 평균 점수를 비교한다. 이것도 설계상 원비율 $\hat r$로 계산하지만, 평균 점수는 분포 전체의 모양에 따라 달라지므로 $\beta$에 대해 불변이라고 증명한 것은 아니다.
- TVD, JSD, $W_1$, 엔트로피는 $\beta$에 따라 변한다.

### 4.4 균등분포 혼합 기준선
"모델 순위 정보가 아니라 그냥 평평하게 만드는 것"만큼 좋은지 비교하는 기준선.

$$
Q_{mpij}(\lambda) = (1 - \lambda)\,P_{mpij}(\beta{=}1) + \lambda\,\frac{\mathrm{mask}_{ij}}{K_i}
$$

$\lambda \in \{0, 0.01, \dots, 1\}$를 같은 방식으로 calibration에서 고르고(같은 파일에 고정), test에서 $\beta^*$ 보정과 **문항별 짝지은 차이**로 비교한다.

### 4.5 균등분포 기준선 (정보 없는 찍기)
$$
U_{ij} = \frac{\mathrm{mask}_{ij}}{K_i}
$$

모델이 이 값보다 사람에 가깝지 않다면 "찍는 것만도 못한" 것이다.

---

## 5. 분포 거리 지표

두 분포 $a, b$ (같은 문항, 보기 수 $K$)에 대해.

### 5.1 TVD (총변동거리) — **주 지표**
$$
\mathrm{TVD}(a,b) = \tfrac12\sum_{j=1}^{K}|a_j - b_j| \in [0,1]
$$

### 5.2 JSD 거리
$m = (a+b)/2$, 로그는 밑 2.

$$
\mathrm{JSD}(a,b) = \sqrt{\ \tfrac12\mathrm{KL}(a\,\|\,m) + \tfrac12\mathrm{KL}(b\,\|\,m)\ }\in[0,1],
\qquad \mathrm{KL}(x\|y) = \sum_j x_j \log_2\frac{x_j}{y_j}
$$

($x_j = 0$인 항은 0으로 둔다)

### 5.3 정규화 Wasserstein-1 (서열형·이진형만)
보기 사이의 **실제 코드 간격**을 반영하고, 문항 척도 범위로 나눠 0~1로 맞춘다.

$$
W_1(a,b) = \frac{\displaystyle\sum_{j=1}^{K-1}\bigl|F_a(j) - F_b(j)\bigr|\,\bigl(x_{j+1} - x_j\bigr)}{x_{\max} - x_{\min}},
\qquad F_a(j) = \sum_{j'\le j} a_{j'}
$$

- 명목형 문항에는 정의하지 않는다(NaN).

### 5.4 최빈값 일치 / 쏠림 지표
$$
\mathrm{mode\_match} = \mathbf 1\Bigl[\arg\max_j P_{j} = \arg\max_j H_{j}\Bigr]
$$

$$
H(p) = -\sum_j p_j \log_2 p_j,\qquad
\text{엔트로피 비} = \frac{H(\hat r)}{H(H_{\text{사람}})},\qquad
\text{top\_share} = \max_j \hat r_j,\qquad
\text{collapse}_{\ge 0.95} = \mathbf 1[\text{top\_share} \ge 0.95]
$$

- 엔트로피 비가 1보다 훨씬 작으면 모델이 사람보다 한 보기에 쏠려 답한다(모드 붕괴).
- 엔트로피·top_share는 평활화 없는 원비율 $\hat r$로 센다.

### 5.5 수준별 집계

| 수준 | 모델 쪽 | 사람 쪽 | 문항 점수 |
|---|---|---|---|
| c0 | $P_{0}$ | $H_{0}$ | $\mathrm{TVD}(P_0, H_0)$ |
| L0 | $P_{1}$ | $H_{1}$ (= $H_0$) | $\mathrm{TVD}(P_1, H_1)$ |
| L1agg | $\sum_c s_c P_c$ | 2.5절 참고 | 합성 분포끼리의 TVD |
| L1cell | 셀별 $P_c$ | 셀별 $H_c$ | $\frac{1}{10}\sum_c \mathrm{TVD}(P_c, H_c)$ |

> 주의: L1cell은 셀마다 TVD를 구한 뒤 **단순 평균**(10셀 동일 가중)이고, L1agg는 **분포를 먼저 합친 뒤** TVD를 구한다. 둘은 다른 양이다.

최종 보고값은 해당 분할 문항 $\mathcal I$에 대한 평균이다.

$$
\overline{\mathrm{TVD}}_{m,\ell} = \frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\mathrm{TVD}_{m,\ell,i}
$$

### 5.6 불확실성: 문항 부트스트랩
같은 문항에서 나온 값은 함께 움직이므로 **문항 단위**로 재표집한다 ($B = 5000$).

$$
\text{95\% 구간} = \bigl[\,q_{2.5\%},\ q_{97.5\%}\,\bigr]\ \text{of}\ \left\{\tfrac1{|\mathcal I|}\sum_{i\in\mathcal I^{*b}} x_i\right\}_{b=1}^{B}
$$

### 5.7 모델 차이·균등분포 대비
같은 문항에서 짝지어 뺀 값의 평균과 구간을 본다.

$$
d_i = \mathrm{TVD}_{\text{Haiku},i} - \mathrm{TVD}_{\text{DeepSeek},i}
\quad(\text{양수면 DeepSeek가 사람에 더 가깝다})
$$

$$
\mathrm{beat\_unif} = \frac{1}{|\mathcal I|}\sum_i \mathbf 1\left[\mathrm{TVD}(P_{mi}, H_i) < \mathrm{TVD}(U_i, H_i)\right]
$$

---

## 6. 순위 지표

분포 모양이 아니라 **순서**를 사람처럼 재현하는지 본다. 서열형·이진형 문항만 쓴다(명목형은 제외).

### 6.1 평균 점수
$$
\mu^{\mathrm{model}}_{pi} = \sum_j \hat r_{pij}\,x_{ij},\qquad \mu^{\mathrm{human}}_{pi}\ \text{은 2.2절}
$$

### 6.2 집단 간 쌍별 순위 일치율
L1의 10셀에서 두 셀 $(a,b)$를 짝지은 45쌍마다, 문항 $i$에서 "어느 셀의 평균이 더 높은가"가 사람과 같은지 센다.

$$
\Delta^{\mathrm{h}}_{abi} = \mu^{\mathrm{human}}_{ai} - \mu^{\mathrm{human}}_{bi},\qquad
\Delta^{\mathrm{m}}_{abi} = \mu^{\mathrm{model}}_{ai} - \mu^{\mathrm{model}}_{bi}
$$

$$
\mathrm{sc}_{abi} =
\begin{cases}
1 & \operatorname{sign}\Delta^{\mathrm m}_{abi} = \operatorname{sign}\Delta^{\mathrm h}_{abi}\\
0.5 & \Delta^{\mathrm m}_{abi} = 0\ \ (\text{모델 동점})\\
0 & \text{그 외}
\end{cases}
$$

$$
\mathrm{AUC}_{\text{pair}} = \frac{\sum_{(a<b),\,i}\ \mathrm{valid}_{abi}\ \mathrm{sc}_{abi}}{\sum_{(a<b),\,i}\ \mathrm{valid}_{abi}},
\qquad \mathrm{valid}_{abi} = \mathbf 1[i\in\text{서열·이진}]\ \mathbf 1[\Delta^{\mathrm h}_{abi}\neq 0]
$$

- 우연 기준은 0.5.
- 구간은 **문항 단위** 부트스트랩이다 (같은 문항의 셀 쌍은 함께 뽑는다).
- 쌍을 성별만 다름 / 연령만 다름 / 둘 다 다름으로 나눠서도 계산한다.

### 6.3 "사람 쪽 차이가 표집 오차 밖인 쌍"만 쓰기 (두 가지 방식)
사람 쪽 차이가 표집 잡음보다 작은 쌍은 정답이 불확실하므로 빼고 다시 계산한다. 노트북과 스크립트가 다른 규칙을 쓴다.

| 파일 | 규칙 |
|---|---|
| `v2_metrics.ipynb` | $\bigl\lvert\Delta^{\mathrm h}_{abi}\bigr\rvert > 2\sqrt{SE_{ai}^2 + SE_{bi}^2}$ |
| `analysis/radius_ties.py` (**최종 규칙**, 7절) | 사람 부트스트랩 차이의 95% 구간이 0을 포함하지 않음 |

### 6.4 셀 간 퍼짐
문항별로 셀 축에서 평균 점수를 센터링해 비교한다.

$$
r_i = \frac{\sum_c (\mu^{\mathrm h}_{ci} - \bar\mu^{\mathrm h}_i)(\mu^{\mathrm m}_{ci} - \bar\mu^{\mathrm m}_i)}{\sqrt{\sum_c (\mu^{\mathrm h}_{ci} - \bar\mu^{\mathrm h}_i)^2\ \sum_c (\mu^{\mathrm m}_{ci} - \bar\mu^{\mathrm m}_i)^2}},
\qquad
\text{퍼짐 비}_i = \frac{\operatorname{sd}_c(\mu^{\mathrm m}_{ci})}{\operatorname{sd}_c(\mu^{\mathrm h}_{ci})}
$$

모델 쪽 점수가 모든 셀에서 같으면 표준편차가 0이라 $r_i$는 계산하지 않는다(NaN).

### 6.5 집단 안 보기 순위 일치율
한 셀 안에서 보기 쌍 $(j, j')$의 순서가 사람과 같은지.

$$
\mathrm{within}_{ci} = \frac{\sum_{j<j'}\mathrm{valid}\cdot a_{jj'}}{\sum_{j<j'}\mathrm{valid}},\quad
a_{jj'} = \begin{cases}1 & \operatorname{sign}(\hat r_j - \hat r_{j'}) = \operatorname{sign}(H_j - H_{j'})\\ 0.5 & \hat r_j = \hat r_{j'}\\ 0&\text{else}\end{cases}
$$

($\mathrm{valid}$: 사람 쪽 $H_j \ne H_{j'}$인 쌍)

**기준선**: 모집단 보기 순서 $\operatorname{sign}(H^{\mathrm{pop}}_j - H^{\mathrm{pop}}_{j'})$를 모든 셀에 그대로 쓴 값. 모델이 이 기준선을 넘어야 "셀에 맞는 순서를 안다"고 말할 수 있다.

---

## 7. RADIUS 동점 규칙

사람 분포도 표본이라 "1위가 어느 보기냐"가 흔들린다. 이를 고려한 최빈값 정확도와 쌍별 순위 규칙이다 (Łajewska et al., 2026, RADIUS의 TRM을 따름). **test 결과를 연 뒤에 정한 사후 규칙**이다.

### 7.1 사람 분포 부트스트랩
페르소나 $p$, 문항 $i$마다 유효 표본 $n_{\mathrm{eff},pi}$로 다항 재추출 ($B = 1000$).

$$
\tilde H^{(b)}_{pi\cdot} = \frac{1}{n_{\mathrm{eff},pi}}\ \mathrm{Multinomial}\bigl(n_{\mathrm{eff},pi},\ H_{pi\cdot}\bigr),\quad b=1,\dots,B
$$

보기별로 2.5%, 97.5% 분위수를 구한다: $\mathrm{lo}_{pij}$, $\mathrm{hi}_{pij}$. (실제 응답자 재추출 대신 $n_{\mathrm{eff}}$로 근사한다.)

### 7.2 사람 1위 그룹
사람 1위 보기 $j^\star = \arg\max_j H_{pij}$의 구간과 겹치는 보기를 모두 "1위 그룹" $G_{pi}$에 넣는다.

$$
G_{pi} = \bigl\{\, j \ :\ \mathrm{hi}_{pij} \ge \mathrm{lo}_{pij^\star}\,\bigr\}
$$

### 7.3 최빈값 정확도 (TRM)
모델의 1위 보기 집합 $T_{pi} = \arg\max_j \hat r_{pij}$ (동점 포함).

$$
\mathrm{TRM}_{pi} = \frac{\left|T_{pi}\cap G_{pi}\right|}{\left|T_{pi}\right|}
$$

모델 1위가 그룹 안에 있으면 1, 동점이면 동점 보기 중 그룹 안에 든 비율이다.

**기준선**

| 기준선 | 값 |
|---|---|
| 우연 (엄격) | $1/K_i$ |
| 우연 (RADIUS) | $\lvert G_{pi}\rvert / K_i$ |
| 항상 1번 | $\mathbf 1[\,1\in G_{pi}]$ |
| 모집단 사람 분포를 모든 페르소나에 | 위 TRM에서 $\hat r \leftarrow H^{\mathrm{pop}}$ |
| 사람 재표집 (상한) | 부트스트랩 분포 $\tilde H^{(b)}$로 같은 TRM |

### 7.4 페르소나 간 쌍
6.2절의 쌍별 순위 일치율에서, 사람 쪽 평균 점수 차이의 **부트스트랩 95% 구간이 0을 포함하는 쌍은 뺀 값**을 주 결과로 쓴다. 전체 쌍 값을 함께 보고한다.

$$
\mathrm{SIG}_{abi} = \mathbf 1\left[\ q_{2.5\%}\!\left(\tilde\Delta^{\mathrm h}_{abi}\right) > 0\ \ \text{or}\ \ q_{97.5\%}\!\left(\tilde\Delta^{\mathrm h}_{abi}\right) < 0\ \right]
$$

$\tilde\Delta^{\mathrm h}$는 7.1절의 부트스트랩($B=1000$)으로 구한 두 셀의 평균 점수 차이다. 일치율 $\mathrm{AUC}_{\text{pair}}$ 자체의 95% 구간은 별도로 문항 단위 부트스트랩($B=2000$)으로 구한다.

---

## 8. 모분포 95% 신뢰영역

(`analysis/POPULATION_ESTIMATION.md`에 예시와 해석이 더 있다.)

KGSS 분포 $\hat Q$도 표본이라, 진짜 모분포 $Q$는 $\hat Q$ 근처 영역 안에 있다. 그 영역을 세우고 모델이 영역에서 얼마나 떨어졌는지 잰다.

### 8.1 Goodman 동시 구간
보기 $K$개를 동시에 95%로 묶기 위해 본페로니식으로 넓힌다 (Goodman, 1965).

$$
z = \Phi^{-1}\!\left(1 - \frac{0.05}{2K}\right),\qquad A = z^2
$$

$$
\mathrm{lo}_j,\ \mathrm{hi}_j = \frac{A + 2n\hat p_j\ \mp\ \sqrt{A\bigl(A + 4n\hat p_j(1-\hat p_j)\bigr)}}{2(n + A)},\qquad n = n_{\mathrm{eff}}
$$

| $K$ | 2 | 3 | 4 | 5 | 7 |
|---|---|---|---|---|---|
| $z$ | 2.241 | 2.394 | 2.498 | 2.576 | 2.690 |

**신뢰영역** = $\{\,q : \mathrm{lo}_j \le q_j \le \mathrm{hi}_j\ \forall j,\ \sum_j q_j = 1\,\}$

### 8.2 최소 TVD (영역까지의 거리)
모델 분포 $P$에서 영역 안 가장 가까운 후보까지의 TVD. "모분포가 영역 안 어디에 있든 모델은 최소 이만큼 떨어져 있다".

$$
\mathrm{minTVD}(P) = \max\Bigl(\ \underbrace{\sum_j \max(\mathrm{lo}_j - P_j,\ 0)}_{\text{모자란 양}},\ \ \underbrace{\sum_j \max(P_j - \mathrm{hi}_j,\ 0)}_{\text{넘치는 양}}\ \Bigr)
$$

### 8.3 포함률
$$
\text{포함률} = \Pr\bigl[\mathrm{minTVD}(P) = 0\bigr]
\quad\text{(페르소나 × 문항 평균)}
$$

완벽한 모델($P = Q$)이면 약 95%여야 한다.

### 8.4 잡음 바닥과 편향 보정 TVD
**잡음 바닥**: 모델이 완벽($P=Q$)해도 표본끼리 비교하면 TVD가 0이 아니다. 사람 응답을 $n_{\mathrm{eff}}$명씩 다시 뽑아 (부트스트랩 300번) 원래 $\hat Q$와의 TVD를 평균한 값.

**편향 보정 TVD**

$$
\mathrm{TVD}_{\text{보정}} = 2\,\mathrm{TVD}(\hat P, \hat Q) - \mathbb E\left[\mathrm{TVD}(\hat P^*, \hat Q^*)\right]
$$

- $\hat P^*$: 모델 응답 100개를 다시 뽑아 **같은 $\beta$로** 보정한 분포
- $\hat Q^*$: 사람 응답을 $n_{\mathrm{eff}}$명 다시 뽑은 분포
- 사람·모델 양쪽의 표집 잡음이 부풀린 만큼을 뺀다. 진짜 거리가 0에 가까우면 음수가 나올 수 있다.

**어림 관계** (직각으로 붙는 두 변처럼 더해진다고 볼 때)

$$
\mathrm{TVD}_{\text{관측}}^2 \approx \mathrm{TVD}_{\text{진짜}}^2 + \text{잡음}^2
$$

---

## 9. 분할과 실행 순서

### 9.1 문항 분할 (이전 재실험과 동일)

| 분할 | 문항 | 용도 |
|---|---|---|
| calibration | 148 | $\beta$, $\lambda$ 그리드 서치 |
| validation | 49 | 그리드 결과 확인, 혼합 기준선 비교 |
| test | 49 | 최종 보고 (설계를 고정한 뒤 한 번만) |

- 층화 기준: 척도 유형 × 보기 수.
- 같은 배터리 문항이 여러 분할에 흩어져 있어, 변수명 접두어로 묶은 5-fold(`group_cv_fold`) 결과도 함께 낸다.

### 9.2 순서 (되돌릴 수 없는 순서)
1. 모든 응답 수집 → 개수 텐서 $c$ 고정
2. calibration 문항에서 $\beta^*$, $\lambda^*$ 선택 → `frozen_beta.json`에 개수 텐서 해시와 함께 기록
3. 환경변수 `KOSMOS_RUN_TEST=1`일 때만 test 49문항을 한 번 평가
4. RADIUS 동점 규칙, 신뢰영역(8절)은 test를 연 뒤에 추가한 **사후 분석**임을 논문에 적는다

---

## 10. 호출 수와 비용

### 10.1 호출 수
$$
\text{프롬프트} = \underbrace{246}_{\mathrm{c0}} + \underbrace{246}_{\mathrm{L0}} + \underbrace{246\times 10}_{\mathrm{L1}} = 2{,}952,\qquad
\text{호출} = 2{,}952 \times k = 295{,}200\ (k=100)
$$

### 10.2 Haiku 배치 예산 상한 (`run_haiku_batch.py`)
제출한 배치의 누적 상한이 `--budget`(기본 $4.70)을 넘으면 제출하지 않는다.

$$
\text{상한}_{\text{배치}} = N_{\mathrm{in}}\times \frac{\$0.05}{10^6} + N_{\mathrm{req}}\cdot 5\times\frac{\$0.25}{10^6}
$$

($N_{\mathrm{in}}$: 입력 토큰 수, 시험 배치 실측값을 쓰고 그 전에는 글자 수 × 1.3. 5토큰 = 출력 상한. 배치 50% 할인가.)

### 10.3 K-EXAONE 팀 분담
표본 번호를 4등분한다. 담당 $r \in \{1,2,3,4\}$는 $s \in [25(r-1),\ 25r - 1]$를 맡고, 호출 수는 $2{,}952 \times 25 = 73{,}800$건이다.

---

## 11. 코드 대응표

| 절 | 식 | 파일·함수 |
|---|---|---|
| 2.1–2.3 | 가중 분포, $SE$, $n_{\mathrm{eff}}$ | `analysis/v2_calibration_test.ipynb` 정답지 셀, `analysis/radius_ties.py` |
| 3.2 | 개수 텐서 | 노트북의 `load_counts`, `analysis/radius_ties.py` `load_counts` |
| 4.1–4.2 | 평활화·지수, $\beta^*$ | 노트북 `to_prob`, 그리드 서치 셀, `frozen_beta.json` |
| 4.4 | 균등 혼합 | 노트북 `best_lam` 셀 |
| 5.1–5.3 | TVD, JSD, $W_1$ | 노트북 `tvd`, `jsd`, `w1_masked`/`w1_full` |
| 5.4 | 엔트로피 비, top_share | `analysis/v2_metrics.ipynb` (`ENT_RATIO`, `TOP`) |
| 5.6–5.7 | 부트스트랩, 모델 차이 | 노트북 `boot_ci`, 모델 차이 셀 |
| 6.2–6.5 | 쌍별 순위, 퍼짐, 집단 안 순위 | `analysis/v2_metrics.ipynb` 쌍별 순위 셀 |
| 7 | RADIUS | `analysis/radius_ties.py` |
| 8 | 신뢰영역 | `analysis/population_region.py`, `analysis/POPULATION_ESTIMATION.md` |
| 10.2 | Haiku 예산 상한 | `runners/run_haiku_batch.py` |
