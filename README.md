# 한국형 실리콘 샘플링 검증

LLM에 인구통계 페르소나를 프롬프트로 부여해 한국 여론조사 문항(KGSS)에
응답시키고, 가상 응답이 실제 응답을 얼마나 재현하는지 정량 분석한다.

> ## ✅ 현재 기준은 **v2** (2026-11-12 확정)
> 실험 설계·조건·상태는 [`research/design_v2_20261112/DESIGN.md`](research/design_v2_20261112/DESIGN.md)가 기준이다.
> 아래 "v2 실험 환경"만 읽어도 실행할 수 있다. 맨 아래 **v1**(초기 구상, Colab GPU 기반)은 접어 두었다.

## 처음 읽는 사람용 요약

**한 줄 요약**: LLM 3종에게 KGSS 2023의 246문항을 페르소나 없이, "한국 성인"으로,
성별×연령 10개 집단으로 각각 100번씩 물어서 실제 한국인 응답 분포와 얼마나
맞는지 잰다.

**읽는 순서**
1. `research/design_v2_20261112/DESIGN.md` — 확정 설계 (무엇을 묻는가, 조건, 상태)
2. [`research/design_v2_20261112/FORMULAS.md`](research/design_v2_20261112/FORMULAS.md) — **파이프라인 수식 정리** (정답지, 개수, 평활화·지수 보정, TVD·순위·RADIUS·신뢰영역)
3. `research/preregistration_haiku_pilot.md`, `research/preregistration_verbalized_arm.md` — 사전등록
4. `gates.md`, `gates_exaone.md`, `gates_exploratory.md` — 단계별 통과 기준
5. `research/design_v2_20261112/analysis/` — 분석 코드와 결과표
6. `audit_outputs/` — 검증·감사 산출물

**데이터와 키**
- KGSS 원본(`.sav`, 코드북)은 성균관대 SRC에서 별도 신청 없이 받아 루트에 둔다(아래 v1의 "데이터" 절 참고).
  현재 파일 목록에는 없지만 과거 커밋 히스토리에는 들어 있다.
- **v2 응답 원본은 `results/raw_gz/`에 gzip으로 올려 두었다**(`gunzip -k <파일>.gz`). 해당 폴더의 구성은 `results/README.md` 참고.
- 10월 재실험 원본(`reexperiment_20261009/`, 약 710MB)과 API 키(`.env*`)는 올리지 않았다.
- `requirements.txt`는 UTF-16LE라 `uv pip install -r`가 바로 안 된다. UTF-8로 변환한 사본을 쓴다.

## v2 실험 환경 (API 호출)

초기(v1)에는 오픈웨이트 모델을 Colab GPU에서 돌렸지만, v2는 **외부 API로 응답을
받아 모은다.** 같은 프롬프트(`prompts_v2.csv`, 2,952개)를 모델마다 프롬프트당 100회씩
물어 응답 원본(jsonl)을 쌓고, 보기 개수 벡터로 집계해 분석한다.

| 모델 | 경로 | 실행기 | 키(환경변수) | 핵심 설정 |
|---|---|---|---|---|
| DeepSeek V4.1 Flash Ultrafast | OpenGateway | `runners/run_deepseek_eco.py` | `OPENGATEWAY_API_KEY` (`.env.opengateway`) | `reasoning_effort: "none"`, temperature 1.0, 최대 16토큰, 동시 60건 |
| K-EXAONE 2.5 | LG 공식 무료 API | `team_template/run_exaone_team.py` | `EXAONE_API_KEY` (폴더의 `.env`) | 추론 `low`(끌 수 없음), temperature 1.0, 최대 4,000토큰 |
| Claude Haiku 5.5 | Anthropic Message Batches | `runners/run_haiku_batch.py` | `ANTHROPIC_API_KEY` (`.env.anthropic`) | thinking 끔, 최대 5토큰, temperature 미전송 |

실행기는 모두 `research/design_v2_20261112/`에 있고, KOSMOS 루트에서 실행한다.
**API 키 파일은 저장소에 없다.** 각자 발급받아 루트에 `.env.*`로 만들면 된다(`.gitignore` 처리됨).

```bash
# DeepSeek (OpenGateway)
set -a; . ./.env.opengateway; set +a
python3 research/design_v2_20261112/runners/run_deepseek_eco.py --test 5   # 시험
python3 research/design_v2_20261112/runners/run_deepseek_eco.py            # 1단계 본실행

# Haiku (Batch) — ANTHROPIC_API_KEY 필요, 예산 가드 기본 $4.70
R="uv run --no-project --with certifi python research/design_v2_20261112/runners/run_haiku_batch.py"
$R submit --samples 0:1 && $R wait          # 시험 배치
$R submit --samples 1:100 --chunk 10 && $R wait

# 응답 원본 → 보기 개수 벡터
python3 research/design_v2_20261112/runners/aggregate_counts.py raw/v2/deepseek_ultrafast_p1.jsonl
```

- **이어 돌리기**: 출력 파일에 이미 있는 `(probe_id, sample_idx)`는 건너뛴다. 끊기면 같은 명령을 다시 실행한다.
- **K-EXAONE는 팀원 4명이 표본 번호로 분담**한다(0–24 / 25–49 / 50–74 / 75–99). 팀원은 패키지 폴더를
  AI 에이전트에 열고 `AGENT_TASK.md를 읽고 그대로 진행해줘`라고만 하면 된다. 실행기는 프롬프트 파일
  해시를 검사해 바뀌면 거부하고, 끝나면 `finalize.py`가 결과 zip 하나를 만든다. 패키지는 `make_team_packages.py`로
  `team_template/`에서 만들고, 팀원 결과는 `merge_check.py`로 합친다.
- **프롬프트·문항·셀은 `MANIFEST.json` 해시로 고정**돼 있다. 바뀌면 설계가 바뀐 것이다.
- DeepSeek는 처음 Eco 등급으로 시작했다가 속도 때문에 Ultrafast로 바꿔 처음부터 다시 돌렸다
  (경위와 비교 결과는 `DESIGN.md`의 "Eco → Ultrafast 전환").
- 응답 원본 위치: 로컬은 `raw/v2/`·`raw/exaone_stage1.jsonl`, 저장소에는 `results/raw_gz/`(압축본).

---

<details>
<summary><b>v1 (초기 구상, 2026-10 이전) — 펼쳐서 보기</b></summary>

> 아래는 오픈웨이트 모델(EXAONE 2.4B, Kanana 2.1B)을 Colab GPU에서 돌리던 **v1**의 설계·실행 순서·결과다.
> 현재 진행 중인 실험은 위의 **v2**다. 재현·참고용으로만 남겨 둔다.

## 연구 설계

관측되는 건 "LLM 응답 분포가 실제와 다르다"는 사실 하나지만 원인은 여러
개일 수 있다. 아래 네 성분으로 분해해서 각각 측정·개입한다.

| 성분 | 측정 | 개입 |
|---|---|---|
| ① 정보 부족 | 셀 최빈응답 상한 대비 개인예측 정확도 | 응답 이력 조건 추가 |
| ② 척도 표현 | 출력 엔트로피, 셀 내 분산 재현율 | SSR, 디코딩 온도 |
| ③ 내용 편향 | 민감 문항 방향성 편차 | **인칭 × 화계 변형 (헤드라인)** |
| ④ 형식 민감도 | TV거리, 인간 기준선 대비 배율 | 정순/역순 교차 |

**대조군 4수준**: 페르소나 없음(C0) / "한국 성인" / 성별+연령 / 6변수
(SEX·AGE·EDUC·MARITAL·REGION·URBAN). 같은 응답자 표본을 4수준 전부에
재사용해 조건 간 대응비교가 가능하게 한다.

③(인칭×화계 조작)이 이 프로젝트의 고유 기여다 — 한국어 화계 체계는
영어권 선행연구에서 시도할 수 없는 통제된 조작을 허용한다.

## 데이터

원본 파일은 저장소에 포함되지 않는다 — 라이선스 제약이 아니라 **용량과
재생성 가능성** 때문이다. 성균관대학교 서베이리서치센터(SRC)에서 별도
신청 절차 없이 받을 수 있다. 다운로드 후 프로젝트 루트에 둘 것.

- `2003-2025_KGSS_kor_public_v2.sav`
- `2003-2025_KGSS_Codebook_v5.pdf`

## 환경 구성

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

오픈웨이트 모델(EXAONE 3.5, Kanana 1.5)을 다루는 스크립트는 GPU가 필요해
로컬이 아니라 Colab/Kaggle에서 실행한다 — `kgss_contam_colab.ipynb` 참고.

## 실행 순서

```powershell
# 1. 데이터 구조 진단 (결측 분해, eta2, 셀 크기, 문항 후보 선정)
uv run kgss_inventory.py --sav "2003-2025_KGSS_kor_public_v2.sav" `
    --out .\inventory --target-year 2023

# 2. 코드북에서 설문 원문 추출 + A/B 분할표본 순서효과 분석
uv run kgss_codebook.py --pdf "2003-2025_KGSS_Codebook_v5.pdf" `
    --inv .\inventory --out .\selection

# 3. (선택) 사람이 읽는 코드북/회차 데이터 변환
uv run kgss_export.py --sav "2003-2025_KGSS_kor_public_v2.sav" --year 2023

# 4. 문항 선정 작업지 생성 (3인 독립 태깅용 xlsx)
uv run kgss_shortlist.py --inv .\inventory --out .\selection --year 2023
#   -> 각자 자기 시트에 민감여부 태깅 후 kgss_agreement.py로 합치기
uv run kgss_agreement.py --out .\selection --files `
    주희=selection\문항선정_작업지_2023_정.csv `
    태우=selection\문항선정_작업지_2023_태우라벨링.xlsx `
    다교=selection\문항선정_작업지_2023_HUR.xlsx

# 5. 선택지 토큰화 점검 (로컬은 토크나이저만, 로짓은 Colab/Kaggle에서)
uv run kgss_token_check.py --tokenizer-only --worded .\selection\variables_worded.csv

# 6. 사전학습 오염 프로브: 생성(로컬) -> 로짓/생성 실행(Colab) -> 채점(로컬)
uv run kgss_contamination.py build --inv .\inventory `
    --worded .\selection\variables_worded.csv --out .\probe
#   Colab에서 kgss_contam_colab.ipynb 실행 -> probe/logit_responses.csv, responses.csv 다운로드 후 여기 probe/ 에 넣기
uv run kgss_contamination.py score --out .\probe

# 7. 6변수 페르소나 표본 생성 (FINALWT 가중 복원추출, 대조군 "6변수" 조건용)
uv run kgss_persona_bootstrap.py --inv .\inventory --out .\persona_sample --n 200 --seed 42

# 8. C0/adult/demo/full 조건 분포 재현 타당성 검증 (운영진 Q1/Q2 대응)
uv run kgss_c0_validity.py build --level c0 --inv .\inventory --out .\c0_check
uv run kgss_c0_validity.py build --level adult --inv .\inventory --out .\c0_check_adult
uv run kgss_c0_validity.py build --level demo --inv .\inventory --out .\c0_check_demo
uv run kgss_c0_validity.py build --level full --inv .\inventory --out .\c0_check_full --n-personas 40
#   Colab에서 kgss_token_check.py --contam-probes probes.csv --out <out디렉토리> --no-contam-generate 실행
#   -> logit_responses.csv 받아서 각 out디렉토리에 넣기
uv run kgss_c0_validity.py score --out .\c0_check
uv run kgss_c0_validity.py score --out .\c0_check_adult
uv run kgss_c0_validity.py score --out .\c0_check_demo
uv run kgss_c0_validity.py score --out .\c0_check_full

# 9. 본실험 프롬프트 생성 (최종 문항 확정 후) + Colab에서 로짓 추출
uv run kgss_experiment_prompt.py --items .\selection\최종문항.csv --out .\experiment
uv run kgss_token_check.py --contam-probes .\experiment\experiment_probes.csv --out .\experiment
```

## 주요 산출물

| 파일 | 내용 |
|---|---|
| `inventory/kgss_clean.parquet` | 결측 정리 완료 데이터. 이후 모든 분석의 입력 |
| `inventory/diagnostics.md` | 데이터 진단 요약 |
| `selection/variables_worded.csv` | 전 변수 설문 원문 + 선택지. 프롬프트 원천 |
| `selection/ab_pairs.csv` | A/B 분할표본과 인간 응답 순서 효과 |
| `selection/문항선정_작업지_2023.xlsx` | 3인 태깅용 문항 선정 작업지 (사실/차별/가치관, 민감여부) |
| `selection/agreement_result.md` | 3인 태깅 일치도(Cohen's/Fleiss' kappa) |
| `selection/문항선정_병합_2023.csv` | 3인 태깅 다수결 병합본 — 최종 문항 뽑을 때 기준 |
| `token_check/*.md` | 모델별 선택지 토큰화 · 첫 토큰 로짓 점검 결과 |
| `probe/README.md` | 오염 프로빙 절차 |
| `probe/contamination_result.md` | 오염 프로브 채점 결과 (A/B/C/E, 모델별) |
| `persona_sample/persona_sample.csv` | FINALWT 가중 복원추출 6변수 페르소나 표본 |
| `c0_check*/c0_validity_result.md` | C0/adult/demo/full 조건별 population Wasserstein거리·출력 엔트로피·subgroup r (운영진 Q1/Q2 대응) |

## 진행 상황

- **문항 선정**: 후보 303개(2023년 회차) 중 사실/차별/가치관 자동 분류, 3인
  독립 태깅(`kgss_agreement.py`로 일치도·다수결 병합) 완료. 배터리 중복 제거하면
  220개로 줄어듦(사실 66 / 차별 11 / 가치관 143, 민감 다수결 Y 38개). 최종
  30개(유형별 10개, 배터리당 1개, eta2 층화) 확정이 다음 단계.
- **사전학습 오염 점검**: 로짓 기반(A)·원문완성(B)·선택지순서(C)·직접회상(E)
  4가지 프로브 + 가짜출처 대조군으로 n=140 재실행 완료. 선택지에 DK/비해당
  코드가 안 걸러지고 섞여 있던 버그를 발견·수정해 A/C/E만 재실행한 결과,
  뚜렷한 오염 증거는 확인되지 않음(C의 선택지순서 정확일치율만 우연보다
  높아 완전한 결백까지는 주장 못 함). 자세한 수치는 `probe/contamination_result.md` 참고.
- **C0/adult/demo/full 조건 분포 재현 타당성 검증**: 성별×연령 10셀 기준 subgroup r을
  "셀평균 − 문항전체평균" 편차로 계산(원값 상관은 문항 자체의 지지율만 재는
  가짜신호라 편차로 교정). 페르소나 없음(C0) → "한국 성인"(adult) → 성별+연령
  (demo) → 6변수 서사형(full) 순으로 정보량을 늘렸을 때:

  | | C0 | adult | demo | full |
  |---|---|---|---|---|
  | Wasserstein — EXAONE 2.4B | 0.85 | 0.82 | 0.81 | 0.73 |
  | Wasserstein — Kanana 2.1B | 0.92 | 0.89 | 0.88 | 0.83 |
  | subgroup r — EXAONE 2.4B | -0.03 | -0.03 | -0.01 | 0.00 |
  | subgroup r — Kanana 2.1B | 0.00 | 0.00 | 0.01 | 0.03 |
  | 출력 엔트로피(bit) — EXAONE | 1.01 | 0.97 | 1.01 | 1.04 |
  | 출력 엔트로피(bit) — Kanana | 0.49 | 0.46 | 0.49 | 0.50 |

  (인간 기준 엔트로피 1.53bit = 유효 선택지 2.89개, 네 조건 공통)

  **통계 검정(대응표본, C0 vs full, 문항/셀 짝지음)**: Wasserstein 개선은
  뚜렷이 유의함(EXAONE t=6.28·p=1.4e-09, Kanana t=4.18·p=4.0e-05, n=255문항).
  subgroup r 개선도 유의하지만 더 약함(EXAONE t=2.26·p=0.050, Kanana
  t=3.59·p=0.006, n=10셀) — 다만 **full 조건의 subgroup r 자체는 여전히 0과
  통계적으로 구분 안 됨**(one-sample t-test, EXAONE p=0.994, Kanana p=0.593).
  즉 정보를 늘릴수록 오르는 추세는 통계적으로 실재하지만, 도달한 절대 수준은
  "하위집단을 구분한다"고 주장할 수 있는 데까지는 못 간 상태.

  **"통계적으로 유의함"이 "실질적으로 쓸만함"을 뜻하진 않음** — 같은 255문항에
  균등분포로 그냥 찍는 기준선(0.677)과 실제 최빈범주를 100% 확신하고 찍는
  오라클급 기준선(0.590)을 계산해 비교하면: **가장 정보가 많은 full 조건까지
  가도 EXAONE(0.731)·Kanana(0.830) 둘 다 균등분포 기준선(0.677)보다 나쁨.**
  통계적으로 유의한 개선이 도달한 지점 자체가 "정보 없이 균등하게 찍는 것"만도
  못하다는 뜻 — "많이 좋아졌다"가 아니라 "훨씬 나빴던 게 조금 덜 나빠졌는데
  여전히 아무 생각 없이 찍는 것보다 못하다"가 정확한 서술.

  **원인 — 출력 엔트로피로 직접 확인(모드 붕괴)**: 두 모델 다 인간 기준(1.53bit)
  보다 뚜렷이 낮고(EXAONE 인간의 0.63-0.68배, Kanana 0.30-0.33배), **그 붕괴
  정도가 네 조건 내내 거의 안 움직임** — 모델이 문항마다 1.4-2개 선택지로
  강하게 쏠려 답하는 성질(모드 붕괴, 성분②) 자체가 페르소나 정보량과 무관함.
  좁게 쏠려서 자신 있게 답하는데 그 확신이 자주 틀린 곳을 가리키다 보니
  균등분포보다 거리 비용이 더 드는 것 — subgroup r이 못 오르는 것과 Wasserstein이
  기준선을 못 넘는 것 둘 다 같은 원인으로 설명됨. Kanana가 EXAONE보다 훨씬
  심하게 붕괴돼 있음(모델 간 차이가 큼)도 확인.

  **모델 크기 때문인지는 이 데이터만으로는 결론 못 냄** — 서울 논문이 훨씬 큰
  오픈웨이트(EXAONE 7.8B, Qwen3-30B)에서도 subgroup r이 0.05-0.12로 비슷하게
  붕괴됐다고 보고해 크기가 지배적 요인이 아닐 가능성을 시사하고, 우리 데이터
  안에서도 크기가 거의 같은 EXAONE(2.4B)과 Kanana(2.1B)의 붕괴 정도가 2배
  넘게 차이 나 — 정렬(alignment) 등 다른 요인 가능성. 직접 확인하려면 같은
  파이프라인을 EXAONE 7.8B/Kanana 8B로도 돌려야 함(미실행, T4 15GB 메모리에
  들어가는지부터 확인 필요).

  **개선 실마리 — 디코딩 온도(temperature)**: 기존 로짓 재계산만으로(재실행 없이)
  온도를 올려 재정규화해보면, C0 조건에서 EXAONE은 T≈3.0, Kanana는 T≈4.0
  부근에서 Wasserstein이 균등분포 기준선(0.677)을 실제로 역전함(EXAONE
  0.660, Kanana 0.646) — 그 지점의 엔트로피도 인간 기준(1.53bit)에 근접함.
  온도를 무한히 올리면 균등분포로 수렴해 기준선에 점근하는 것 자체는 당연하지만
  기준선을 "역전"하는 건 자명하지 않음 — 모델이 가진 순위 정보(어느 선택지가
  상대적으로 더 그럴듯한지)가 확신도만 낮추면 실제로 쓸모 있다는 뜻. 단 이건
  population Wasserstein 기준 탐색이고, **더 중요한 subgroup r이 온도 조정으로
  같이 좋아지는지는 아직 미확인**(다음 단계). 자세한 수치는
  `c0_check*/c0_validity_result.md` 참고.
- **본실험**: 프롬프트 생성 파이프라인(`kgss_experiment_prompt.py`)까지 준비
  완료, 최종 문항 확정 후 실행 예정.

## 재실험과 정정 (2026-10-09 ~ 10-18)

위 "C0/adult/demo/full 조건 분포 재현 타당성 검증"의 수치는 아래 오류가 있던
파이프라인의 결과다. 최신 결과는
[`reexperiment_20261009/report_20261018/KOSMOS_REEXPERIMENT_REPORT_20261018.md`](reexperiment_20261009/report_20261018/KOSMOS_REEXPERIMENT_REPORT_20261018.md)를
기준으로 한다(수치 대조 스크립트 동봉).

- **비교 구조**: 페르소나를 6변수까지 늘려도 인간 정답은 모집단 전체(Wasserstein)나
  성별×연령 10셀(subgroup r, 정규화 정확도)이었다. Wasserstein은 페르소나 행별
  거리를 평균 냈다.
- **발견·수정한 오류**: 행별 거리 평균 → 예측분포를 먼저 섞은 뒤 비교. Demo 셀
  가중치 누락과 subgroup r 모델 쪽 미중심화를 수정했다. 명목형 문항에는 W1을
  쓰지 않는다. 프롬프트 결함 9문항을 제외했다(정제 246문항). Full 페르소나의
  "학교를 졸업했고" 단정이 GRADUATE 정답을 누출하던 문제를 고쳤다. 중간보고서
  표 3-8의 정확도 차이 표기도 정정 대상이다.
- **결과**: 오류를 고치고 인간 정답을 페르소나 변수에 맞춰도 보정 전 성능은 균등분포
  기준선보다 나빴다. 성능을 바꾼 것은 저장된 로짓에 적용한 사후 온도 보정이다.
  성별×연령 고정 비교 Test49에서 TVD가 EXAONE 0.4178→0.3067, Kanana
  0.5608→0.3080으로 낮아져 균등 기준선(0.3716)을 넘었다. 다만 균등분포 혼합
  기준선보다의 이득은 0.02~0.04로 작다. 학력 추가는 개선을 보였지만, 혼인·6변수의
  추가 이득은 검출되지 않았다.
- **미완료**: Core29 본실험(현재 4문항 파일럿), T4 신규 추론, 외부 검증, 인칭×화계.
- **환경 메모**: `requirements.txt`는 UTF-16LE로 저장돼 있어 `uv pip install -r`가
  실패한다. UTF-8로 변환한 사본을 쓰면 `tests/` 45개가 통과한다(2026-10-18 확인).

</details>
