"""Render the report and Slack draft from the verified closeout data."""
from pathlib import Path
import json
import pandas as pd

b = Path(__file__).resolve().parent
s = pd.read_csv(b/'evidence/summary.csv')
r = json.loads((b/'evidence/subgroup.json').read_text())
m = json.loads((b/'evidence/evidence.json').read_text())
tests = json.loads((b/'evidence/final_tests.json').read_text())


def row(subset, condition, weighting='cell_poststratified'):
    result = s[(s.subset == subset)&(s.condition == condition)&(s.weighting == weighting)]
    if len(result) != 1:
        raise ValueError('Missing or ambiguous publication figure')
    return result.iloc[0]


def n(x):
    return f'{x:.4f}'


p125 = row('common125_all', 'paired_demo')
p118 = row('common125_valid118', 'paired_demo')
p96 = row('ordinal96', 'paired_demo')
p5 = row('common5_no_graduate', 'paired_demo', 'equal_persona')
f5 = row('common5_no_graduate', 'full', 'equal_persona')
p5w = row('common5_no_graduate', 'paired_demo')
f5w = row('common5_no_graduate', 'full')
p6 = row('common6_all', 'paired_demo', 'equal_persona')
f6 = row('common6_all', 'full', 'equal_persona')
ba = row('binary_attitude6', 'paired_demo')

message = f'''KOSMOS 평가 오류 수정 및 EXAONE 캐시 재검증 결과 공유드립니다.

• 개인별 거리 평균을 모집단 재현 지표로 해석한 문제, Demo 셀 가중치 중복 적용, subgroup r의 모델 중심화 누락을 수정했습니다. 명목형 문항은 코드 평균 상관과 주 Wasserstein 지표에서 제외했습니다.
• 공통 125문항의 Demo 모집단 TVD는 {n(p125.tvd)}이며 균등 기준선 {n(p125.uniform_tvd)}보다 높습니다. 서열형·이진형 118문항의 Wasserstein도 {n(p118.w1)}이며 균등 {n(p118.uniform_w1)}보다 높습니다(오차는 낮을수록 좋음).
• 명목형 제외 subgroup r은 {n(r['common125_valid118']['centered_r'])}입니다. 다점 서열형 96문항에서는 r={n(r['ordinal96']['centered_r'])}, 척도폭 정규화 r={n(r['ordinal96']['range_normalized_r'])}, 문항 내 집단 상관 평균={n(r['ordinal96']['within_item_r'])}이 관측됐습니다. 제한된 캐시의 탐색적 결과입니다.
• Full 비교는 6문항만 확보됐습니다. 졸업 여부를 묻는 GRADUATE에 대해 기존 Full 페르소나 문구가 졸업을 이미 단정하고 있어 주효과 해석에서 제외했습니다. 나머지 이진형 5문항의 동일가중 TVD는 Demo {n(p5.tvd)} → Full {n(f5.tvd)}로 거의 같습니다. Full의 일반적 우위나 여론·가치관 재현 성공을 입증한 결과는 아닙니다.
• 실제 표본은 40페르소나·37고유 응답자입니다. 4개 캐시의 실제 채점 결과를 독립 계산과 대조했습니다. 이번 마감 범위는 평가 수정과 확보된 EXAONE 캐시 재평가이며, 전체 정제 문항·Kanana·29문항 개인 정확도 CI·인칭×화계 효과는 아직 검증되지 않았습니다.

첨부 공유본과 근거 표를 기준으로 수치를 사용해 주세요. 후속 실험은 문항·표본·가중체계를 고정하고 답을 포함하지 않는 페르소나로 다시 실행할 예정입니다.
'''
(b/'slack_message.md').write_text(message)

report = f'''# KOSMOS 캐시 재검증 공유본 — 2026-10-09

판정: **확보된 캐시의 수정 평가를 공유할 수 있음. 전체 연구의 모델 성능 검증은 미완료.**
이 문서는 Antigravity V1–V3 수치를 그대로 인용하지 않고, 수정된 실제 채점 실행과 별도 계산으로 확인한 결과를 사용한다. 기존 감사 보고서는 이력으로 보존한다.

## 범위와 정의

원본 GitHub 기준 커밋은 `da22600f18f33cb085744c8c972cb5a1af6c6795`이다. 원본 결과 파일 4개 해시를 확인했으며 변경하지 않았다. EXAONE 캐시는 C0 197문항, Adult 145문항, paired Demo 125문항×40명, Full 6문항×40명이다. 조건 비교는 공통 문항에서만 수행한다. paired Demo/Full은 같은 40페르소나·37고유 응답자이며, 원본 대표연령 10셀 Demo와 구분한다. 성별×연령 셀 중 남성 50–59세는 페르소나가 1명이라 셀 내 표집 변동을 충분히 추정할 수 없다.

모집단 지표는 조건부 예측분포를 먼저 혼합한 뒤 인간 분포와 비교한다. 모든 확률벡터를 정규화한다. Demo의 셀 사후층화 행 가중치는 셀 가중치÷셀 인원수다. Full의 기본은 이미 가중 복원추출한 페르소나의 동일가중 평균이며, 조건 간 비교는 같은 가중 방식끼리 제시한다. 인간 참조분포와 셀 가중치는 문항 유효응답 기준이다. 정확한 목표 모집단/무응답 처리 가정의 한계는 후속 전수 실험에서 고정해야 한다.

TVD를 전체 범주형 문항의 주 분포 지표로 사용한다. Wasserstein과 코드 평균 상관은 분류표의 정제된 서열형·이진형에 한정한다. 서열형 Wasserstein은 범주 사이 동일 비용을 가정한다. ACPTCULT·NORTHWHO 등의 의미 축은 검토 대상이며, 이번 TVD 주결과는 그 순서 가정에 의존하지 않는다.

## 검증된 주요 결과

| 비교 집합 / 지표 | C0 | Adult | paired Demo | 균등 기준선 |
| --- | ---: | ---: | ---: | ---: |
| 공통 125문항 TVD | {n(row('common125_all','c0').tvd)} | {n(row('common125_all','adult').tvd)} | {n(p125.tvd)} | {n(p125.uniform_tvd)} |
| 서열·이진 118문항 W1 | {n(row('common125_valid118','c0').w1)} | {n(row('common125_valid118','adult').w1)} | {n(p118.w1)} | {n(p118.uniform_w1)} |
| 다점 서열형 96문항 W1 | {n(row('ordinal96','c0').w1)} | {n(row('ordinal96','adult').w1)} | {n(p96.w1)} | {n(p96.uniform_w1)} |

각 지표는 문항별 오차의 산술평균이고 낮을수록 좋다. Demo는 C0/Adult 대비 오차가 낮지만 이 집합에서는 균등 기준선에 미치지 못한다. 이 비교는 사후 캐시 교집합에 대한 기술적 비교이며 전체 문항에 대한 성능이나 인과적 효과를 확정하지 않는다.

| subgroup r 유효 집합 | 문항 수 | 중심화 r | 척도폭 정규화 r | 문항 내 셀 상관 평균 |
| --- | ---: | ---: | ---: | ---: |
'''
for key,label in [('common125_valid118','서열형·이진형 전체'),('ordinal96','다점 서열형'),
                  ('attitude101_valid','태도·가치·인식, 유효 척도'),('factual17_valid','사실·상태·행동, 유효 척도')]:
    x=r[key]
    report += f"| {label} | {x['n_items']} | {n(x['centered_r'])} | {n(x['range_normalized_r'])} | {n(x['within_item_r'])} |\n"
report += f'''
평균 셀 상관의 제곱을 설명분산 R²로 해석하지 않는다. 배터리 단위 2,000회 부트스트랩 CI는 [evidence/subgroup.json](evidence/subgroup.json)에 보존했으며 관측된 인간·페르소나 셀 추정값을 고정한 탐색적 구간이다. 조사 설계·페르소나 표집 불확실성을 모두 반영한 구간이 아니다. V3의 별도 2원·3원 CI는 이번 공유본의 주근거로 채택하지 않는다.

이진형 태도 6문항의 Demo TVD {n(ba.tvd)}는 균등 {n(ba.uniform_tvd)}보다 좋지만 최빈 원핫 {n(ba.modal_tvd)}보다 나쁘다. 모든 기준선에 대해 실패했다고 표현하지 않는다. 내용 영역별 차이는 수동 분류와 제한된 표본의 탐색적 관찰이다.

## Full 추가정보 비교와 GRADUATE 문제

| 같은 가중 방식·공통 집합 / TVD | paired Demo | Full | Full−Demo |
| --- | ---: | ---: | ---: |
| 6문항, 동일가중 (졸업 단정 포함; 진단용) | {n(p6.tvd)} | {n(f6.tvd)} | {n(f6.tvd-p6.tvd)} |
| GRADUATE 제외 5문항, 동일가중 | {n(p5.tvd)} | {n(f5.tvd)} | {n(f5.tvd-p5.tvd)} |
| GRADUATE 제외 5문항, 셀 사후층화 | {n(p5w.tvd)} | {n(f5w.tvd)} | {n(f5w.tvd-p5w.tvd)} |

기존 Full GRADUATE 프롬프트 40건 모두 페르소나에 ‘학교를 졸업했고’라는 단정을 포함한다. GRADUATE는 졸업/중퇴/재학을 묻기 때문에 이 결과는 일반 예측 능력의 근거에서 제외한다. EDUC가 원인이라는 주장은 졸업 단정을 제거한 재추론 없이 확정하지 않는다. 생성기의 문구는 최종학력만 표현하도록 수정했으나 기존 캐시에 새 추론 결과를 섞지 않았다.

나머지 5개는 이진형이고 다점 태도 문항이 없다. 동일가중 평균 차이는 {n(f5.tvd-p5.tvd)}, 셀가중 차이는 {n(f5w.tvd-p5w.tvd)}이며 방향도 일치하지 않는다. 이 수치는 통계적 동등성 증명이나 Full의 일반적 우위 근거가 아니다.

## 수정·실행·검증·미완료 상태 구분

- **수정 적용 (Code Fix Applied)**:
  - `kgss_metrics.py`·`kgss_c0_validity.py`: Demo 다중 페르소나 셀 가중치 보존($w_i = \tilde{{w}}_c / n_c$), `subgroup r` 인간·모델 양측 동일 관측셀·동일 가중 중심화, 확률분포 정규화 강제, 분류표(`--classification`) 기반 부적절한 $W_1$·코드 평균 상관 제외(진단 열 `diagnostic_*` 분리), 누락·중복 로짓 응답 거부, `--clean-prompts` 옵션(`SAMPLEAB` 제외, 결측 보기 필터링, `NaN` 원문 임시 라벨 대체 및 `prompt_requires_review=True` 표시).
  - `kgss_persona_bootstrap.py`: Full 페르소나 생성 문구에서 졸업 단정(`학교를 졸업했고`)을 제거하고 최종학력 수준만 서술(`최종학력은 {{EDUC}}이며`)하도록 수정.
- **실제 실행 (Executed)**:
  - 단위·통합 테스트 {tests['test_count']}개 실행(명시적 skip {tests['explicit_skip_mentions']}개).
  - 확보된 EXAONE 4개 캐시(`c0_cached` 197문항, `adult_cached` 145문항, `paired_demo_125items` 125문항×40명, `full_6items` 6문항×40명)를 별도 출력 경로(`evidence/scored/`)에서 `--classification` 적용 채점 실행 및 `reproduce.py` 독립 계산(배터리 부트스트랩 2,000회, `seed=42`) 실행.
  - `kgss_c0_validity.py build --clean-prompts` 실행으로 254문항·2,540셀 프롬프트 생성 및 결함 필터링 작동 확인(5개 임시 라벨 대체 문항 `HATOPSEX`, `HHSEX`, `HHMARI`, `HHAWAY`, `HHEMPLY`는 `prompt_requires_review=True`로 분리).
- **검증 완료 (Verified)**:
  - 실제 `score()` 채점 결과와 `reproduce.py` 독립 계산 간 전 지표(`population_tvd`, `population_wasserstein`, 조건부/모집단 엔트로피, 상호정보량) 대조 일치 확인(최대 차이 약 $1.6\\times 10^{{-15}}$, [score_crosscheck.json](evidence/score_crosscheck.json)).
  - 원본 결과 파일 4개(`c0_check*/c0_validity_result.md`) 및 캐시 입력 파일 SHA-256 해시 보존 확인([evidence.json](evidence/evidence.json)).
  - 과거 수치 대조: 명목형을 포함한 125문항 $W_1$ `0.6020` 및 중심화 $r$ `0.1833`은 이력·민감도 값으로만 보존하며, 공유 주결과는 위 유효 척도 118문항 $W_1$ `0.5970`, $r$ `0.1494` 및 125문항 전체 TVD `0.4000`으로 확정.
- **미완료·미검증 (Unverified / Pending)**:
  - 결함 9개를 제외한 전체 정제 문항(246문항, 또는 공식 질문 복원 후 254문항)에 대한 `original_demo`·`paired_demo`·`Full` 전수 재추론 및 재채점.
  - 졸업 단정을 제거한 새 Full 페르소나 프롬프트 기준의 모델 재추론(`GRADUATE` 등).
  - Kanana-1.5-2.1B의 원시 로짓 확보 또는 재추론을 통한 수정 지표 성능 검증.
  - 본실험 29문항 개인 응답 예측 정확도 및 페르소나 클러스터 부트스트랩 CI(원시 개인별 예측 로그 미확보; 기존 요약값만으로 CI를 복원하지 않음).
  - RQ3 한국어 인칭×화계(2×2) 조건의 민감 문항 효과 검증.

## 후속 실험 계약

1. 문항 목록·척도 근거·질문 원문·결측 처리·목표 모집단·평균 가중치를 실행 전에 고정한다. 행정변수(`SAMPLEAB`)와 입력에 답을 포함한 문항은 일반 예측 성능 집합에서 제외하거나 별도 진단한다.
2. 같은 응답자·문항으로 C0/Adult/Demo/Full을 비교하고 Full의 학교 졸업 단정을 제거한다. 생성한 입력과 모델 리비전·추론 설정·선택지 매핑을 함께 보존한다.
3. 명목형은 TVD, 서열형은 TVD와 사전에 정의한 W1, subgroup r은 의미 있는 척도에서만 보고한다. 필수 응답이 하나라도 누락·중복이면 전수 완료로 처리하지 않는다.
4. 원시 로그를 확보해 기존 실행을 재채점하거나, 자원이 확보된 환경에서 동일 계약으로 새 추론을 수행한다. 이 공유본은 그 실행을 완료했다고 주장하지 않는다.

## 재현과 공유 (입력·원시 로그·실행 환경·명령)

- **실행 환경**: Python `{m['python']}`, NumPy `{m['numpy']}`, Pandas `{m['pandas']}`, SciPy (`audit-venv`: `/Users/taewoo/Documents/Codex/2026-10-08/new-chat/work/audit-venv/bin/python`), 난수 시드 `seed={m['seed']}`, 부트스트랩 `n_boot={m['n_boot']}`.
- **현재 마감 재현에 필요한 입력 및 원시 로그**:
  - KGSS 참조 및 메타데이터: `inventory/kgss_clean.parquet`, `inventory/variables.csv`, `inventory/wave_items.csv`, `selection/variables_worded.csv`, `c0_check/probes.csv`, `audit_outputs/item_classification_255.csv`
  - EXAONE 캐시 원시 로그 4종: `audit_outputs/{{c0_cached,adult_cached,paired_demo_125items,full_6items}}/{{probes.csv,logit_responses.csv,real_cells.csv}}`
  - 원본 보존 대조용: `c0_check*/c0_validity_result.md` (4개 파일)
- **후속 전수 재실험에 추가로 필요한 입력 및 원시 로그**:
  - `persona_sample/persona_sample.csv` (수정된 `kgss_persona_bootstrap.py`로 재생성한 졸업 단정 제거 표본)
  - 정제 246문항(또는 공식 원문 복원 후 254문항) 전체에 대한 4조건(`c0`, `adult`, `demo`, `full`)×2모델(`exaone`, `kanana`) 전수 `logit_responses.csv` 및 본실험 29문항 인칭×화계 원시 로짓 로그
- **재현 및 후속 실행 명령**:
  1. 테스트 스위트(17개): `python -m unittest discover -s tests -p "test_*.py" -v`
  2. 독립 계산 및 근거 파일 재생성: `python audit_outputs/closeout_20261009/reproduce.py --source . --output audit_outputs/closeout_20261009/evidence --n-boot 2000`
  3. 실제 채점기 실행(분류표 적용): `python kgss_c0_validity.py score --out <캐시_복사본_경로> --classification audit_outputs/item_classification_255.csv`
  4. 공유본 및 슬랙 문안 렌더링: `python audit_outputs/closeout_20261009/write_shareout.py`
  5. 후속 실험용 페르소나·프롬프트 생성: `python kgss_persona_bootstrap.py --n 200 --seed 42 --out ./persona_sample` 및 `python kgss_c0_validity.py build --level <c0|adult|demo|full> --clean-prompts --out <출력_경로>`

[문항별 지표](evidence/item_metrics.csv), [집합별 요약](evidence/summary.csv), [하위집단 상관·부트스트랩](evidence/subgroup.json), [입력/코드 해시·표본 범위](evidence/evidence.json), [실제 채점과 독립 계산 대조](evidence/score_crosscheck.json), [슬랙 문안](slack_message.md).
공유 패킷에는 개인별 원시 응답이나 페르소나 행을 넣지 않았다. 재현하려면 원본 입력이 필요하다. 원본 자료는 [GitHub 저장소](https://github.com/7897nikey/DAT_2026_2nd_KOSMOS)와 별도로 확보한 KGSS/모델 캐시다.
'''
(b/'closeout.md').write_text(report)
print('Rendered closeout.md and slack_message.md from verified tables')
