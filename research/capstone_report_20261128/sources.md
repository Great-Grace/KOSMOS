# 캡스톤 최종보고서 (2026-11-28 제출본) 만들기와 수치 출처

## 파일

| 파일 | 내용 |
|---|---|
| `DAT 8회 캡스톤 프로젝트 보고서_KOSMOS팀.docx` | 제출본. DAT 양식(`template.docx`)의 표지·속표지·요약·목차·꼬리말을 그대로 썼다 |
| `미리보기_대체글꼴.pdf` | 확인용. 한국외대체가 없는 환경이라 산돌고딕으로 바꿔 LibreOffice로 렌더링했다. 제출용 아님 |
| `content.py` | 원고 전체(요약, 본문, 표, 참고문헌) |
| `build_report.py` | 양식에 원고를 끼워 넣는다 |
| `preview.py`, `pages.py` | 미리보기 렌더링, 목차 쪽 번호 계산(`toc_pages.json`) |
| `verify_numbers.py` | 본문 수치 133개를 원본 결과 파일과 대조(2026-11-27 통과). 그 밖의 수치 약 110개는 같은 날 따로 원본과 대조해 고쳤다 |
| `make_fig_reasoning.py` | 그림 4-2. 입력 해시와 값은 `fig_reasoning_shift.json` |
| `make_fig_beta.py` | 그림 4-1. `counts_k100.npz`에서 다시 그린다. 값은 `fig_beta.json` |
| `baselines_human.py` | 표 4-1의 사람 쪽 기준선(소표본, 표집 잡음, 전체 분포 적용). 사람 자료만 쓴다. 결과 `baselines_human.csv`, 기록 `baselines_human.run.json`·`.run.log` |

## 다시 만들기

```bash
python3 baselines_human.py
python3 make_fig_beta.py   # matplotlib 필요
python3 verify_numbers.py
python3 build_report.py template.docx "DAT 8회 캡스톤 프로젝트 보고서_KOSMOS팀.docx" toc_pages.json
```

미리보기는 LibreOffice 임시 프로필에 산돌고딕을 넣고 `LO_PROFILE=<프로필 경로> python3 preview.py <docx> <폴더>`로 돌린다. 쪽 번호는 미리보기 기준이라 실제 한국외대체로 열면 한두 쪽 다를 수 있다.

## 수치 출처

| 본문 | 파일 |
|---|---|
| 표 4-1, 표 4-2 위 두 행, 4장 1절 모델 간 차이 | `research/design_v2_20261112/analysis/out_final_k100/REPORT.md` (시험용 49문항, k=100) |
| 표 4-2 분산비·정렬, 기울기 | `experiments/20261126-model-exaggeration/out/primary.csv`, `explore.csv` (브랜치 `exp/20261126-model-exaggeration`) |
| 소형 모델 재계산 0.292·0.310 | `research/design_v2_20261112/analysis/out_calib/legacy_vs_v2_summary.csv` |
| 예비 실험 수치 | `Woostudy/Silicon Sampling/report/sections/03_pilot.tex` (11/26 1차 최종본에서 대조 완료) |
| 표 4-3 | `research/design_v2_20261112/smoke_haiku_cot/summary.txt`, `four_levels/out/quality.csv` (브랜치 `exp/20261126-haiku-think`) |
| 표 4-4, 표 4-5, 그림 4-2, 보정 후 사람 TVD | `research/summary_register_reasoning_20261126/four_levels/out/` |
| DeepSeek 분산비·정렬 추세(high − low) | `experiments/20261123-reasoning-high-full/` |
| 표 4-6, 4장 3절 4항 | `experiments/20261127-haiku-think-control/out/` (대조군, 커밋 `499c1d7`) |
| 표 4-7 | `experiments/20261123-register-final/out/effects.csv` |
| 표 4-1 W1 행 | `research/design_v2_20261112/analysis/out_final_k100/metrics_by_setting.csv` (test, w1, L0, β*) |
| 표 4-1 기준선 행, 4장 1절 소표본 비교 | `baselines_human.csv` |
| 부록 표 A-1·A-2 | `research/design_v2_20261112/build_prompts.py`, `runners/run_haiku_batch.py`, `register_v1/build_register_prompts.py` |
| 부록 표 A-3 | `research/design_v2_20261112/DESIGN.md` 4절, 수집 로그 시각 |
| 문구 민감도, 표 4-8, 거부 민감도, 요소 분해 | `research/summary_register_reasoning_20261126/README.md` 2·3절 (원 출처 `20261119-register-effect`, `20261123-register-decomp-*`) |
| β 보정만의 효과 0.443 → 0.285 등 | 같은 README 5절 (`20261115-baseline-calibration`) |
