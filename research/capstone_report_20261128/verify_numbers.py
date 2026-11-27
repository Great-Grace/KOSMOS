# -*- coding: utf-8 -*-
"""보고서 수치 대조: content.py에 적힌 숫자를 원본 결과 파일에서 다시 계산해 문자열로 맞춰 본다.
실행: python verify_numbers.py   (KOSMOS 루트 기준 경로, 브랜치 파일은 git show로 읽는다)
하나라도 틀리면 종료 코드 1."""
import csv, io, json, subprocess, sys, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import content as C

TEXT = "\n".join(str(x) for b in C.T for x in (b if isinstance(b, tuple) else [b]))
TEXT += "\n".join(C.ABSTRACT_KO + C.ABSTRACT_EN)
ok = fail = 0
def check(label, s):
    global ok, fail
    if s in TEXT: ok += 1
    else: fail += 1; print("없음:", label, "→", repr(s))

def rows(path=None, git=None):
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", git]).decode("utf-8-sig") if git else open(ROOT / path, encoding="utf-8-sig").read()
    return list(csv.DictReader(io.StringIO(raw)))
f3 = lambda x: f"{float(x):.3f}"
sgn = lambda x: ("+" if float(x) >= 0 else "−") + f"{abs(float(x)):.3f}"

# 1) 세 모델 과장 (브랜치 exp/20261126-model-exaggeration)
for r in rows(git="exp/20261126-model-exaggeration:experiments/20261126-model-exaggeration/out/primary.csv"):
    check("분산비 " + r["모델"], f"{float(r['분산비']):.2f} [{float(r['분산비_하한']):.2f}, {float(r['분산비_상한']):.2f}]")
    check("정렬 " + r["모델"], f"{f3(r['정렬'])} [{f3(r['정렬_하한'])}, {f3(r['정렬_상한'])}]")

# 2) 인칭·화계 확정 효과
for r in rows("experiments/20261123-register-final/out/effects.csv"):
    s = f"{sgn(r['차이'])} [{sgn(r['하한'])}, {sgn(r['상한'])}]"
    check("register " + r["효과"] + r["설정"], s)
    if r["p_Holm"]:
        check("Holm " + r["효과"] + r["설정"], f"Holm p {float(r['p_Holm']):.2f}")

# 3) 추론 4단계 (four_levels)
FL = "research/summary_register_reasoning_20261126/four_levels/out/"
lv = rows(FL + "concentration_levels.csv")
for r in lv:
    if r["지표"] == "최빈 보기 비율" and "그대로" not in r["설정"]:
        check(f"쏠림 {r['모델']} {r['설정']} {r['페르소나']}", f3(r["값"]))
sh = rows(FL + "shift.csv")
P = "성별×연령(10개 평균)"
for r in sh:
    if r["페르소나"] != P or r["비교"] not in ("끔 → low", "low → medium", "medium → high", "끔 → medium"): continue
    if r["모델"] == "DeepSeek" and r["비교"] == "끔 → medium": continue
    if r["지표"] == "전후 TVD": check("전후 " + r["모델"] + r["비교"], f3(r["값"]))
    if r["지표"] == "잡음 TVD": check("잡음 " + r["모델"] + r["비교"], f"({f3(r['값'])})")
    if r["지표"] == "사람 쪽 순변화": check("순변화 " + r["모델"] + r["비교"], sgn(r["값"]))
    if r["지표"] == "잡음 밖 흔들림 비율": check("흔들림 " + r["모델"] + r["비교"], f"{round(float(r['값'])*100)}%")
    if r["지표"] == "최빈 보기 뒤바뀜 비율": check("뒤바뀜 " + r["모델"] + r["비교"], f"{round(float(r['값'])*100)}%")
q = {r["설정"]: r for r in rows(FL + "quality.csv")}
check("Haiku medium 생각", f"{float(q['medium']['생각블록비율'])*100:.1f}%")
check("Haiku high 생각", f"{float(q['high']['생각블록비율'])*100:.1f}%")
check("비용", f"${float(q['medium']['비용USD']) + float(q['high']['비용USD']):.2f}")
tc = rows(FL + "tvd_calibrated.csv")
for r in tc:
    if r["모델"] == "Haiku" and r["비교"] == "high − 끔" and r["페르소나"] == "페르소나 없음":
        check("Haiku 페르소나 없음 high − 끔 보정 후", f"{sgn(r['값'])}, [{sgn(r['하한'])}, {sgn(r['상한'])}]")

# 4) 세 모델 최종 비교 (REPORT.md 표의 숫자)
rep = open(ROOT / "research/design_v2_20261112/analysis/out_final_k100/REPORT.md", encoding="utf-8").read()
for s in ["0.254", "0.235", "0.288", "0.200", "0.228", "0.271", "0.222", "0.257", "0.284", "0.124", "0.209", "0.246", "0.075", "0.726", "0.735", "0.641", "−0.035 [−0.070, −0.003]".replace("−", "-"), "0.355", "0.315", "0.245"]:
    if s.replace("-", "−") not in rep.replace("-", "−"):
        fail += 1; print("REPORT.md에 없음:", s)
    else: ok += 1
    check("본문 " + s, s.replace("-", "−").replace("−0.035 ", ""))   # 본문은 "−0.035, 95% 구간 [−0.070, −0.003]"로 쓴다

# 5) 소형 모델 재계산
for r in rows("research/design_v2_20261112/analysis/out_calib/legacy_vs_v2_summary.csv"):
    if r["metric"] == "TVD" and r["level"] == "L0" and r["setting"].startswith("보정 후") and r["model"] in ("EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"):
        check("소형 " + r["model"], f3(r["value"]))

# 6) Haiku 대조군 (표 4-6)
for r in rows("experiments/20261127-haiku-think-control/out/shift.csv"):
    if r["페르소나"] != P or r["비교"] not in ("끔 → 끔(생각 지시)", "끔(생각 지시) → medium", "끔(생각 지시) → high", "끔 → medium"): continue
    if r["지표"] == "전후 TVD": check("대조 전후 " + r["비교"], f3(r["값"]))
    if r["지표"] == "잡음 TVD": check("대조 잡음 " + r["비교"], f"({f3(r['값'])})")
    if r["지표"] in ("사람 쪽 순변화", "사람 TVD 변화"): check("대조 " + r["지표"] + r["비교"], sgn(r["값"]))
    if r["지표"] == "잡음 밖 흔들림 비율": check("대조 흔들림 " + r["비교"], f"{round(float(r['값'])*100)}%")
    if r["지표"] == "최빈 보기 뒤바뀜 비율": check("대조 뒤바뀜 " + r["비교"], f"{round(float(r['값'])*100)}%")
for r in rows("experiments/20261127-haiku-think-control/out/tvd_calibrated.csv"):
    if r["비교"] == "high − 끔(생각 지시)" and r["페르소나"] == "한국 성인":
        check("대조 보정 후 한국 성인", f"{sgn(r['값'])}, [{sgn(r['하한'])}, {sgn(r['상한'])}]")
for r in rows("experiments/20261127-haiku-think-control/out/concentration_compare.csv"):
    if r["비교"] == "끔(생각 지시) − 끔" and r["지표"] == "최빈 보기 비율" and r["페르소나"] == "한국 성인":
        check("대조 쏠림 한국 성인", sgn(r["차이"]))

# 7) 사람 쪽 기준선 (표 4-1 아래 행, baselines_human.py)
B = {(r["기준선"], r["페르소나"]): float(r["값"]) for r in rows("research/capstone_report_20261128/baselines_human.csv")}
check("기준선 소표본", f"{B[('사람 20명 소표본', '한국 성인')]:.3f} / {B[('사람 100명 소표본', '한국 성인')]:.3f}")
check("기준선 20명 본문", f"TVD는 {B[('사람 20명 소표본', '한국 성인')]:.3f}이고 100명이면 {B[('사람 100명 소표본', '한국 성인')]:.3f}")
check("기준선 전체 분포", f"TVD가 {B[('전체 분포를 모든 셀에 적용', '성별×연령')]:.3f}인데")
check("기준선 셀 잡음", f"다시 뽑기만 해도 {B[('표집 잡음(유효 표본 수)', '성별×연령')]:.3f}")

# 8) 서열 거리 W1 (표 4-1, 한국 성인, 시험용, β*)
W = {r["model"]: float(r["mean"]) for r in rows("research/design_v2_20261112/analysis/out_final_k100/metrics_by_setting.csv")
     if r["split"] == "test" and r["metric"] == "w1" and r["level"] == "L0" and (r["setting"] == "β*" or r["model"] == "uniform")}
for k, fmt in (("exaone", "**{:.3f}**"), ("deepseek", "{:.3f}"), ("haiku", "{:.3f}"), ("uniform", "균등 {:.3f}")):
    check("W1 " + k, fmt.format(W[k]))

print(f"대조 통과 {ok}개, 실패 {fail}개")
sys.exit(1 if fail else 0)
