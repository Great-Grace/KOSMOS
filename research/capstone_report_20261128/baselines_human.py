# -*- coding: utf-8 -*-
"""사람 쪽 기준선 (표 4-1 아래 행): 보정 후 TVD가 얼마나 큰지 견줄 값.
사람 자료만 쓴다(모델 응답은 열지 않는다). 시험용 49문항, 표 4-1과 같은 페르소나 단계.
  1) 사람 n명 소표본: 사람 분포에서 n명을 다시 뽑아 원래 분포와의 TVD (n = 20, 100)
  2) 표집 잡음: 유효 표본 수(NEFF)만큼 다시 뽑았을 때의 TVD
  3) 전체 분포를 모든 셀에 적용: 한국 성인 전체 분포로 각 성별×연령 셀 분포를 맞힐 때의 TVD
입력: research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz
출력: baselines_human.csv, baselines_human.run.json, baselines_human.run.log"""
import numpy as np, csv, json, hashlib, pathlib, datetime, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[0] / "design_v2_20261112/analysis/out_final_k100/counts_k100.npz"
SEED, REPS = 20261128, 1000
log = open(HERE / "baselines_human.run.log", "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

z = np.load(SRC, allow_pickle=True)
H, NEFF, CODES, split = z["H"], z["NEFF"], z["CODES"], z["split"]
MASK = ~np.isnan(CODES); test = np.where(split == "test")[0]
L0, CELLS = 1, list(range(2, 12))
rng = np.random.default_rng(SEED)
tvd = lambda p, q: 0.5 * np.abs(p - q).sum(-1)

def resample_tvd(j, i, n):
    k = int(MASK[i].sum()); h = H[j, i, :k] / H[j, i, :k].sum()
    return tvd(rng.multinomial(n, h, size=REPS) / n, h).mean()

rows = []
def add(name, level, val, note):
    rows.append(dict(기준선=name, 페르소나=level, 값=round(float(val), 6), 비고=note)); say(f"{name:28s} {level:10s} {val:.4f}  {note}")

for n in (20, 100):
    add(f"사람 {n}명 소표본", "한국 성인", np.mean([resample_tvd(L0, i, n) for i in test]), "시험용 49문항 평균")
    add(f"사람 {n}명 소표본", "성별×연령", np.mean([[resample_tvd(j, i, n) for i in test] for j in CELLS]), "셀마다 n명, 10개 셀 단순 평균")
add("표집 잡음(유효 표본 수)", "한국 성인", np.mean([resample_tvd(L0, i, int(NEFF[L0, i])) for i in test]),
    f"유효 표본 수 중앙값 {int(np.median(NEFF[L0, test]))}")
add("표집 잡음(유효 표본 수)", "성별×연령", np.mean([[resample_tvd(j, i, int(NEFF[j, i])) for i in test] for j in CELLS]),
    f"셀별 유효 표본 수 {int(NEFF[2:, test].min())}~{int(NEFF[2:, test].max())}")
pop = np.mean([[tvd(H[L0, i], H[j, i]) for i in test] for j in CELLS])
add("전체 분포를 모든 셀에 적용", "성별×연령", pop, "한국 성인 전체 분포로 각 셀을 맞힘. 표 4-2의 사람 0.075와 같은 양")
unif = lambda j: np.mean([tvd(np.where(MASK[i], 1 / MASK[i].sum(), 0), H[j, i]) for i in test])
add("균등분포", "한국 성인", unif(L0), "점검: REPORT.md 0.360")
add("균등분포", "성별×연령", np.mean([unif(j) for j in CELLS]), "점검: REPORT.md 0.372")

with open(HERE / "baselines_human.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True).stdout.strip()
json.dump(dict(script="baselines_human.py", time=datetime.datetime.now().isoformat(timespec="seconds"), seed=SEED, reps=REPS,
               input=str(SRC.relative_to(HERE.parents[1])), sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
               n_test_items=int(len(test)), uses_model_responses=False, git_commit=git("rev-parse", "--short", "HEAD"), rows=rows),
          open(HERE / "baselines_human.run.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
