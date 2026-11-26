"""20261126-reasoning-concentration — 추론 수준(끔·low·medium·high)별 응답 쏠림(최빈 보기 비율(top_share)·엔트로피 비 (FORMULAS.md 5.4)).
실행 (KOSMOS 루트): uv run -q --no-project --python 3.12 --with pandas --with numpy python experiments/20261126-reasoning-concentration/run.py

출처: 데이터 경로·load는 experiments/20261123-reasoning-high-full/run.py 복사(197문항, 표본 0~49, 중복·api_error 제외).
β는 experiments/20261123-reasoning-high-full/out/trend.csv 값, 보정식 prob는 같은 파일의 prob.
"""
import csv
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS = DESIGN / "prompts_v2.csv"
CELLS = DESIGN / "cells.csv"
SUBSETS = [DESIGN / "subsets/high60.csv", DESIGN / "subsets/high_rest137.csv"]
TREND = ROOT / "experiments/20261123-reasoning-high-full/out/trend.csv"
RAW = {"끔": [ROOT / "raw/v2/deepseek_eco_p1.jsonl"],
       "low": [ROOT / "raw/v2/deepseek_eco_low_p1.jsonl"],
       "medium": [ROOT / "raw/v2/deepseek_eco_medium_p1.jsonl"],
       "high": [ROOT / "raw/v2/deepseek_eco_high_high60_p1.jsonl", ROOT / "raw/v2/deepseek_eco_high_high_rest137_p1.jsonl"]}
K, B, NULL_REPS, N_MATCH, SEED = 50, 2000, 200, 50, 20261126
rng = np.random.default_rng(SEED)
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# ── 1. 데이터 ───────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
H, CODES, SPLIT = d["H"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NK = MASK.sum(1)                                              # 문항별 보기 수
NI = len(VARS)
SUBV = [[r["var"] for r in csv.DictReader(f.open(encoding="utf-8-sig"))] for f in SUBSETS]
SUB = np.isin(VARS, SUBV[0] + SUBV[1])
assert SUB.sum() == 197 and (SUB == (SPLIT != "test")).all()
IDX = np.where(SUB)[0]
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}
BETA = {r["설정"]: (float(r["β_페르소나없음"]), float(r["β_페르소나있음"])) for r in csv.DictReader(TREND.open(encoding="utf-8-sig"))}


def load(paths):
    """출처: experiments/20261123-reasoning-high-full/run.py load."""
    C = np.zeros((len(PERSONAS), NI, CODES.shape[1]))
    seen = set()
    for line in (ln for path in paths for ln in open(path, encoding="utf-8")):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in probes:
            continue
        p = probes[r["probe_id"]]
        i = VARS.index(p["var"])
        if not SUB[i]:
            continue
        seen.add(key)
        if r["valid"]:
            C[PERSONAS.index(p["cell_id"]), i, list(CODES[i]).index(float(r["parsed"]))] += 1
    return C


SET = list(RAW)
DATA = {n: load(p) for n, p in RAW.items()}


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def prob(X, beta):
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def top(P):
    """FORMULAS.md 5.4 top_share = max_j r_j."""
    return P.max(-1)


def H_(P):
    """엔트로피(자연로그). 엔트로피 비 = H(모델)/H(사람)이라 로그 밑과 무관하다(FORMULAS.md 5.4)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0).sum(-1)


UNITS = ["페르소나 없음", "한국 성인", "10셀 셀별", "10셀 합성"]


def units(P, f):
    """P: [12, 문항, 보기] → 단위별 문항 값.
    10셀 셀별 = FORMULAS.md 5.5 L1cell: 셀마다 계산해 10셀 단순 평균.
    10셀 합성 = FORMULAS.md 2.5 L1agg: Σ pop_share·P_c로 분포를 먼저 합친 뒤 계산."""
    return {"페르소나 없음": f(P[0]), "한국 성인": f(P[1]),
            "10셀 셀별": f(P[L1J]).mean(0),
            "10셀 합성": f((W[:, None, None] * P[L1J]).sum(0))}


HP = np.nan_to_num(H.copy())
assert np.allclose(HP[0][IDX], HP[1][IDX])                    # 페르소나 없음 정답 = 한국 성인 정답 = 전체
HUM_H = {"페르소나 없음": H_(HP[0]), "한국 성인": H_(HP[1]),
         "10셀 셀별": H_(HP[L1J]),                              # [10, 문항] 셀마다 비를 낸 뒤 평균
         "10셀 합성": H_((W[:, None, None] * HP[L1J]).sum(0))}   # 사람 합성 분포 = Σ pop_share·H_c (FORMULAS.md 2.5, calibration 노트북 정의)


def ent_ratio(P):
    m = {"페르소나 없음": H_(P[0]), "한국 성인": H_(P[1]), "10셀 셀별": H_(P[L1J]),
         "10셀 합성": H_((W[:, None, None] * P[L1J]).sum(0))}
    r = {u: m[u] / np.where(HUM_H[u] > 0, HUM_H[u], np.nan) for u in UNITS}
    r["10셀 셀별"] = np.nanmean(r["10셀 셀별"], 0)
    return r


# 사람 기준: 분포 그대로 / 모델과 같은 50명을 다시 뽑았을 때의 기대값(200회)
def resample(p_, k_):
    x = np.zeros((NULL_REPS, CODES.shape[1]))
    x[:, :k_] = rng.multinomial(N_MATCH, p_[:k_] / p_[:k_].sum(), size=NULL_REPS) / N_MATCH
    return x


RS = np.zeros((12, NI, NULL_REPS, CODES.shape[1]))
for j in [1] + L1J:
    for i in IDX:
        RS[j, i] = resample(HP[j, i], NK[i])
RS[0] = RS[1]
HUM50 = {"top": {}, "ent": {}}
for u, sel in (("페르소나 없음", 1), ("한국 성인", 1)):
    HUM50["top"][u] = top(RS[sel]).mean(-1)
    HUM50["ent"][u] = (H_(RS[sel]) / np.where(HUM_H[u] > 0, HUM_H[u], np.nan)[:, None]).mean(-1)
HUM50["top"]["10셀 셀별"] = top(RS[L1J]).mean(-1).mean(0)
HUM50["ent"]["10셀 셀별"] = np.nanmean((H_(RS[L1J]) / np.where(HUM_H["10셀 셀별"] > 0, HUM_H["10셀 셀별"], np.nan)[..., None]).mean(-1), 0)
mixRS = (W[:, None, None, None] * RS[L1J]).sum(0)
HUM50["top"]["10셀 합성"] = top(mixRS).mean(-1)
HUM50["ent"]["10셀 합성"] = (H_(mixRS) / np.where(HUM_H["10셀 합성"] > 0, HUM_H["10셀 합성"], np.nan)[:, None]).mean(-1)


# ── 2. 설정별 값 ─────────────────────────────────────────
V = {}                                                        # (설정, 분포종류, 지표) → 단위별 문항 값
for n_ in SET:
    C = DATA[n_]
    b0, b1 = BETA[n_]
    for dist, P in (("원비율", share(C)), ("β 보정 후", np.concatenate([prob(C[:1], b0), prob(C[1:], b1)]))):
        V[(n_, dist, "top")] = units(P, top)
        V[(n_, dist, "ent")] = ent_ratio(P)
V[("사람", "그대로", "top")] = units(HP, top)
V[("사람", "그대로", "ent")] = {u: np.ones(NI) for u in UNITS}
V[("사람", "50명 다시 뽑음", "top")] = HUM50["top"]
V[("사람", "50명 다시 뽑음", "ent")] = HUM50["ent"]

BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
KNAME = {"top": "최빈 보기 비율(top_share)", "ent": "엔트로피 비"}

rows = []
for (n_, dist, kind), u in V.items():
    for un in UNITS:
        x = u[un]
        lo, hi = ci(np.nanmean(x[BIX], 1))
        rows.append(dict(설정=n_, 분포=dist, 지표=KNAME[kind], 단위=un, 값=float(np.nanmean(x[IDX])), 하한=lo, 상한=hi))
lv = pd.DataFrame(rows)
lv.to_csv(out / "levels.csv", index=False, encoding="utf-8-sig")

crows = []
for dist in ("원비율", "β 보정 후"):
    for kind in ("top", "ent"):
        for a, b in [("끔", "low"), ("low", "medium"), ("medium", "high"), ("끔", "high")]:
            for un in UNITS:
                x = V[(b, dist, kind)][un] - V[(a, dist, kind)][un]
                bs = np.nanmean(x[BIX], 1)
                p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
                lo, hi = ci(bs)
                crows.append(dict(분포=dist, 지표=KNAME[kind], 단위=un, 비교=f"{b} − {a}", 차이=float(np.nanmean(x[IDX])),
                                  하한=lo, 상한=hi, p=float(min(p, 1))))
cp = pd.DataFrame(crows)
cp.to_csv(out / "compare.csv", index=False, encoding="utf-8-sig")

# 보기 수별 (원비율, 10셀 셀별 최빈 보기 비율)
krows = []
for k_ in sorted(set(NK[IDX])):
    ix = IDX[NK[IDX] == k_]
    r = dict(보기수=int(k_), 문항수=len(ix))
    for n_ in SET:
        r[n_] = float(V[(n_, "원비율", "top")]["10셀 셀별"][ix].mean())
    r["사람(50명 다시 뽑음)"] = float(V[("사람", "50명 다시 뽑음", "top")]["10셀 셀별"][ix].mean())
    r["사람(그대로)"] = float(V[("사람", "그대로", "top")]["10셀 셀별"][ix].mean())
    krows.append(r)
kd = pd.DataFrame(krows)
kd.to_csv(out / "by_k.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
log(f"197문항 (test 제외), 설정 {', '.join(SET)}, β {BETA}")
for kind in ("top", "ent"):
    for dist in ("원비율", "β 보정 후"):
        t = lv[(lv.지표 == KNAME[kind]) & ((lv.분포 == dist) | (lv.설정 == "사람"))].pivot_table(index=["설정", "분포"], columns="단위", values="값", sort=False)[UNITS]
        log(f"\n[{KNAME[kind]} — 모델 {dist}]")
        log(t.round(3).to_string())
log("\n[설정 사이 차이 (탐색, 보정 없음)]")
log(cp.round(4).to_string(index=False))
log("\n[보기 수별 10셀 셀별 최빈 보기 비율 (원비율)]")
log(kd.round(3).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
rel = HERE.relative_to(ROOT)
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){rel}/out")) or bool(git("ls-files", "--others", "--exclude-standard", "--", str(rel), f":(exclude){rel}/out")),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: [{"path": str(x.relative_to(ROOT)), "sha256_16": hashlib.sha256(x.read_bytes()).hexdigest()[:16]} for x in v] for k, v in RAW.items()},
           "beta_source": {"path": str(TREND.relative_to(ROOT)), "sha256_16": hashlib.sha256(TREND.read_bytes()).hexdigest()[:16]},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "n_match": N_MATCH, "seed": SEED},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
