"""20261123-reasoning-high — 추론 high(60문항 층화)에서 셀 간 분산비·셀 편차 정렬이 low보다 커지는지 (k=50).
실행 (KOSMOS 루트): uv run -q --no-project --python 3.12 --with pandas --with numpy python experiments/20261123-reasoning-high/run.py

대상: subsets/high60.csv 60문항 중 순서형·이진 문항. 점수 s = 응답 코드를 0~1로 바꾼 평균. 10셀 가중치 pop_share.
설정: 추론 끔·low·high. 추론 medium 파일이 있으면 추세 표에 넣는다(주 지표는 high − low).
지표 정의 출처: experiments/20261123-reasoning-explain/run.py (stats, parts, agg, 이동 분해)
TVD·β 출처: experiments/20261118-reasoning-effect/run.py (prob, tvd, BETAS)
"""
import csv
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS = DESIGN / "prompts_v2.csv"
MANIFEST = DESIGN / "item_manifest_v2.csv"
CELLS = DESIGN / "cells.csv"
SUBSET = DESIGN / "subsets/high60.csv"
RAW = {"끔": ROOT / "raw/v2/deepseek_eco_p1.jsonl",
       "low": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl",
       "medium": ROOT / "raw/v2/deepseek_eco_medium_p1.jsonl",
       "high": Path(os.environ.get("HIGH_PATH", ROOT / "raw/v2/deepseek_eco_high_high60_p1.jsonl"))}  # HIGH_PATH는 코드 점검용
K, B, NULL_REPS = 50, 2000, 200
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
rng = np.random.default_rng(20261123)
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# ── 1. 데이터 ───────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
H, NEFF, CODES, SPLIT = d["H"], d["NEFF"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NI = len(VARS)
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
SUB = np.isin(VARS, [r["var"] for r in csv.DictReader(SUBSET.open(encoding="utf-8-sig"))])
assert SUB.sum() == 60 and not (SUB & (SPLIT == "test")).any()
SEL = SUB & ORD
CAL, VAL = SUB & (SPLIT == "calibration"), SUB & (SPLIT == "validation")
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}
NORM = np.zeros(CODES.shape)                                   # 코드 → 0~1
for i in range(NI):
    c = CODES[i][MASK[i]]
    NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)


def load(path):
    """출처: experiments/20261123-reasoning-explain/run.py load (표본 0~49, 중복·api_error 제외). 60문항만 센다."""
    C = np.zeros((len(PERSONAS), NI, CODES.shape[1]))
    n = np.zeros((len(PERSONAS), NI))
    q = dict(n=0, valid=0, rtok=0, length=0)
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in probes:
            continue
        p = probes[r["probe_id"]]
        i = VARS.index(p["var"])
        if not SUB[i]:
            continue
        seen.add(key)
        j = PERSONAS.index(p["cell_id"])
        n[j, i] += 1
        q["n"] += 1
        q["rtok"] += r.get("reasoning_tokens", 0) or 0
        q["length"] += r.get("finish_reason") == "length"
        if r["valid"]:
            q["valid"] += 1
            C[j, i, list(CODES[i]).index(float(r["parsed"]))] += 1
    return C, n, q


DATA = {}
for name, path in RAW.items():
    if path.exists():
        DATA[name] = load(path)
    else:
        log(f"{name}: 파일 없음 ({path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}) → 건너뜀")
SET = list(DATA)
log(f"60문항 중 순서형·이진 {int(SEL.sum())}개, calibration {int(CAL.sum())}개, validation {int(VAL.sum())}개 | 설정: {', '.join(SET)}")


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def prob(X, beta=1.0):
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def stats(P, n):
    s = (P * NORM).sum(-1)
    var = (P * (NORM - s[..., None]) ** 2 * MASK).sum(-1)
    return s, var / np.maximum(n, 1)


def parts(s, sig2):
    sbar = (W[:, None] * s).sum(0)
    dv = s - sbar
    V = (W[:, None] * dv ** 2).sum(0) - ((W * (1 - W))[:, None] * sig2).sum(0)
    return V, dv


# ── 2. 셀 지표 ───────────────────────────────────────────
sH, sigH = stats(H[L1J], NEFF[L1J])
VH, dH = parts(sH, sigH)
DH2 = (W[:, None] * dH ** 2).sum(0)
M = {}
for name in SET:
    C = DATA[name][0]
    s, sig = stats(share(C[L1J]), C[L1J].sum(-1))
    V, dm = parts(s, sig)
    M[name] = dict(s=s, V=V, num=(W[:, None] * dm * dH).sum(0), dm2=(W[:, None] * dm ** 2).sum(0))
IDX = np.where(SEL)[0]
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]


def agg(name, ix):
    m = M[name]
    return dict(분산비=m["V"][ix].sum(-1) / VH[ix].sum(-1),
                정렬=m["num"][ix].sum(-1) / np.sqrt(m["dm2"][ix].sum(-1) * DH2[ix].sum(-1)),
                기울기=m["num"][ix].sum(-1) / DH2[ix].sum(-1))


A_pt = {n: agg(n, IDX) for n in SET}
A_bs = {n: agg(n, BIX) for n in SET}
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def compare(key, a, b):
    d_ = A_bs[b][key] - A_bs[a][key]
    p = max(2 * min((d_ <= 0).mean(), (d_ >= 0).mean()), 1 / B)
    return dict(지표=key, 비교=f"{b} − {a}", 기준=float(A_pt[a][key]), 대상=float(A_pt[b][key]),
                차이=float(A_pt[b][key] - A_pt[a][key]), 하한=ci(d_)[0], 상한=ci(d_)[1], p=float(min(p, 1)))


def holm(ps):
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i]))
        adj[i] = run
    return adj


trend = []
for n_ in SET:
    row = dict(설정=n_)
    for key in ("분산비", "정렬", "기울기"):
        row[key], (row[f"{key}_하한"], row[f"{key}_상한"]) = float(A_pt[n_][key]), ci(A_bs[n_][key])
    trend.append(row)

if {"low", "high"} <= set(SET):
    pr = pd.DataFrame([compare("분산비", "low", "high"), compare("정렬", "low", "high")])
    pr["p_Holm"] = holm(pr["p"].to_numpy())
    pr["있음"] = pr["p_Holm"] < 0.05
    pr.to_csv(out / "primary.csv", index=False, encoding="utf-8-sig")
    log("\n[주 지표 (10셀, 60문항 중 순서형·이진, high − low, Holm 2개)]")
    log(pr.round(4).to_string(index=False))
else:
    log("\n[주 지표] high 또는 low 파일이 없어 계산하지 않음")

# ── 3. 탐색: 보정 후 사람 TVD (β: calibration 45 → validation 15) ──
xrows = [compare("기울기", "low", "high")] if {"low", "high"} <= set(SET) else []
for a, b in [("끔", "high"), ("medium", "high"), ("끔", "low")]:
    if {a, b} <= set(SET):
        xrows += [compare(k, a, b) for k in ("분산비", "정렬", "기울기")]
TV, BETA = {}, {}
for n_ in SET:
    C = DATA[n_][0]
    g0 = tvd(prob(C[0][None], BETAS[:, None, None]), H[0][None])                           # 페르소나 없음 [β, 문항]
    gL0 = tvd(prob(C[1][None], BETAS[:, None, None]), H[1][None])                          # 한국 성인
    gL1 = (W[None, :, None] * tvd(prob(C[L1J][None], BETAS[:, None, None, None]), H[L1J][None])).sum(1)   # 10셀 가중
    b0 = int(g0[:, CAL].mean(1).argmin())
    b1 = int(((gL0[:, CAL].mean(1) + gL1[:, CAL].mean(1)) / 2).argmin())                    # 한국 성인과 10셀이 β 하나
    BETA[n_] = (float(BETAS[b0]), float(BETAS[b1]))
    TV[n_] = {"페르소나 없음": g0[b0], "한국 성인": gL0[b1], "10셀": gL1[b1]}
    trend[SET.index(n_)].update({"β_페르소나없음": BETA[n_][0], "β_페르소나있음": BETA[n_][1],
                                 **{f"TVD_{u}": float(TV[n_][u][VAL].mean()) for u in TV[n_]}})
VI = np.where(VAL)[0]
VB = VI[rng.integers(0, len(VI), (B, len(VI)))]
for a, b in [("low", "high"), ("끔", "high")]:
    if {a, b} <= set(SET):
        for u in ("페르소나 없음", "한국 성인", "10셀"):
            x = TV[b][u] - TV[a][u]
            xrows.append(dict(지표=f"보정 후 사람 TVD {u} (validation {len(VI)})", 비교=f"{b} − {a}", 기준=float(TV[a][u][VAL].mean()),
                              대상=float(TV[b][u][VAL].mean()), 차이=float(x[VAL].mean()), 하한=ci(x[VB].mean(1))[0], 상한=ci(x[VB].mean(1))[1]))

# 이동 분해 low → high
if {"low", "high"} <= set(SET):
    Ca, Cb = DATA["low"][0], DATA["high"][0]
    delta = M["high"]["s"] - M["low"]["s"]
    common = (W[:, None] * delta).sum(0)
    spec = (W[:, None] * (delta - common) ** 2).sum(0)
    pooled = share(Ca[L1J] + Cb[L1J])
    na, nb = Ca[L1J].sum(-1).astype(int), Cb[L1J].sum(-1).astype(int)
    nc, ns = np.zeros(NI), np.zeros(NI)
    for i in IDX:
        k_ = MASK[i].sum()
        sa = np.stack([rng.multinomial(na[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(na[c, i], 1) for c in range(10)])
        sb = np.stack([rng.multinomial(nb[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(nb[c, i], 1) for c in range(10)])
        dl = sb - sa
        cm = (W[:, None] * dl).sum(0)
        nc[i], ns[i] = (cm ** 2).mean(), (W[:, None] * (dl - cm) ** 2).sum(0).mean()
    s_h = (W[:, None] * sH).sum(0)
    toward = np.abs((W[:, None] * M["high"]["s"]).sum(0) - s_h) - np.abs((W[:, None] * M["low"]["s"]).sum(0) - s_h)
    for lab, x in [("이동 분해: 공통 이동 제곱 − 잡음", common ** 2 - nc), ("이동 분해: 셀 고유 이동 제곱합 − 잡음", spec - ns),
                   ("이동 분해: 10셀 평균이 사람 평균에 가까워진 정도(음수=가까워짐)", toward),
                   ("이동 분해: 셀 고유 이동이 사람 셀 편차와 같은 방향(공분산)", (W[:, None] * (delta - common) * dH).sum(0))]:
        xrows.append(dict(지표=lab, 비교="high − low", 차이=float(x[IDX].mean()), 하한=ci(x[BIX].mean(1))[0], 상한=ci(x[BIX].mean(1))[1]))
xd = pd.DataFrame(xrows)
xd.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색 (보정 없음)]")
log(xd.round(4).to_string(index=False))

td = pd.DataFrame(trend)
td.to_csv(out / "trend.csv", index=False, encoding="utf-8-sig")
log("\n[추세: 설정별 지표 (60문항)]")
log(td.round(4).to_string(index=False))

qd = pd.DataFrame([dict(설정=n_, 응답수=DATA[n_][2]["n"], 무효율=1 - DATA[n_][2]["valid"] / max(DATA[n_][2]["n"], 1),
                        평균_추론_토큰=DATA[n_][2]["rtok"] / max(DATA[n_][2]["n"], 1),
                        길이_초과_비율=DATA[n_][2]["length"] / max(DATA[n_][2]["n"], 1)) for n_ in SET])
qd.to_csv(out / "quality.csv", index=False, encoding="utf-8-sig")
log("\n[응답 품질 (60문항, 12페르소나)]")
log(qd.round(4).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){HERE.relative_to(ROOT)}/out")) or bool(git("ls-files", "--others", "--exclude-standard", "--", str(HERE.relative_to(ROOT)), f":(exclude){HERE.relative_to(ROOT)}/out")),  # out/은 이 스크립트의 산출물이라 판정에서 뺀다
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: {"path": str(RAW[k].relative_to(ROOT) if RAW[k].is_relative_to(ROOT) else RAW[k].name), "sha256_16": hashlib.sha256(RAW[k].read_bytes()).hexdigest()[:16]} for k in SET},
           "subset": {"path": str(SUBSET.relative_to(ROOT)), "sha256_16": hashlib.sha256(SUBSET.read_bytes()).hexdigest()[:16]},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "seed": 20261123},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
