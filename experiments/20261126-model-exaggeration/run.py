"""20261126-model-exaggeration — 작업 C: 셀 간 분산비·셀 편차 정렬을 세 모델(k=100 기존 데이터)에서 잰다.
실행 (KOSMOS 루트): uv run -q --no-project --python 3.12 --with pandas --with numpy python experiments/20261126-model-exaggeration/run.py

지표 정의 출처: experiments/20261123-reasoning-high-full/run.py (stats, parts, agg). 바꾼 점: 원응답 대신 counts_k100.npz의
개수 텐서 C[모델, 페르소나, 문항, 보기]를 쓰고, 설정 대신 모델을 비교한다.
대상: test 제외 197문항 중 순서형·이진 문항. 신규 API 호출 없음.
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
MANIFEST = DESIGN / "item_manifest_v2.csv"
CELLS = DESIGN / "cells.csv"
B, SEED = 2000, 20261126
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
C_ALL, H, NEFF, CODES, SPLIT = d["C"], d["H"], d["NEFF"], d["CODES"], d["split"]
MODELS, PERSONAS, VARS = list(d["models"]), list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NI = len(VARS)
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
SUB = SPLIT != "test"
assert SUB.sum() == 197
SEL = SUB & ORD
L1J = list(range(2, 12))
assert all(PERSONAS[j].startswith("L1_") for j in L1J)
cells = {r["cell_id"]: r for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
NORM = np.zeros(CODES.shape)                                   # 코드 → 0~1
for i in range(NI):
    c = CODES[i][MASK[i]]
    NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)
C_ALL = np.nan_to_num(C_ALL)
log(f"모델: {', '.join(MODELS)} | test 제외 197문항 중 순서형·이진 {int(SEL.sum())}개 | 셀 {len(L1J)}개")
for mi, m in enumerate(MODELS):
    n = C_ALL[mi][L1J][:, SEL].sum(-1)
    log(f"  {m}: 셀·문항당 유효 표본 수 평균 {n.mean():.1f}, 최소 {n.min():.0f}")


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


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
sH, sigH = stats(np.nan_to_num(H[L1J]), NEFF[L1J])
VH, dH = parts(sH, sigH)
DH2 = (W[:, None] * dH ** 2).sum(0)
M = {}
for mi, m in enumerate(MODELS):
    C = C_ALL[mi]
    s, sig = stats(share(C[L1J]), C[L1J].sum(-1))
    V, dm = parts(s, sig)
    M[m] = dict(V=V, num=(W[:, None] * dm * dH).sum(0), dm2=(W[:, None] * dm ** 2).sum(0))
IDX = np.where(SEL)[0]
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]


def agg(m, ix):
    x = M[m]
    return dict(분산비=x["V"][ix].sum(-1) / VH[ix].sum(-1),
                정렬=x["num"][ix].sum(-1) / np.sqrt(x["dm2"][ix].sum(-1) * DH2[ix].sum(-1)),
                기울기=x["num"][ix].sum(-1) / DH2[ix].sum(-1))


A_pt = {m: agg(m, IDX) for m in MODELS}
A_bs = {m: agg(m, BIX) for m in MODELS}
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

rows = []
for m in MODELS:
    r = dict(모델=m)
    for key in ("분산비", "정렬"):
        r[key] = float(A_pt[m][key])
        r[f"{key}_하한"], r[f"{key}_상한"] = ci(A_bs[m][key])
    r["과장_있음"] = r["분산비_하한"] > 1
    rows.append(r)
pr = pd.DataFrame(rows)
pr.to_csv(out / "primary.csv", index=False, encoding="utf-8-sig")
log("\n[주 지표 (10셀, test 제외 순서형·이진)] 판정: 분산비 95% 하한 > 1이면 과장 있음")
log(pr.round(4).to_string(index=False))
log(f"→ 세 모델 공통 과장: {bool(pr['과장_있음'].all())}")

# ── 3. 탐색 ───────────────────────────────────────────
xrows = []
for m in MODELS:
    xrows.append(dict(지표="기울기", 비교=m, 차이=float(A_pt[m]["기울기"]), 하한=ci(A_bs[m]["기울기"])[0], 상한=ci(A_bs[m]["기울기"])[1]))
for a, b in [("exaone", "deepseek"), ("exaone", "haiku"), ("deepseek", "haiku")]:
    for key in ("분산비", "정렬"):
        d_ = A_bs[b][key] - A_bs[a][key]
        p = max(2 * min((d_ <= 0).mean(), (d_ >= 0).mean()), 1 / B)
        xrows.append(dict(지표=key, 비교=f"{b} − {a}", 차이=float(A_pt[b][key] - A_pt[a][key]),
                          하한=ci(d_)[0], 상한=ci(d_)[1], p=float(min(p, 1))))
xd = pd.DataFrame(xrows)
xd.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색 (보정 없음)] 기울기, 모델 간 짝지은 차이")
log(xd.round(4).to_string(index=False))
log("\n참고: DeepSeek Eco 추론 끔(k=50, 20261123-reasoning-explain) 분산비 3.38, 정렬 0.479")
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
rel = HERE.relative_to(ROOT)
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){rel}/out")) or bool(git("ls-files", "--others", "--exclude-standard", "--", str(rel), f":(exclude){rel}/out")),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": [{"path": str(f.relative_to(ROOT)), "sha256_16": hashlib.sha256(f.read_bytes()).hexdigest()[:16]} for f in (NPZ, MANIFEST, CELLS)],
           "params": {"bootstrap": B, "seed": SEED, "items": int(SEL.sum()), "models": MODELS},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
