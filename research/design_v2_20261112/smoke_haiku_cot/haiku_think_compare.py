#!/usr/bin/env python3
"""[Haiku 판] Haiku 생각 지시 medium·high(k=20, 197문항) vs 추론 끔(k=100에서 k=20 재표본). 원본 설명: DeepSeek 추론 효과(끔 → medium, 끔 → high)가 몇 k·몇 문항부터 유의하게 잡히는가 (무료, 기존 원응답 재표본).

지표 정의는 experiments/20261123-reasoning-high-full/run.py 그대로(10셀, 순서형·이진, 잡음 뺀 셀 간 분산비, 셀 편차 정렬).
재표본: 프롬프트마다 원응답 50개(표본 0~49, 무효 포함)에서 k개를 비복원 추출 → 실제로 k번 호출한 것과 같다.
판정: 문항 부트스트랩 95% 구간이 0을 안 포함하면 유의. k<50은 20회 반복해 유의 비율과 차이 평균을 낸다.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "research/design_v2_20261112"
OUT = Path(__file__).resolve().parent / "out_haiku_think_k20"
OUT.mkdir(exist_ok=True)
d = np.load(DESIGN / "analysis/out_final_k100/counts_k100.npz", allow_pickle=True)
H, NEFF, CODES, SPLIT = d["H"], d["NEFF"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES); NI, NC = CODES.shape
items = list(csv.DictReader((DESIGN / "item_manifest_v2.csv").open(encoding="utf-8-sig")))
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
sub = lambda f: np.isin(VARS, [r["var"] for r in csv.DictReader((DESIGN / "subsets" / f).open(encoding="utf-8-sig"))])
S60, S137 = sub("high60.csv"), sub("high_rest137.csv")
SCOPES = {"60문항": S60 & ORD, "197문항": (S60 | S137) & ORD}
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader((DESIGN / "cells.csv").open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J]); W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig")) if r["phase"] == "1"}
NORM = np.zeros(CODES.shape)
for i in range(NI):
    c = CODES[i][MASK[i]]; NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)
RAW = {"끔": ["haiku_p1.jsonl"], "medium": ["haiku_medium_think_197items_p1.jsonl"], "high": ["haiku_high_think_197items_p1.jsonl"]}
SMAX = {"끔": 100, "medium": 20, "high": 20}


def load(files, smax=50):
    """ans[j, i, s] = 보기 위치(0..), 무효 -1, 없음 -2. 표본 0~49, api_error·중복 제외(출처 run.py load)."""
    A = np.full((len(PERSONAS), NI, smax), -2, np.int8)
    for f in files:
        for line in open(ROOT / "raw/v2" / f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("invalid_type") == "api_error" or r["sample_idx"] >= smax or r["probe_id"] not in probes:
                continue
            p = probes[r["probe_id"]]; i, j = VARS.index(p["var"]), PERSONAS.index(p["cell_id"])
            if A[j, i, r["sample_idx"]] != -2:
                continue
            A[j, i, r["sample_idx"]] = list(CODES[i]).index(float(r["parsed"])) if r["valid"] else -1
    return A


def counts(A, k, rng):
    if k < A.shape[-1]:
        idx = np.argsort(rng.random(A.shape), axis=-1)[..., :k]
        A = np.take_along_axis(A, idx, -1)
    return (A[..., None] == np.arange(NC)).sum(-2).astype(float)


def stats(P, n):
    s = (P * NORM).sum(-1); var = (P * (NORM - s[..., None]) ** 2 * MASK).sum(-1)
    return s, var / np.maximum(n, 1)


def parts(s, sig2):
    dv = s - (W[:, None] * s).sum(0)
    return (W[:, None] * dv ** 2).sum(0) - ((W * (1 - W))[:, None] * sig2).sum(0), dv


sH, sigH = stats(H[L1J], NEFF[L1J]); VH, dH = parts(sH, sigH); DH2 = (W[:, None] * dH ** 2).sum(0)


def per_item(C):
    C = C[L1J]; n = C.sum(-1)
    s, sig = stats(C / np.maximum(n[..., None], 1), n); V, dm = parts(s, sig)
    return V, (W[:, None] * dm * dH).sum(0), (W[:, None] * dm ** 2).sum(0)


def metrics(m, ix):
    V, num, dm2 = m
    return {"분산비": V[ix].sum(-1) / VH[ix].sum(-1), "정렬": num[ix].sum(-1) / np.sqrt(dm2[ix].sum(-1) * DH2[ix].sum(-1))}



DATA = {c: load(v, SMAX[c]) for c, v in RAW.items()}
for c, A in DATA.items():
    sub197 = A[:, SCOPES["197문항"] | (S60 | S137)]
    print(c, "없음 비율", float((A[:, S60 | S137] == -2).mean()), "무효 비율", float((A[:, S60 | S137] == -1).mean()))
B, REPS, K = 2000, 20, 20
rng = np.random.default_rng(20261126)
sc = "197문항"; ix = np.where(SCOPES[sc])[0]
BIX = ix[rng.integers(0, len(ix), (B, len(ix)))]
Mt = {c: per_item(counts(DATA[c], K, rng)) for c in ("medium", "high")}
rows, lvl = [], []
for rep in range(REPS):
    Mo = per_item(counts(DATA["끔"], K, rng))
    for c in ("끔", "medium", "high"):
        m = Mo if c == "끔" else Mt[c]
        for key in ("분산비", "정렬"):
            lvl.append(dict(rep=rep, 설정=c, 지표=key, 값=float(metrics(m, ix)[key])))
    for tgt in ("medium", "high"):
        for key in ("분산비", "정렬"):
            bs = metrics(Mt[tgt], BIX)[key] - metrics(Mo, BIX)[key]
            pt = metrics(Mt[tgt], ix)[key] - metrics(Mo, ix)[key]
            p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(rep=rep, 비교=f"{tgt} − 끔", 지표=key, 차이=pt, 하한=lo, 상한=hi, p=p, 유의=bool(lo > 0 or hi < 0)))
for key in ("분산비", "정렬"):                                    # medium → high (재표본 없음)
    bs = metrics(Mt["high"], BIX)[key] - metrics(Mt["medium"], BIX)[key]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(rep=-1, 비교="high − medium", 지표=key, 차이=float(metrics(Mt["high"], ix)[key] - metrics(Mt["medium"], ix)[key]), 하한=lo, 상한=hi,
                     p=max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B), 유의=bool(lo > 0 or hi < 0)))
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT / "reps.csv", index=False, encoding="utf-8-sig")
summ = df.groupby(["비교", "지표"]).agg(차이=("차이", "mean"), 하한=("하한", "mean"), 상한=("상한", "mean"), p중앙=("p", "median"), 유의비율=("유의", "mean"), 반복=("rep", "count")).reset_index()
summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
lv = pd.DataFrame(lvl).groupby(["설정", "지표"])["값"].mean().unstack(); lv.to_csv(OUT / "levels.csv", encoding="utf-8-sig")
print("[수준, 끔은 k=20 재표본 20회 평균]"); print(lv.round(3).to_string())
print("[차이]"); print(summ.round(4).to_string(index=False))
(OUT / "run.json").write_text(json.dumps({"script": Path(__file__).name, "B": B, "reps": REPS, "k": K, "seed": 20261126, "n_items": int(len(ix)),
    "raw": RAW}, ensure_ascii=False, indent=1))
