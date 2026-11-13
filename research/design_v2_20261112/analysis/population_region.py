"""KGSS 표본에서 모분포의 95% 신뢰영역을 세우고, 모델 분포와 다시 비교한다.

계산 과정 설명: analysis/POPULATION_ESTIMATION.md

1. 페르소나 × 문항마다 유효 표본 n_eff = (Σw)² / Σw² (FINALWT, Kish)
2. Goodman(1965) 동시 95% 구간: 보기 K개가 한꺼번에 구간 안에 들 확률이 95%
3. 지표
   - TVD(P, Q̂): 기존 값
   - 최소 TVD: 신뢰영역 안의 Q 중 P에 가장 가까운 것과의 거리 = max(모자란 양, 넘치는 양)
   - 포함률: P가 신뢰영역 안에 완전히 들어간 (페르소나, 문항)의 비율
   - 잡음 바닥: 모델이 완벽해도(P = Q) 표집 잡음만으로 생기는 TVD 기댓값
   - 편향 보정 TVD: 2·TVD(P̂, Q̂) − 평균 TVD(P̂*, Q̂*) (사람·모델 양쪽 재표집 부트스트랩)
"""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "inventory/kgss_clean.parquet").exists())
DESIGN = ROOT / "research/design_v2_20261112"
OUT = DESIGN / "analysis/out_calib"
MODELS = {"deepseek": ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl", "haiku": ROOT / "raw/v2/haiku_p1.jsonl"}
CONF = 0.95
B = 300
rng = np.random.default_rng(20261112)

# ── 문항 축 ──────────────────────────────────────────────
items = pd.read_csv(DESIGN / "item_manifest_v2.csv", encoding="utf-8-sig")
items["codes"] = items["codes"].apply(json.loads)
I, K = len(items), items["codes"].str.len().max()
CODES = np.full((I, K), np.nan)
for i, c in enumerate(items["codes"]):
    CODES[i, :len(c)] = c
MASK = ~np.isnan(CODES)
NOPT = MASK.sum(1)
UNIF = MASK / NOPT[:, None]
SPLIT = {"test": (items["split_60_20_20"] == "test").to_numpy(), "전체": np.ones(I, bool)}

# ── 페르소나 축, 사람 분포 Q̂, 유효 표본 n_eff ─────────────
cells = pd.read_csv(DESIGN / "cells.csv", encoding="utf-8-sig").query("level == 'L1'").reset_index(drop=True)
PERSONAS = ["c0", "L0", *cells["cell_id"]]
kgss = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet").query("YEAR == 2023")
kgss = kgss.assign(age_band=pd.cut(kgss["AGE"], [18, 30, 40, 50, 60, 200], right=False,
                                   labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str))
SEX = {"남자": 1, "여자": 2}
resp = {"c0": kgss, "L0": kgss} | {c.cell_id: kgss[(kgss["SEX"] == SEX[c.sex]) & (kgss["age_band"] == c.age_band)]
                                   for c in cells.itertuples()}
NP = len(PERSONAS)
Qhat = np.zeros((NP, I, K))
NRAW = np.zeros((NP, I), dtype=int)
NEFF = np.zeros((NP, I))
for j, p in enumerate(PERSONAS):
    for i, v in enumerate(items["var"]):
        d = resp[p][[v, "FINALWT"]].dropna()
        d = d[d[v].isin(CODES[i, MASK[i]])]
        w = d.groupby(v)["FINALWT"].sum().reindex(CODES[i, MASK[i]]).fillna(0).to_numpy()
        Qhat[j, i, :NOPT[i]] = w / w.sum()
        NRAW[j, i] = len(d)
        NEFF[j, i] = d["FINALWT"].sum() ** 2 / (d["FINALWT"] ** 2).sum()

# ── Goodman 동시 신뢰구간 ────────────────────────────────
# A = χ²(1) 분포의 1 − α/K 분위수 = z² (z: 표준정규의 1 − α/(2K) 분위수)
alpha = 1 - CONF
Z = np.array([NormalDist().inv_cdf(1 - alpha / (2 * k)) for k in NOPT])     # [i]
A = (Z ** 2)[None, :, None]                                                 # [1, i, 1]
n = NEFF[..., None]
root = np.sqrt(A * (A + 4 * n * Qhat * (1 - Qhat)))
LO = np.where(MASK, (A + 2 * n * Qhat - root) / (2 * (n + A)), 0.0)
HI = np.where(MASK, (A + 2 * n * Qhat + root) / (2 * (n + A)), 0.0)


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def tvd_min(p):
    # 신뢰영역(구간 상자 ∩ 합 1) 안에서 p에 가장 가까운 분포까지의 TVD
    short = np.clip(LO - p, 0, None).sum(-1)      # 구간 아래로 모자란 양
    over = np.clip(p - HI, 0, None).sum(-1)       # 구간 위로 넘치는 양
    return np.maximum(short, over)


# ── 모델 개수와 보정 ─────────────────────────────────────
P_IDX = {p: j for j, p in enumerate(PERSONAS)}
I_IDX = {v: i for i, v in enumerate(items["var"])}
K_IDX = [{c: k for k, c in enumerate(cs)} for cs in items["codes"]]


def load_counts(path):
    C = np.zeros((NP, I, K))
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or key in seen:
            continue
        seen.add(key)
        if r["valid"]:
            i = I_IDX[r["var"]]
            C[P_IDX[r["cell_id"]], i, K_IDX[i][r["parsed"]]] += 1
    return C


frozen = json.loads((OUT / "frozen_beta.json").read_text())
LEVEL_OF = np.array(["c0", "L0", *["L1cell"] * 10])
LEVELS = ["c0", "L0", "L1cell"]


def calibrate(C, beta):
    # p ∝ (c + 1)^β (라플라스), β는 페르소나 단계별
    x = (C + MASK) ** beta[:, None, None] * MASK
    return x / x.sum(-1, keepdims=True)


def resample_counts(C):
    out = np.zeros_like(C)
    k = C.sum(-1)
    for j in range(NP):
        for i in range(I):
            if k[j, i] > 0:
                out[j, i, :NOPT[i]] = rng.multinomial(int(k[j, i]), C[j, i, :NOPT[i]] / k[j, i])
    return out


def resample_human():
    out = np.zeros_like(Qhat)
    for j in range(NP):
        for i in range(I):
            m = max(int(round(NEFF[j, i])), 1)
            out[j, i, :NOPT[i]] = rng.multinomial(m, Qhat[j, i, :NOPT[i]]) / m
    return out


Cs = {m: load_counts(p) for m, p in MODELS.items()}
BETA = {m: np.array([frozen["beta"][f"laplace|{m}|{lv}"] for lv in LEVEL_OF]) for m in MODELS}

# 비교할 분포: 모델 β*·β=1, 기준선 균등분포·모집단 사람 분포 복사
cands = {}
for m, C in Cs.items():
    cands[f"{m} β*"] = ("model", m, BETA[m])
    cands[f"{m} β=1"] = ("model", m, np.ones(NP))
cands["균등분포"] = ("fixed", np.broadcast_to(UNIF, Qhat.shape), None)
cands["모집단 사람 분포 복사"] = ("fixed", np.broadcast_to(Qhat[0], Qhat.shape), None)


def point(c):
    kind, a, b = cands[c]
    return calibrate(Cs[a], b) if kind == "model" else np.asarray(a)


# 부트스트랩: 사람·모델 양쪽 재표집 → 편향 보정
boot_sum = {c: np.zeros((NP, I)) for c in cands}
floor_sum = np.zeros((NP, I))
for b in range(B):
    Hs = resample_human()
    floor_sum += tvd(Hs, Qhat)
    Cb = {m: resample_counts(C) for m, C in Cs.items()}
    for c, (kind, a, beta) in cands.items():
        Pb = calibrate(Cb[a], beta) if kind == "model" else np.asarray(a)
        boot_sum[c] += tvd(Pb, Hs)
FLOOR = floor_sum / B

per = {}   # 지표 → 후보 → [p, i]
for c in cands:
    P = point(c)
    t = tvd(P, Qhat)
    per.setdefault("TVD(P, Q̂)", {})[c] = t
    per.setdefault("편향 보정 TVD", {})[c] = 2 * t - boot_sum[c] / B
    per.setdefault("최소 TVD (95% 영역)", {})[c] = tvd_min(P)
    per.setdefault("포함률", {})[c] = (tvd_min(P) == 0).astype(float)


def level_mean(x, sel):
    x = x[:, sel]
    return np.array([x[0].mean(), x[1].mean(), x[2:].mean()])


def level_ci(x, sel, nb=2000):
    x = x[:, sel]
    pi = np.stack([x[0], x[1], x[2:].mean(0)])      # [단계, 문항]
    idx = rng.integers(0, pi.shape[1], (nb, pi.shape[1]))
    bs = pi[:, idx].mean(-1)                         # [단계, nb]
    return np.percentile(bs, [2.5, 97.5], axis=1)    # [2, 단계]


rows = []
for split, sel in SPLIT.items():
    for met, d in per.items():
        for c, x in d.items():
            mean, (lo, hi) = level_mean(x, sel), level_ci(x, sel)
            for l, lv in enumerate(LEVELS):
                rows.append(dict(split=split, 지표=met, 후보=c, 단계=lv, 값=mean[l], lo=lo[l], hi=hi[l]))
    f = level_mean(FLOOR, sel)
    for l, lv in enumerate(LEVELS):
        rows.append(dict(split=split, 지표="잡음 바닥 (사람 쪽)", 후보="—", 단계=lv, 값=f[l], lo=np.nan, hi=np.nan))
res = pd.DataFrame(rows)
res.to_csv(OUT / "v2_population_region.csv", index=False)

# 모델 차이 (편향 보정 TVD, test): Haiku − DeepSeek
diff_rows = []
for met in ["TVD(P, Q̂)", "편향 보정 TVD", "최소 TVD (95% 영역)"]:
    x = per[met]["haiku β*"] - per[met]["deepseek β*"]
    m, (lo, hi) = level_mean(x, SPLIT["test"]), level_ci(x, SPLIT["test"])
    for l, lv in enumerate(LEVELS):
        diff_rows.append(dict(지표=met, 단계=lv, haiku_minus_deepseek=m[l], lo=lo[l], hi=hi[l]))
diffs = pd.DataFrame(diff_rows)
diffs.to_csv(OUT / "v2_population_region_diffs.csv", index=False)

pd.set_option("display.width", 220)
print(res[res.split == "test"].pivot_table(index=["지표", "후보"], columns="단계", values="값", sort=False)
      .reindex(columns=LEVELS).round(3).to_string())
print()
print(diffs.round(3).to_string(index=False))
print("\nn_eff: 모집단", NEFF[0].min().round(0), "~", NEFF[0].max().round(0),
      "| L1", NEFF[2:].min().round(0), "~", NEFF[2:].max().round(0))
print("구간 평균 폭(보기당):", {lv: round(float(level_mean(((HI - LO) * MASK).sum(-1) / NOPT, SPLIT['전체'])[l]), 3)
                         for l, lv in enumerate(LEVELS)})

# ── 문서용 계산 예시: ACPTCULT × 여 18–29 ────────────────
ex_i = I_IDX["ACPTCULT"]; ex_j = P_IDX["L1_2_18-29"]; k = NOPT[ex_i]
Pds = point("deepseek β*")[ex_j, ex_i, :k]
print("\n[예시] ACPTCULT × 여 18–29")
print("  표본 n =", NRAW[ex_j, ex_i], "| n_eff =", round(NEFF[ex_j, ex_i], 1), "| 보기 수 K =", k,
      "| z =", round(Z[ex_i], 4), "| A = z² =", round(Z[ex_i] ** 2, 4))
print("  Q̂ =", Qhat[ex_j, ex_i, :k].round(4))
print("  하한 =", LO[ex_j, ex_i, :k].round(4))
print("  상한 =", HI[ex_j, ex_i, :k].round(4))
print("  모델 개수(DeepSeek L1 이 셀) =", Cs["deepseek"][ex_j, ex_i, :k], "| β* =", BETA["deepseek"][ex_j])
print("  P(β*) =", Pds.round(4))
print("  TVD(P, Q̂) =", round(float(tvd(Pds, Qhat[ex_j, ex_i, :k])), 4),
      "| 최소 TVD =", round(float(tvd_min(point('deepseek β*'))[ex_j, ex_i]), 4),
      "| 잡음 바닥 =", round(float(FLOOR[ex_j, ex_i]), 4),
      "| 편향 보정 =", round(float(per['편향 보정 TVD']['deepseek β*'][ex_j, ex_i]), 4))
