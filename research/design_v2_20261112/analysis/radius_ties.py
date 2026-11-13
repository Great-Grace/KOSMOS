"""RADIUS식 동점 규칙으로 최빈값 정확도와 페르소나 간 쌍별 순위를 다시 계산한다.

규칙 (Łajewska et al., 2026, RADIUS의 TRM을 따름)
- 사람 분포를 부트스트랩(B=1000)해 보기별 비율의 95% 구간을 구한다.
- 사람 1위 보기의 구간과 겹치는 보기를 모두 "1위 그룹"으로 본다.
- 모델 1위 보기가 1위 그룹 안에 있으면 정답(모델 동점이면 동점 보기 중 그룹 안 비율).
- 페르소나 간 쌍: 사람 쪽 평균점수 차이의 95% 구간이 0을 포함하면 그 쌍은 뺀다.

사람 부트스트랩은 응답자 재추출 대신 문항·페르소나별 유효 표본 크기
n_eff = (Σw)² / Σw² (FINALWT, Kish)로 다항 재추출해 근사한다.
모델 분포는 보정 전 원비율(k=100)이다. 지수 보정은 보기 순서를 바꾸지 않아 결과가 같다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "inventory/kgss_clean.parquet").exists())
DESIGN = ROOT / "research/design_v2_20261112"
OUT = DESIGN / "analysis/out_calib"
MODELS = {"deepseek": ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl", "haiku": ROOT / "raw/v2/haiku_p1.jsonl"}
B = 1000
rng = np.random.default_rng(20261112)

# 문항 축
items = pd.read_csv(DESIGN / "item_manifest_v2.csv", encoding="utf-8-sig")
items["codes"] = items["codes"].apply(json.loads)
I, K = len(items), items["codes"].str.len().max()
CODES = np.full((I, K), np.nan)
for i, c in enumerate(items["codes"]):
    CODES[i, :len(c)] = c
MASK = ~np.isnan(CODES)
NOPT = MASK.sum(1)
ORDINAL = items["scale_type"].isin(["ordinal_monotone", "binary"]).to_numpy()
TEST = (items["split_60_20_20"] == "test").to_numpy()

# 페르소나 축과 사람 분포·유효 표본
cells = pd.read_csv(DESIGN / "cells.csv", encoding="utf-8-sig").query("level == 'L1'").reset_index(drop=True)
PERSONAS = ["c0", "L0", *cells["cell_id"]]
kgss = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet").query("YEAR == 2023")
kgss = kgss.assign(age_band=pd.cut(kgss["AGE"], [18, 30, 40, 50, 60, 200], right=False,
                                   labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str))
SEX = {"남자": 1, "여자": 2}
resp = {"c0": kgss, "L0": kgss} | {c.cell_id: kgss[(kgss["SEX"] == SEX[c.sex]) & (kgss["age_band"] == c.age_band)]
                                   for c in cells.itertuples()}

H = np.zeros((len(PERSONAS), I, K))
NEFF = np.zeros((len(PERSONAS), I), dtype=int)
for j, p in enumerate(PERSONAS):
    for i, v in enumerate(items["var"]):
        d = resp[p][[v, "FINALWT"]].dropna()
        d = d[d[v].isin(CODES[i, MASK[i]])]
        w = d.groupby(v)["FINALWT"].sum().reindex(CODES[i, MASK[i]]).fillna(0).to_numpy()
        H[j, i, :NOPT[i]] = w / w.sum()
        NEFF[j, i] = round(d["FINALWT"].sum() ** 2 / (d["FINALWT"] ** 2).sum())

# 모델 개수 텐서 (노트북과 같은 규칙: api_error 제외, 같은 표본 번호 중복은 처음 것)
P_IDX = {p: j for j, p in enumerate(PERSONAS)}
I_IDX = {v: i for i, v in enumerate(items["var"])}
K_IDX = [{c: k for k, c in enumerate(cs)} for cs in items["codes"]]


def load_counts(path):
    C = np.zeros((len(PERSONAS), I, K))
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


R = {m: (lambda C: C / np.maximum(C.sum(-1, keepdims=True), 1))(load_counts(p)) for m, p in MODELS.items()}

# 사람 부트스트랩 [B, p, i, k]
Hb = np.zeros((B, *H.shape))
for j in range(len(PERSONAS)):
    for i in range(I):
        Hb[:, j, i, :NOPT[i]] = rng.multinomial(NEFF[j, i], H[j, i, :NOPT[i]], size=B) / NEFF[j, i]
lo, hi = np.percentile(Hb, [2.5, 97.5], axis=0)                       # [p, i, k]

top_h = np.where(MASK, H, -np.inf).argmax(-1)                          # [p, i]
lo_top = np.take_along_axis(lo, top_h[..., None], -1)[..., 0]
GROUP = MASK & (hi >= lo_top[..., None])                               # 사람 1위 그룹 [p, i, k]
GROUP_SIZE = GROUP.sum(-1)


def trm(P):
    # 모델 1위(동점 포함)가 사람 1위 그룹 안에 있는 비율
    m = np.where(MASK, P, -np.inf)
    is_top = MASK & (m == m.max(-1, keepdims=True))
    return (is_top & GROUP).sum(-1) / is_top.sum(-1)


def strict(P):
    return (np.where(MASK, P, -np.inf).argmax(-1) == top_h).astype(float)


def by_level(x, sel):
    # [p, i] → c0, L0, L1cell(셀·문항 평균)
    x = x[:, sel]
    return [x[0].mean(), x[1].mean(), x[2:].mean()]


LEVELS = ["c0", "L0", "L1cell"]
rows = []
for split, sel in {"전체 246": np.ones(I, bool), "test 49": TEST}.items():
    add = lambda name, x: rows.append(dict(split=split, 기준=name, **dict(zip(LEVELS, by_level(x, sel)))))
    for m, P in R.items():
        add(f"{m} · 엄격(1위 일치)", strict(P))
        add(f"{m} · RADIUS 동점 규칙", trm(P))
    add("우연 · 엄격 (1/보기 수)", np.broadcast_to(1 / NOPT, top_h.shape))
    add("우연 · RADIUS (그룹 크기/보기 수)", GROUP_SIZE / NOPT)
    add("항상 1번 · RADIUS", GROUP[..., 0].astype(float))
    pop = np.broadcast_to(H[0], H.shape)                               # 모집단 사람 분포를 모든 페르소나에
    add("모집단 사람 분포 · RADIUS", trm(pop))
    add("사람 재표집 · RADIUS (상한)", np.mean([trm(Hb[b]) for b in range(0, B, 10)], 0))
    add("사람 1위 그룹 크기 (평균)", GROUP_SIZE.astype(float))
mode_tbl = pd.DataFrame(rows)

# 페르소나 간 쌍별 순위: 사람 차이 구간이 0을 포함하는 쌍 제외
L1 = slice(2, None)
up = np.triu(np.ones((10, 10), bool), 1)
sex = cells["sex"].to_numpy(); age = cells["age_band"].to_numpy()
KIND = {"전체": np.ones((10, 10), bool), "성별만 다름": age[:, None] == age[None, :], "연령만 다름": sex[:, None] == sex[None, :]}
score = lambda P: (P * np.nan_to_num(CODES)).sum(-1)                   # 평균점수 [..., i]
cd = lambda x: x[..., :, None, :] - x[..., None, :, :]                 # 셀 쌍 차이 [..., 10, 10, i]
dh = cd(score(H[L1]))
dhb = cd(score(Hb[:, L1]))
dlo, dhi = np.percentile(dhb, [2.5, 97.5], axis=0)
SIG = (dlo > 0) | (dhi < 0)                                            # 사람 쪽 차이가 표집 오차 밖


def auc(dm, valid):
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    return (s * valid).sum() / valid.sum(), int(valid.sum())


def auc_ci(dm, valid, nb=2000):
    # 문항 단위 부트스트랩 95% 구간 (같은 문항의 셀 쌍은 함께 뽑는다)
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    hit, n = (s * valid).sum((0, 1)), valid.sum((0, 1)).astype(float)   # 문항별 [i]
    idx = rng.integers(0, I, (nb, I))
    return np.percentile(hit[idx].sum(1) / np.maximum(n[idx].sum(1), 1), [2.5, 97.5])


pair_rows = []
for kind, km in KIND.items():
    base = up[..., None] & km[..., None] & ORDINAL[None, None, :] & (dh != 0)
    for m, P in R.items():
        dm = cd(score(P[L1]))
        a_all, n_all = auc(dm, base)
        a_sig, n_sig = auc(dm, base & SIG)
        (l1, h1), (l2, h2) = auc_ci(dm, base), auc_ci(dm, base & SIG)
        pair_rows.append(dict(쌍=kind, model=m, AUC_전체쌍=a_all, 구간_전체=f"[{l1:.3f}, {h1:.3f}]", 쌍수_전체=n_all,
                              AUC_유의쌍만=a_sig, 구간_유의=f"[{l2:.3f}, {h2:.3f}]", 쌍수_유의=n_sig))
pair_tbl = pd.DataFrame(pair_rows)

mode_tbl.to_csv(OUT / "v2_mode_radius.csv", index=False)
pair_tbl.to_csv(OUT / "v2_pairs_radius.csv", index=False)
pd.set_option("display.width", 200)
print(mode_tbl.round(3).to_string(index=False))
print()
print(pair_tbl.round(3).to_string(index=False))
print("\n사람 쪽 유의 쌍 비율(서열·이진, 전체 셀 쌍):",
      round((SIG & up[..., None] & ORDINAL).sum() / (up[..., None] & ORDINAL[None, None, :]).sum(), 3))
print("n_eff 범위 L1:", NEFF[2:].min(), "~", NEFF[2:].max(), "| 모집단:", NEFF[0].min(), "~", NEFF[0].max())
