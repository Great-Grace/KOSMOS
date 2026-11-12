#!/usr/bin/env python3
"""설계 v2 1단계: DeepSeek Ultrafast vs Haiku 5.5 비교 (탐색용, test 분할 제외).

입력: raw/v2/{deepseek_ultrafast,haiku}_p1_counts.csv, inventory/kgss_clean.parquet (YEAR==2023),
      item_manifest_v2.csv, cells.csv
출력: analysis/out/ 아래 CSV와 summary.json

- test 49문항은 DESIGN.md 6장에 따라 열지 않는다(calibration+validation 197문항만).
- 쌍별 순위 일치율의 동점 처리 규칙은 아직 미정이라 두 규칙(A: 사람 평균이 다른 모든 쌍,
  B: 사람 차이가 2SE를 넘는 쌍)을 모두 계산한다. 잠정 값이다.

실행 (KOSMOS 루트):
  uv run --no-project --with pandas --with pyarrow python research/design_v2_20261112/analysis/compare_deepseek_haiku.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "research/design_v2_20261112"
OUT = DESIGN / "analysis/out"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"deepseek": "raw/v2/deepseek_ultrafast_p1_counts.csv", "haiku": "raw/v2/haiku_p1_counts.csv"}
ALPHA = 0.5
T_GRID = np.round(np.arange(0.5, 6.01, 0.05), 2)
rng = np.random.default_rng(20261112)


def tvd(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def ent(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def smooth(c, T=1.0):
    p = (np.asarray(c, float) + ALPHA) ** (1.0 / T)
    return p / p.sum()


def raw(c):
    c = np.asarray(c, float)
    return c / c.sum() if c.sum() > 0 else np.full(len(c), 1 / len(c))


# ---------- 사람 정답지 ----------
man = pd.read_csv(DESIGN / "item_manifest_v2.csv", encoding="utf-8-sig")
man["codes_l"] = man["codes"].apply(json.loads)
man = man[man["split_60_20_20"] != "test"].copy()
ITEMS = man.set_index("var")
cells = pd.read_csv(DESIGN / "cells.csv", encoding="utf-8-sig")
cells = cells[cells["level"] == "L1"].copy()

df = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet")
df = df[df["YEAR"] == 2023].copy()
df["band"] = pd.cut(df["AGE"], [18, 30, 40, 50, 60, 200], right=False,
                    labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str)
SEXMAP = {"남자": 1.0, "여자": 2.0}
cells["sexc"] = cells["sex"].map(SEXMAP)
cells["band"] = cells["age_band"].replace({"60-": "60+"})


def wdist(sub, var, codes):
    s = sub[[var, "FINALWT"]].dropna()
    s = s[s[var].isin([float(c) for c in codes]) & (s["FINALWT"] > 0)]
    w = s["FINALWT"].to_numpy()
    p = np.array([w[s[var].to_numpy() == float(c)].sum() for c in codes]) / w.sum()
    # 평균과 Kish 유효 n 기반 SE
    x = s[var].to_numpy()
    m = float((w * x).sum() / w.sum())
    var_w = float((w * (x - m) ** 2).sum() / w.sum())
    neff = w.sum() ** 2 / (w ** 2).sum()
    return p, m, (var_w / neff) ** 0.5, neff


human_pop, human_cell = {}, {}
for v, r in ITEMS.iterrows():
    human_pop[v] = wdist(df, v, r.codes_l)
    for _, c in cells.iterrows():
        sub = df[(df["SEX"] == c.sexc) & (df["band"] == c.band)]
        human_cell[(v, c.cell_id)] = wdist(sub, v, r.codes_l)
bands = cells["band"].unique().tolist()
assert len(bands) == 5 and "nan" not in bands, bands

# ---------- 모델 개수 ----------
cnt = {}
inval = []
for m, path in MODELS.items():
    d = pd.read_csv(ROOT / path, encoding="utf-8-sig")
    d["counts"] = d["counts_json"].apply(json.loads)
    for _, r in d.iterrows():
        cnt[(m, r["level"], r["cell_id"], r["var"])] = r["counts"]
    g = d.groupby("level")[["n_total", "n_valid", "n_refusal", "n_other_invalid"]].sum()
    g["model"] = m
    inval.append(g.reset_index())
inval = pd.concat(inval)
inval["valid_rate"] = inval["n_valid"] / inval["n_total"]
inval.to_csv(OUT / "validity_by_level.csv", index=False)

# ---------- 1. 모집단 재현 (c0, L0, L1 합성) ----------
pop_share = cells.set_index("cell_id")["pop_share"]
pop_share = pop_share / pop_share.sum()
rows = []
for m in MODELS:
    for v, r in ITEMS.iterrows():
        h = human_pop[v][0]
        k = len(r.codes_l)
        conds = {
            "c0": cnt[(m, "c0", "c0", v)],
            "L0": cnt[(m, "L0", "L0", v)],
        }
        for lv, c in conds.items():
            ps, pr = smooth(c), raw(c)
            rows.append(dict(model=m, level=lv, var=v, split=r.split_60_20_20, scale=r.scale_type, topic=r.topic,
                             sens=r.sensitivity_level, k=k, n_valid=int(sum(c)), tvd=tvd(ps, h), tvd_raw=tvd(pr, h),
                             tvd_unif=tvd(np.full(k, 1 / k), h), ent_ratio=ent(pr) / max(ent(h), 1e-9),
                             top_share=float(pr.max()), mode_match=int(np.argmax(pr) == np.argmax(h)),
                             counts=json.dumps(c)))
        # L1 셀 분포를 인구 비중으로 합성
        mix = sum(pop_share[cid] * smooth(cnt[(m, "L1", cid, v)]) for cid in cells.cell_id)
        mixr = sum(pop_share[cid] * raw(cnt[(m, "L1", cid, v)]) for cid in cells.cell_id)
        rows.append(dict(model=m, level="L1agg", var=v, split=r.split_60_20_20, scale=r.scale_type, topic=r.topic,
                         sens=r.sensitivity_level, k=k, n_valid=None, tvd=tvd(mix, h), tvd_raw=tvd(mixr, h),
                         tvd_unif=tvd(np.full(k, 1 / k), h), ent_ratio=ent(mixr) / max(ent(h), 1e-9),
                         top_share=float(mixr.max()), mode_match=int(np.argmax(mixr) == np.argmax(h)), counts=""))
pop = pd.DataFrame(rows)
pop.to_csv(OUT / "population_item_level.csv", index=False)


def boot_ci(x, n=5000):
    x = np.asarray(x)
    b = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


summ_pop = (pop.groupby(["model", "level"])
            .agg(tvd=("tvd", "mean"), tvd_raw=("tvd_raw", "mean"), tvd_unif=("tvd_unif", "mean"),
                 ent_ratio=("ent_ratio", "median"), top_share=("top_share", "mean"),
                 collapse_ge95=("top_share", lambda s: float((s >= 0.95).mean())),
                 mode_match=("mode_match", "mean"),
                 beat_unif=("tvd", lambda s: np.nan))
            .reset_index())
summ_pop["beat_unif"] = [float((pop[(pop.model == a) & (pop.level == b)].eval("tvd < tvd_unif")).mean())
                         for a, b in zip(summ_pop.model, summ_pop.level)]
summ_pop.to_csv(OUT / "population_summary.csv", index=False)

# 모델 간 쌍대 차이 (같은 문항): Haiku − DeepSeek
paired = []
for lv in ["c0", "L0", "L1agg"]:
    a = pop[(pop.model == "deepseek") & (pop.level == lv)].set_index("var")["tvd"]
    b = pop[(pop.model == "haiku") & (pop.level == lv)].set_index("var")["tvd"]
    d = (b - a).loc[a.index]
    paired.append(dict(level=lv, mean_diff_haiku_minus_ds=float(d.mean()), ci95=boot_ci(d.values),
                       haiku_better_share=float((d < 0).mean())))
# 두 모델 분포끼리 거리
mm = []
for lv in ["c0", "L0"]:
    for v in ITEMS.index:
        mm.append(dict(level=lv, var=v, tvd_ds_haiku=tvd(raw(cnt[("deepseek", lv, lv, v)]), raw(cnt[("haiku", lv, lv, v)])),
                       same_mode=int(np.argmax(cnt[("deepseek", lv, lv, v)]) == np.argmax(cnt[("haiku", lv, lv, v)]))))
mm = pd.DataFrame(mm)
mm.to_csv(OUT / "model_vs_model.csv", index=False)

# 하위 집단별 (L0 기준)
def sub_table(col):
    t = pop[pop.level.isin(["c0", "L0"])].pivot_table(index=[col], columns=["model", "level"], values="tvd", aggfunc="mean")
    t["n_items"] = pop[(pop.level == "L0") & (pop.model == "haiku")].groupby(col).size()
    return t


pop["sens_bin"] = pd.cut(pop["sens"], [-1, 1.5, 2.0, 9], labels=["<1.5", "1.5-2", ">=2"])
sub_topic = sub_table("topic")
sub_scale = sub_table("scale")
sub_sens = sub_table("sens_bin")
sub_topic.to_csv(OUT / "by_topic.csv")
sub_scale.to_csv(OUT / "by_scale.csv")
sub_sens.to_csv(OUT / "by_sensitivity.csv")

# ---------- 2. 한 줄의 효과 (c0 → L0) ----------
eff = []
for m in MODELS:
    a = pop[(pop.model == m) & (pop.level == "c0")].set_index("var")
    b = pop[(pop.model == m) & (pop.level == "L0")].set_index("var")
    d = b["tvd"] - a["tvd"]
    shift = [tvd(raw(cnt[(m, "c0", "c0", v)]), raw(cnt[(m, "L0", "L0", v)])) for v in a.index]
    eff.append(dict(model=m, mean_tvd_change=float(d.mean()), ci95=boot_ci(d.values), improved_share=float((d < 0).mean()),
                    mean_shift_c0_L0=float(np.mean(shift)), mode_changed=float(np.mean(
                        [np.argmax(cnt[(m, 'c0', 'c0', v)]) != np.argmax(cnt[(m, 'L0', 'L0', v)]) for v in a.index])),
                    refusal_c0=int(inval[(inval.model == m) & (inval.level == "c0")]["n_refusal"].sum()),
                    refusal_L0=int(inval[(inval.model == m) & (inval.level == "L0")]["n_refusal"].sum())))

# ---------- 3. 집단 구분 (L1) ----------
cid = cells.cell_id.tolist()
cell_rows, pair_rows, spread_rows, within_rows = [], [], [], []
for m in MODELS:
    for v, r in ITEMS.iterrows():
        codes = np.array(r.codes_l, float)
        hpop = human_pop[v][0]
        for c in cid:
            h, hm, hse, neff = human_cell[(v, c)]
            pm = smooth(cnt[(m, "L1", c, v)])
            cell_rows.append(dict(model=m, var=v, cell=c, tvd=tvd(pm, h), tvd_popbase=tvd(hpop, h)))
            # 셀 안 보기 순위: 보기 쌍마다 순서가 사람과 같은가 (사람 동률 쌍 제외, 모델 동률 0.5)
            pr = raw(cnt[(m, "L1", c, v)])
            agree, base, n = 0.0, 0.0, 0
            for i in range(len(h)):
                for j in range(i + 1, len(h)):
                    hs = np.sign(h[i] - h[j])
                    if hs == 0:
                        continue
                    ms = np.sign(pr[i] - pr[j])
                    bs = np.sign(hpop[i] - hpop[j])
                    agree += 1.0 if ms == hs else (0.5 if ms == 0 else 0.0)
                    base += 1.0 if bs == hs else (0.5 if bs == 0 else 0.0)
                    n += 1
            if n:
                within_rows.append(dict(model=m, var=v, cell=c, agree=agree / n, base=base / n))
        if r.scale_type not in ("ordinal_monotone", "binary"):
            continue
        hmeans = {c: human_cell[(v, c)][1] for c in cid}
        hses = {c: human_cell[(v, c)][2] for c in cid}
        mmeans = {c: float((codes * raw(cnt[(m, "L1", c, v)])).sum()) for c in cid}
        hm_arr = np.array([hmeans[c] for c in cid])
        mm_arr = np.array([mmeans[c] for c in cid])
        spread_rows.append(dict(model=m, var=v, human_sd=float(hm_arr.std()), model_sd=float(mm_arr.std()),
                                corr=float(np.corrcoef(hm_arr, mm_arr)[0, 1]) if mm_arr.std() > 0 else np.nan,
                                scale_range=float(codes.max() - codes.min())))
        for i in range(len(cid)):
            for j in range(i + 1, len(cid)):
                a, b = cid[i], cid[j]
                hd = hmeans[a] - hmeans[b]
                md = mmeans[a] - mmeans[b]
                if hd == 0:
                    continue
                sc = 1.0 if np.sign(md) == np.sign(hd) else (0.5 if md == 0 else 0.0)
                sig = abs(hd) > 2 * (hses[a] ** 2 + hses[b] ** 2) ** 0.5
                same_sex = a.split("_")[1] == b.split("_")[1]
                same_age = a.split("_")[2] == b.split("_")[2]
                pair_rows.append(dict(model=m, var=v, a=a, b=b, score=sc, sig=sig, model_tie=md == 0,
                                      kind="sex" if same_age else ("age" if same_sex else "both")))
cellt = pd.DataFrame(cell_rows)
pairs = pd.DataFrame(pair_rows)
spread = pd.DataFrame(spread_rows)
within = pd.DataFrame(within_rows)
cellt.to_csv(OUT / "L1_cell_tvd.csv", index=False)
pairs.to_csv(OUT / "L1_pairs.csv", index=False)
spread.to_csv(OUT / "L1_spread.csv", index=False)


def pair_summary(p):
    out = {}
    for name, q in [("A_all", p), ("B_sig2se", p[p.sig])]:
        item_mean = q.groupby("var")["score"].mean()
        out[name] = dict(pairs=int(len(q)), agree=float(q["score"].mean()), item_mean=float(item_mean.mean()),
                         ci95_items=boot_ci(item_mean.values), model_tie_share=float(q["model_tie"].mean()),
                         by_kind={k: float(g["score"].mean()) for k, g in q.groupby("kind")})
    return out


L1 = {}
for m in MODELS:
    L1[m] = dict(
        cell_tvd=float(cellt[cellt.model == m]["tvd"].mean()),
        cell_tvd_popbase=float(cellt[cellt.model == m]["tvd_popbase"].mean()),
        pairs=pair_summary(pairs[pairs.model == m]),
        spread_ratio_median=float((spread[spread.model == m].model_sd / spread[spread.model == m].human_sd).median()),
        spread_corr_median=float(spread[spread.model == m]["corr"].median()),
        spread_corr_nan_share=float(spread[spread.model == m]["corr"].isna().mean()),
        within_agree=float(within[within.model == m]["agree"].mean()),
        within_base=float(within[within.model == m]["base"].mean()),
    )

# ---------- 4. T 보정 (calibration에서 고르고 validation에서 확인) ----------
calib = {}
for m in MODELS:
    for lv in ["c0", "L0"]:
        def mean_tvd(T, split):
            vs = ITEMS.index[ITEMS.split_60_20_20 == split]
            return float(np.mean([tvd(smooth(cnt[(m, lv, lv, v)], T), human_pop[v][0]) for v in vs]))
        curve = [mean_tvd(T, "calibration") for T in T_GRID]
        Tb = float(T_GRID[int(np.argmin(curve))])
        lam_grid = np.round(np.arange(0, 1.001, 0.02), 2)

        def mix_tvd(lam, split):
            vs = ITEMS.index[ITEMS.split_60_20_20 == split]
            out = []
            for v in vs:
                p = smooth(cnt[(m, lv, lv, v)])
                k = len(p)
                out.append(tvd((1 - lam) * p + lam / k, human_pop[v][0]))
            return float(np.mean(out))
        lc = [mix_tvd(l, "calibration") for l in lam_grid]
        lb = float(lam_grid[int(np.argmin(lc))])
        vs = ITEMS.index[ITEMS.split_60_20_20 == "validation"]
        calib[f"{m}_{lv}"] = dict(T=Tb, calib_T1=mean_tvd(1.0, "calibration"), calib_Tbest=min(curve),
                                  val_T1=mean_tvd(1.0, "validation"), val_Tbest=mean_tvd(Tb, "validation"),
                                  lam=lb, val_lam=mix_tvd(lb, "validation"),
                                  val_unif=float(np.mean([tvd(np.full(len(human_pop[v][0]), 1 / len(human_pop[v][0])), human_pop[v][0]) for v in vs])))

summary = dict(
    note="탐색 분석. test 49문항 제외(197문항). 쌍별 순위 규칙 미정이라 A/B 병기.",
    n_items=int(len(ITEMS)),
    population=summ_pop.to_dict("records"),
    paired_haiku_minus_deepseek=paired,
    model_vs_model={lv: dict(mean_tvd=float(g.tvd_ds_haiku.mean()), same_mode=float(g.same_mode.mean()))
                    for lv, g in mm.groupby("level")},
    persona_line_effect=eff,
    L1=L1,
    calibration=calib,
    validity=inval.to_dict("records"),
)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
print("\n[topic]\n", sub_topic.round(3).to_string())
print("\n[scale]\n", sub_scale.round(3).to_string())
print("\n[sens]\n", sub_sens.round(3).to_string())
