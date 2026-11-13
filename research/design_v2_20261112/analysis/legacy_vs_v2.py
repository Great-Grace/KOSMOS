"""기존 소형 모델(EXAONE-3.5-2.4B, Kanana-1.5-2.1B)을 v2 지표로 다시 계산해 DeepSeek·Haiku와 비교한다.

같은 것: 문항 246개·분할(item_manifest_v2.csv)·FINALWT 가중 정답지·성별×연령 10셀·β 그리드·
         calibration 문항 평균 TVD 기준·RADIUS 동점 규칙·셀 쌍 순위 규칙
다른 것: 소형 모델 분포는 보기 로짓 softmax(정확한 확률, 표집 없음). 보정은 p^β 정규화
         v2 모델은 k=100 개수 + Laplace α=1, (c+1)^β 정규화 (v2_calibration_test.ipynb와 같음)
         프롬프트 문구가 다르다(기존 c0_clean / adult_clean / demo10_clean).
대응: 기존 c0_clean → c0, adult_clean → L0, demo10_clean(성별×연령 10셀) → L1

검증: DeepSeek·Haiku 값이 기존 v2 산출물(out_calib/test_tvd.csv, v2_mode_radius.csv, v2_pairs_radius.csv)과
      같은지 assert 한다. 통과해야 소형 모델 값을 저장한다.

실행: .venv-nb/bin/python research/design_v2_20261112/analysis/legacy_vs_v2.py   (API 호출 없음)
산출: out_calib/legacy_vs_v2_summary.csv, out_calib/legacy_vs_v2_diffs.csv, out_calib/legacy_vs_v2_diagnostics.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "inventory/kgss_clean.parquet").exists())
D = ROOT / "research/design_v2_20261112"
OUT = D / "analysis/out_calib"
RE = ROOT / "reexperiment_20261009"
rng = np.random.default_rng(20261112)

# ---------------- 문항·셀·정답지 (v2와 같은 규칙) ----------------
man = pd.read_csv(D / "item_manifest_v2.csv", encoding="utf-8-sig")
man["codes_l"] = man["codes"].apply(json.loads)
ITEMS = man["var"].tolist(); I = len(ITEMS); K = int(man.codes_l.str.len().max())
CODES = np.full((I, K), np.nan)
for i, cs in enumerate(man.codes_l):
    CODES[i, :len(cs)] = cs
MASK = ~np.isnan(CODES); NOPT = MASK.sum(1)
SPLIT = man.split_60_20_20.to_numpy(); TRAIN, TEST = SPLIT == "calibration", SPLIT == "test"
ORD = man.scale_type.isin(["ordinal_monotone", "binary"]).to_numpy()
IV = {v: i for i, v in enumerate(ITEMS)}
C2K = [{c: k for k, c in enumerate(cs)} for cs in man.codes_l]

cells = pd.read_csv(D / "cells.csv", encoding="utf-8-sig").query("level=='L1'").reset_index(drop=True)
PERS = ["c0", "L0"] + cells.cell_id.tolist(); P = len(PERS); PI = {p: j for j, p in enumerate(PERS)}
L1 = slice(2, P)

kg = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet").query("YEAR == 2023").copy()
kg["band"] = pd.cut(kg.AGE, [18, 30, 40, 50, 60, 200], right=False,
                    labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str)
grp = {"c0": kg, "L0": kg} | {c.cell_id: kg[(kg.SEX == {"남자": 1.0, "여자": 2.0}[c.sex]) & (kg.band == c.age_band)]
                              for c in cells.itertuples()}
H = np.zeros((P, I, K)); NEFF = np.zeros((P, I), int)
for p, g in grp.items():
    for i, v in enumerate(ITEMS):
        x, w = g[v].to_numpy(), g.FINALWT.to_numpy()
        ok = np.isin(x, CODES[i, MASK[i]]) & (w > 0); x, w = x[ok], w[ok]
        H[PI[p], i] = (w[:, None] * (x[:, None] == CODES[i][None])).sum(0) / w.sum()
        NEFF[PI[p], i] = round(w.sum() ** 2 / (w ** 2).sum())


# ---------------- 모델 분포 ----------------
def v2_counts(path):
    C = np.zeros((P, I, K)); seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or key in seen:
            continue
        seen.add(key)
        if r["valid"]:
            i = IV[r["var"]]; C[PI[r["cell_id"]], i, C2K[i][r["parsed"]]] += 1
    return C


def legacy_dist(model):
    X = np.zeros((P, I, K)); filled = np.zeros((P, I), int)
    for sub, how in [("c0_clean", "c0"), ("adult_clean", "L0"), ("demo10_clean", "L1")]:
        probes = {r["probe_id"]: r for r in csv.DictReader(open(RE / sub / "probes.csv", encoding="utf-8-sig"))}
        for r in csv.DictReader(open(RE / sub / "logit_responses.csv", encoding="utf-8-sig")):
            if r["model"] != model or r["var"] not in IV:
                continue
            i = IV[r["var"]]; d = np.array(json.loads(r["model_dist"]), float)
            pr = probes[r["probe_id"]]
            assert json.loads(pr["codes"]) == man.codes_l[i], f"보기 코드 불일치 {r['var']}"
            p = f"L1_{int(float(pr['sex']))}_{pr['age_grp']}" if how == "L1" else how
            X[PI[p], i, :len(d)] = d / d.sum(); filled[PI[p], i] += 1
    assert (filled == 1).all(), "페르소나×문항이 정확히 한 번씩 채워지지 않음"
    return X


MODELS = {
    "EXAONE-3.5-2.4B": ("legacy", legacy_dist("EXAONE-3.5-2.4B")),
    "Kanana-1.5-2.1B": ("legacy", legacy_dist("Kanana-1.5-2.1B")),
    "DeepSeek V4.1 Flash": ("v2", v2_counts(ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl")),
    "Haiku 5.5": ("v2", v2_counts(ROOT / "raw/v2/haiku_p1.jsonl")),
}


def prob(kind, X, beta):
    base = (X + 1.0 * MASK) if kind == "v2" else np.where(MASK, X, 0)
    x = base ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def raw_share(kind, X):
    # 보정·평활 없는 원비율 (최빈값·순위·쏠림 진단용, radius_ties.py와 같음)
    return X if kind == "legacy" else X / np.maximum(X.sum(-1, keepdims=True), 1)


tvd = lambda a, b: 0.5 * np.abs(a - b).sum(-1)
LEVELS = ["c0", "L0", "L1cell"]


def tvd_lv(Pm):  # [3, i]
    return np.stack([tvd(Pm[0], H[0]), tvd(Pm[1], H[1]), tvd(Pm[L1], H[L1]).mean(0)])


BG = np.round(np.arange(0.05, 1.5001, 0.005), 3)


def boot(x, B=5000):
    idx = rng.integers(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])


# ---------------- RADIUS 동점 그룹·셀 쌍 유의성 (radius_ties.py와 같은 규칙) ----------------
Hb = np.zeros((1000, P, I, K))
for j in range(P):
    for i in range(I):
        pv = np.clip(H[j, i, :NOPT[i]], 0, 1); pv = pv / pv.sum() * (1 - 1e-12)
        Hb[:, j, i, :NOPT[i]] = rng.multinomial(NEFF[j, i], pv, size=1000) / NEFF[j, i]
lo, hi = np.percentile(Hb, [2.5, 97.5], axis=0)
top_h = np.where(MASK, H, -np.inf).argmax(-1)
GROUP = MASK & (hi >= np.take_along_axis(lo, top_h[..., None], -1))


def trm(Pm):
    m = np.where(MASK, Pm, -np.inf); t = MASK & (m == m.max(-1, keepdims=True))
    return (t & GROUP).sum(-1) / t.sum(-1)


def strict(Pm):
    return (np.where(MASK, Pm, -np.inf).argmax(-1) == top_h).astype(float)


up = np.triu(np.ones((10, 10), bool), 1)
score = lambda X: (X * np.nan_to_num(CODES)).sum(-1)
cd = lambda x: x[..., :, None, :] - x[..., None, :, :]
dh = cd(score(H[L1]))
dlo, dhi = np.percentile(cd(score(Hb[:, L1])), [2.5, 97.5], axis=0)
SIG = (dlo > 0) | (dhi < 0)
BASE = up[..., None] & ORD[None, None] & (dh != 0)


def pair_hits(Pm, valid):
    dm = cd(score(Pm[L1]))
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    return (s * valid).sum((0, 1)), valid.sum((0, 1)).astype(float)   # 문항별 [i]


def auc_ci(hit, n, B=2000):
    idx = rng.integers(0, I, (B, I))
    return hit.sum() / n.sum(), *np.percentile(hit[idx].sum(1) / np.maximum(n[idx].sum(1), 1), [2.5, 97.5])


# ---------------- 계산 ----------------
summary, per, diag = [], {}, []
hs = score(H[L1])
for name, (kind, X) in MODELS.items():
    G = np.stack([tvd_lv(prob(kind, X, b)) for b in BG])                 # [β, 3, i]
    bi = G[:, :, TRAIN].mean(-1).argmin(0)
    raw_t = tvd_lv(prob(kind, X, 1.0)); cal_t = np.stack([G[bi[l], l] for l in range(3)])
    R = raw_share(kind, X)
    st, tr = strict(R), trm(R)
    h_all, n_all = pair_hits(R, BASE); h_sig, n_sig = pair_hits(R, BASE & SIG)
    per[name] = dict(cal=cal_t, strict_L0=st[1], strict_L1=st[L1].mean(0), h_sig=h_sig, n_sig=n_sig)
    for l, lv in enumerate(LEVELS):
        for stage, arr, b in [("보정 전 (β=1)", raw_t, 1.0), ("보정 후 (β*)", cal_t, BG[bi[l]])]:
            ci = boot(arr[l, TEST])
            summary.append(dict(model=name, metric="TVD", level=lv, setting=stage, beta=b,
                                value=arr[l, TEST].mean(), ci_lo=ci[0], ci_hi=ci[1], n_items=int(TEST.sum())))
    for lv, xs, xr in [("L0", st[1], tr[1]), ("L1cell", st[L1].mean(0), tr[L1].mean(0))]:
        summary.append(dict(model=name, metric="최빈값 엄격", level=lv, setting="원비율", value=xs[TEST].mean(), n_items=int(TEST.sum())))
        summary.append(dict(model=name, metric="최빈값 RADIUS", level=lv, setting="원비율", value=xr[TEST].mean(), n_items=int(TEST.sum())))
    for lab, (h, n) in {"셀 쌍 AUC 전체": (h_all, n_all), "셀 쌍 AUC 유의쌍": (h_sig, n_sig)}.items():
        a, l_, h_ = auc_ci(h, n)
        summary.append(dict(model=name, metric=lab, level="L1", setting="원비율", value=a, ci_lo=l_, ci_hi=h_,
                            n_items=I, n_pairs=int(n.sum())))
    mx = np.where(MASK, R, -np.inf).max(-1)
    ms = score(R[L1]); mode_l1 = np.where(MASK, R[L1], -np.inf).argmax(-1)
    r = [np.corrcoef(ms[:, i], hs[:, i])[0, 1] for i in range(I) if ORD[i] and ms[:, i].std() > 0 and hs[:, i].std() > 0]
    diag.append({"model": name, "최빈 보기 확률 L0 (246)": mx[1].mean(), "최빈 보기 확률 L1 (246)": mx[L1].mean(),
                 "한 보기 100% 비율 L0": (mx[1] >= 0.999).mean(), "한 보기 100% 비율 L1": (mx[L1] >= 0.999).mean(),
                 "셀 간 평균점수 SD (서열·이진)": ms.std(0)[ORD].mean(),
                 "10셀 최빈값 모두 같은 문항 비율": (mode_l1 == mode_l1[0]).all(0).mean(),
                 "셀 점수-사람 상관 중앙값": float(np.median(r))})
U = MASK / NOPT[:, None]
diag.append({"model": "사람 (KGSS 2023)", "최빈 보기 확률 L0 (246)": np.where(MASK, H, -np.inf).max(-1)[1].mean(),
             "최빈 보기 확률 L1 (246)": np.where(MASK, H, -np.inf).max(-1)[L1].mean(),
             "셀 간 평균점수 SD (서열·이진)": hs.std(0)[ORD].mean()})
for lv, x in [("c0", tvd(U, H[0])), ("L0", tvd(U, H[1])), ("L1cell", tvd(U, H[L1]).mean(0))]:
    summary.append(dict(model="균등분포", metric="TVD", level=lv, setting="기준선", value=x[TEST].mean(), n_items=int(TEST.sum())))
for lv, x in [("L0", (GROUP.sum(-1) / NOPT)[1]), ("L1cell", (GROUP.sum(-1) / NOPT)[L1].mean(0))]:
    summary.append(dict(model="우연", metric="최빈값 RADIUS", level=lv, setting="기준선", value=x[TEST].mean(), n_items=int(TEST.sum())))
summary = pd.DataFrame(summary); diag = pd.DataFrame(diag)

# ---------------- 기존 v2 산출물과 대조 ----------------
ref = pd.read_csv(OUT / "test_tvd.csv")
for m_ref, m in [("deepseek", "DeepSeek V4.1 Flash"), ("haiku", "Haiku 5.5")]:
    for setting_ref, setting in [("laplace β=1", "보정 전 (β=1)"), ("laplace β*", "보정 후 (β*)")]:
        for lv in LEVELS:
            a = ref.query("setting==@setting_ref and model==@m_ref and level==@lv").tvd.item()
            b = summary.query("model==@m and metric=='TVD' and setting==@setting and level==@lv").value.item()
            assert abs(a - b) < 1e-9, (m, setting, lv, a, b)
mref = pd.read_csv(OUT / "v2_mode_radius.csv").query("split=='test 49'").set_index("기준")
for m_ref, m in [("deepseek", "DeepSeek V4.1 Flash"), ("haiku", "Haiku 5.5")]:
    for crit, metric in [("엄격(1위 일치)", "최빈값 엄격"), ("RADIUS 동점 규칙", "최빈값 RADIUS")]:
        for lv in ["L0", "L1cell"]:
            a = mref.loc[f"{m_ref} · {crit}", lv]
            b = summary.query("model==@m and metric==@metric and level==@lv").value.item()
            assert abs(a - b) < 1e-9, (m, metric, lv, a, b)
pref = pd.read_csv(OUT / "v2_pairs_radius.csv").query("쌍=='전체'").set_index("model")
for m_ref, m in [("deepseek", "DeepSeek V4.1 Flash"), ("haiku", "Haiku 5.5")]:
    a = pref.loc[m_ref, "AUC_유의쌍만"]
    b = summary.query("model==@m and metric=='셀 쌍 AUC 유의쌍'").value.item()
    assert abs(a - b) < 1e-9, (m, "AUC 유의쌍", a, b)
    # 전체 쌍은 부동소수점 0 차이 판정으로 쌍 수가 조금 다를 수 있어 허용 오차를 둔다
    a = pref.loc[m_ref, "AUC_전체쌍"]; b = summary.query("model==@m and metric=='셀 쌍 AUC 전체'").value.item()
    assert abs(a - b) < 0.005, (m, "AUC 전체", a, b)
print("대조 통과: DeepSeek·Haiku TVD(β=1, β*), 최빈값, 유의쌍 AUC가 기존 v2 산출물과 일치")

# ---------------- 짝지은 차이 ----------------
diffs = []
for a in ["DeepSeek V4.1 Flash", "Haiku 5.5"]:
    for b in ["EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"]:
        for lab, x in [("보정 후 TVD L0", per[a]["cal"][1] - per[b]["cal"][1]),
                       ("보정 후 TVD L1cell", per[a]["cal"][2] - per[b]["cal"][2]),
                       ("최빈값 엄격 L0", per[a]["strict_L0"] - per[b]["strict_L0"]),
                       ("최빈값 엄격 L1cell", per[a]["strict_L1"] - per[b]["strict_L1"])]:
            x = x[TEST]; ci = boot(x)
            diffs.append(dict(비교=f"{a} − {b}", 지표=lab, 차이=x.mean(), ci_lo=ci[0], ci_hi=ci[1], 단위="test 49문항"))
        idx = rng.integers(0, I, (5000, I))
        f = lambda m, ix: per[m]["h_sig"][ix].sum(-1) / per[m]["n_sig"][ix].sum(-1)
        d = f(a, idx) - f(b, idx)
        diffs.append(dict(비교=f"{a} − {b}", 지표="셀 쌍 AUC 유의쌍", 차이=f(a, np.arange(I)) - f(b, np.arange(I)),
                          ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), 단위="246문항 셀 쌍"))
diffs = pd.DataFrame(diffs)

summary.to_csv(OUT / "legacy_vs_v2_summary.csv", index=False, encoding="utf-8-sig")
diffs.to_csv(OUT / "legacy_vs_v2_diffs.csv", index=False, encoding="utf-8-sig")
diag.to_csv(OUT / "legacy_vs_v2_diagnostics.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print(summary.round(4).to_string(index=False)); print(); print(diffs.round(4).to_string(index=False))
print(); print(diag.round(3).to_string(index=False))
