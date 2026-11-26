"""20261126-haiku-think-original — 실행: python experiments/20261126-haiku-think-original/run.py  (KOSMOS 루트에서, 2~4분)

Haiku 5.5 생각 지시 추론(medium·high, k=20) vs 추론 끔(k=100 → k=20 재표본), DeepSeek Eco(끔·medium·high, k=50 → k=20)를
같은 197문항(순서형·이진 188), 같은 지표·같은 k로 비교한다.
지표 정의 출처: experiments/20261123-reasoning-high-full/run.py (stats·parts·agg). 원응답이 필요하다(raw/v2, git 무시).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
sys.path.insert(0, str(DESIGN / "dataset"))
import silicon as sl  # noqa: E402  (save_run: 커밋·해시 기록)

K, B, REPS, SEED = 20, 2000, 20, 20261126
RAW = {
    "Haiku": {"끔": (["haiku_p1.jsonl"], 100),
              "medium": (["haiku_medium_think_197items_p1.jsonl"], 20),
              "high": (["haiku_high_think_197items_p1.jsonl"], 20)},
    "DeepSeek": {"끔": (["deepseek_eco_p1.jsonl"], 50),
                 "medium": (["deepseek_eco_medium_p1.jsonl"], 50),
                 "high": (["deepseek_eco_high_high60_p1.jsonl", "deepseek_eco_high_high_rest137_p1.jsonl"], 50)},
}
LEDGERS = {"medium": "haiku_medium_think_197items_p1_batches.json", "high": "haiku_high_think_197items_p1_batches.json"}

# ── 1. 정답·문항 ─────────────────────────────────────────
d = np.load(DESIGN / "analysis/out_final_k100/counts_k100.npz", allow_pickle=True)
H, NEFF, CODES, SPLIT = d["H"], d["NEFF"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NI, NC = CODES.shape
items = list(csv.DictReader((DESIGN / "item_manifest_v2.csv").open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
subv = [r["var"] for f in ("high60.csv", "high_rest137.csv") for r in csv.DictReader((DESIGN / "subsets" / f).open(encoding="utf-8-sig"))]
SUB = np.isin(VARS, subv)
assert SUB.sum() == 197 and not (SUB & (SPLIT == "test")).any()
IX = np.where(SUB & ORD)[0]
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader((DESIGN / "cells.csv").open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig")) if r["phase"] == "1"}
NORM = np.zeros(CODES.shape)
for i in range(NI):
    c = CODES[i][MASK[i]]
    NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)


def load(files, smax):
    """ans[셀, 문항, 표본] = 보기 위치, 무효 -1, 없음 -2. api_error·중복 제외. 197문항만."""
    A = np.full((len(PERSONAS), NI, smax), -2, np.int8)
    for f in files:
        for line in open(ROOT / "raw/v2" / f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("invalid_type") == "api_error" or r["sample_idx"] >= smax or r["probe_id"] not in probes:
                continue
            p = probes[r["probe_id"]]
            i, j = VARS.index(p["var"]), PERSONAS.index(p["cell_id"])
            if not SUB[i] or A[j, i, r["sample_idx"]] != -2:
                continue
            A[j, i, r["sample_idx"]] = list(CODES[i]).index(float(r["parsed"])) if r["valid"] else -1
    assert (A[:, SUB] != -2).all(), f"{files}: 197문항 중 빈 칸"
    return A


def counts(A, k, rng):
    """프롬프트마다 표본 k개를 비복원 추출해 보기별 개수(무효는 빠짐). 표본이 k개뿐이면 그대로."""
    if k < A.shape[-1]:
        A = np.take_along_axis(A, np.argsort(rng.random(A.shape), axis=-1)[..., :k], -1)
    return (A[..., None] == np.arange(NC)).sum(-2).astype(float)


# ── 2. 지표 (출처: 20261123-reasoning-high-full/run.py) ──
def stats(P, n):
    s = (P * NORM).sum(-1)
    var = (P * (NORM - s[..., None]) ** 2 * MASK).sum(-1)
    return s, var / np.maximum(n, 1)


def parts(s, sig2):
    dv = s - (W[:, None] * s).sum(0)
    return (W[:, None] * dv ** 2).sum(0) - ((W * (1 - W))[:, None] * sig2).sum(0), dv


sH, sigH = stats(H[L1J], NEFF[L1J])
VH, dH = parts(sH, sigH)
DH2 = (W[:, None] * dH ** 2).sum(0)


def per_item(C):
    C = C[L1J]
    n = C.sum(-1)
    s, sig = stats(C / np.maximum(n[..., None], 1), n)
    V, dm = parts(s, sig)
    return V, (W[:, None] * dm * dH).sum(0), (W[:, None] * dm ** 2).sum(0)


def metric(m, ix, key):
    V, num, dm2 = m
    if key == "분산비":
        return V[ix].sum(-1) / VH[ix].sum(-1)
    return num[ix].sum(-1) / np.sqrt(dm2[ix].sum(-1) * DH2[ix].sum(-1))


def holm(ps):
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i]))
        adj[i] = run
    return adj


# ── 3. 비교 ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)
BIX = IX[rng.integers(0, len(IX), (B, len(IX)))]          # 문항 재표본 (두 조건에 같은 인덱스 → 짝지은 부트스트랩)
KEYS, COMPS = ("분산비", "정렬"), (("medium", "끔"), ("high", "끔"), ("high", "medium"))
rows, lvl, qual = [], [], []
for model, conds in RAW.items():
    A = {c: load(f, s) for c, (f, s) in conds.items()}
    for c, a in A.items():
        x = a[:, SUB]
        qual.append(dict(모델=model, 설정=c, 원래k=a.shape[-1], 무효비율=float((x == -1).mean())))
    for rep in range(REPS):
        M = {c: per_item(counts(a, K, rng)) for c, a in A.items()}
        for c in A:
            for key in KEYS:
                lvl.append(dict(모델=model, rep=rep, 설정=c, 지표=key, 값=float(metric(M[c], IX, key))))
        block = []
        for tgt, ref in COMPS:
            for key in KEYS:
                bs = metric(M[tgt], BIX, key) - metric(M[ref], BIX, key)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                p = min(1.0, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B))
                block.append(dict(모델=model, rep=rep, 비교=f"{tgt} − {ref}", 지표=key,
                                  차이=float(metric(M[tgt], IX, key) - metric(M[ref], IX, key)), 하한=lo, 상한=hi, p=p))
        b = pd.DataFrame(block)
        for comp in b["비교"].unique():                     # Holm ①: 비교마다 지표 2개 (DeepSeek 추론 실험과 같은 묶음)
            m_ = b["비교"] == comp
            b.loc[m_, "p_Holm2"] = holm(b.loc[m_, "p"].to_numpy())
        m_ = b["비교"].str.endswith("끔")                     # Holm ②: 끔 대비 2비교 × 2지표 = 4개 (민감도)
        b.loc[m_, "p_Holm4"] = holm(b.loc[m_, "p"].to_numpy())
        rows.append(b)
    print(model, "끝", flush=True)

# 민감도: Haiku 끔을 재표본하지 않고 k=100 전부 쓰기 (지표가 표집 잡음을 빼므로 k가 달라도 비교 가능해야 한다)
A_h = {c: load(f, s) for c, (f, s) in RAW["Haiku"].items()}
M_full = {c: per_item(counts(a, a.shape[-1], rng)) for c, a in A_h.items()}
sens = []
for tgt in ("medium", "high"):
    for key in KEYS:
        bs = metric(M_full[tgt], BIX, key) - metric(M_full["끔"], BIX, key)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sens.append(dict(비교=f"{tgt}(k=20) − 끔(k=100)", 지표=key, 차이=float(metric(M_full[tgt], IX, key) - metric(M_full["끔"], IX, key)),
                         하한=lo, 상한=hi, p=min(1.0, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B))))

R = pd.concat(rows, ignore_index=True)
R["유의_raw"], R["유의_Holm2"], R["유의_Holm4"] = R["p"] < 0.05, R["p_Holm2"] < 0.05, R["p_Holm4"] < 0.05
comp = (R.groupby(["모델", "비교", "지표"], sort=False)
        .agg(차이=("차이", "mean"), 하한=("하한", "mean"), 상한=("상한", "mean"), p중앙=("p", "median"),
             p_Holm2중앙=("p_Holm2", "median"), Holm2유의비율=("유의_Holm2", "mean"),
             p_Holm4중앙=("p_Holm4", "median"), Holm4유의비율=("유의_Holm4", "mean")).reset_index())
levels = pd.DataFrame(lvl).groupby(["모델", "설정", "지표"], sort=False)["값"].mean().unstack().reset_index()

for c, f in LEDGERS.items():                                 # Haiku 신규 수집 품질·비용
    rs = [json.loads(l) for l in open(ROOT / "raw/v2" / f.replace("_batches.json", ".jsonl"), encoding="utf-8")]
    led = json.loads((ROOT / "raw/v2" / f).read_text(encoding="utf-8"))
    qual.append(dict(모델="Haiku 수집", 설정=c, 원래k=20, 무효비율=1 - np.mean([r["valid"] for r in rs]),
                     건수=len(rs), thinking비율=np.mean([r["thinking_blocks"] > 0 for r in rs]),
                     출력토큰평균=np.mean([r["output_tokens"] for r in rs]), 비용USD=sum(b["cost_actual"] for b in led["batches"])))
quality, sens = pd.DataFrame(qual), pd.DataFrame(sens)

pd.set_option("display.width", 200)
print("\n[수준: 20회 평균]\n", levels.round(3).to_string(index=False))
print("\n[비교]\n", comp.round(4).to_string(index=False))
print("\n[민감도: Haiku 끔 k=100 그대로]\n", sens.round(4).to_string(index=False))
print("\n[품질]\n", quality.round(4).to_string(index=False))
run = sl.save_run(HERE / "out", tables={"levels": levels.set_index("모델"), "comparisons": comp.set_index("모델"),
                                        "reps": R.set_index("모델"), "sensitivity_k100": sens.set_index("비교"),
                                        "quality": quality.set_index("모델")},
                  params={"k": K, "bootstrap": B, "reps": REPS, "seed": SEED, "n_items": int(len(IX)), "raw": RAW})
print(f"\n기록: out/run.json (commit {run['git_commit']}, dirty={run['git_dirty']})")
