"""20261126-haiku-think-original — Claude Haiku 5.5 생각 지시 추론(medium·high)의 원문 문체 응답 분포를 추론 끔과 직접 비교.
실행 (KOSMOS 루트): uv run -q --no-project --python 3.12 --with pandas --with numpy python experiments/20261126-haiku-think-original/run.py

분석 틀은 DeepSeek의 두 실험을 그대로 따른다.
  쏠림     experiments/20261126-reasoning-concentration/run.py (최빈 보기 비율(top_share)·엔트로피 비, FORMULAS.md 5.4)
  전후 비교 experiments/20261126-reasoning-shift/run.py (전후 TVD·잡음 밖 흔들림·최빈 보기 뒤바뀜·사람 쪽 순변화·사람 TVD 변화)
  보정 후 사람 TVD  experiments/20261123-reasoning-high-full/run.py (β: calibration 148 → validation 49)
바뀐 점: Haiku 추론은 k=20이라 모든 설정을 표본 0~19(k=20)로 맞춘다. DeepSeek도 같은 k=20으로 다시 계산해 나란히 놓는다.
  전후 비교의 최소 유효 응답 수는 20(k=50 기준) → 10(k=20의 절반). 사람 기준 다시 뽑기는 50명 → 20명.
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
ROOT = Path("/Users/taewoo/Developer/KOSMOS")
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS = DESIGN / "prompts_v2.csv"
CELLS = DESIGN / "cells.csv"
SUBSETS = [DESIGN / "subsets/high60.csv", DESIGN / "subsets/high_rest137.csv"]
R2 = ROOT / "raw/v2"
RAW = {
    "Haiku": {"끔": [R2 / "haiku_p1.jsonl"],
              "medium": [R2 / "haiku_medium_think_197items_p1.jsonl"],
              "high": [R2 / "haiku_high_think_197items_p1.jsonl"]},
    "DeepSeek": {"끔": [R2 / "deepseek_eco_p1.jsonl"],
                 "low": [R2 / "deepseek_eco_low_p1.jsonl"],
                 "medium": [R2 / "deepseek_eco_medium_p1.jsonl"],
                 "high": [R2 / "deepseek_eco_high_high60_p1.jsonl", R2 / "deepseek_eco_high_high_rest137_p1.jsonl"]},
}
LEDGERS = {"medium": R2 / "haiku_medium_think_197items_p1_batches.json", "high": R2 / "haiku_high_think_197items_p1_batches.json"}
COMPS = [("끔", "low"), ("low", "medium"), ("끔", "medium"), ("medium", "high"), ("끔", "high")]
K, B, NULL_REPS, MIN_N, N_MATCH, SEED = 20, 2000, 200, 10, 20, 20261126
S0 = 0                                                        # 쓰는 표본 번호 S0 ~ S0+K-1 (Haiku 추론은 0~19뿐)
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
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
H, CODES, SPLIT = np.nan_to_num(d["H"]), d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NK = MASK.sum(1)
NI = len(VARS)
SUBV = [[r["var"] for r in csv.DictReader(f.open(encoding="utf-8-sig"))] for f in SUBSETS]
SUB = np.isin(VARS, SUBV[0] + SUBV[1])
assert SUB.sum() == 197 and (SUB == (SPLIT != "test")).all()
IDX = np.where(SUB)[0]
CAL, VAL = SUB & (SPLIT == "calibration"), SUB & (SPLIT == "validation")
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}


def load(paths):
    """출처: 20261126-reasoning-concentration/run.py load. 표본 번호 S0~S0+K-1만, 중복·api_error 제외, 197문항만."""
    C = np.zeros((len(PERSONAS), NI, CODES.shape[1]))
    n_all = np.zeros((len(PERSONAS), NI))
    seen = set()
    for line in (ln for path in paths for ln in open(path, encoding="utf-8")):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or not (S0 <= r["sample_idx"] < S0 + K) or key in seen or r["probe_id"] not in probes:
            continue
        p = probes[r["probe_id"]]
        i = VARS.index(p["var"])
        if not SUB[i]:
            continue
        seen.add(key)
        j = PERSONAS.index(p["cell_id"])
        n_all[j, i] += 1
        if r["valid"]:
            C[j, i, list(CODES[i]).index(float(r["parsed"]))] += 1
    assert (n_all[:, SUB] == K).all(), f"{paths[0].name}: 표본 {K}개가 아닌 칸이 있음"
    return C


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def prob(X, beta):
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def H_(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0).sum(-1)


ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]
VI = np.where(VAL)[0]
VB = VI[rng.integers(0, len(VI), (B, len(VI)))]
LEVELS = {"페르소나 없음": [0], "한국 성인": [1], "성별×연령(10개 평균)": L1J}


def per_level(X):
    """[12, 문항] → 단계별 문항 값 (성별×연령은 10개 단순 평균, FORMULAS.md 5.5 L1cell)."""
    return {lv: np.nanmean(X[js], 0) for lv, js in LEVELS.items()}


def boot(x, ix=IDX, bix=BIX):
    bs = np.nanmean(x[bix], 1)
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    return float(np.nanmean(x[ix])), *ci(bs), float(min(p, 1))


# ── 2. 사람 기준 (쏠림): 그대로 / 모델과 같은 20명을 다시 뽑았을 때의 기대값 ──
HUM_H = H_(H)                                                  # [12, 문항]
RS = np.zeros((12, NI, NULL_REPS, CODES.shape[1]))
for j in [1] + L1J:
    for i in IDX:
        k_ = NK[i]
        RS[j, i, :, :k_] = rng.multinomial(N_MATCH, H[j, i, :k_] / H[j, i, :k_].sum(), size=NULL_REPS) / N_MATCH
RS[0] = RS[1]
safe = lambda h: np.where(h > 0, h, np.nan)
HUMAN_LV = {"사람 (그대로)": {"최빈 보기 비율": per_level(H.max(-1)), "엔트로피 비": per_level(np.ones((12, NI)))},
            f"사람 ({N_MATCH}명 다시 뽑음)": {"최빈 보기 비율": per_level(RS.max(-1).mean(-1)),
                                         "엔트로피 비": per_level((H_(RS) / safe(HUM_H)[..., None]).mean(-1))}}

# ── 3. 모델별 계산 ───────────────────────────────────────
conc_rows, cmp_rows, shift_rows, tvd_rows = [], [], [], []
for model, raws in RAW.items():
    DATA = {s: load(p) for s, p in raws.items()}
    # 3-1 쏠림 (원비율)
    CONC = {s: {"최빈 보기 비율": per_level(share(C).max(-1)), "엔트로피 비": per_level(H_(share(C)) / safe(HUM_H))} for s, C in DATA.items()}
    for s, vals in list(CONC.items()) + (list(HUMAN_LV.items()) if model == "Haiku" else []):
        for met, lvs in vals.items():
            for lv, x in lvs.items():
                v, lo, hi, _ = boot(x)
                conc_rows.append(dict(모델=model if s in DATA else "사람", 설정=s, 지표=met, 페르소나=lv, 값=v, 하한=lo, 상한=hi))
    for a, b in [c for c in COMPS if c[0] in DATA and c[1] in DATA]:
        for met in ("최빈 보기 비율", "엔트로피 비"):
            for lv in LEVELS:
                v, lo, hi, p = boot(CONC[b][met][lv] - CONC[a][met][lv])
                cmp_rows.append(dict(모델=model, 비교=f"{b} − {a}", 지표=met, 페르소나=lv, 차이=v, 하한=lo, 상한=hi, p=p))
    # 3-2 전후 비교 (출처: 20261126-reasoning-shift/run.py)
    for a, b in [c for c in COMPS if c[0] in DATA and c[1] in DATA]:
        CA, CB = DATA[a], DATA[b]
        na, nb = CA.sum(-1), CB.sum(-1)
        sel = (na >= MIN_N) & (nb >= MIN_N) & SUB[None, :]
        shA, shB = share(CA), share(CB)
        obs, pooled = tvd(shA, shB), share(CA + CB)
        null_mean, null95 = np.full(obs.shape, np.nan), np.full(obs.shape, np.nan)
        for j in range(12):
            ix = np.where(sel[j])[0]
            naj, nbj = na[j, ix].astype(int), nb[j, ix].astype(int)
            nul = np.empty((NULL_REPS, len(ix)))
            for r_ in range(NULL_REPS):
                nul[r_] = tvd(rng.multinomial(naj, pooled[j, ix]) / naj[:, None], rng.multinomial(nbj, pooled[j, ix]) / nbj[:, None])
            null_mean[j, ix], null95[j, ix] = nul.mean(0), np.percentile(nul, 95, axis=0)
        hm = H.argmax(-1)
        ra, rb = (shA.argmax(-1) == hm).astype(float), (shB.argmax(-1) == hm).astype(float)
        ms = lambda sh: sh == sh.max(-1, keepdims=True)
        flip = (~(ms(shA) & ms(shB)).any(-1)).astype(float)
        m = lambda x: np.where(sel, x, np.nan)
        vals = {"전후 TVD": m(obs), "잡음 TVD": m(null_mean), "잡음 밖 흔들림 비율": m((obs > null95).astype(float)),
                "사람 TVD 전": m(tvd(shA, H)), "사람 TVD 변화": m(tvd(shB, H) - tvd(shA, H)),
                "최빈 보기 뒤바뀜 비율": m(flip), "사람 최빈 일치 전": m(ra), "사람 쪽 순변화": m(rb - ra)}
        log(f"{model} {a} → {b}: 쓰인 (페르소나, 문항) {int(sel.sum())}개 / {12 * len(IDX)}")
        for key, X in vals.items():
            for lv, x in per_level(X).items():
                v, lo, hi, p = boot(x)
                shift_rows.append(dict(모델=model, 비교=f"{a} → {b}", 페르소나=lv, 지표=key, 값=v, 하한=lo, 상한=hi, p=p,
                                       문항수=int(np.isfinite(x[IDX]).sum())))
    # 3-3 보정 후 사람 TVD (출처: 20261123-reasoning-high-full/run.py)
    TV = {}
    for s, C in DATA.items():
        g0 = tvd(prob(C[0][None], BETAS[:, None, None]), H[0][None])
        gL0 = tvd(prob(C[1][None], BETAS[:, None, None]), H[1][None])
        gL1 = (W[None, :, None] * tvd(prob(C[L1J][None], BETAS[:, None, None, None]), H[L1J][None])).sum(1)
        b0 = int(g0[:, CAL].mean(1).argmin())
        b1 = int(((gL0[:, CAL].mean(1) + gL1[:, CAL].mean(1)) / 2).argmin())
        TV[s] = {"페르소나 없음": g0[b0], "한국 성인": gL0[b1], "성별×연령(10셀 가중)": gL1[b1]}
        for u, x in TV[s].items():
            tvd_rows.append(dict(모델=model, 비교=s, 페르소나=u, β=float(BETAS[b0 if u == "페르소나 없음" else b1]),
                                 값=float(x[VAL].mean()), 하한=ci(x[VB].mean(1))[0], 상한=ci(x[VB].mean(1))[1], p=np.nan))
    for a, b in [c for c in COMPS if c[0] in DATA and c[1] in DATA]:
        for u in TV[a]:
            v, lo, hi, p = boot(TV[b][u] - TV[a][u], VI, VB)
            tvd_rows.append(dict(모델=model, 비교=f"{b} − {a}", 페르소나=u, β=np.nan, 값=v, 하한=lo, 상한=hi, p=p))

conc, cmp_, shift, tv = map(pd.DataFrame, (conc_rows, cmp_rows, shift_rows, tvd_rows))
conc.to_csv(out / "concentration_levels.csv", index=False, encoding="utf-8-sig")
cmp_.to_csv(out / "concentration_compare.csv", index=False, encoding="utf-8-sig")
shift.to_csv(out / "shift.csv", index=False, encoding="utf-8-sig")
tv.to_csv(out / "tvd_calibrated.csv", index=False, encoding="utf-8-sig")

qrows = []
for s, f in LEDGERS.items():
    rs = [json.loads(l) for l in open(f.with_name(f.name.replace("_batches.json", ".jsonl")), encoding="utf-8")]
    led = json.loads(f.read_text(encoding="utf-8"))
    qrows.append(dict(설정=s, 건수=len(rs), 무효율=1 - np.mean([r["valid"] for r in rs]), 생각블록비율=np.mean([r["thinking_blocks"] > 0 for r in rs]),
                      출력토큰평균=np.mean([r["output_tokens"] for r in rs]), 비용USD=sum(b["cost_actual"] for b in led["batches"])))
qd = pd.DataFrame(qrows)
qd.to_csv(out / "quality.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
fmt = lambda t, v="값": t.assign(표시=t.apply(lambda r: f"{r[v]:.3f} [{r.하한:.3f}, {r.상한:.3f}]", axis=1))
log(f"\n표본 {S0}~{S0 + K - 1} (k={K}), 197문항")
log("\n[쏠림: 원비율 최빈 보기 비율]")
log(fmt(conc[conc.지표 == "최빈 보기 비율"]).pivot_table(index=["모델", "설정"], columns="페르소나", values="표시", aggfunc="first", sort=False).to_string())
log("\n[쏠림 차이]")
log(fmt(cmp_, "차이").pivot_table(index=["모델", "지표", "비교"], columns="페르소나", values="표시", aggfunc="first", sort=False).to_string())
SHOW = ["전후 TVD", "잡음 TVD", "잡음 밖 흔들림 비율", "사람 TVD 전", "사람 TVD 변화", "최빈 보기 뒤바뀜 비율", "사람 최빈 일치 전", "사람 쪽 순변화"]
for model in RAW:
    for lv in LEVELS:
        t = fmt(shift[(shift.모델 == model) & (shift.페르소나 == lv)])
        log(f"\n[전후 비교: {model} {lv}]")
        log(t.pivot_table(index="지표", columns="비교", values="표시", aggfunc="first", sort=False).reindex(SHOW).to_string())
log("\n[보정 후 사람 TVD (validation 49)]")
log(fmt(tv).pivot_table(index=["모델", "비교"], columns="페르소나", values="표시", aggfunc="first", sort=False).to_string())
log("\n[Haiku 수집 품질]")
log(qd.round(4).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
rel = Path("experiments/20261126-haiku-think-original")
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){rel}/out")) or bool(git("ls-files", "--others", "--exclude-standard", "--", str(rel), f":(exclude){rel}/out")),
           "note": "copy of experiments/20261126-haiku-think-original/run.py + DeepSeek low", "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {f"{m}/{s}": [{"path": str(x.relative_to(ROOT)), "sha256_16": hashlib.sha256(x.read_bytes()).hexdigest()[:16]} for x in v]
                      for m, r in RAW.items() for s, v in r.items()},
           "params": {"k": K, "sample_start": S0, "bootstrap": B, "null_reps": NULL_REPS, "min_n": MIN_N, "n_match": N_MATCH, "seed": SEED},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
