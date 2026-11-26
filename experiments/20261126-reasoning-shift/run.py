"""20261126-reasoning-shift — 같은 페르소나·같은 문항의 분포를 추론 전후(끔→low→medium→high)로 직접 비교.
실행 (KOSMOS 루트): uv run -q --no-project --python 3.12 --with pandas --with numpy python experiments/20261126-reasoning-shift/run.py

출처: 데이터 경로·load는 experiments/20261126-reasoning-concentration/run.py (← 20261123-reasoning-high-full/run.py).
지표 정의(잡음 밖 흔들림, 최빈 보기 뒤바뀜, 사람 최빈 일치, 사람 TVD 변화, MIN_N=20)는 experiments/20261119-item-overlap/run.py.
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
SUBSETS = [DESIGN / "subsets/high60.csv", DESIGN / "subsets/high_rest137.csv"]
RAW = {"끔": [ROOT / "raw/v2/deepseek_eco_p1.jsonl"],
       "low": [ROOT / "raw/v2/deepseek_eco_low_p1.jsonl"],
       "medium": [ROOT / "raw/v2/deepseek_eco_medium_p1.jsonl"],
       "high": [ROOT / "raw/v2/deepseek_eco_high_high60_p1.jsonl", ROOT / "raw/v2/deepseek_eco_high_high_rest137_p1.jsonl"]}
COMPS = [("끔", "low"), ("low", "medium"), ("medium", "high"), ("끔", "high")]
K, B, NULL_REPS, MIN_N, SEED = 50, 2000, 200, 20, 20261126
rng = np.random.default_rng(SEED)
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


d = np.load(NPZ, allow_pickle=True)
H, CODES, SPLIT = np.nan_to_num(d["H"]), d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
NI = len(VARS)
SUBV = [[r["var"] for r in csv.DictReader(f.open(encoding="utf-8-sig"))] for f in SUBSETS]
SUB = np.isin(VARS, SUBV[0] + SUBV[1])
assert SUB.sum() == 197 and (SUB == (SPLIT != "test")).all()
IDX = np.where(SUB)[0]
L1J = list(range(2, 12))
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}


def load(paths):
    """출처: experiments/20261126-reasoning-concentration/run.py load."""
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


DATA = {n: load(p) for n, p in RAW.items()}


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def mode_sets(sh):
    return sh == sh.max(-1, keepdims=True)


# ── 비교마다 (페르소나, 문항) 값 ─────────────────────────
VALS = {}                                                      # (비교, 지표) → [12, 문항] (쓰지 않는 칸은 NaN)
for a, b in COMPS:
    CA, CB = DATA[a], DATA[b]
    na, nb = CA.sum(-1), CB.sum(-1)
    sel = (na >= MIN_N) & (nb >= MIN_N) & SUB[None, :]
    shA, shB = share(CA), share(CB)
    obs = tvd(shA, shB)
    pooled = share(CA + CB)
    null_mean = np.full(obs.shape, np.nan)
    null95 = np.full(obs.shape, np.nan)
    for j in range(12):
        ix = np.where(sel[j])[0]
        nul = np.empty((NULL_REPS, len(ix)))
        naj, nbj = na[j, ix].astype(int), nb[j, ix].astype(int)
        for r_ in range(NULL_REPS):
            x = rng.multinomial(naj, pooled[j, ix]) / naj[:, None]
            y = rng.multinomial(nbj, pooled[j, ix]) / nbj[:, None]
            nul[r_] = tvd(x, y)
        null_mean[j, ix], null95[j, ix] = nul.mean(0), np.percentile(nul, 95, axis=0)
    hm = H.argmax(-1)                                          # [12, 문항] 사람 최빈 보기 (페르소나별 정답)
    ra = (shA.argmax(-1) == hm).astype(float)
    rb = (shB.argmax(-1) == hm).astype(float)
    flip = (~(mode_sets(shA) & mode_sets(shB)).any(-1)).astype(float)
    m = lambda x: np.where(sel, x, np.nan)
    VALS[(a, b)] = {
        "전후 TVD": m(obs), "잡음 TVD": m(null_mean), "전후 TVD − 잡음": m(obs - null_mean),
        "잡음 밖 흔들림 비율": m((obs > null95).astype(float)),
        "사람 TVD 전": m(tvd(shA, H)), "사람 TVD 후": m(tvd(shB, H)), "사람 TVD 변화": m(tvd(shB, H) - tvd(shA, H)),
        "최빈 보기 뒤바뀜 비율": m(flip),
        "사람 최빈 일치 전": m(ra), "사람 최빈 일치 후": m(rb), "사람 쪽 순변화": m(rb - ra),
    }
    log(f"{b} − {a}: 쓰인 (페르소나, 문항) {int(sel.sum())}개 / {12 * len(IDX)}")

LEVELS = {"페르소나 없음": [0], "한국 성인": [1], "성별×연령(10개 평균)": L1J}
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]
rows = []
for (a, b), vals in VALS.items():
    for lv, js in LEVELS.items():
        for key, X in vals.items():
            per_item = np.nanmean(X[js], 0)                    # 페르소나 평균 → 문항 값
            pt = float(np.nanmean(per_item[IDX]))
            bs = np.nanmean(per_item[BIX], 1)
            rows.append(dict(비교=f"{a} → {b}", 페르소나=lv, 지표=key, 값=pt,
                             하한=float(np.percentile(bs, 2.5)), 상한=float(np.percentile(bs, 97.5)),
                             문항수=int(np.isfinite(per_item[IDX]).sum())))
sd = pd.DataFrame(rows)
sd.to_csv(out / "shift.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
SHOW = ["전후 TVD", "잡음 TVD", "잡음 밖 흔들림 비율", "사람 TVD 전", "사람 TVD 변화", "최빈 보기 뒤바뀜 비율", "사람 최빈 일치 전", "사람 쪽 순변화"]
for lv in LEVELS:
    t = sd[sd.페르소나 == lv].copy()
    t["표시"] = t.apply(lambda r: f"{r.값:.3f} [{r.하한:.3f}, {r.상한:.3f}]", axis=1)
    log(f"\n[{lv}]")
    log(t.pivot_table(index="지표", columns="비교", values="표시", aggfunc="first", sort=False).reindex(SHOW)[[f"{a} → {b}" for a, b in COMPS]].to_string())
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
rel = HERE.relative_to(ROOT)
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){rel}/out")) or bool(git("ls-files", "--others", "--exclude-standard", "--", str(rel), f":(exclude){rel}/out")),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: [{"path": str(x.relative_to(ROOT)), "sha256_16": hashlib.sha256(x.read_bytes()).hexdigest()[:16]} for x in v] for k, v in RAW.items()},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "min_n": MIN_N, "seed": SEED},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
