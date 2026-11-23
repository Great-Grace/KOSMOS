"""20261123-reasoning-medium — DeepSeek Eco 추론 low → medium, 페르소나 없음·한국 성인·성별×연령 10셀, 246문항, k=50.
실행 (KOSMOS 루트): python experiments/20261123-reasoning-medium/run.py   (pandas·numpy 필요)

주 지표 4개 (medium − low, 문항 부트스트랩 2,000회, Holm 4개):
  1. 보정 후 사람 TVD, 한국 성인 (validation)      2. 보정 후 사람 TVD, 10셀 pop_share 가중 (validation)
  3. 셀 간 분산비 (모델/사람, 잡음 뺀 값)           4. 셀 편차 정렬 (가중 상관)   — 3·4는 test 뺀 순서형·이진 문항
탐색(보정 없음): medium − 끔, 기울기, 이동 분해, 무효율, 평균 추론 토큰, 끔→low→medium 추세표.

출처: 지표 1·2·β 격자·load_counts는 experiments/20261118-reasoning-effect/run.py, β 묶음 규칙(페르소나 없음 하나, 한국 성인+10셀 하나)은
      experiments/20261123-beta-subgroup/run.py, 지표 3·4·이동 분해는 experiments/20261123-reasoning-explain/run.py 의 정의를 복사했다.
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
MANIFEST = DESIGN / "item_manifest_v2.csv"
CELLS = DESIGN / "cells.csv"
RAW = {"끔": ROOT / "raw/v2/deepseek_eco_p1.jsonl",
       "low": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl",
       "medium": ROOT / "raw/v2/deepseek_eco_medium_p1.jsonl"}
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
NI, NJ = len(VARS), len(PERSONAS)
L1J = list(range(2, 12))
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
EV, CAL, VAL = SPLIT != "test", SPLIT == "calibration", SPLIT == "validation"
SEL = ORD & EV
cells = {r["cell_id"]: float(r["pop_share"]) for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([cells[PERSONAS[j]] for j in L1J])
W /= W.sum()
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}
NORM = np.zeros(CODES.shape)                                   # 코드 → 0~1 (출처: reasoning-explain)
for i in range(NI):
    c = CODES[i][MASK[i]]
    NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)


def load_counts(path):
    """출처: 20261118-reasoning-effect load_counts (표본 0~49, 중복·api_error 제외). 무효율·추론 토큰도 같이 센다."""
    C = np.zeros((NJ, NI, CODES.shape[1]))
    seen, n, bad, rt = set(), 0, 0, 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in probes:
            continue
        seen.add(key)
        p = probes[r["probe_id"]]
        if p["cell_id"] not in PERSONAS:
            continue
        n += 1
        rt += r.get("reasoning_tokens") or 0
        if r["valid"]:
            C[PERSONAS.index(p["cell_id"]), VARS.index(p["var"]), list(CODES[VARS.index(p["var"])]).index(float(r["parsed"]))] += 1
        else:
            bad += 1
    return C, dict(호출=n, 무효율=bad / max(n, 1), 평균_추론_토큰=rt / max(n, 1))


COUNTS, INFO = {}, {}
for s, p in RAW.items():
    COUNTS[s], INFO[s] = load_counts(p)
    nn = COUNTS[s][:, EV].sum(-1)
    log(f"{s}: 응답 {INFO[s]['호출']:,}건, 무효율 {INFO[s]['무효율']:.4f}, 평균 추론 토큰 {INFO[s]['평균_추론_토큰']:.1f}, "
        f"프롬프트당 유효 표본 최소 {nn.min():.0f}, 중앙 {np.median(nn):.0f}, 최대 {nn.max():.0f}")


def prob(X, beta=1.0):
    x = (X + MASK) ** beta * MASK                               # 출처: register-effect prob
    return x / x.sum(-1, keepdims=True)


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


# ── 2. β (설정마다: 페르소나 없음 하나, 한국 성인 + 10셀 하나. calibration 148문항에서 TVD 최소) ──
BETA, PER_ITEM = {}, {}
for s, C in COUNTS.items():
    G = {j: tvd(prob(C[j][None], BETAS[:, None, None]), H[j][None]) for j in range(NJ)}     # [β, 문항]
    g_none = G[0]
    g_pers = 0.5 * (G[1] + sum(w * G[j] for w, j in zip(W, L1J)))                         # 한국 성인과 10셀을 같은 가중
    b0, b1 = BETAS[g_none[:, CAL].mean(1).argmin()], BETAS[g_pers[:, CAL].mean(1).argmin()]
    BETA[s] = dict(페르소나_없음=float(b0), 한국성인_10셀=float(b1))
    t_adult = tvd(prob(C[1], b1), H[1])
    t_cell = sum(w * tvd(prob(C[j], b1), H[j]) for w, j in zip(W, L1J))
    PER_ITEM[s] = {"보정 TVD 한국 성인": t_adult, "보정 TVD 10셀 가중": t_cell}
    log(f"β {s}: 페르소나 없음 {b0:.3f}, 한국 성인·10셀 {b1:.3f}")


# ── 3. 셀 간 분산비·정렬 (출처: reasoning-explain) ─────────────────
def stats(P, n):
    s = (P * NORM).sum(-1)
    var = (P * (NORM - s[..., None]) ** 2 * MASK).sum(-1)
    return s, var / np.maximum(n, 1)


def parts(s, sig2):
    sbar = (W[:, None] * s).sum(0)
    dv = s - sbar
    V = (W[:, None] * dv ** 2).sum(0) - (W * (1 - W))[:, None].__mul__(sig2).sum(0)
    return V, dv


sH, sigH = stats(H[L1J], NEFF[L1J])
VH, dH = parts(sH, sigH)
M = {}
for s, C in COUNTS.items():
    sc, sig = stats(share(C[L1J]), C[L1J].sum(-1))
    V, dm = parts(sc, sig)
    M[s] = dict(s=sc, V=V, d=dm, num=(W[:, None] * dm * dH).sum(0), dm2=(W[:, None] * dm ** 2).sum(0))
DH2 = (W[:, None] * dH ** 2).sum(0)

IDX = np.where(SEL)[0]
VIDX = np.where(VAL)[0]
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]            # 지표 3·4·기울기용 (순서형·이진, test 뺌)
BVX = VIDX[rng.integers(0, len(VIDX), (B, len(VIDX)))]         # 지표 1·2용 (validation 49문항)


def metric(s, key, ix):
    """ix: 문항 번호 배열 [..., n]. 같은 부트스트랩 표본으로 설정끼리 짝짓는다."""
    if key in PER_ITEM[s]:
        return PER_ITEM[s][key][ix].mean(-1)
    m = M[s]
    if key == "셀 간 분산비":
        return m["V"][ix].sum(-1) / VH[ix].sum(-1)
    if key == "셀 편차 정렬":
        return m["num"][ix].sum(-1) / np.sqrt(m["dm2"][ix].sum(-1) * DH2[ix].sum(-1))
    if key == "기울기":
        return m["num"][ix].sum(-1) / DH2[ix].sum(-1)
    raise KeyError(key)


PRIMARY = ["보정 TVD 한국 성인", "보정 TVD 10셀 가중", "셀 간 분산비", "셀 편차 정렬"]
SCOPE = {"보정 TVD 한국 성인": (VIDX, BVX), "보정 TVD 10셀 가중": (VIDX, BVX), "셀 간 분산비": (IDX, BIX), "셀 편차 정렬": (IDX, BIX), "기울기": (IDX, BIX)}


def compare(a, b, key):
    pt, bs = SCOPE[key]
    va, vb = metric(a, key, pt), metric(b, key, pt)
    diff = metric(b, key, bs) - metric(a, key, bs)
    p = min(max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / B), 1.0)
    return dict(비교=f"{b} − {a}", 지표=key, **{a: float(va), b: float(vb)}, 차이=float(vb - va),
                하한=float(np.percentile(diff, 2.5)), 상한=float(np.percentile(diff, 97.5)), p=float(p))


def holm(p):
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r_) * p[i]))
        adj[i] = run
    return adj


pr = pd.DataFrame([compare("low", "medium", k) for k in PRIMARY])
pr["p_Holm"] = holm(pr["p"].to_numpy())
pr["있음"] = pr["p_Holm"] < 0.05
pr.to_csv(out / "primary.csv", index=False, encoding="utf-8-sig")
log("\n[주 지표 (medium − low, Holm 4개)]")
log(pr.round(4).to_string(index=False))

# ── 4. 탐색 (보정 없음) ───────────────────────────────────
ex = pd.DataFrame([compare("끔", "medium", k) for k in PRIMARY + ["기울기"]] + [compare("low", "medium", "기울기")])
ex["용도"] = "탐색"
ex.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: medium − 끔 4지표, 기울기]")
log(ex.round(4).to_string(index=False))

# 추세표 끔→low→medium
tr = []
for s in ("끔", "low", "medium"):
    row = {"설정": s}
    for k in PRIMARY + ["기울기"]:
        row[k] = float(metric(s, k, SCOPE[k][0]))
    row.update({"무효율": INFO[s]["무효율"], "평균 추론 토큰": INFO[s]["평균_추론_토큰"],
                "β 페르소나 없음": BETA[s]["페르소나_없음"], "β 한국 성인·10셀": BETA[s]["한국성인_10셀"]})
    tr.append(row)
trd = pd.DataFrame(tr)
trd.to_csv(out / "trend.csv", index=False, encoding="utf-8-sig")
log("\n[끔 → low → medium 추세]")
log(trd.round(4).to_string(index=False))


# 이동 분해 (출처: reasoning-explain, 두 설정 쌍마다)
def decompose(a, b):
    delta = M[b]["s"] - M[a]["s"]
    common = (W[:, None] * delta).sum(0)
    spec = (W[:, None] * (delta - common) ** 2).sum(0)
    pooled = share(COUNTS[a][L1J] + COUNTS[b][L1J])
    na, nb = COUNTS[a][L1J].sum(-1).astype(int), COUNTS[b][L1J].sum(-1).astype(int)
    nc, ns = np.zeros(NI), np.zeros(NI)
    for i in IDX:
        k_ = MASK[i].sum()
        sa = np.stack([rng.multinomial(na[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(na[c, i], 1) for c in range(10)])
        sb = np.stack([rng.multinomial(nb[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(nb[c, i], 1) for c in range(10)])
        dl = sb - sa
        cm = (W[:, None] * dl).sum(0)
        nc[i] = (cm ** 2).mean()
        ns[i] = (W[:, None] * (dl - cm) ** 2).sum(0).mean()
    s_h = (W[:, None] * sH).sum(0)
    toward = np.abs((W[:, None] * M[b]["s"]).sum(0) - s_h) - np.abs((W[:, None] * M[a]["s"]).sum(0) - s_h)
    rows = []
    for lab, x in [("공통 이동 제곱", common ** 2), ("공통 이동 제곱 잡음", nc), ("공통 이동 제곱 − 잡음", common ** 2 - nc),
                   ("셀 고유 이동 제곱합", spec), ("셀 고유 이동 잡음", ns), ("셀 고유 이동 − 잡음", spec - ns),
                   ("10셀 평균이 사람 평균에 가까워진 정도(음수=가까워짐)", toward),
                   ("셀 고유 이동이 사람 셀 편차와 같은 방향(공분산)", (W[:, None] * (delta - common) * dH).sum(0))]:
        v = x[BIX].mean(1)
        rows.append(dict(비교=f"{b} − {a}", 항목=lab, 평균=float(x[IDX].mean()), 하한=float(np.percentile(v, 2.5)), 상한=float(np.percentile(v, 97.5))))
    return rows


dd = pd.DataFrame(decompose("low", "medium") + decompose("끔", "medium"))
dd.to_csv(out / "decomp.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 이동 분해 (문항 평균, 10셀)]")
log(dd.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)), f":(exclude){HERE.relative_to(ROOT).as_posix()}/out")),  # out/은 이 스크립트가 만드는 결과라 제외
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for k, p in RAW.items()},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "seed": 20261123, "beta_grid": [0.05, 1.5, 0.005], "beta": BETA},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
