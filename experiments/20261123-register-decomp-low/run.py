"""20261123-register-decomp-low — 인칭 요소 분해(페르소나 문장·문항 호칭·지시문·주체 높임), DeepSeek Eco 추론 low, 한국 성인, k=50.
실행 (KOSMOS 루트): PYTHONUTF8=1 python experiments/20261123-register-decomp-low/run.py

주 지표: validation 49문항 보정 후 사람 TVD (β는 register-final의 한국 성인 β, 10조건 공통).
출처: boot·holm은 20261123-register-final/run.py, load·prob·tvd는 20261119-register-effect/run.py.
"""
import csv
import datetime
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# register-effect의 데이터 준비부(H, CODES, VARS, 분할, prob, tvd)를 불러 쓴다
REG = ROOT / "experiments/20261119-register-effect/run.py"
src = REG.read_text(encoding="utf-8").split("rows, info = [], {}")[0]
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "logf = io.StringIO()")
g = {"__file__": str(REG), "io": io}
exec(src, g)
H, CODES, VARS, MASK, CAL, VAL, EV, K = (g[k] for k in ("H", "CODES", "VARS", "MASK", "CAL", "VAL", "EV", "K"))
prob, tvd, share = g["prob"], g["tvd"], g["share"]
HU = H[1]                                    # 한국 성인 사람 분포
B = 2000
rng = np.random.default_rng(20261123)

# ── 조건 정의 ─────────────────────────────────────────────
# (페르소나 문장 '이 사람', 문항 호칭 '이 사람', 지시문 '이 사람이 고를') 2×2×2
FACT = {"원문": (0, 0, 0), "persona": (1, 0, 0), "address": (0, 1, 0), "instr": (0, 0, 1),
        "persona_address": (1, 1, 0), "persona_instr": (1, 0, 1), "address_instr": (0, 1, 1), "all_keep_si": (1, 1, 1)}
LABEL = {"원문": "원문(당신·귀하·원문 지시문)", "persona": "페르소나 문장만 이 사람", "address": "문항 호칭만 이 사람",
         "instr": "지시문만 이 사람이 고를", "persona_address": "페르소나 문장+문항 호칭", "persona_instr": "페르소나 문장+지시문",
         "address_instr": "문항 호칭+지시문", "all_keep_si": "세 가지 모두 이 사람(주체 높임 유지)",
         "이사람하십시오": "이 사람·하십시오체(주체 높임 제거)", "dangsin_address": "문항 호칭만 당신(귀하→당신)"}
CONDS = list(LABEL)

RAWF = {"원문": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl", "이사람하십시오": ROOT / "raw/v2/deepseek_eco_low_register_p1.jsonl",
        "decomp": ROOT / "raw/v2/deepseek_eco_low_decomp_p1.jsonl"}
DECOMP_CSV = ROOT / "research/design_v2_20261112/register_decomp/out/prompts_decomp.csv"
decomp_cond = {r["probe_id"]: r["cond"] for r in csv.DictReader(DECOMP_CSV.open(encoding="utf-8-sig"))}

C = {c: np.zeros((len(VARS), CODES.shape[1])) for c in CONDS}
N = {c: np.zeros(len(VARS)) for c in CONDS}
REFUSAL = {c: 0 for c in CONDS}
seen = set()
for kind, path in RAWF.items():
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["level"] != "L0" or r.get("invalid_type") == "api_error" or r["sample_idx"] >= K:
            continue
        pid = r["probe_id"]
        if kind == "원문":
            c = "원문"
        elif kind == "이사람하십시오":
            if not pid.endswith("__C"):
                continue
            c = "이사람하십시오"
        else:
            c = decomp_cond.get(pid)
            if c is None:
                continue
        key = (kind, pid, r["sample_idx"])
        if key in seen:
            continue
        seen.add(key)
        i = VARS.index(r["var"])
        N[c][i] += 1
        if r["valid"]:
            C[c][i, list(CODES[i]).index(float(r["parsed"]))] += 1
        elif r.get("invalid_type") == "refusal":
            REFUSAL[c] += 1

# β: register-final이 정한 한국 성인 β (low)
bt = pd.read_csv(ROOT / "experiments/20261123-register-final/out/betas.csv")
BETA = float(bt[(bt["설정"] == "DeepSeek Eco 추론 low") & (bt["단계"] == "한국 성인")]["β"].iloc[0])
log(f"β(한국 성인, low) = {BETA}")
cnt = pd.DataFrame({LABEL[c]: {"문항 평균 표본수": N[c].mean(), "표본 최소": N[c].min(), "표본 최대": N[c].max()} for c in CONDS}).T
log(cnt.round(1).to_string())

T = {c: tvd(prob(C[c], BETA), HU) for c in CONDS}          # 문항별 보정 후 사람 TVD
T1 = {c: tvd(prob(C[c]), HU) for c in CONDS}               # 보정 없음(β=1)

# 주체 높임 있는 문항(-시- 있던 151문항)
flags = {r["var"] for r in csv.DictReader((ROOT / "research/design_v2_20261112/register_decomp/out/flags.csv").open(encoding="utf-8-sig"))
         if r["flag"].startswith("주체 높임 없는")}
HAS_SI = np.array([v not in flags for v in VARS])
log(f"주체 높임 문항 {HAS_SI.sum()}개(없는 문항 {(~HAS_SI).sum()}개), 그중 test 제외 {(HAS_SI & EV).sum()}개")


def boot(x, sel):
    idx = np.where(sel & ~np.isnan(x))[0]
    bs = x[idx[rng.integers(0, len(idx), (B, len(idx)))]].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    return float(x[idx].mean()), float(lo), float(hi), float(min(p, 1.0)), len(idx)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main_eff(TT, axis):
    return sum(TT[c] for c, f in FACT.items() if f[axis] == 1) / 4 - sum(TT[c] for c, f in FACT.items() if f[axis] == 0) / 4


def inter(TT, a, b):
    return sum((1 if f[a] == f[b] else -1) * TT[c] for c, f in FACT.items()) / 4    # 부호곱 ¼ 가중 대비


AXIS = {"페르소나 문장(당신→이 사람)": 0, "문항 호칭(귀하→이 사람)": 1, "지시문(원문→이 사람이 고를)": 2}
MAIN = {**{k: (lambda TT, a=a: main_eff(TT, a)) for k, a in AXIS.items()},
        "주체 높임 제거(-시- 유지→제거)": lambda TT: TT["이사람하십시오"] - TT["all_keep_si"]}

# ── 주 가설군 (Holm 4개) ─────────────────────────────────
rows = []
for eff, f in MAIN.items():
    m, lo, hi, p, n_ = boot(f(T), VAL)
    rows.append(dict(효과=eff, 차이=m, 하한=lo, 상한=hi, p=p, 문항수=n_))
ed = pd.DataFrame(rows)
ed["p_Holm"] = holm(ed["p"].to_numpy())
ed["유의"] = ed["p_Holm"] < 0.05
ed.to_csv(out / "effects.csv", index=False, encoding="utf-8-sig")
log("\n[주 가설군: 보정 후 사람 TVD 차이(validation 49), Holm 4개. 음수 = 사람에 가까워짐]")
log(ed.round(4).to_string(index=False))

# 조건별 TVD 표(참고)
crow = []
for c in CONDS:
    crow.append(dict(조건=LABEL[c], validation=float(T[c][VAL].mean()), 계산_197문항=float(T[c][EV].mean()),
                     보정없음_validation=float(T1[c][VAL].mean()), 무효율_197문항=float((1 - C[c].sum(-1) / np.maximum(N[c], 1))[EV].mean()),
                     거부_건수=REFUSAL[c]))
cd = pd.DataFrame(crow)
cd.to_csv(out / "cond_tvd.csv", index=False, encoding="utf-8-sig")
log("\n[조건별 TVD]")
log(cd.round(4).to_string(index=False))

# ── 탐색 (보정 없음) ─────────────────────────────────────
xrows = []


def xadd(kind_, eff, x, sel):
    m, lo, hi, p, n_ = boot(x, sel)
    xrows.append(dict(분석=kind_, 효과=eff, 차이=m, 하한=lo, 상한=hi, 구간_0_제외=bool(lo > 0 or hi < 0), 문항수=n_))


xadd("validation", "호칭 귀하→당신(dangsin_address − 원문)", T["dangsin_address"] - T["원문"], VAL)
xadd("197문항", "호칭 귀하→당신(dangsin_address − 원문)", T["dangsin_address"] - T["원문"], EV)
for nm, (a, b) in {"페르소나 문장×문항 호칭": (0, 1), "페르소나 문장×지시문": (0, 2), "문항 호칭×지시문": (1, 2)}.items():
    xadd("validation", f"2차 상호작용 {nm}", inter(T, a, b), VAL)
for eff, f in MAIN.items():
    xadd("197문항", eff, f(T), EV)
    xadd("197문항 β=1", eff, f(T1), EV)
xadd("validation, -시- 문항만", "주체 높임 제거(151문항 중)", MAIN["주체 높임 제거(-시- 유지→제거)"](T), VAL & HAS_SI)
xadd("197문항, -시- 문항만", "주체 높임 제거(151문항 중)", MAIN["주체 높임 제거(-시- 유지→제거)"](T), EV & HAS_SI)
xadd("회귀 점검 validation", "이 사람·하십시오체 − 원문", T["이사람하십시오"] - T["원문"], VAL)
xadd("회귀 점검 197문항", "이 사람·하십시오체 − 원문", T["이사람하십시오"] - T["원문"], EV)
for c in CONDS[1:]:
    xadd("단순 대비 validation", f"{LABEL[c]} − 원문", T[c] - T["원문"], VAL)
xd = pd.DataFrame(xrows)
xd.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색 (보정 없음)]")
log(xd.round(4).to_string(index=False))

old = pd.read_csv(ROOT / "experiments/20261123-register-final/out/explore.csv")
ref = old[(old["설정"] == "DeepSeek Eco 추론 low") & (old["분석"] == "단순 대비 validation β 규칙") & (old["효과"].str.contains("이 사람·하십시오체"))]
log("\n[회귀 점검: register-final 기록]")
log(ref[["효과", "차이", "하한", "상한"]].round(4).to_string(index=False))

# 최빈 보기
top = lambda x: np.where(MASK, x, -np.inf).argmax(-1)
hm, base = top(HU), top(C["원문"])
mrows = []
for c in CONDS[1:]:
    mc = top(C[c])
    ch = (mc != base) & EV
    mrows.append(dict(조건=LABEL[c], 최빈_바뀐_비율=float(ch.sum() / EV.sum()),
                      바뀐_문항에서_새_최빈이_사람과_같음=float((mc[ch] == hm[ch]).mean()) if ch.any() else np.nan,
                      바뀐_문항에서_원문_최빈이_사람과_같음=float((base[ch] == hm[ch]).mean()) if ch.any() else np.nan))
md = pd.DataFrame(mrows)
md.to_csv(out / "mode.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 원문 대비 최빈 보기가 바뀐 문항 비율 (197문항)]")
log(md.round(3).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for k, p in RAWF.items()},
           "params": {"k": K, "bootstrap": B, "seed": 20261123, "beta": BETA, "beta_source": "20261123-register-final/out/betas.csv 한국 성인 low"},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
