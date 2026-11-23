"""사후 점검(결과를 본 뒤 추가): 문장이 완전히 같은 문항에서 '세 요소 모두 이 사람(주체 높임 유지)'과 '이 사람·하십시오체'를 비교한다.
실행 (KOSMOS 루트): PYTHONUTF8=1 python experiments/20261123-register-decomp-off/drift.py [off|low]

-시-가 없는 95문항은 두 조건의 프롬프트가 글자까지 같다. 따라서 차이는 표집 잡음과 수집 시점 차이(2026-11-19 vs 11-23)뿐이다.
1) 보정 후 사람 TVD 차이(문항 짝 부트스트랩) 2) 두 조건 사이 TVD를 같은 분포에서 50개씩 두 번 뽑은 잡음 기준선(1,000회)과 비교.
low는 중앙 PR #2(joohee)의 원응답이 raw/v2에 풀려 있을 때만 돈다. test 49문항은 쓰지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
which = sys.argv[1] if len(sys.argv) > 1 else "off"
src = (HERE / "run.py").read_text(encoding="utf-8").split("# ── 주 가설군")[0]
src = src.replace('SETTING = "끔"', f'SETTING = "{"끔" if which == "off" else "low"}"')
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "import io as _io; logf = _io.StringIO()", 1)
src = src.replace("    print(s)\n", "")
g = {"__file__": str(HERE / "run.py")}
exec(src, g)
T, C, VAL, EV, HAS_SI, MASK, boot, VARS = (g[k] for k in ("T", "C", "VAL", "EV", "HAS_SI", "MASK", "boot", "VARS"))
A_, B_ = C["all_keep_si"], C["이사람하십시오"]
same = ~HAS_SI
import csv
reg = {r["var"]: r["user"] for r in csv.DictReader((g["ROOT"] / "research/design_v2_20261112/register_v1/out/prompts_register_v1.csv").open(encoding="utf-8-sig")) if r["level"] == "L0" and r["cond"] == "C"}
dec = {r["var"]: r["user"] for r in csv.DictReader(g["DECOMP_CSV"].open(encoding="utf-8-sig")) if r["cond"] == "all_keep_si"}
assert all(reg[v] == dec[v] for v, s in zip(VARS, same) if s)          # 글자까지 같은지 확인
rng = np.random.default_rng(20261123)
rows = []
for nm, sel in {"validation 49 전체": VAL, "validation 같은 문장": VAL & same, "validation 다른 문장(-시- 제거)": VAL & HAS_SI,
                "197 전체": EV, "197 같은 문장": EV & same, "197 다른 문장(-시- 제거)": EV & HAS_SI}.items():
    m, lo, hi, p, n = boot(T["이사람하십시오"] - T["all_keep_si"], sel)
    rows.append(dict(문항=nm, 문항수=n, TVD_차이=m, 하한=lo, 상한=hi))
pd.DataFrame(rows).to_csv(HERE / "out" / f"drift_tvd_{which}.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).round(4).to_string(index=False))

share = lambda x: x / np.maximum(x.sum(-1, keepdims=True), 1)
obs = 0.5 * np.abs(share(A_) - share(B_)).sum(-1)
idx = np.where(EV & same)[0]
null = np.zeros((1000, len(idx)))
for j, i in enumerate(idx):
    k = int(MASK[i].sum())
    p = share(A_[i] + B_[i])[:k]
    na, nb = int(A_[i].sum()), int(B_[i].sum())
    null[:, j] = 0.5 * np.abs(rng.multinomial(na, p, 1000) / na - rng.multinomial(nb, p, 1000) / nb).sum(-1)
nm = null.mean(1)
r = dict(문항수=len(idx), 조건_사이_TVD_평균=obs[idx].mean(), 잡음_기준선_평균=nm.mean(), 잡음_97_5백분위=np.percentile(nm, 97.5),
         잡음_95백분위_넘은_문항=int((obs[idx] > np.percentile(null, 95, axis=0)).sum()), p=float((nm >= obs[idx].mean()).mean()))
pd.DataFrame([r]).to_csv(HERE / "out" / f"drift_noise_{which}.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame([r]).round(4).to_string(index=False))
