# -*- coding: utf-8 -*-
"""그림 4-1: 보정 지수 β에 따른 보정용 148문항의 평균 TVD (한국 성인, k=100). 점은 각 모델의 최적 β.
입력: research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz"""
import numpy as np, json, hashlib, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[0] / "design_v2_20261112/analysis/out_final_k100/counts_k100.npz"
z = np.load(SRC, allow_pickle=True)
C, H, CODES, split = z["C"], z["H"], z["CODES"], z["split"]
MASK = ~np.isnan(CODES); cal = split == "calibration"; L0 = 1
def calibrated(c, beta):
    x = np.where(MASK, (c + 1.0) ** beta, 0.0); return x / x.sum(-1, keepdims=True)
tvd = lambda p, q: 0.5 * np.abs(p - q).sum(-1)
grid = np.round(np.arange(0.05, 1.5001, 0.005), 3)
names = {"exaone": "K-EXAONE", "deepseek": "DeepSeek", "haiku": "Haiku"}
curves = {names[m]: np.array([tvd(calibrated(C[a, L0], b), H[L0])[cal].mean() for b in grid]) for a, m in enumerate(z["models"])}
unif = tvd(np.where(MASK, 1 / MASK.sum(1, keepdims=True), 0), H[L0])[cal].mean()

for f in ["AppleSDGothicNeo-Regular", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
COL = {"K-EXAONE": "#2a78d6", "DeepSeek": "#eb6834", "Haiku": "#1baf7a"}
INK, MUTED = "#1f1f1e", "#6b6b66"
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
best = {}
for n, y in curves.items():
    ax.plot(grid, y, color=COL[n], lw=2)
    i = int(y.argmin()); best[n] = (float(grid[i]), float(y[i]))
    ax.plot(grid[i], y[i], "o", ms=8, color=COL[n], mec="white", mew=2, zorder=5)
    ax.text(1.515, y[-1], f"{n}\n최적 β {grid[i]:.3f}", color=INK, fontsize=10, va="center", linespacing=1.35)
ax.axhline(unif, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(0.06, unif + 0.006, f"균등분포 {unif:.3f}", color=MUTED, fontsize=9.5, va="bottom")
ax.axvline(1.0, color="#e4e3df", lw=1); ax.text(1.0, 0.212, "보정 전 (β = 1)", color=MUTED, fontsize=9, ha="center")
ax.set_xlim(0.03, 1.5); ax.set_ylim(0.2, 0.56)
ax.set_xlabel("보정 지수 β (작을수록 강하게 편다)", fontsize=10, color=MUTED)
ax.set_ylabel("평균 TVD (보정용 148문항)", fontsize=10, color=MUTED)
ax.tick_params(labelsize=9.5, colors=MUTED)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color(MUTED)
ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.1, right=0.84, top=0.97, bottom=0.12)
out = HERE / "fig_beta.png"; fig.savefig(out)
json.dump({"input": str(SRC.relative_to(HERE.parents[1])), "sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "uniform": float(unif), "best": best, "at_beta_1": {n: float(y[list(grid).index(1.0)]) for n, y in curves.items()}},
          open(HERE / "fig_beta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(best, round(float(unif), 4))
