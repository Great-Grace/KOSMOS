"""fig_temperature_role.py의 차트만 남긴 버전 (제목·부제·주석 없음, 슬라이드에 붙여 쓰는 용도).

산출 (out_calib/figs/plain/):
  temperature_<모델>_<문항>_before_T1.png      한 문항, 보정 전 모델 분포 vs KGSS
  temperature_<모델>_<문항>_after_T<값>.png    같은 문항, 보정 후 (두 그림 y축 동일)
  tvd_vs_temperature_L0_calib148.png           T에 따른 calibration 148문항 평균 TVD (4개 모델 + 균등분포)

실행: .venv-nb/bin/python research/design_v2_20261112/analysis/fig_temperature_role_plain.py [모델] [문항]
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
exec((HERE / "legacy_vs_v2.py").read_text(encoding="utf-8").split("# ---------------- RADIUS")[0])

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EXAONE-3.5-2.4B"
VAR = sys.argv[2] if len(sys.argv) > 2 else "PERTRT5"
OUTD = HERE / "out_calib/figs/plain"; OUTD.mkdir(parents=True, exist_ok=True)

BG_COLOR, INK2, MUTED, GRID, AXIS = "#ffffff", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HUMAN = "#b9b8b0"
SERIES = {"EXAONE-3.5-2.4B": "#2a78d6", "Kanana-1.5-2.1B": "#eb6834",
          "DeepSeek V4.1 Flash": "#1baf7a", "Haiku 5.5": "#eda100"}
plt.rcParams.update({"font.family": "Pretendard", "axes.unicode_minus": False, "figure.facecolor": BG_COLOR,
                     "axes.facecolor": BG_COLOR, "savefig.facecolor": BG_COLOR, "text.color": INK2,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=11)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def option_labels(var):
    for r in csv.DictReader(open(RE / "adult_clean/probes.csv", encoding="utf-8-sig")):
        if r["var"] == var:
            block = r["prompt"].split("선택지:")[1].split("번호로만")[0]
            return [re.sub(r"^\d+\.\s*", "", s).strip() for s in block.strip().splitlines() if s.strip()]
    raise KeyError(var)


def best_beta(kind, X):
    G = np.stack([tvd_lv(prob(kind, X, b)) for b in BG])
    return BG[G[:, 1, TRAIN].mean(-1).argmin()], G


# ---- 막대: 보정 전 / 후 ----
kind, X = MODELS[MODEL]
i = ITEMS.index(VAR); k = NOPT[i]
beta, _ = best_beta(kind, X)
h = H[1, i, :k]
labels = option_labels(VAR)
ymax = max(prob(kind, X, 1.0)[1, i, :k].max(), prob(kind, X, beta)[1, i, :k].max(), h.max()) * 1.12
tag = MODEL.split("-")[0] + "-" + MODEL.split("-")[-1]          # EXAONE-2.4B
outs = []
for name, b in [("before_T1", 1.0), (f"after_T{1 / beta:.1f}", beta)]:
    p = prob(kind, X, b)[1, i, :k]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(k); w = 0.36
    ax.bar(x - w / 2 - 0.01, h, w, color=HUMAN, label="KGSS 2023", zorder=3)
    ax.bar(x + w / 2 + 0.01, p, w, color=SERIES[MODEL], label=MODEL, zorder=3)
    for xx, v in zip(x, h):
        ax.text(xx - w / 2 - 0.01, v + ymax * 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color=INK2)
    for xx, v in zip(x, p):
        ax.text(xx + w / 2 + 0.01, v + ymax * 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color="#0b0b0b")
    ax.set_xticks(x, [l.replace(" ", "\n", 1) if len(l) > 7 else l for l in labels], fontsize=11, color=INK2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_ylim(0, ymax)
    style(ax)
    lg = ax.legend(loc="upper left", frameon=True, fontsize=10.5, handlelength=1.0, handleheight=1.0, labelcolor=INK2)
    lg.get_frame().set(facecolor=BG_COLOR, edgecolor="none", alpha=1.0); lg.set_zorder(5)
    fig.tight_layout()
    out = OUTD / f"temperature_{tag}_{VAR}_{name}.png"; fig.savefig(out, dpi=220); plt.close(fig); outs.append(out)
    print(out, f"TVD={tvd(p, h):.4f}")

# ---- 선: T별 평균 TVD ----
fig, ax = plt.subplots(figsize=(8.5, 5))
T = 1 / BG; sel = (T >= 1) & (T <= 20)
for name, (kd, Xm) in MODELS.items():
    b, G = best_beta(kd, Xm); curve = G[:, 1, TRAIN].mean(-1)
    ax.plot(T[sel], curve[sel], color=SERIES[name], lw=2, zorder=3, label=name)
    ax.scatter([1], [curve[list(BG).index(1.0)]], s=40, color=BG_COLOR, edgecolor=SERIES[name], linewidth=2, zorder=4)
    ax.scatter([1 / b], [curve[list(BG).index(b)]], s=64, color=SERIES[name], edgecolor=BG_COLOR, linewidth=2, zorder=4)
ax.axhline(tvd(MASK / NOPT[:, None], H[1])[TRAIN].mean(), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2, label="균등분포")
ax.set_xscale("log"); ax.set_xlim(0.95, 21)
ax.set_xticks([1, 2, 3, 5, 10, 20], ["1", "2", "3", "5", "10", "20"]); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("T", fontsize=11); ax.set_ylabel("TVD", fontsize=11)
style(ax)
ax.legend(loc="upper right", frameon=False, fontsize=10.5, labelcolor=INK2)
fig.tight_layout()
out = OUTD / "tvd_vs_temperature_L0_calib148.png"; fig.savefig(out, dpi=220); plt.close(fig); outs.append(out)
print(out)
