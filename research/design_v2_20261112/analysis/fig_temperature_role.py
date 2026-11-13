"""온도(T = 1/β)의 역할을 보여 주는 발표용 그림 두 장.

그림 1  temperature_role_<모델>_<문항>.png
        한 문항에서 모델 분포와 KGSS 분포를 같은 축에 겹쳐 그린다. 왼쪽은 보정 전(T=1), 오른쪽은 보정 후(T*).
그림 2  temperature_curve_L0.png
        T를 키울 때 calibration 문항의 평균 TVD가 어떻게 변하는지(모델별), 고른 T*와 균등분포 기준선.

수치는 legacy_vs_v2.py와 같은 규칙으로 계산한다(그 파일의 데이터 적재 부분을 그대로 실행).
T*는 calibration 148문항에서 고른 값이고, 그림 1의 문항은 test 문항이다.

실행: .venv-nb/bin/python research/design_v2_20261112/analysis/fig_temperature_role.py [모델] [문항]
      기본값: EXAONE-3.5-2.4B PERTRT5  (EXAONE의 test 문항 보정 이득 중앙값 0.167 근처, 쏠림이 뚜렷한 사례)
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
_src = (HERE / "legacy_vs_v2.py").read_text(encoding="utf-8").split("# ---------------- RADIUS")[0]
exec(_src)   # ITEMS, H, MODELS, prob, tvd, tvd_lv, BG, TRAIN, TEST, MASK, NOPT, man, ROOT, RE 등

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EXAONE-3.5-2.4B"
VAR = sys.argv[2] if len(sys.argv) > 2 else "PERTRT5"
FIG = HERE / "out_calib/figs"; FIG.mkdir(parents=True, exist_ok=True)

# ---- 디자인 토큰 (dataviz 기본 팔레트, 라이트) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
HUMAN = "#b9b8b0"                      # KGSS: 기준(정답)이라 중립 회색
SERIES = {"EXAONE-3.5-2.4B": "#2a78d6", "Kanana-1.5-2.1B": "#eb6834",
          "DeepSeek V4.1 Flash": "#1baf7a", "Haiku 5.5": "#eda100"}   # 고정 순서, 모든 그림에서 같은 색
plt.rcParams.update({"font.family": "Pretendard", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, ygrid=True):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def option_labels(var):
    for r in csv.DictReader(open(RE / "adult_clean/probes.csv", encoding="utf-8-sig")):
        if r["var"] == var:
            block = r["prompt"].split("선택지:")[1].split("번호로만")[0]
            stem = r["prompt"].split("문항:")[1].split("선택지:")[0].strip()
            return [re.sub(r"^\d+\.\s*", "", s).strip() for s in block.strip().splitlines() if s.strip()], stem
    raise KeyError(var)


def best_beta(kind, X, level=1):
    G = np.stack([tvd_lv(prob(kind, X, b)) for b in BG])
    return BG[G[:, level, TRAIN].mean(-1).argmin()], G


# =================== 그림 1: 한 문항, 보정 전·후 ===================
kind, X = MODELS[MODEL]
i = ITEMS.index(VAR); k = NOPT[i]
beta, _ = best_beta(kind, X)
p_raw, p_cal, h = prob(kind, X, 1.0)[1, i, :k], prob(kind, X, beta)[1, i, :k], H[1, i, :k]
labels, stem = option_labels(VAR)
assert len(labels) == k
split = man.split_60_20_20[i]
color = SERIES[MODEL]

fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
x = np.arange(k); w = 0.36
ymax = max(p_raw.max(), h.max(), p_cal.max())
for ax, p, title in [(axs[0], p_raw, "보정 전  (T = 1)"), (axs[1], p_cal, f"보정 후  (T = {1 / beta:.1f})")]:
    ax.bar(x - w / 2 - 0.01, h, w, color=HUMAN, label="KGSS 2023 (한국 성인, 가중)", zorder=3)
    ax.bar(x + w / 2 + 0.01, p, w, color=color, label=f"{MODEL} (‘한국 성인’ 페르소나)", zorder=3)
    for xx, v in zip(x, p):
        ax.text(xx + w / 2 + 0.01, v + ymax * 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9.5, color=INK)
    for xx, v in zip(x, h):
        ax.text(xx - w / 2 - 0.01, v + ymax * 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xticks(x, [l.replace(" ", "\n", 1) if len(l) > 7 else l for l in labels], fontsize=10, color=INK2)
    ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, ymax * 1.15)
    style(ax)
    t = tvd(p, h)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, f"TVD {t:.3f}   ·   최빈 보기 확률 {p.max():.0%} (사람 {h.max():.0%})",
            transform=ax.transAxes, fontsize=10.5, color=INK2)
axs[0].set_ylabel("응답 비율", fontsize=10.5)
h1, l1 = axs[0].get_legend_handles_labels()
fig.legend(h1, l1, loc="upper right", bbox_to_anchor=(0.985, 0.92), ncol=2, frameon=False, fontsize=10.5,
           handlelength=1.0, handleheight=1.0, labelcolor=INK2)
fig.text(0.012, 0.975, "온도 보정은 모델의 쏠린 분포를 사람 분포 쪽으로 편다", fontsize=17, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.912, f"{VAR} · {man.label[i]}  ({split} 문항)", fontsize=11, color=INK2, va="top")
fig.text(0.012, 0.018, f"보정: p_model^(1/T)을 합이 1이 되게 정규화.  T는 calibration 148문항의 평균 TVD가 가장 작은 값(β* = {beta:.3f}).  "
                       f"보기 순서는 바뀌지 않고 폭만 바뀐다.  정답: KGSS 2023, FINALWT 가중.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.86))
out1 = FIG / f"temperature_role_{MODEL.split('-')[0]}_{VAR}.png"
fig.savefig(out1, dpi=200); plt.close(fig)

# =================== 그림 2: T에 따른 평균 TVD ===================
fig, ax = plt.subplots(figsize=(10, 5.6))
T = 1 / BG
unif_tr = tvd(MASK / NOPT[:, None], H[1])[TRAIN].mean()
ends = []
for name, (kd, Xm) in MODELS.items():
    b, G = best_beta(kd, Xm)
    curve = G[:, 1, TRAIN].mean(-1)
    sel = (T >= 1) & (T <= 20)
    ax.plot(T[sel], curve[sel], color=SERIES[name], lw=2, zorder=3)
    ax.scatter([1 / b], [curve[list(BG).index(b)]], s=64, color=SERIES[name], edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.scatter([1], [curve[list(BG).index(1.0)]], s=40, color=SURFACE, edgecolor=SERIES[name], linewidth=2, zorder=4)
    ends.append((curve[list(BG).index(1.0)], name, 1 / b, curve[list(BG).index(b)]))
ax.axhline(unif_tr, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(20, unif_tr + 0.006, f"균등분포 {unif_tr:.3f}", ha="right", va="bottom", fontsize=10, color=INK2)
# 직접 라벨: 시작점(T=1) 왼쪽에 모델명과 T*·최저값. 시작점이 가까운 쌍은 최소 간격만큼 벌린다
ends.sort(key=lambda e: -e[0]); ys = []
for y0, *_ in ends:
    ys.append(min(y0, ys[-1] - 0.034) if ys else y0)
for (y0, name, tb, vb), y in zip(ends, ys):
    ax.annotate(f"{name}\nT*={tb:.1f} → {vb:.3f}", (1, y0), xytext=(0.93, y), textcoords="data",
                ha="right", va="center", fontsize=9.5, color=INK, linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if abs(y - y0) > 1e-9 else None)
ax.set_xscale("log"); ax.set_xlim(0.42, 21)
ax.set_xticks([1, 2, 3, 5, 10, 20], ["1", "2", "3", "5", "10", "20"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("온도 T  (= 1/β, 로그 눈금)", fontsize=10.5); ax.set_ylabel("평균 TVD (calibration 148문항, 낮을수록 좋음)", fontsize=10.5)
style(ax)
ax.xaxis.grid(False)
ax.spines["bottom"].set_bounds(1, 20)
fig.text(0.012, 0.975, "온도를 올리면 과신이 풀리고, 지나치면 다시 나빠진다", fontsize=16, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.915, "‘한국 성인’(L0) 조건 · 빈 점 = 보정 전(T=1), 찬 점 = 고른 T*", fontsize=11, color=INK2, va="top")
fig.text(0.012, 0.018, "소형 모델(EXAONE·Kanana)은 로짓 분포, DeepSeek·Haiku는 k=100 표집 개수 + Laplace(+1). 같은 꼴의 보정 x^(1/T) 정규화.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.88))
out2 = FIG / "temperature_curve_L0.png"
fig.savefig(out2, dpi=200); plt.close(fig)
print(out1); print(out2)
print(f"{MODEL} {VAR}: β*={beta} T*={1/beta:.2f}  raw TVD {tvd(p_raw, h):.4f} → cal {tvd(p_cal, h):.4f}")
