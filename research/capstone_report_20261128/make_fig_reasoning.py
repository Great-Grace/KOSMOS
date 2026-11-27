"""그림 4-2: 추론 단계별 '모델이 움직인 거리'와 '사람과의 거리 변화' (성별×연령, 197문항, k=20, 원비율).
입력: research/summary_register_reasoning_20261126/four_levels/out/shift.csv"""
import csv, json, hashlib, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SRC = pathlib.Path(__file__).resolve().parents[1] / "summary_register_reasoning_20261126/four_levels/out/shift.csv"
rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
P = "성별×연령(10개 평균)"
def get(m, c, k):
    for r in rows:
        if r["모델"] == m and r["비교"] == c and r["페르소나"] == P and r["지표"] == k:
            return float(r["값"]), float(r["하한"]), float(r["상한"])
    raise KeyError((m, c, k))

steps = [("DeepSeek", "끔 → low"), ("DeepSeek", "low → medium"), ("DeepSeek", "medium → high"),
         ("Haiku", "끔 → medium"), ("Haiku", "medium → high")]
data = [(m, c, get(m, c, "전후 TVD")[0], get(m, c, "잡음 TVD")[0], get(m, c, "사람 TVD 변화")) for m, c in steps]

for f in ["AppleSDGothicNeo-Regular", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b66"

fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
h = 0.36
ys = []
for i, (m, c, move, noise, (dh, lo, hi)) in enumerate(data):
    y = (len(data) - 1 - i) * 1.15 - (0.45 if i >= 3 else 0) + 0.45
    ys.append(y)
    ax.barh(y + h/2 + 0.02, move, h, color=BLUE)
    ax.plot([noise, noise], [y + 0.04, y + h + 0.0], color=INK, lw=1.6)
    ax.barh(y - h/2 - 0.02, dh, h, color=ORANGE)
    ax.errorbar(dh, y - h/2 - 0.02, xerr=[[dh - lo], [hi - dh]], fmt="none", ecolor=INK, lw=0.9, capsize=2)
    ax.text(move + 0.006, y + h/2 + 0.02, f"{move:.3f}", va="center", fontsize=10, color=INK)
    tx = (max(hi, 0) + 0.006)
    ax.text(tx, y - h/2 - 0.02, f"{dh:+.3f}", va="center", fontsize=10, color=INK)
ax.set_yticks(ys)
ax.set_yticklabels([f"{m}\n{c}" for m, c, *_ in data], fontsize=10.5, color=INK)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.06, 0.40)
ax.set_xlabel("TVD", fontsize=10, color=MUTED)
ax.tick_params(axis="x", labelsize=9.5, colors=MUTED)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=BLUE, label="모델 분포가 움직인 거리 (전후 TVD)"),
                   Line2D([0], [0], color=INK, lw=1.6, label="같은 분포에서 다시 뽑을 때의 잡음 TVD"),
                   Patch(color=ORANGE, label="사람과의 거리 변화 (음수 = 가까워짐, 막대는 95% 구간)")],
          fontsize=9, loc="lower right", frameon=False)
fig.tight_layout()
out = pathlib.Path(__file__).with_name("fig_reasoning_shift.png")
fig.savefig(out)
json.dump({"input": str(SRC), "sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "values": [{"model": m, "step": c, "shift_tvd": mv, "noise_tvd": nz, "human_tvd_change": list(d)} for m, c, mv, nz, d in data]},
          open(out.with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
print(out)
