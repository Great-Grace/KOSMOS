"""20261123-reasoning-high-full — 실행: python experiments/20261123-reasoning-high-full/run.py  (KOSMOS 루트에서, 수 초)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl  # noqa: E402

df = sl.load()                         # test 49문항 제외
beta = sl.fit_beta(df)                 # calibration에서 모델×단계별 β*
val = df[df["split"] == "validation"]

tables = {
    "validation_tvd_beta_star": sl.summary(val, beta),
    "validation_tvd_beta1": sl.summary(val, 1.0),
}
for name, t in tables.items():
    print(f"\n[{name}]\n{t.round(3)}")

run = sl.save_run(HERE / "out", params={"beta_star": beta, "alpha": sl.ALPHA}, tables=tables)
print(f"\n기록: {HERE / 'out/run.json'} (commit {run['git_commit']}, dirty={run['git_dirty']})")
