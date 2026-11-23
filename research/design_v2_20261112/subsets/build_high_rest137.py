"""추론 high 확장용 나머지 137문항 — 실행 (KOSMOS 루트):
  python3 research/design_v2_20261112/subsets/build_high_rest137.py

test가 아닌 197문항 중 high60.csv에 없는 문항. high60(이미 수집)과 합치면 197문항 전부가 된다.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
items = list(csv.DictReader((HERE.parent / "items_246.csv").open(encoding="utf-8-sig")))
done = {r["var"] for r in csv.DictReader((HERE / "high60.csv").open(encoding="utf-8-sig"))}
rest = sorted((r for r in items if r["split_60_20_20"] != "test" and r["var"] not in done), key=lambda r: r["var"])
assert len(done) == 60 and len(rest) == 137
cols = ["var", "split_60_20_20", "scale_type", "content_domain"]
with (HERE / "high_rest137.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rest)
print(len(rest), "문항 →", HERE / "high_rest137.csv")
