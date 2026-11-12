#!/usr/bin/env python3
"""모델 원응답 jsonl → 프롬프트별 보기 개수 벡터 CSV.

api_error 줄은 버리고, (probe_id, sample_idx)가 겹치면 먼저 나온 것을 쓴다.
DeepSeek(run_deepseek_eco.py)와 K-EXAONE 병합본(merge_check.py 출력) 모두 읽는다.

사용 (KOSMOS 루트에서):
  python3 research/design_v2_20261112/runners/aggregate_counts.py raw/v2/deepseek_ultrafast_p1.jsonl
출력: 같은 위치에 <이름>_counts.csv
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

DESIGN = Path(__file__).resolve().parent.parent


def main() -> None:
    src = Path(sys.argv[1])
    probes = {p["probe_id"]: p for p in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig"))}
    seen, recs = set(), []
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error":
            continue
        k = (r["probe_id"], r["sample_idx"])
        if k in seen:
            continue
        seen.add(k)
        recs.append(r)

    by = collections.defaultdict(list)
    for r in recs:
        by[r["probe_id"]].append(r)

    out = src.with_name(src.stem + "_counts.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_id", "phase", "level", "cell_id", "var", "codes", "n_total", "n_valid",
                    "n_refusal", "n_other_invalid", "counts_json"])
        for pid, rs in sorted(by.items()):
            p = probes[pid]
            codes = json.loads(p["codes"])
            c = collections.Counter(r["parsed"] for r in rs if r["valid"])
            n_ref = sum(r["invalid_type"] == "refusal" for r in rs)
            n_valid = sum(c.values())
            w.writerow([pid, p["phase"], p["level"], p["cell_id"], p["var"], p["codes"], len(rs), n_valid,
                        n_ref, len(rs) - n_valid - n_ref, json.dumps([c.get(x, 0) for x in codes])])
    print(f"{out}: 프롬프트 {len(by)}개, 응답 {len(recs)}건")


if __name__ == "__main__":
    main()
