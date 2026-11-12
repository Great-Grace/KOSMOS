"""설계 v2의 페르소나 × 문항 프롬프트를 만든다.

입력: items_246.csv, cells.csv (같은 폴더)
출력: prompts_v2.csv
  probe_id, phase(1=c0·L0·L1, 2=L2), level, cell_id, var, codes, system, user

머리말 문체는 "당신은 …입니다"(2인칭·하십시오체) 하나다.
대표 나이는 DESIGN.md 3절을 따른다.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = "보기 번호 숫자 하나만 출력하고 다른 말은 하지 마십시오."

L1_AGE = {"18-29": 26, "30-39": 35, "40-49": 45, "50-59": 55, "60+": 66}
L2_AGE = {"18-39": 30, "40-59": 50, "60+": 67}
CAPITAL_TEXT = {"수도권": "수도권(서울·인천·경기)", "비수도권": "비수도권"}


def main() -> None:
    items = list(csv.DictReader((HERE / "items_246.csv").open(encoding="utf-8-sig")))
    cells = list(csv.DictReader((HERE / "cells.csv").open(encoding="utf-8-sig")))

    headers = [(1, "c0", "c0", ""), (1, "L0", "L0", "당신은 한국 성인입니다.")]
    for c in cells:
        if c["level"] == "L1":
            age = L1_AGE[c["age_band"]]
            headers.append((1, "L1", c["cell_id"], f"당신은 {age}세 {c['sex']}입니다."))
        elif c["level"] == "L2":
            age = L2_AGE[c["age_band"]]
            headers.append((2, "L2", c["cell_id"],
                            f"당신은 {CAPITAL_TEXT[c['capital']]}에 사는 {age}세 {c['sex']}입니다."))

    rows = []
    for phase, level, cell_id, head in headers:
        for it in items:
            user = f"{head}\n\n{it['item_prompt']}" if head else it["item_prompt"]
            rows.append({
                "probe_id": f"{cell_id}__{it['var']}",
                "phase": phase, "level": level, "cell_id": cell_id, "var": it["var"],
                "codes": it["codes"], "system": SYSTEM, "user": user,
            })

    out = HERE / "prompts_v2.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n1 = sum(r["phase"] == 1 for r in rows)
    print(f"{out.name}: {len(rows)}행 (1단계 {n1}, 2단계 {len(rows) - n1})")


if __name__ == "__main__":
    main()
