"""설계 v2 manifest를 고정한다.

입력: items_246.csv, cells.csv, prompts_v2.csv,
      item_classification_jev-omni.csv(주 분류), item_classification_decider-4b.csv(교차 확인)
출력:
  item_manifest_v2.csv  문항별 고정 정보 + 분류
  MANIFEST.json         파일 해시, 개수, 실험 설정. 이 파일의 해시가 바뀌면 설계가 바뀐 것이다.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    return list(csv.DictReader((HERE / name).open(encoding="utf-8-sig")))


def main() -> None:
    items = rows("items_246.csv")
    J = {r["var"]: r for r in rows("item_classification_jev-omni.csv")}
    D = {r["var"]: r for r in rows("item_classification_decider-4b.csv")}

    out = []
    for it in items:
        v = it["var"]
        j, d = J[v], D[v]
        flags = []
        if j["topic"] != d["topic"]:
            flags.append("topic_disagree")
        if j["desirable_end"] != d["desirable_end"]:
            flags.append("desirable_disagree")
        if float(j.get("topic_confidence") or 0) < 0.7:
            flags.append("topic_low_conf")
        if it["scale_type"] == "nominal_unordered" and j["desirable_end"] in ("first", "last"):
            flags.append("nominal_with_direction")
        out.append({
            "var": v, "label": it["label"], "k": it["k"], "codes": it["codes"],
            "scale_type": it["scale_type"], "content_domain": it["content_domain"],
            "battery": it["battery"], "split_60_20_20": it["split_60_20_20"],
            "cv_fold": it["cv_fold"], "group_cv_fold": it["group_cv_fold"],
            "topic": j["topic"], "topic_confidence": j["topic_confidence"], "topic_decider": d["topic"],
            "sensitivity_level": j["sensitivity_level"], "sensitivity_level_decider": d["sensitivity_level"],
            "desirable_end": j["desirable_end"], "desirable_end_decider": d["desirable_end"],
            "review_flags": ";".join(flags),
        })

    mpath = HERE / "item_manifest_v2.csv"
    with mpath.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    prompts = rows("prompts_v2.csv")
    manifest = {
        "name": "KOSMOS design v2",
        "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": {n: sha256(HERE / n) for n in
                  ["items_246.csv", "cells.csv", "prompts_v2.csv", "item_manifest_v2.csv"]},
        "counts": {
            "items": len(items),
            "prompts_total": len(prompts),
            "prompts_phase1": sum(p["phase"] == "1" for p in prompts),
            "prompts_phase2": sum(p["phase"] == "2" for p in prompts),
            "k": 100,
            "calls_per_model_phase1": sum(p["phase"] == "1" for p in prompts) * 100,
        },
        "conditions": {
            "phase1": ["c0", "L0", "L1 (성별×연령5, 10셀)"],
            "phase2": ["L2 (성별×연령3×수도권, 12셀) — 1단계 결과를 보고 결정"],
        },
        "system_prompt": prompts[0]["system"],
        "classification": {
            "primary": "sionic-ai/jev-omni (OpenGateway Decisions)",
            "cross_check": "sionic-ai/decider-4b",
            "run_date": datetime.date.today().isoformat(),
            "note": "분류는 분석용 라벨이며 프롬프트에는 영향이 없다.",
        },
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nflag = sum(bool(r["review_flags"]) for r in out)
    print(f"item_manifest_v2.csv {len(out)}행 (검수 플래그 {nflag}개), MANIFEST.json 저장")


if __name__ == "__main__":
    main()
