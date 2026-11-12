#!/usr/bin/env python3
"""실행이 끝난 뒤 결과를 점검하고 공유용 zip 한 개로 묶는다.

사용:  python3 finalize.py
출력:  KOSMOS_EXAONE_결과_m{담당}_{이름}.zip
       (결과 jsonl, 실행 로그, summary.json — API 키는 절대 들어가지 않는다)

완료 기준을 못 채우면 zip을 만들지 않고 이유를 알려준다.
"""
from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    m, name, phase = cfg["member"], cfg["name"], cfg.get("phase", 1)
    lo, hi = cfg["sample_from"], cfg["sample_to"]
    out = HERE / f"exaone_m{m}_p{phase}.jsonl"
    log = HERE / f"exaone_m{m}_p{phase}.log"
    if not out.exists():
        sys.exit(f"{out.name}이 없습니다. 먼저 run_exaone_team.py를 실행하세요.")

    probes = {r["probe_id"] for r in csv.DictReader((HERE / "prompts_v2.csv").open(encoding="utf-8-sig"))
              if int(r["phase"]) == phase}
    expected = len(probes) * (hi - lo + 1)

    ok, err, dup, bad = {}, 0, 0, 0
    inv = collections.Counter()
    dates, models = set(), set()
    for line in out.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error":
            err += 1
            continue
        if not (lo <= r["sample_idx"] <= hi) or r["probe_id"] not in probes:
            bad += 1
            continue
        key = (r["probe_id"], r["sample_idx"])
        if key in ok:
            dup += 1
            continue
        ok[key] = r
        inv[r["invalid_type"] or "valid"] += 1
        dates.add(r.get("run_date"))
        models.add((r.get("model_id"), r.get("reasoning_effort")))

    missing = expected - len(ok)
    summary = {
        "member": m, "name": name, "phase": phase, "samples": f"{lo}-{hi}",
        "expected": expected, "completed": len(ok), "missing": missing,
        "valid_rate": round(inv["valid"] / max(len(ok), 1), 4),
        "invalid_breakdown": dict(inv), "api_error_lines": err, "duplicates_ignored": dup,
        "out_of_scope_ignored": bad, "models": sorted(map(list, models)), "run_dates": sorted(d for d in dates if d),
        "prompts_sha256": sha256(HERE / "prompts_v2.csv"),
        "prompts_sha256_expected": cfg.get("prompts_sha256"),
        "results_sha256": sha256(out),
        "finalized_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    problems = []
    if missing > 0:
        problems.append(f"아직 {missing}건이 남았습니다. run_exaone_team.py를 다시 실행하세요.")
    if summary["prompts_sha256"] != cfg.get("prompts_sha256"):
        problems.append("prompts_v2.csv가 배포본과 다릅니다.")
    if len(models) > 1:
        problems.append("모델 설정이 섞여 있습니다.")
    if problems:
        sys.exit("zip을 만들지 않았습니다:\n  - " + "\n  - ".join(problems))

    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    zname = HERE / f"KOSMOS_EXAONE_결과_m{m}_{name}.zip"
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, out.name)
        if log.exists():
            z.write(log, log.name)
        z.write(HERE / "summary.json", "summary.json")
    print(f"\n완료: {zname.name} ({zname.stat().st_size / 1e6:.1f} MB) — 이 파일 하나만 태우에게 보내면 됩니다.")


if __name__ == "__main__":
    main()
