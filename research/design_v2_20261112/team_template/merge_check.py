#!/usr/bin/env python3
"""팀원 4명의 K-EXAONE 결과를 합치고 점검한다.

사용:
  python3 merge_check.py --phase 1 exaone_m1_p1.jsonl KOSMOS_EXAONE_결과_m2_정주희.zip ...
  (jsonl과 팀원 결과 zip을 섞어 넣어도 된다)

점검:
  - 담당 번호와 표본 번호 구간이 맞는지
  - (프롬프트, 표본) 중복
  - 프롬프트별 유효 표본 수 (목표 100)
  - 모델·추론 설정이 모두 같은지

출력 (이 폴더):
  exaone_p{phase}_merged.jsonl   중복을 뺀 전체 원응답 (api_error 제외)
  exaone_p{phase}_counts.csv     프롬프트별 보기 개수 벡터
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_TOTAL, N_MEMBERS = 100, 4
PER = K_TOTAL // N_MEMBERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    probes = {r["probe_id"]: r for r in csv.DictReader((HERE / "prompts_v2.csv").open(encoding="utf-8-sig"))
              if int(r["phase"]) == args.phase}

    seen, recs, problems = {}, [], collections.Counter()
    settings = collections.Counter()
    def lines_of(fn):
        """jsonl 파일 또는 팀원 결과 zip(KOSMOS_EXAONE_결과_*.zip) 안의 jsonl을 읽는다."""
        if fn.endswith(".zip"):
            import io
            import zipfile
            with zipfile.ZipFile(fn) as z:
                for n in z.namelist():
                    if n.endswith(".jsonl"):
                        yield from io.TextIOWrapper(z.open(n), encoding="utf-8")
        else:
            yield from open(fn, encoding="utf-8")

    for fn in args.files:
        for line in lines_of(fn):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("invalid_type") == "api_error":
                problems["api_error(재실행 대상)"] += 1
                continue
            m, s = r["member"], r["sample_idx"]
            if not ((m - 1) * PER <= s < m * PER):
                problems["담당 구간 밖 표본"] += 1
                continue
            if r["probe_id"] not in probes:
                problems["프롬프트 파일에 없는 probe_id"] += 1
                continue
            key = (r["probe_id"], s)
            if key in seen:
                problems["중복 (뒤의 것 버림)"] += 1
                continue
            seen[key] = True
            settings[(r["model_id"], r["reasoning_effort"])] += 1
            recs.append(r)

    per_probe = collections.Counter(r["probe_id"] for r in recs)
    valid = collections.defaultdict(list)
    for r in recs:
        if r["valid"]:
            valid[r["probe_id"]].append(r["parsed"])
    per_member = collections.Counter(r["member"] for r in recs)

    print(f"레코드 {len(recs)}건 (목표 {len(probes) * K_TOTAL})")
    print("담당별:", dict(sorted(per_member.items())), f"(1인 목표 {len(probes) * PER})")
    print("설정:", dict(settings))
    if len(settings) > 1:
        print("  경고: 모델이나 추론 설정이 섞여 있습니다.")
    for k, v in problems.items():
        print(f"  {k}: {v}")
    counts = sorted(per_probe.get(p, 0) for p in probes)
    print(f"프롬프트별 표본 수: 최소 {counts[0]}, 중앙 {counts[len(counts) // 2]}, 최대 {counts[-1]}")
    n_valid = sum(len(v) for v in valid.values())
    print(f"유효 응답률: {n_valid / max(len(recs), 1):.1%}")

    with (HERE / f"exaone_p{args.phase}_merged.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (HERE / f"exaone_p{args.phase}_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_id", "level", "cell_id", "var", "codes", "n_total", "n_valid", "counts_json"])
        for pid, p in probes.items():
            codes = json.loads(p["codes"])
            c = collections.Counter(valid.get(pid, []))
            w.writerow([pid, p["level"], p["cell_id"], p["var"], p["codes"], per_probe.get(pid, 0),
                        len(valid.get(pid, [])), json.dumps([c.get(x, 0) for x in codes])])
    print(f"저장: exaone_p{args.phase}_merged.jsonl, exaone_p{args.phase}_counts.csv")


if __name__ == "__main__":
    main()
