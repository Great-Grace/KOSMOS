"""K-EXAONE 팀원별 배포 패키지를 만든다.

출력: team_packages/KOSMOS_EXAONE_m{n}_{이름}/ 와 같은 이름의 .zip
  START_HERE.md   사람용 안내 (1쪽)
  AGENT_TASK.md   AI 에이전트용 작업 지시서
  config.json     담당 번호·표본 구간·프롬프트 해시
  run_exaone_team.py, finalize.py, prompts_v2.csv
  (담당 1에게만) merge_check.py — 4명 결과 병합용
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TPL = HERE / "team_template"
OUT = HERE / "team_packages"
MEMBERS = [(1, "태우"), (2, "정주희"), (3, "허다교"), (4, "이지은")]
PER = 25


def main() -> None:
    prompts = HERE / "prompts_v2.csv"
    phash = hashlib.sha256(prompts.read_bytes()).hexdigest()
    n_p1 = sum(r["phase"] == "1" for r in csv.DictReader(prompts.open(encoding="utf-8-sig")))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for m, name in MEMBERS:
        folder = f"KOSMOS_EXAONE_m{m}_{name}"
        d = OUT / folder
        d.mkdir()
        lo, hi = (m - 1) * PER, m * PER - 1
        fmt = dict(member=m, name=name, lo=lo, hi=hi, expected=n_p1 * PER, folder=folder)
        for doc in ("START_HERE.md", "AGENT_TASK.md"):
            (d / doc).write_text((TPL / doc).read_text(encoding="utf-8").format(**fmt), encoding="utf-8")
        cfg = {"member": m, "name": name, "phase": 1, "sample_from": lo, "sample_to": hi,
               "expected_calls": n_p1 * PER, "prompts_sha256": phash,
               "model": "k-exaone_v2.5", "reasoning_effort": "low", "temperature": 1.0}
        (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for f in ("run_exaone_team.py", "finalize.py"):
            shutil.copy2(TPL / f, d / f)
        shutil.copy2(prompts, d / "prompts_v2.csv")
        if m == 1:
            shutil.copy2(TPL / "merge_check.py", d / "merge_check.py")

        z = OUT / f"{folder}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(d.iterdir()):
                zf.write(f, f"{folder}/{f.name}")
        print(f"{z.name}: {', '.join(sorted(p.name for p in d.iterdir()))}")


if __name__ == "__main__":
    main()
