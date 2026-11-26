#!/usr/bin/env python3
"""Haiku 5.5 추론 유도 프롬프트 스모크 (2026-11-26, 7차 세션 피드백 2번 후속).

피드백: "프롬프트에 '추론해서 답하라'고 넣으면 항상 추론하게 할 수 있지 않나?"
기존 v2 시스템 프롬프트는 "보기 번호 숫자 하나만 출력하고 다른 말은 하지 마십시오"라서
adaptive thinking(low·medium)에서도 거의 생각하지 않았다(생각 블록 low 1/300, medium 17/300).

조건 (같은 smoke_probes 300개, 프롬프트당 1회, 동기 Messages API):
  cot    thinking 끔. 시스템·마지막 줄을 "생각을 먼저 적고 마지막 줄에 '답: 번호'"로 바꾼다.
  think  thinking adaptive(effort medium). 시스템에 "답하기 전에 충분히 생각하라"를 넣고 최종 출력은 숫자 하나.

출력: raw/v2/haiku_{cot,think}_p1_smoke.jsonl, 같은 이름 .log, run.json
사용: uv run --no-project --with certifi python research/design_v2_20261112/runners/smoke_haiku_cot.py --arm cot
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_haiku_batch as hb  # noqa: E402
from run_deepseek_eco import parse_response, smoke_probes  # noqa: E402

API = "https://api.anthropic.com/v1/messages"
OUT_DIR = hb.OUT_DIR
PRICE_IN, PRICE_OUT = 0.10, 0.50  # 정가(동기 호출) USD/1M

SYS_COT = ("설문 응답자로서 답하십시오. 먼저 응답자의 입장에서 어떤 보기를 고를지 2~4문장으로 생각을 적으십시오. "
           "그다음 마지막 줄에 '답: 번호' 형식으로 보기 번호 하나를 적으십시오.")
LAST_COT = "생각을 먼저 적고, 마지막 줄에 '답: 번호'로 답하십시오."
SYS_THINK = ("답하기 전에 응답자의 입장에서 어떤 보기를 고를지 충분히 생각하십시오. "
             "생각을 마친 뒤 최종 출력에는 보기 번호 숫자 하나만 적고 다른 말은 하지 마십시오.")
EFFORT = "medium"
ANS_RE = re.compile(r"답\s*[:：]\s*\**\s*(\d+)")


def build(p, arm):
    user = p["user"]
    if arm == "cot":
        user = user.replace("번호로만 답하십시오.", LAST_COT)
        assert LAST_COT in user, p["probe_id"]
        return {"model": hb.MODEL, "max_tokens": 600, "thinking": {"type": "disabled"},
                "system": SYS_COT, "messages": [{"role": "user", "content": user}]}
    return {"model": hb.MODEL, "max_tokens": 4000, "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT},
            "system": SYS_THINK, "messages": [{"role": "user", "content": user}]}


def parse(arm, text, codes):
    if arm == "cot":
        m = ANS_RE.findall(text or "")
        if not m:
            return None, False, "no_answer_line"
        v = int(m[-1])
        return (v, True, "") if v in codes else (v, False, "out_of_range")
    return parse_response(text, codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("cot", "think"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    a = ap.parse_args()
    global EFFORT
    EFFORT = a.effort

    probes = smoke_probes([r for r in hb.read_prompts("v2") if r["phase"] == "1"], a.n)
    tag = f"haiku_{a.arm}_p1_smoke" if a.arm == "cot" or a.effort == "medium" else f"haiku_{a.arm}_{a.effort}_p1_smoke{a.n}"
    out, logf = OUT_DIR / f"{tag}.jsonl", OUT_DIR / f"{tag}.log"
    done = set()
    if out.exists():
        done = {json.loads(l)["probe_id"] for l in out.open()}
    todo = [p for p in probes if p["probe_id"] not in done]
    key = hb.api_key()
    lock, t0 = threading.Lock(), time.monotonic()
    stats = {"n": 0, "in": 0, "out": 0}

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with logf.open("a") as f:
            f.write(line + "\n")

    def one(p):
        body = build(p, a.arm)
        msg = json.loads(hb.http("POST", API, key, body, timeout=120))
        texts = [c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text"]
        thinks = [c.get("thinking", "") for c in msg.get("content", []) if c.get("type") == "thinking"]
        text = "".join(texts).strip()
        codes = json.loads(p["codes"])
        v, ok, why = parse(a.arm, text, codes)
        u = msg.get("usage", {})
        return {"model_id": msg.get("model"), "arm": a.arm, "params": {k: body[k] for k in body if k not in ("messages",)},
                "run_date": datetime.date.today().isoformat(), "probe_id": p["probe_id"], "level": p["level"],
                "cell_id": p["cell_id"], "var": p["var"], "sample_idx": 0, "raw_text": text,
                "thinking_text": "\n".join(thinks), "thinking_blocks": sum(c.get("type") in ("thinking", "redacted_thinking") for c in msg.get("content", [])),
                "parsed": v, "valid": ok, "invalid_type": why, "stop_reason": msg.get("stop_reason"),
                "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "request_id": msg.get("id")}

    log(f"시작 arm={a.arm} 남은 {len(todo)}/{len(probes)}")
    last = time.monotonic()
    with cf.ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        futs = [ex.submit(one, p) for p in todo]
        for fu in cf.as_completed(futs):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                stats["n"] += 1; stats["in"] += r["input_tokens"]; stats["out"] += r["output_tokens"]
                if time.monotonic() - last > 30:
                    last = time.monotonic()
                    log(f"진행 {stats['n']}/{len(todo)} 출력 {stats['out']}토큰")
    cost = stats["in"] / 1e6 * PRICE_IN + stats["out"] / 1e6 * PRICE_OUT
    log(f"끝 {stats['n']}건 입력 {stats['in']} 출력 {stats['out']}토큰 ${cost:.4f} {time.monotonic()-t0:.0f}s")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=hb.ROOT).stdout.strip()
    (OUT_DIR / f"{tag}.run.json").write_text(json.dumps({
        "script": "smoke_haiku_cot.py", "arm": a.arm, "commit": commit, "n_probes": len(probes),
        "new_rows": stats["n"], "input_tokens": stats["in"], "output_tokens": stats["out"], "cost_usd": round(cost, 4),
        "system": SYS_COT if a.arm == "cot" else SYS_THINK, "seed": "smoke_probes 기본(20261118)",
        "finished": datetime.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
