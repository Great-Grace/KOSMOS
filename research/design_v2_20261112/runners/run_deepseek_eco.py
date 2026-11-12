#!/usr/bin/env python3
"""DeepSeek V4.1 Flash Eco (OpenGateway) 본실행기 — KOSMOS 설계 v2.

설정 (2026-11-12 시험 호출로 확인):
  model                deepseek/deepseek-v4.1-flash-eco
  reasoning_effort     "none"  (빼면 추론이 켜져 답이 잘린다)
  temperature          1.0
  max_completion_tokens 16     (실측 출력 2토큰)
  system               prompts_v2.csv의 system 열
  logprobs             제공 안 됨 → 표본 반복으로 분포 추정

표본 순서: 표본 번호를 바깥 루프로 돈다. 멈춰도 모든 프롬프트의 표본 수가 고르다.
이어 돌리기: 출력 파일에 이미 있는 (probe_id, sample_idx)는 건너뛴다(api_error 제외).

사용 (KOSMOS 루트에서):
  set -a; . ./.env.opengateway; set +a
  python3 research/design_v2_20261112/runners/run_deepseek_eco.py --test 5
  python3 research/design_v2_20261112/runners/run_deepseek_eco.py            # 1단계 본실행
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
ROOT = DESIGN.parents[1]
PROMPTS = DESIGN / "prompts_v2.csv"
OUT_DIR = ROOT / "raw" / "v2"

URL = "https://apis.opengateway.ai/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash-eco"
PARAMS = {"reasoning_effort": "none", "temperature": 1.0, "max_completion_tokens": 16}
PRICE = {"input": 0.030, "cached": 0.0006, "output": 0.120}  # USD / 1M tokens, 2026-11-12 할인가
K = 100

SINGLE_NUM_RE = re.compile(r"^\s*(?:정답|답|번호)?\s*[:：]?\s*(\d+)\s*\.?\s*$")
REFUSAL_RE = re.compile(
    r"(죄송|답변드리기|답변하기|할\s*수\s*없|선택할\s*수\s*없|적절하지|판단하기\s*어렵|"
    r"모르겠|무응답|거부|응답할\s*수\s*없|제공할\s*수\s*없|중립|특정\s*번호를|"
    r"저는\s*AI|I\s*don't|I'm\s*not|I'm\s*an\s*A|I\s*cannot|정답을\s*(?:고|알|특|하)|"
    r"이\s*(?:질문|문항|설문)|(?:질문|설문|문항)(?:에|이)\s*(?:대|제|는)|지역\s*정보|설명에\s*|나는\s*이)",
    re.I,
)


def parse_response(raw, codes):
    txt = (raw or "").strip()
    if not txt:
        return None, False, "refusal"
    m = SINGLE_NUM_RE.match(txt)
    if m:
        v = int(m.group(1))
        return (v, True, "") if v in codes else (v, False, "out_of_range")
    if REFUSAL_RE.search(txt):
        return None, False, "refusal"
    return None, False, "explanation"


RETRY = {"429": 0, "529": 0, "5xx": 0, "net": 0}


class Gate:
    """429를 받으면 모든 작업자를 Retry-After 동안 멈춘다."""

    def __init__(self):
        self.lock, self.until = threading.Lock(), 0.0

    def wait(self):
        while True:
            with self.lock:
                d = self.until - time.monotonic()
            if d <= 0:
                return
            time.sleep(min(d, 5))

    def pause(self, s):
        with self.lock:
            self.until = max(self.until, time.monotonic() + s)


def call(key, gate, system, user, log):
    body = json.dumps({"model": MODEL, **PARAMS, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    err = ""
    attempt = 0
    while attempt < 10:
        gate.wait()
        req = urllib.request.Request(URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read()), "", time.monotonic() - t0
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            err = f"HTTP {e.code}: {msg}"
            if e.code in (401, 402, 403):
                log(f"중단: {err}")
                os._exit(2)
            ra = e.headers.get("retry-after")
            if e.code == 429 and "concurrency_limit" in msg:
                # 키당 동시 한도: 요청이 업스트림에 가지 않았다. 이 요청만 기다렸다 재시도하고 시도 횟수는 세지 않는다.
                RETRY["429"] += 1
                time.sleep(float(ra) if ra else 1.0)
                continue
            if e.code in (429, 529) or e.code >= 500:
                RETRY["429" if e.code == 429 else "529" if e.code == 529 else "5xx"] += 1
                gate.pause(float(ra) if ra else min(60, 2 ** attempt))
                attempt += 1
                continue
            return None, err, time.monotonic() - t0
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            RETRY["net"] += 1
            gate.pause(min(60, 2 ** attempt))
            attempt += 1
    return None, err, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    ap.add_argument("--workers", type=int, default=48, help="동시 요청 수 (키당 한도 64)")
    ap.add_argument("--test", type=int, default=0, help="n건만 돌려 확인 (별도 파일에 저장)")
    ap.add_argument("--model", choices=("eco", "ultrafast"), default="eco",
                    help="eco = deepseek-v4.1-flash-eco (기본), ultrafast = deepseek-v4.1-flash-ultrafast")
    args = ap.parse_args()
    global MODEL, PRICE
    if args.model == "ultrafast":
        MODEL = "deepseek/deepseek-v4.1-flash-ultrafast"
        PRICE = {"input": 0.090, "cached": 0.0018, "output": 0.360}  # 2026-11-12 10:00 KST 전 할인가(-70%)

    key = os.environ.get("OPENGATEWAY_API_KEY", "").strip()
    if not key:
        sys.exit("OPENGATEWAY_API_KEY가 없습니다.  set -a; . ./.env.opengateway; set +a")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"deepseek_{args.model}_p{args.phase}" + ("_test" if args.test else "")
    out, log_path = OUT_DIR / f"{tag}.jsonl", OUT_DIR / f"{tag}.log"

    probes = [r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if int(r["phase"]) == args.phase]
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("invalid_type") != "api_error":
                    done.add((r["probe_id"], r["sample_idx"]))
    tasks = [(s, p) for s in range(K) for p in probes if (p["probe_id"], s) not in done]
    if args.test:
        tasks = tasks[: args.test]

    lock = threading.Lock()

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        with lock:
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    total = len(probes) * K
    log(f"{MODEL} | 단계 {args.phase} | 설정 {json.dumps(PARAMS)} | 전체 {total}건 중 완료 {len(done)}건, "
        f"이번 실행 {len(tasks)}건 | 동시 {args.workers}")
    if not tasks:
        log("할 일이 없습니다.")
        return

    gate = Gate()
    st = {"n": 0, "valid": 0, "err": 0, "in": 0, "cached": 0, "out": 0, "rt": 0}
    t0 = time.monotonic()
    run_date = datetime.date.today().isoformat()

    with out.open("a", encoding="utf-8") as fout:
        def work(task):
            s, p = task
            codes = json.loads(p["codes"])
            resp, err, lat = call(key, gate, p["system"], p["user"], log)
            rec = {"model_id": MODEL, "params": PARAMS, "run_date": run_date, "phase": args.phase,
                   "probe_id": p["probe_id"], "level": p["level"], "cell_id": p["cell_id"], "var": p["var"],
                   "sample_idx": s, "latency_s": round(lat, 2)}
            if resp is None:
                rec.update({"raw_text": "", "parsed": None, "valid": False, "invalid_type": "api_error", "error": err})
            else:
                ch = resp["choices"][0]
                msg = ch.get("message", {})
                raw = msg.get("content") or ""
                parsed, valid, inv = parse_response(raw, codes)
                if ch.get("finish_reason") == "length" and not valid:
                    inv = "length"
                u = resp.get("usage") or {}
                rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                rec.update({"raw_text": raw, "parsed": parsed, "valid": valid, "invalid_type": inv,
                            "finish_reason": ch.get("finish_reason"),
                            "input_tokens": u.get("prompt_tokens", 0),
                            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                            "output_tokens": u.get("completion_tokens", 0), "reasoning_tokens": rt,
                            "request_id": resp.get("id", "")})
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                st["n"] += 1
                st["valid"] += bool(rec["valid"])
                st["err"] += rec["invalid_type"] == "api_error"
                for k_, f_ in (("in", "input_tokens"), ("cached", "cached_tokens"),
                               ("out", "output_tokens"), ("rt", "reasoning_tokens")):
                    st[k_] += rec.get(f_, 0) or 0
                n = st["n"]
            if args.test:
                log(f"시험 {n}: {p['probe_id']} s{s} → {rec['raw_text']!r} ({rec['invalid_type'] or 'ok'}, {lat:.1f}s)")
            elif n % 2000 == 0 or n == len(tasks):
                el = time.monotonic() - t0
                cost = ((st["in"] - st["cached"]) * PRICE["input"] + st["cached"] * PRICE["cached"]
                        + st["out"] * PRICE["output"]) / 1e6
                log(f"진행 {len(done) + n}/{total} | 유효 {st['valid'] / n:.1%} | api_error {st['err']} | "
                    f"추론 토큰 합 {st['rt']} | 누적 비용 ${cost:.3f} | {n / el:.1f}건/s | "
                    f"남은 시간 약 {(len(tasks) - n) * el / n / 60:.0f}분 | 재시도 {RETRY}")
                if st["rt"] > 0:
                    log("경고: 추론 토큰이 발생했습니다. 설정을 확인하세요.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(work, t) for t in tasks]):
                fut.result()
    log("완료" if not st["err"] else f"끝 (api_error {st['err']}건 — 같은 명령으로 다시 실행하면 재시도)")


if __name__ == "__main__":
    main()
