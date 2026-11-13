#!/usr/bin/env python3
"""
K-EXAONE 2.5 무료 API로 Haiku 파일럿 Stage 1과 같은 설계를 돌린다 (탐색 분석).

- 프롬프트: experiment_expand/experiment_probes.csv의 c0 250 + adult 1,000 = 1,250개 (Haiku Stage 1과 같음)
- system 문구: Haiku와 같음 ("보기 번호 숫자 하나만 출력하고 다른 말은 하지 마십시오.")
- K-EXAONE 2.5는 추론을 끌 수 없다. reasoning_effort=low로 고정한다.
- 표본 순서: 표본 번호 s를 바깥 루프로 돌린다. 어느 시점에 멈춰도 모든 프롬프트의 표본 수가 고르게 남는다.
- 이어 돌리기: 출력 파일에 이미 있는 (프롬프트, 표본 번호)는 건너뛴다.
- 호출 한도: 응답 헤더 기준 초당 1회·분당 60회·하루 86,400회. 출발 간격을 1.05초로 지킨다.

사용:
  set -a; . ./.env.local; set +a
  python3 kgss_exaone_run.py --k 50            # 끝까지
  python3 kgss_exaone_run.py --k 20 --limit 10 # 시험
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

import kgss_haiku_pilot as pilot

URL = "https://api.exaone.ai/v1/chat/completions"
MODEL = "k-exaone_v2.5"
REASONING_EFFORT = "low"
MAX_TOKENS = 4000
MIN_INTERVAL = 1.0
OUT = pilot.RAW_DIR / "exaone_stage1.jsonl"
LOG = pilot.RAW_DIR / "exaone_stage1.log"


class Pacer:
    """출발 간격을 최소 MIN_INTERVAL초로 유지한다. 429를 받으면 잠시 전체를 멈춘다."""

    def __init__(self, interval: float):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        if at > now:
            time.sleep(at - now)

    def pause(self, seconds: float):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


def answer_top_logprobs(logprobs: dict | None):
    """</think> 이후 첫 숫자 토큰 위치의 상위 로그확률. 없으면 None."""
    if not logprobs or not logprobs.get("content"):
        return None
    toks = logprobs["content"]
    start = 0
    for i, t in enumerate(toks):
        if t.get("token") == "</think>":
            start = i + 1
    for t in toks[start:]:
        if t.get("token", "").strip().isdigit():
            return [[x["token"], x["logprob"]] for x in t.get("top_logprobs", [])]
    return None


def call(key: str, ctx, pacer: Pacer, prompt: str, system: str, log) -> tuple[dict | None, str, float]:
    body = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": 1.0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": 20,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    data = json.dumps(body).encode("utf-8")
    last_err = ""
    for attempt in range(8):
        pacer.wait()
        req = urllib.request.Request(URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
                return json.loads(r.read()), "", time.monotonic() - t0
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {msg}"
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("retry-after-ms")
                wait = float(ra) / 1000 if ra else float(e.headers.get("retry-after") or min(60, 2 ** attempt))
                if "day" in msg.lower():
                    log(f"일일 한도 초과로 보임: {msg}")
                    wait = max(wait, 600)
                pacer.pause(wait)
                continue
            return None, last_err, time.monotonic() - t0
        except Exception as e:  # 네트워크·타임아웃
            last_err = f"{type(e).__name__}: {e}"
            pacer.pause(min(60, 2 ** attempt))
    return None, last_err, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50, help="프롬프트당 표본 수")
    ap.add_argument("--workers", type=int, default=24, help="동시 요청 수 (출발 간격은 별도로 1.05초)")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 처리할 호출 수 상한")
    args = ap.parse_args()

    key = os.environ.get("EXAONE_API_KEY", "").strip()
    if not key:
        sys.exit("EXAONE_API_KEY가 없습니다. set -a; . ./.env.local; set +a")
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    cfg = pilot.load_config()
    system = cfg["system_prompt"]
    probes = pilot.load_stage_probes("stage1", cfg)

    pilot.RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("invalid_type") != "api_error":
                    done.add((r["probe_id"], r["sample_idx"]))

    tasks = [(s, p) for s in range(args.k) for p in probes if (p["probe_id"], s) not in done]
    if args.limit:
        tasks = tasks[: args.limit]

    log_lock = threading.Lock()

    def log(msg: str):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        with log_lock:
            print(line, flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    log(f"시작: 프롬프트 {len(probes)}개 × k={args.k}, 기완료 {len(done)}건, 이번 실행 {len(tasks)}건, "
        f"예상 소요 약 {len(tasks) * MIN_INTERVAL / 3600:.1f}시간")
    if not tasks:
        return

    pacer = Pacer(MIN_INTERVAL)
    out_lock = threading.Lock()
    stats = {"n": 0, "valid": 0, "err": 0, "reason_tok": 0, "lat": 0.0}
    t_start = time.monotonic()
    run_date = datetime.date.today().isoformat()

    with open(OUT, "a", encoding="utf-8") as fout:
        def work(task):
            s, p = task
            codes = json.loads(p["codes"])
            resp, err, lat = call(key, ctx, pacer, p["prompt"], system, log)
            rec = {
                "model_id": MODEL, "reasoning_effort": REASONING_EFFORT, "run_date": run_date,
                "stage": "exaone_stage1", "probe_id": p["probe_id"], "cond": p["level"], "item_id": p["var"],
                "persona_id": p["persona_id"], "person": p["addressee"], "speech": p["register"],
                "sample_idx": s, "latency_s": round(lat, 2),
            }
            if resp is None:
                rec.update({"raw_text": "", "parsed": None, "valid": False, "invalid_type": "api_error", "error": err})
            else:
                ch = resp["choices"][0]
                msg = ch.get("message", {})
                raw = msg.get("content") or ""
                parsed, valid, inv = pilot.parse_response(raw, codes)
                if ch.get("finish_reason") == "length":
                    parsed, valid, inv = None, False, "length"
                usage = resp.get("usage") or {}
                rec.update({
                    "raw_text": raw, "parsed": parsed, "valid": valid, "invalid_type": inv,
                    "finish_reason": ch.get("finish_reason"),
                    "reasoning_content": msg.get("reasoning_content") or "",
                    "answer_top_logprobs": answer_top_logprobs(ch.get("logprobs")),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    "request_id": resp.get("id", ""),
                })
            with out_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                stats["n"] += 1
                stats["valid"] += bool(rec["valid"])
                stats["err"] += rec["invalid_type"] == "api_error"
                stats["reason_tok"] += rec.get("reasoning_tokens", 0) or 0
                stats["lat"] += lat
                n = stats["n"]
                if n % 250 == 0 or n == len(tasks):
                    el = time.monotonic() - t_start
                    rem = (len(tasks) - n) * el / n
                    log(f"진행 {n}/{len(tasks)} | 유효 {stats['valid'] / n:.1%} | api_error {stats['err']} | "
                        f"평균 지연 {stats['lat'] / n:.1f}s | 평균 추론 토큰 {stats['reason_tok'] / n:.0f} | "
                        f"잔여 약 {rem / 3600:.1f}시간")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(work, t) for t in tasks]):
                fut.result()
    log("완료")


if __name__ == "__main__":
    main()
