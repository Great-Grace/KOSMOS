#!/usr/bin/env python3
"""K-EXAONE 2.5 팀 분담 실행기 (KOSMOS 설계 v2).

4명이 같은 프롬프트(prompts_v2.csv)를 표본 번호 구간만 나눠 돌린다.
  담당 1: 표본 0–24 / 담당 2: 25–49 / 담당 3: 50–74 / 담당 4: 75–99
표본 번호를 바깥 루프로 돌리므로, 중간에 멈춰도 모든 프롬프트의 표본 수가 고르게 남는다.
끊기면 같은 명령을 다시 실행하면 이어서 돈다.

필요한 것: Python 3.9 이상. 외부 패키지 없음(certifi가 있으면 인증서로 사용).

사용:
  export EXAONE_API_KEY=본인키
  python3 run_exaone_team.py --member 1 --test     # 3건만 시험
  python3 run_exaone_team.py --member 1            # 본실행
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts_v2.csv"

URL = "https://api.exaone.ai/v1/chat/completions"
MODEL = "k-exaone_v2.5"
REASONING_EFFORT = "low"   # K-EXAONE 2.5는 추론을 끌 수 없다
TEMPERATURE = 1.0
MAX_TOKENS = 4000
MIN_INTERVAL = 1.0         # 응답 헤더 기준 초당 1회 한도
K_TOTAL = 100
N_MEMBERS = 4
SAMPLES_PER_MEMBER = K_TOTAL // N_MEMBERS

SINGLE_NUM_RE = re.compile(r"^\s*(?:정답|답|번호)?\s*[:：]?\s*(\d+)\s*\.?\s*$")
REFUSAL_RE = re.compile(
    r"(죄송|답변드리기|답변하기|할\s*수\s*없|선택할\s*수\s*없|적절하지|판단하기\s*어렵|"
    r"모르겠|무응답|거부|응답할\s*수\s*없|제공할\s*수\s*없|중립|특정\s*번호를|"
    r"저는\s*AI|I\s*don't|I'm\s*not|I'm\s*an\s*A|I\s*cannot|정답을\s*(?:고|알|특|하)|"
    r"이\s*(?:질문|문항|설문)|(?:질문|설문|문항)(?:에|이)\s*(?:대|제|는)|지역\s*정보|설명에\s*|나는\s*이)",
    re.I,
)


def parse_response(raw: str | None, codes: list[int]) -> tuple[int | None, bool, str]:
    """kgss_haiku_pilot.parse_response와 같은 규칙."""
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


def ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile:
        try:
            import certifi  # type: ignore
            cafile = certifi.where()
        except ImportError:
            pass
    return ssl.create_default_context(cafile=cafile)


class Pacer:
    """출발 간격을 MIN_INTERVAL초 이상으로 유지한다. 429를 받으면 전체를 잠시 멈춘다."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.next_at = interval, threading.Lock(), 0.0

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


def call(key, ctx, pacer, system, user, log):
    body = {
        "model": MODEL, "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS, "logprobs": True, "top_logprobs": 20,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
            if e.code in (401, 403):
                log(f"인증 실패: {msg} — EXAONE_API_KEY를 확인하세요.")
                os._exit(2)
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("retry-after-ms")
                wait = float(ra) / 1000 if ra else float(e.headers.get("retry-after") or min(60, 2 ** attempt))
                if "day" in msg.lower():
                    log(f"일일 한도 초과로 보임: {msg}")
                    wait = max(wait, 600)
                pacer.pause(wait)
                continue
            return None, last_err, time.monotonic() - t0
        except ssl.SSLError as e:
            log(f"SSL 오류: {e}\n  → 'pip install certifi' 후 다시 실행하거나, macOS라면 "
                f"/Applications/Python 3.x/Install Certificates.command 를 실행하세요.")
            os._exit(2)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if "CERTIFICATE_VERIFY_FAILED" in last_err:
                log("인증서 검증 실패 → 'pip install certifi' 후 다시 실행하세요.")
                os._exit(2)
            pacer.pause(min(60, 2 ** attempt))
    return None, last_err, 0.0


def load_config() -> dict:
    p = HERE / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_dotenv() -> None:
    """폴더의 .env에서 EXAONE_API_KEY를 읽는다 (환경변수가 이미 있으면 그대로 둔다)."""
    p = HERE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sha256_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, default=cfg.get("member"), choices=range(1, N_MEMBERS + 1),
                    help="담당 번호 1~4 (config.json에 있으면 생략)")
    ap.add_argument("--phase", type=int, default=cfg.get("phase", 1), choices=(1, 2),
                    help="1 = c0·L0·L1 (기본), 2 = L2")
    ap.add_argument("--workers", type=int, default=150,
                    help="동시 요청 수. 출발 간격은 별도로 1초를 지킨다")
    ap.add_argument("--test", action="store_true", help="3건만 돌려 설정을 확인한다")
    ap.add_argument("--status", action="store_true", help="호출 없이 진행 상황만 보여준다")
    args = ap.parse_args()
    if args.member is None:
        sys.exit("담당 번호가 없습니다. --member 1~4 를 주거나 config.json을 확인하세요.")

    if cfg.get("prompts_sha256") and sha256_file(PROMPTS) != cfg["prompts_sha256"]:
        sys.exit("prompts_v2.csv가 배포본과 다릅니다. 파일을 고치지 말고 원래 파일로 되돌리세요.")

    suffix = "_test" if args.test else ""
    out = HERE / f"exaone_m{args.member}_p{args.phase}{suffix}.jsonl"
    log_path = HERE / f"exaone_m{args.member}_p{args.phase}{suffix}.log"
    lo = (args.member - 1) * SAMPLES_PER_MEMBER
    samples = list(range(lo, lo + SAMPLES_PER_MEMBER))

    with PROMPTS.open(encoding="utf-8-sig") as f:
        probes = [r for r in csv.DictReader(f) if int(r["phase"]) == args.phase]

    done = set()
    n_err_lines = 0
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("invalid_type") != "api_error":
                    done.add((r["probe_id"], r["sample_idx"]))
                else:
                    n_err_lines += 1

    if args.status:
        total = len(probes) * len(samples)
        print(json.dumps({"member": args.member, "phase": args.phase, "done": len(done), "total": total,
                          "percent": round(100 * len(done) / total, 2), "api_error_lines": n_err_lines,
                          "complete": len(done) == total, "output": out.name}, ensure_ascii=False))
        return

    load_dotenv()
    key = os.environ.get("EXAONE_API_KEY", "").strip()
    if not key:
        sys.exit("EXAONE_API_KEY가 없습니다. 이 폴더의 .env 파일에 EXAONE_API_KEY=본인키 를 넣으세요.")

    tasks = [(s, p) for s in samples for p in probes if (p["probe_id"], s) not in done]
    if args.test:
        tasks = tasks[:3]

    lock = threading.Lock()

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        with lock:
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")

    total = len(probes) * len(samples)
    log(f"담당 {args.member} | 단계 {args.phase} | 표본 {samples[0]}–{samples[-1]} | "
        f"프롬프트 {len(probes)}개 | 전체 {total}건 중 완료 {len(done)}건, 이번 실행 {len(tasks)}건")
    if not tasks:
        log("할 일이 없습니다. 완료된 상태입니다.")
        return

    ctx = ssl_context()
    pacer = Pacer(MIN_INTERVAL)
    stats = {"n": 0, "valid": 0, "err": 0, "lat": 0.0, "rt": 0}
    t_start = time.monotonic()
    run_date = datetime.date.today().isoformat()

    with out.open("a", encoding="utf-8") as fout:
        def work(task):
            s, p = task
            codes = json.loads(p["codes"])
            resp, err, lat = call(key, ctx, pacer, p["system"], p["user"], log)
            rec = {"model_id": MODEL, "reasoning_effort": REASONING_EFFORT, "run_date": run_date,
                   "member": args.member, "phase": args.phase, "probe_id": p["probe_id"],
                   "level": p["level"], "cell_id": p["cell_id"], "var": p["var"],
                   "sample_idx": s, "latency_s": round(lat, 2)}
            if resp is None:
                rec.update({"raw_text": "", "parsed": None, "valid": False,
                            "invalid_type": "api_error", "error": err})
            else:
                ch = resp["choices"][0]
                msg = ch.get("message", {})
                raw = msg.get("content") or ""
                parsed, valid, inv = parse_response(raw, codes)
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
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                stats["n"] += 1
                stats["valid"] += bool(rec["valid"])
                stats["err"] += rec["invalid_type"] == "api_error"
                stats["lat"] += lat
                stats["rt"] += rec.get("reasoning_tokens", 0) or 0
                n = stats["n"]
            if args.test:
                log(f"시험 {n}: {p['probe_id']} 표본 {s} → 답 {rec['raw_text']!r} "
                    f"(유효 {rec['valid']}, {rec['invalid_type'] or 'ok'}, {lat:.1f}s)")
            elif n % 250 == 0 or n == len(tasks):
                el = time.monotonic() - t_start
                log(f"진행 {len(done) + n}/{total} | 유효 {stats['valid'] / n:.1%} | api_error {stats['err']} | "
                    f"평균 지연 {stats['lat'] / n:.1f}s | 남은 시간 약 {(len(tasks) - n) * el / n / 3600:.1f}시간")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1 if args.test else args.workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(work, t) for t in tasks]):
                fut.result()
    log("이번 실행 끝. 남은 건이 있으면 같은 명령을 다시 실행하세요." if stats["err"] else "완료")


if __name__ == "__main__":
    main()
