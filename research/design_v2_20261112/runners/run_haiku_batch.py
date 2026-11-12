#!/usr/bin/env python3
"""Claude Haiku 5.5 (Anthropic Message Batches) 본실행기 — KOSMOS 설계 v2, 1단계.

설정 (Haiku 파일럿에서 검증한 값 그대로):
  model        claude-haiku-5-5
  thinking     {"type": "disabled"}   (기본값이 켜짐이라 반드시 끈다 — 켜지면 비용이 몇 배가 된다)
  max_tokens   5
  temperature  보내지 않음 (Haiku 5.5는 기본값 1.0 외 400)
  system/user  prompts_v2.csv의 system·user 열 (DeepSeek과 같은 프롬프트)

예산 가드: 제출할 때마다 "이미 제출한 배치의 상한 + 이번 배치의 상한"이 --budget(기본 $4.70)을
넘으면 제출하지 않는다. 상한 = 입력 토큰 × $0.05/M + max_tokens(5) 전부 × $0.25/M (배치 50% 할인가).
입력 토큰은 시험 배치(표본 0번) 실측값을 쓰고, 그 전에는 글자 수 × 1.3으로 넉넉히 잡는다.

중복 방지: 원장(raw/v2/haiku_p1_batches.json)에 제출한 표본 번호 범위를 적고, 겹치면 거부한다.

사용 (KOSMOS 루트에서, 키는 ANTHROPIC_API_KEY 환경변수 또는 .env.anthropic):
  R="uv run --no-project --with certifi python research/design_v2_20261112/runners/run_haiku_batch.py"
  $R submit --samples 0:1          # 시험 배치 (2,952건)
  $R wait                          # 끝날 때까지 기다렸다가 회수
  $R estimate                      # 실측 토큰으로 전체 비용 추정
  $R submit --samples 1:100 --chunk 10   # 본 배치 (표본 10개씩 배치 하나)
  $R wait
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
ROOT = DESIGN.parents[1]
sys.path.insert(0, str(HERE))
from run_deepseek_eco import parse_response  # noqa: E402  (v2 공통 파서)

PROMPTS = DESIGN / "prompts_v2.csv"
OUT_DIR = ROOT / "raw" / "v2"
LEDGER = OUT_DIR / "haiku_p1_batches.json"
OUT = OUT_DIR / "haiku_p1.jsonl"
API = "https://api.anthropic.com/v1/messages/batches"
VERSION = "2023-06-01"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 5
PRICE_IN, PRICE_OUT = 0.05, 0.25  # USD / 1M tokens, Batch 50% 할인가 (정가 $0.10 / $0.50)
K = 100

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()


def api_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        f = ROOT / ".env.anthropic"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        sys.exit("ANTHROPIC_API_KEY가 없습니다. 환경변수로 넘기거나 KOSMOS/.env.anthropic에 적어 주세요.")
    return k


def http(method, url, key, body=None, timeout=300):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    for attempt in range(8):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "x-api-key": key, "anthropic-version": VERSION, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < 7:
                time.sleep(min(60, 2 ** attempt))
                continue
            sys.exit(f"HTTP {e.code} {method} {url}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 7:
                time.sleep(min(60, 2 ** attempt))
                continue
            sys.exit(f"네트워크 오류: {e}")


def load_probes():
    probes = [r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"]
    digest = hashlib.sha256("\n".join(p["probe_id"] + p["system"] + p["user"] for p in probes).encode()).hexdigest()[:16]
    return probes, digest


def params(p):
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "thinking": {"type": "disabled"},
            "system": p["system"], "messages": [{"role": "user", "content": p["user"]}]}


def load_ledger():
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"batches": []}


def save_ledger(led):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LEDGER)


def measured_input_tokens():
    """수집한 결과에서 프롬프트별 입력 토큰 실측값."""
    m = {}
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("input_tokens"):
                m[r["probe_id"]] = r["input_tokens"]
    return m


def per_sample_cost_upper(probes):
    """표본 1회분(프롬프트 2,952개)의 비용 상한과 근거."""
    m = measured_input_tokens()
    if len(m) == len(probes):
        tin, src = sum(m[p["probe_id"]] for p in probes), "실측"
    else:
        tin, src = sum(int((len(p["system"]) + len(p["user"])) * 1.3) + 20 for p in probes), "글자수×1.3 추정"
    return (tin * PRICE_IN + len(probes) * MAX_TOKENS * PRICE_OUT) / 1e6, tin, src


def cmd_submit(a):
    probes, digest = load_probes()
    s0, s1 = map(int, a.samples.split(":"))
    if not (0 <= s0 < s1 <= K):
        sys.exit("--samples 범위가 잘못됐습니다")
    led = load_ledger()
    if led.get("digest") and led["digest"] != digest:
        sys.exit("prompts_v2.csv가 이전 제출 때와 다릅니다. 중단합니다.")
    used = set()
    for b in led["batches"]:
        used.update(range(b["samples"][0], b["samples"][1]))
    if used & set(range(s0, s1)):
        sys.exit(f"이미 제출한 표본 번호와 겹칩니다: {sorted(used & set(range(s0, s1)))[:10]}")

    unit, tin, src = per_sample_cost_upper(probes)
    committed = sum(b["cost_upper"] for b in led["batches"])
    new = unit * (s1 - s0)
    print(f"표본 1회분 상한 ${unit:.4f} (입력 {tin:,}토큰, {src}) | 이미 제출 상한 ${committed:.3f} | 이번 상한 ${new:.3f} | 합계 ${committed + new:.3f} / 예산 ${a.budget:.2f}")
    if committed + new > a.budget:
        sys.exit("예산 상한을 넘습니다. 제출하지 않습니다.")
    if a.dry_run:
        return

    key = api_key()
    led["digest"] = digest
    for c0 in range(s0, s1, a.chunk):
        c1 = min(s1, c0 + a.chunk)
        reqs = [{"custom_id": f"p{i:04d}_s{s:03d}", "params": params(p)}
                for s in range(c0, c1) for i, p in enumerate(probes)]
        info = json.loads(http("POST", API, key, {"requests": reqs}))
        led["batches"].append({"id": info["id"], "samples": [c0, c1], "n": len(reqs),
                               "cost_upper": round(unit * (c1 - c0), 5),
                               "created_at": info.get("created_at"), "status": info["processing_status"]})
        save_ledger(led)
        print(f"  제출 {info['id']} 표본 {c0}–{c1 - 1} ({len(reqs):,}건) 상한 ${unit * (c1 - c0):.3f}", flush=True)


def refresh(key, led):
    for b in led["batches"]:
        if b["status"] == "ended" and b.get("collected"):
            continue
        info = json.loads(http("GET", f"{API}/{b['id']}", key))
        b["status"], b["counts"] = info["processing_status"], info["request_counts"]
        b["results_url"], b["ended_at"] = info.get("results_url"), info.get("ended_at")
    save_ledger(led)


def collect(key, led, probes):
    run_date = datetime.date.today().isoformat()
    for b in led["batches"]:
        if b["status"] != "ended" or b.get("collected") or not b.get("results_url"):
            continue
        raw = http("GET", b["results_url"], key, timeout=600)
        rows = []
        tin = tout = 0
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            pi, si = o["custom_id"].split("_")
            p, s = probes[int(pi[1:])], int(si[1:])
            res = o.get("result", {})
            if res.get("type") == "succeeded":
                msg = res["message"]
                text = "".join(c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text")
                parsed, valid, inv = parse_response(text, json.loads(p["codes"]))
                u = msg.get("usage", {})
                stop, rid, model = msg.get("stop_reason"), msg.get("id", ""), msg.get("model", MODEL)
            else:
                text, parsed, valid, inv, u, stop, rid, model = "", None, False, "api_error", {}, None, "", MODEL
            tin += u.get("input_tokens", 0)
            tout += u.get("output_tokens", 0)
            rows.append({"model_id": model, "params": {"thinking": "disabled", "max_tokens": MAX_TOKENS, "temperature": "default"},
                         "run_date": run_date, "phase": 1, "probe_id": p["probe_id"], "level": p["level"],
                         "cell_id": p["cell_id"], "var": p["var"], "sample_idx": s, "raw_text": text,
                         "parsed": parsed, "valid": valid, "invalid_type": inv, "stop_reason": stop,
                         "result_type": res.get("type"), "input_tokens": u.get("input_tokens", 0),
                         "output_tokens": u.get("output_tokens", 0), "request_id": rid, "batch_id": b["id"]})
        with OUT.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        b["collected"], b["rows"] = True, len(rows)
        b["input_tokens"], b["output_tokens"] = tin, tout
        b["cost_actual"] = round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5)
        save_ledger(led)
        print(f"  회수 {b['id']}: {len(rows):,}행, 입력 {tin:,} / 출력 {tout:,}토큰, ${b['cost_actual']:.4f}", flush=True)


def cmd_status(a, quiet=False):
    key = api_key()
    probes, _ = load_probes()
    led = load_ledger()
    refresh(key, led)
    collect(key, led, probes)
    done = all(b["status"] == "ended" and b.get("collected") for b in led["batches"])
    if not quiet or done:
        for b in led["batches"]:
            print(f"  {b['id']} 표본 {b['samples'][0]}–{b['samples'][1] - 1} {b['status']} {b.get('counts', {})}"
                  + (f" ${b['cost_actual']:.4f}" if "cost_actual" in b else ""))
        act = sum(b.get("cost_actual", 0) for b in led["batches"])
        print(f"실측 비용 합계 ${act:.4f} (회수분)")
    return done


def cmd_wait(a):
    while not cmd_status(a, quiet=True):
        led = load_ledger()
        c = sum((b.get("counts") or {}).get("succeeded", 0) + (b.get("counts") or {}).get("errored", 0)
                for b in led["batches"])
        n = sum(b["n"] for b in led["batches"])
        print(f"{datetime.datetime.now():%H:%M:%S} 처리 {c:,}/{n:,}", flush=True)
        time.sleep(a.interval)


def cmd_estimate(a):
    probes, _ = load_probes()
    unit, tin, src = per_sample_cost_upper(probes)
    rows = [json.loads(l) for l in OUT.open(encoding="utf-8")] if OUT.exists() else []
    if rows:
        n = len(rows)
        ain = sum(r["input_tokens"] for r in rows) / n
        aout = sum(r["output_tokens"] for r in rows) / n
        val = sum(r["valid"] for r in rows) / n
        from collections import Counter
        print(f"수집 {n:,}행 | 평균 입력 {ain:.1f} / 출력 {aout:.2f}토큰 | 유효 {val:.1%} | 무효 유형 {dict(Counter(r['invalid_type'] for r in rows if not r['valid']))}")
        print(f"stop_reason {dict(Counter(r['stop_reason'] for r in rows))}")
        exp = K * len(probes) * (ain * PRICE_IN + aout * PRICE_OUT) / 1e6
        print(f"전체 {K}회 예상 비용 ${exp:.3f} (실측 평균 기준)")
    print(f"전체 {K}회 상한 ${unit * K:.3f} (입력 {src}, 출력 {MAX_TOKENS}토큰 전부 가정)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--samples", required=True, help="표본 번호 범위 a:b (b 미포함)")
    s.add_argument("--chunk", type=int, default=10, help="배치 하나에 넣을 표본 수")
    s.add_argument("--budget", type=float, default=4.70)
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("--interval", type=int, default=60)
    sub.add_parser("estimate")
    a = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "wait": cmd_wait, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
