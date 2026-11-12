#!/usr/bin/env python3
"""DeepSeek V4.1 Flash Eco vs Ultrafast 동등성 소량 검증.

1단계 프롬프트 중 60개를 무작위로 뽑아(seed 20261112) 두 모델에 50회씩 같은 설정으로 묻는다.
설정은 본실행과 같다: reasoning_effort none, temperature 1.0, max_completion_tokens 16.

판정 아이디어: 같은 모델의 반쪽 표본끼리(25 vs 25) TVD와, 다른 모델끼리(25 vs 25) TVD를 비교한다.
두 모델이 같은 분포라면 교차 TVD가 같은-모델 TVD와 비슷해야 한다.

출력: raw/v2/compare_eco_ultrafast.jsonl, 화면 요약
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]
sys.path.insert(0, str(HERE))
from run_deepseek_eco import parse_response, PARAMS  # noqa: E402

URL = "https://apis.opengateway.ai/v1/chat/completions"
MODELS = {"eco": "deepseek/deepseek-v4.1-flash-eco", "uf": "deepseek/deepseek-v4.1-flash-ultrafast"}
N_PROMPTS, K, SEED = 60, 50, 20261112
OUT = ROOT / "raw" / "v2" / "compare_eco_ultrafast.jsonl"


def call(key, model, system, user):
    body = json.dumps({"model": model, **PARAMS, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in range(10):
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read()), time.monotonic() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) or e.code >= 500:
                time.sleep(float(e.headers.get("retry-after") or min(30, 2 ** attempt)))
                continue
            raise
        except Exception:
            time.sleep(min(30, 2 ** attempt))
    return None, 0.0


def tvd(a, b, codes):
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    return 0.5 * sum(abs(a[c] / na - b[c] / nb) for c in codes)


def main():
    key = os.environ["OPENGATEWAY_API_KEY"]
    probes = [r for r in csv.DictReader((HERE.parent / "prompts_v2.csv").open(encoding="utf-8-sig")) if r["phase"] == "1"]
    random.seed(SEED)
    sample = random.sample(probes, N_PROMPTS)

    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            r = json.loads(l)
            done.add((r["m"], r["probe_id"], r["s"]))
    tasks = [(m, p, s) for s in range(K) for p in sample for m in MODELS if (m, p["probe_id"], s) not in done]
    print(f"프롬프트 {N_PROMPTS} × {K}회 × 모델 2 = {N_PROMPTS * K * 2}건, 이번 {len(tasks)}건", flush=True)

    lock = threading.Lock()
    t_start = {m: None for m in MODELS}
    t_end = {}
    with OUT.open("a", encoding="utf-8") as f:
        def work(t):
            m, p, s = t
            if t_start[m] is None:
                t_start[m] = time.monotonic()
            resp, lat = call(key, MODELS[m], p["system"], p["user"])
            raw = (resp or {}).get("choices", [{}])[0].get("message", {}).get("content") if resp else None
            parsed, valid, inv = parse_response(raw, json.loads(p["codes"]))
            u = (resp or {}).get("usage") or {}
            rec = {"m": m, "model": MODELS[m], "probe_id": p["probe_id"], "s": s, "raw": raw, "parsed": parsed,
                   "valid": valid, "inv": inv if resp else "api_error", "lat": round(lat, 2),
                   "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                   "rt": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                t_end[m] = time.monotonic()

        # 모델별 동시 30건씩
        with cf.ThreadPoolExecutor(60) as ex:
            list(ex.map(work, tasks))

    recs = [json.loads(l) for l in OUT.open(encoding="utf-8")]
    codes_of = {p["probe_id"]: json.loads(p["codes"]) for p in sample}
    by = collections.defaultdict(list)
    for r in recs:
        if r["valid"]:
            by[(r["m"], r["probe_id"])].append((r["s"], r["parsed"]))

    within, cross, mode_agree, mean_diff = [], [], 0, []
    for pid, codes in codes_of.items():
        half = {}
        for m in MODELS:
            xs = by[(m, pid)]
            half[(m, "a")] = collections.Counter(v for s, v in xs if s < K // 2)
            half[(m, "b")] = collections.Counter(v for s, v in xs if s >= K // 2)
            half[(m, "all")] = collections.Counter(v for s, v in xs)
        w = [tvd(half[("eco", "a")], half[("eco", "b")], codes), tvd(half[("uf", "a")], half[("uf", "b")], codes)]
        c = [tvd(half[("eco", "a")], half[("uf", "b")], codes), tvd(half[("uf", "a")], half[("eco", "b")], codes)]
        within += [x for x in w if x is not None]
        cross += [x for x in c if x is not None]
        e, u = half[("eco", "all")], half[("uf", "all")]
        if e and u and e.most_common(1)[0][0] == u.most_common(1)[0][0]:
            mode_agree += 1
        if e and u:
            mean_diff.append(abs(sum(k * v for k, v in e.items()) / sum(e.values())
                                 - sum(k * v for k, v in u.items()) / sum(u.values())))

    st = {}
    for m in MODELS:
        rs = [r for r in recs if r["m"] == m]
        lats = sorted(r["lat"] for r in rs)
        st[m] = dict(n=len(rs), valid=sum(r["valid"] for r in rs) / len(rs), reasoning=sum(r["rt"] for r in rs),
                     lat_p50=lats[len(lats) // 2], lat_p90=lats[int(len(lats) * .9)],
                     avg_in=sum(r["in"] for r in rs) / len(rs), avg_out=sum(r["out"] for r in rs) / len(rs))
    mean = lambda x: sum(x) / len(x)
    print(json.dumps({
        "per_model": st,
        "tvd_within_same_model_25v25": round(mean(within), 4),
        "tvd_cross_model_25v25": round(mean(cross), 4),
        "cross_minus_within": round(mean(cross) - mean(within), 4),
        "mode_agreement": f"{mode_agree}/{len(codes_of)}",
        "mean_score_abs_diff_avg": round(mean(mean_diff), 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
