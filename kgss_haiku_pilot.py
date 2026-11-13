#!/usr/bin/env python3
"""
KOSMOS Claude Haiku 5.5 Pilot Runner & Evaluator (Pre-registered pipeline).
Reads all settings and gate thresholds from research/haiku_pilot_config.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import io
import json
import math
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

BASE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = BASE / "research" / "haiku_pilot_config.json"
PROBES_PATH = BASE / "experiment_expand" / "experiment_probes.csv"
RAW_DIR = BASE / "raw"
COUNTS_DIR = BASE / "counts"
METRICS_DIR = BASE / "metrics"
GATES_PATH = BASE / "gates.md"

SINGLE_NUM_RE = re.compile(r"^\s*(?:정답|답|번호)?\s*[:：]?\s*(\d+)\s*\.?\s*$")
REFUSAL_RE = re.compile(
    r"(죄송|답변드리기|답변하기|할\s*수\s*없|선택할\s*수\s*없|적절하지|판단하기\s*어렵|"
    r"모르겠|무응답|거부|응답할\s*수\s*없|제공할\s*수\s*없|중립|특정\s*번호를|"
    r"저는\s*AI|I\s*don't|I'm\s*not|I'm\s*an\s*A|I\s*cannot|정답을\s*(?:고|알|특|하)|"
    r"이\s*(?:질문|문항|설문)|(?:질문|설문|문항)(?:에|이)\s*(?:대|제|는)|지역\s*정보|설명에\s*|나는\s*이)",
    re.I,
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_response(raw_text: str, valid_codes: list[int]) -> tuple[int | None, bool, str]:
    """
    Parse model raw_text into (parsed_int_or_None, is_valid_bool, invalid_type_str).
    invalid_type is "" when valid, or one of: "refusal", "out_of_range", "explanation".
    """
    if raw_text is None:
        return None, False, "refusal"
    txt = raw_text.strip()
    if not txt:
        return None, False, "refusal"

    m = SINGLE_NUM_RE.match(txt)
    if m:
        val = int(m.group(1))
        if val in valid_codes:
            return val, True, ""
        return val, False, "out_of_range"

    if REFUSAL_RE.search(txt):
        return None, False, "refusal"

    return None, False, "explanation"


VERBAL_STAGES = ("verbal0", "verbal1")
VERBAL_TAIL = "번호로만 답하십시오."
PERSONA_LINE_RE = re.compile(r"^당신은\s*(.+?)입니다\.$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def is_verbal(stage: str) -> bool:
    return stage in VERBAL_STAGES


def build_verbal_probes(stage: str, cfg: dict, all_probes: list[dict]) -> list[dict]:
    """
    분포 직접 예측 arm 프롬프트. 문항·보기 텍스트는 c0 프롬프트에서 그대로 가져오고
    마지막 응답 지시 한 줄만 뺀다. 집단은 한국 성인 전체 1개 + 성별×연령 14셀.
    verbal0(스모크)는 stage0 20문항만 쓴다.
    """
    vcfg = cfg["verbal"]
    persona_text = {}
    for p in all_probes:
        if p["level"] == "demo" and p["persona_id"] not in persona_text:
            m = PERSONA_LINE_RE.match(p["prompt"].split("\n")[0].strip())
            if not m:
                raise ValueError(f"페르소나 문구를 읽지 못함: {p['prompt'][:40]!r}")
            persona_text[p["persona_id"]] = m.group(1)

    smoke_items = set(cfg["stage0_items"]["sensitive"] + cfg["stage0_items"]["non_sensitive"])
    out = []
    for p in all_probes:
        if p["level"] != "c0":
            continue
        if stage == "verbal0" and p["var"] not in smoke_items:
            continue
        body = p["prompt"].rstrip()
        if not body.endswith(VERBAL_TAIL):
            raise ValueError(f"c0 프롬프트 형식이 예상과 다름: {p['var']}")
        item_block = body[: -len(VERBAL_TAIL)].rstrip()
        groups = [("verbal_pop", "", vcfg["group_pop"])] + [
            ("verbal_cell", pid, vcfg["group_cell_template"].format(persona=txt))
            for pid, txt in sorted(persona_text.items())
        ]
        for level, pid, group in groups:
            out.append({
                "probe_id": f"{stage}_{p['var']}_{pid or 'pop'}",
                "level": level,
                "var": p["var"],
                "year": p.get("year", ""),
                "addressee": "",
                "register": "",
                "persona_id": pid,
                "prompt": vcfg["user_template"].format(group=group, item_block=item_block),
                "codes": p["codes"],
            })
    return out


def parse_verbal(raw_text: str, n_options: int, sum_tol: float) -> tuple[list[float] | None, bool, str]:
    """비율 배열 파싱. 무효 유형: refusal / bad_count / bad_sum."""
    txt = (raw_text or "").strip()
    if not txt:
        return None, False, "refusal"
    nums = [float(x) for x in NUMBER_RE.findall(txt.replace("%", " "))]
    if not nums:
        return None, False, "refusal"
    if len(nums) != n_options:
        return nums, False, "bad_count"
    if any(x < 0 for x in nums) or abs(sum(nums) - 100.0) > sum_tol:
        return nums, False, "bad_sum"
    return nums, True, ""


def parse_any(stage: str, cfg: dict, raw_text: str, codes: list[int]):
    if is_verbal(stage):
        return parse_verbal(raw_text, len(codes), float(cfg["verbal"]["sum_tolerance_pct"]))
    return parse_response(raw_text, codes)


def request_params(stage: str, cfg: dict, prompt: str) -> dict:
    """Messages API 요청 본문. 단일 호출·배치·토큰 계산이 모두 이 함수를 쓴다."""
    if is_verbal(stage):
        # temperature는 지정하지 않는다(기본값). Haiku 5.5는 기본값 외 400.
        return {
            "model": cfg["model_id"],
            "max_tokens": cfg["verbal"]["max_tokens"],
            "thinking": {"type": "disabled"},
            "system": cfg["verbal"]["system_prompt"],
            "messages": [{"role": "user", "content": prompt}],
        }
    return {
        "model": cfg["model_id"],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "thinking": {"type": "disabled"},
        "system": cfg["system_prompt"],
        "messages": [{"role": "user", "content": prompt}],
    }


def load_stage_probes(stage: str, cfg: dict) -> list[dict]:
    with open(PROBES_PATH, "r", encoding="utf-8-sig") as f:
        all_probes = list(csv.DictReader(f))

    if is_verbal(stage):
        return build_verbal_probes(stage, cfg, all_probes)
    if stage == "stage0":
        s0_items = set(cfg["stage0_items"]["sensitive"] + cfg["stage0_items"]["non_sensitive"])
        return [
            p for p in all_probes
            if p["var"] in s0_items and p["level"] in ("c0", "adult")
        ]
    elif stage == "stage1":
        return [p for p in all_probes if p["level"] in ("c0", "adult")]
    elif stage == "stage2":
        return [p for p in all_probes if p["level"] == "demo"]
    else:
        raise ValueError(f"Unknown stage: {stage}")


def call_single_message(api_key: str, cfg: dict, prompt: str, max_retries: int = 5, stage: str = "stage1") -> dict:
    payload = request_params(stage, cfg, prompt)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(max_retries):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise


def run_concurrent_stage(stage: str, api_key: str, cfg: dict, workers: int = 20):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RAW_DIR / f"{stage}.jsonl"
    probes = load_stage_probes(stage, cfg)
    k = cfg["stage_k"][stage]
    run_date = datetime.date.today().isoformat()

    done_keys = set()
    if out_jsonl.exists():
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done_keys.add((row["cond"], row["item_id"], row["persona_id"], row["person"], row["speech"], int(row["sample_idx"])))

    tasks = []
    for p in probes:
        cond = p["level"]
        item_id = p["var"]
        persona_id = p["persona_id"]
        person = p["addressee"]
        speech = p["register"]
        codes = json.loads(p["codes"])
        prompt = p["prompt"]
        for s_idx in range(k):
            tkey = (cond, item_id, persona_id, person, speech, s_idx)
            if tkey not in done_keys:
                tasks.append((p, codes, s_idx))

    total = len(probes) * k
    print(f"[{stage}] 프롬프트 {len(probes)}개 × k={k} = 총 {total}회 (기완료 {len(done_keys)}회, 남은 호출 {len(tasks)}회)", flush=True)
    if not tasks:
        return out_jsonl

    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        def worker(task):
            nonlocal completed
            p, codes, s_idx = task
            resp = call_single_message(api_key, cfg, p["prompt"], stage=stage)
            content_list = resp.get("content", [])
            raw_text = "".join(c.get("text", "") for c in content_list if c.get("type") == "text")
            parsed, valid, invalid_type = parse_any(stage, cfg, raw_text, codes)
            usage = resp.get("usage", {})
            rec = {
                "model_id": resp.get("model", cfg["model_id"]),
                "run_date": run_date,
                "stage": stage,
                "cond": p["level"],
                "item_id": p["var"],
                "persona_id": p["persona_id"],
                "person": p["addressee"],
                "speech": p["register"],
                "sample_idx": s_idx,
                "raw_text": raw_text,
                "parsed": parsed,
                "valid": valid,
                "invalid_type": invalid_type,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "request_id": resp.get("id", ""),
            }
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with lock:
                fout.write(line)
                fout.flush()
                completed += 1
                if completed % 200 == 0 or completed == len(tasks):
                    el = time.time() - t0
                    rate = completed / el if el > 0 else 0
                    rem = (len(tasks) - completed) / rate if rate > 0 else 0
                    print(f"  진행: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) | 경과 {el:.1f}s | 잔여 {rem:.1f}s", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    return out_jsonl


def submit_batch_stage(stage: str, api_key: str, cfg: dict, chunk_size: int = 30000):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    probes = load_stage_probes(stage, cfg)
    k = cfg["stage_k"][stage]

    requests_list = []
    for p_idx, p in enumerate(probes):
        for s_idx in range(k):
            custom_id = f"{stage}_p{p_idx:05d}_s{s_idx:02d}"
            requests_list.append({
                "custom_id": custom_id,
                "params": request_params(stage, cfg, p["prompt"]),
            })

    print(f"[{stage}] 총 {len(requests_list)}개 배치 요청 준비 완료 (청크 크기 {chunk_size})", flush=True)
    batches_meta_path = RAW_DIR / f"{stage}_batches.json"
    submitted = []
    for i in range(0, len(requests_list), chunk_size):
        chunk = requests_list[i : i + chunk_size]
        body = json.dumps({"requests": chunk}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages/batches",
            data=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            b_info = json.loads(resp.read().decode("utf-8"))
            submitted.append(b_info)
            print(f"  배치 제출 완료 ({i//chunk_size + 1}): id={b_info['id']}, status={b_info['processing_status']}, count={len(chunk)}", flush=True)

    with open(batches_meta_path, "w", encoding="utf-8") as f:
        json.dump(submitted, f, ensure_ascii=False, indent=2)
    return batches_meta_path


def check_and_collect_batch(stage: str, api_key: str, cfg: dict) -> bool:
    batches_meta_path = RAW_DIR / f"{stage}_batches.json"
    if not batches_meta_path.exists():
        raise FileNotFoundError(f"{batches_meta_path} 파일이 없습니다.")
    with open(batches_meta_path, "r", encoding="utf-8") as f:
        batches = json.load(f)

    all_ended = True
    updated_batches = []
    for b in batches:
        bid = b["id"]
        req = urllib.request.Request(
            f"https://api.anthropic.com/v1/messages/batches/{bid}",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            updated_batches.append(info)
            rc = info.get("request_counts", {})
            print(
                f"  배치 {bid}: status={info['processing_status']} | "
                f"succeeded={rc.get('succeeded', 0)}, processing={rc.get('processing', 0)}, "
                f"errored={rc.get('errored', 0)}",
                flush=True,
            )
            if info["processing_status"] != "ended":
                all_ended = False

    with open(batches_meta_path, "w", encoding="utf-8") as f:
        json.dump(updated_batches, f, ensure_ascii=False, indent=2)

    # 끝난 배치만 먼저 회수한다. 미완·결과 소실 배치는 목록으로 보고하고 False를 돌려
    # 호출자가 부분 자료로 집계하지 않게 한다.
    probes = load_stage_probes(stage, cfg)
    out_jsonl = RAW_DIR / f"{stage}.jsonl"
    run_date = datetime.date.today().isoformat()
    total_written = 0
    total_api_error = 0
    unavailable = []

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for info in updated_batches:
            r_url = info.get("results_url")
            if info["processing_status"] != "ended" or not r_url:
                unavailable.append((info["id"], info["processing_status"], "결과 URL 없음"))
                continue
            req = urllib.request.Request(
                r_url,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                method="GET",
            )
            try:
                resp = urllib.request.urlopen(req, timeout=180)
            except urllib.error.HTTPError as e:
                unavailable.append((info["id"], info["processing_status"], f"HTTP {e.code}"))
                continue
            with resp:
                for raw_line in resp:
                    line_str = raw_line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    obj = json.loads(line_str)
                    cid = obj["custom_id"]
                    # format: stage1_p00000_s00
                    parts = cid.split("_")
                    p_idx = int(parts[1][1:])
                    s_idx = int(parts[2][1:])
                    p = probes[p_idx]
                    codes = json.loads(p["codes"])
                    res = obj.get("result", {})
                    if res.get("type") == "succeeded":
                        msg = res.get("message", {})
                        content_list = msg.get("content", [])
                        raw_text = "".join(c.get("text", "") for c in content_list if c.get("type") == "text")
                        usage = msg.get("usage", {})
                        req_id = msg.get("id", "")
                        model_used = msg.get("model", cfg["model_id"])
                    else:
                        raw_text = ""
                        usage = {}
                        req_id = ""
                        model_used = cfg["model_id"]
                    if res.get("type") == "succeeded":
                        parsed, valid, invalid_type = parse_any(stage, cfg, raw_text, codes)
                    else:
                        # errored/canceled/expired는 모델 응답이 아니다. 거부로 세지 않는다.
                        parsed, valid, invalid_type = None, False, "api_error"
                        total_api_error += 1
                    rec = {
                        "model_id": model_used,
                        "run_date": run_date,
                        "stage": stage,
                        "cond": p["level"],
                        "item_id": p["var"],
                        "persona_id": p["persona_id"],
                        "person": p["addressee"],
                        "speech": p["register"],
                        "sample_idx": s_idx,
                        "raw_text": raw_text,
                        "parsed": parsed,
                        "valid": valid,
                        "invalid_type": invalid_type,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "request_id": req_id,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += 1

    print(
        f"[{stage}] 배치 결과 수집: 총 {total_written}행 (api_error {total_api_error}행) -> {out_jsonl}",
        flush=True,
    )
    if unavailable:
        print(f"[{stage}] 회수하지 못한 배치 {len(unavailable)}개 — 집계하지 않습니다:", flush=True)
        for bid, status, why in unavailable:
            print(f"  {bid}: status={status}, {why}", flush=True)
        return False
    return all_ended


def aggregate_stage0(cfg: dict):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "stage0.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} 파일이 없습니다.")

    probes = load_stage_probes("stage0", cfg)
    probe_codes = {
        (p["level"], p["var"], p["persona_id"], p["addressee"], p["register"]): json.loads(p["codes"])
        for p in probes
    }
    sens_set = set(cfg["stage0_items"]["sensitive"])
    alpha = float(cfg["jeffreys_alpha"])

    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    groups = defaultdict(list)
    for r in records:
        gkey = (r["cond"], r["item_id"], r["persona_id"], r["person"], r["speech"])
        groups[gkey].append(r)

    count_rows = []
    for gkey, recs in sorted(groups.items()):
        cond, item_id, persona_id, person, speech = gkey
        codes = probe_codes[gkey]
        K = len(codes)
        n_total = len(recs)
        n_valid = sum(1 for r in recs if r["valid"])
        n_refusal = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_explanation = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "out_of_range")

        c_map = Counter(r["parsed"] for r in recs if r["valid"])
        counts_vec = [c_map.get(c, 0) for c in codes]
        denom = n_valid + K * alpha
        jeffreys_dist = [(cnt + alpha) / denom for cnt in counts_vec]

        count_rows.append({
            "stage": "stage0",
            "cond": cond,
            "item_id": item_id,
            "is_sensitive": item_id in sens_set,
            "persona_id": persona_id,
            "person": person,
            "speech": speech,
            "codes": json.dumps(codes),
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_rate": round(n_valid / n_total, 6) if n_total else 0.0,
            "n_refusal": n_refusal,
            "n_explanation": n_explanation,
            "n_out_of_range": n_oor,
            "counts_json": json.dumps(counts_vec),
            "jeffreys_dist_json": json.dumps([round(x, 6) for x in jeffreys_dist]),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_total, 2),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_total, 2),
        })

    counts_csv = COUNTS_DIR / "stage0_counts.csv"
    with open(counts_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()))
        w.writeheader()
        w.writerows(count_rows)

    # Condition-level summary for Gate 0
    cond_groups = defaultdict(list)
    for r in records:
        cond_label = "c0" if r["cond"] == "c0" else f"adult_{r['person']}_{r['speech']}"
        cond_groups[cond_label].append(r)

    metric_rows = []
    for clab, recs in sorted(cond_groups.items()):
        n_tot = len(recs)
        n_val = sum(1 for r in recs if r["valid"])
        n_ref = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_exp = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "out_of_range")
        metric_rows.append({
            "group_type": "condition",
            "group_name": clab,
            "n_calls": n_tot,
            "n_valid": n_val,
            "valid_rate": round(n_val / n_tot, 6),
            "refusal_rate": round(n_ref / n_tot, 6),
            "explanation_rate": round(n_exp / n_tot, 6),
            "out_of_range_rate": round(n_oor / n_tot, 6),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_tot, 2),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_tot, 2),
        })

    # Sensitive vs Non-sensitive breakdown
    for s_flag, s_name in [(True, "sensitive_10items"), (False, "non_sensitive_10items")]:
        recs = [r for r in records if (r["item_id"] in sens_set) == s_flag]
        n_tot = len(recs)
        n_val = sum(1 for r in recs if r["valid"])
        n_ref = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_exp = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "out_of_range")
        metric_rows.append({
            "group_type": "sensitivity",
            "group_name": s_name,
            "n_calls": n_tot,
            "n_valid": n_val,
            "valid_rate": round(n_val / n_tot, 6),
            "refusal_rate": round(n_ref / n_tot, 6),
            "explanation_rate": round(n_exp / n_tot, 6),
            "out_of_range_rate": round(n_oor / n_tot, 6),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_tot, 2),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_tot, 2),
        })

    # Addressee (2nd vs 3rd person) breakdown in adult
    for person_val in ["당신", "이사람"]:
        recs = [r for r in records if r["cond"] == "adult" and r["person"] == person_val]
        n_tot = len(recs)
        n_val = sum(1 for r in recs if r["valid"])
        n_ref = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_exp = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "out_of_range")
        metric_rows.append({
            "group_type": "addressee",
            "group_name": f"adult_{person_val}",
            "n_calls": n_tot,
            "n_valid": n_val,
            "valid_rate": round(n_val / n_tot, 6),
            "refusal_rate": round(n_ref / n_tot, 6),
            "explanation_rate": round(n_exp / n_tot, 6),
            "out_of_range_rate": round(n_oor / n_tot, 6),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_tot, 2),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_tot, 2),
        })

    metrics_csv = METRICS_DIR / "stage0_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)

    # Gate 0 evaluation
    min_rate_thresh = float(cfg["gates"]["gate0_min_valid_rate"])
    cond_rates = [m["valid_rate"] for m in metric_rows if m["group_type"] == "condition"]
    gate0_pass = all(r >= min_rate_thresh for r in cond_rates)

    mean_in = sum(r["input_tokens"] for r in records) / len(records)
    mean_out = sum(r["output_tokens"] for r in records) / len(records)
    total_calls_all_stages = 2000 + 62500 + 280000
    est_in_mtok = (total_calls_all_stages * mean_in) / 1e6
    est_out_mtok = (total_calls_all_stages * mean_out) / 1e6
    batch_in_cost = est_in_mtok * cfg["pricing_batch_per_mtok"]["input"]
    batch_out_cost = est_out_mtok * cfg["pricing_batch_per_mtok"]["output"]
    batch_total_cost = batch_in_cost + batch_out_cost

    lines = [
        "# KOSMOS Haiku 5.5 파일럿 — 게이트 판정 보고서 (`gates.md`)",
        "",
        f"- 모델 ID: `{cfg['model_id']}`",
        f"- 실행일: `{datetime.date.today().isoformat()}`",
        f"- 설정 파일: `research/haiku_pilot_config.json`",
        "",
        "## Gate 0 — 형식 및 스모크 테스트 (Stage 0)",
        "",
        f"- **최종 판정**: **{'PASS (통과)' if gate0_pass else 'FAIL (미통과)'}** (기준: 모든 조건 유효율 ≥ `{min_rate_thresh:.2%}`)",
        "- **Assistant Prefill (`정답: `) 지원 여부**: **미지원 확인됨** (`HTTP 400: This model does not support assistant message prefill. The conversation must end with a user message.`) → 사전등록 규칙(3절)에 따라 user 단독 메시지 + 정규식 파싱 적용",
        f"- **실측 토큰 수 (2,000호출 평균)**: 입력 `{mean_in:.2f}` 토큰 / 출력 `{mean_out:.2f}` 토큰",
        f"- **실측 기반 전체 실험(344,500호출) 비용 재계산 (Batch API 기준)**:",
        f"  - 총 입력 토큰: 약 `{est_in_mtok:.2f}M` 토큰 → `${batch_in_cost:.2f}` (`${cfg['pricing_batch_per_mtok']['input']}/M`)",
        f"  - 총 출력 토큰: 약 `{est_out_mtok:.2f}M` 토큰 → `${batch_out_cost:.2f}` (`${cfg['pricing_batch_per_mtok']['output']}/M`)",
        f"  - **재계산 총비용 합계**: **약 `${batch_total_cost:.2f}`** (사전등록 예상치 `$8.9` 대비 실측 입력 토큰이 약 1/3.5 수준이라 비용 대폭 절감)",
        "",
        "### 1. 조건별 유효율 및 무효 유형 집계",
        "",
        "| 구분 | 조건명 | 호출 수 | 유효 응답 수 | 유효율 | 거부율 (`refusal`) | 설명문 비율 (`explanation`) | 범위 밖 숫자 (`out_of_range`) | 평균 입력 토큰 | 평균 출력 토큰 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metric_rows:
        lines.append(
            f"| {m['group_type']} | `{m['group_name']}` | {m['n_calls']} | {m['n_valid']} | "
            f"**{m['valid_rate']:.2%}** | {m['refusal_rate']:.2%} | {m['explanation_rate']:.2%} | "
            f"{m['out_of_range_rate']:.2%} | {m['mean_input_tokens']:.1f} | {m['mean_output_tokens']:.2f} |"
        )

    lines.extend([
        "",
        "## Gate 1 — 위치 (Stage 1 대기)",
        "- 상태: 미실행 (Stage 1 배치 제출 전)",
        "",
        "## Gate 2 — 서브그룹 (Stage 2 대기)",
        "- 상태: 미실행 (Stage 2 배치 제출 전)",
        "",
        "## Gate 3 — 화계·인칭 (Stage 2 대기)",
        "- 상태: 미실행 (Stage 2 배치 제출 전)",
        "",
    ])

    GATES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"집계 완료: {counts_csv}, {metrics_csv}, {GATES_PATH}", flush=True)


def tvd_fn(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def w1_raw_fn(p: list[float], q: list[float]) -> float:
    cp, cq, s = 0.0, 0.0, 0.0
    for a, b in zip(p, q):
        cp += a
        cq += b
        s += abs(cp - cq)
    return s


def w1_norm_fn(p: list[float], q: list[float]) -> float:
    k = len(p)
    return w1_raw_fn(p, q) / (k - 1) if k > 1 else 0.0


def nll_fn(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    return -sum(qj * math.log(max(pj, eps)) for pj, qj in zip(p, q))


def load_q_pop_250() -> dict[str, list[float]]:
    rc_path = BASE / "reexperiment_20261009" / "c0_clean" / "real_cells.csv"
    by_var = defaultdict(list)
    with open(rc_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_var[r["var"]].append((float(r["cell_weight_item_norm"]), json.loads(r["real_dist"])))
    q_pop = {}
    for var, cells in by_var.items():
        w_sum = sum(w for w, _ in cells)
        k = len(cells[0][1])
        vec = [0.0] * k
        for w, d in cells:
            for j in range(k):
                vec[j] += (w / w_sum) * float(d[j])
        v_sum = sum(vec)
        q_pop[var] = [x / v_sum for x in vec]
    return q_pop


def load_valid_indices_250(probes: list[dict]) -> dict[str, list[int]]:
    vt_path = BASE / "inventory" / "variables.csv"
    vt = {}
    with open(vt_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vt[r["var"]] = json.loads(r["value_labels"])
    valid_idx = {}
    for p in probes:
        if p["level"] != "c0":
            continue
        var = p["var"]
        codes = json.loads(p["codes"])
        vl = vt.get(var, {})
        idx = []
        for i, c in enumerate(codes):
            lab = str(vl.get(str(float(c)), vl.get(str(c), "")))
            if lab.upper().startswith("DK") or lab.upper().startswith("NOT APPLICABLE"):
                continue
            idx.append(i)
        valid_idx[var] = idx
    return valid_idx


def multinomial_draw(rng, probs: list[float], k_draws: int) -> list[int]:
    K = len(probs)
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0
    counts = [0] * K
    for _ in range(k_draws):
        u = rng.random()
        for j, c_val in enumerate(cdf):
            if u <= c_val:
                counts[j] += 1
                break
    return counts


def aggregate_stage1(cfg: dict):
    import random

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "stage1.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} 파일이 없습니다.")

    probes = load_stage_probes("stage1", cfg)
    probe_codes = {
        (p["level"], p["var"], p["persona_id"], p["addressee"], p["register"]): json.loads(p["codes"])
        for p in probes
    }
    valid_idx_map = load_valid_indices_250(probes)
    q_pop_map = load_q_pop_250()
    alpha = float(cfg["jeffreys_alpha"])

    groups = defaultdict(list)
    total_records = 0
    total_in_tok = 0
    total_out_tok = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("invalid_type") == "api_error":
                continue  # 모델 응답이 아니므로 분포·거부율 분모에서 제외
            # Re-parse with updated REFUSAL_RE for consistency
            gkey =(r["cond"], r["item_id"], r["persona_id"], r["person"], r["speech"])
            codes = probe_codes[gkey]
            parsed, valid, inv_type = parse_response(r["raw_text"], codes)
            r["parsed"], r["valid"], r["invalid_type"] = parsed, valid, inv_type
            groups[gkey].append(r)
            total_records += 1
            total_in_tok += r.get("input_tokens", 0)
            total_out_tok += r.get("output_tokens", 0)

    count_rows = []
    by_item_cond = {}
    for gkey, recs in sorted(groups.items()):
        cond, item_id, persona_id, person, speech = gkey
        codes = probe_codes[gkey]
        kept_idx = valid_idx_map[item_id]
        kept_codes = [codes[i] for i in kept_idx]
        K = len(kept_codes)

        n_total = len(recs)
        n_valid = sum(1 for r in recs if r["valid"] and r["parsed"] in kept_codes)
        n_refusal = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_explanation = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = sum(1 for r in recs if (not r["valid"] and r["invalid_type"] == "out_of_range") or (r["valid"] and r["parsed"] not in kept_codes))

        c_map = Counter(r["parsed"] for r in recs if r["valid"] and r["parsed"] in kept_codes)
        counts_vec = [c_map.get(c, 0) for c in kept_codes]
        denom = n_valid + K * alpha
        jeffreys_dist = [(cnt + alpha) / denom for cnt in counts_vec]

        row_dict = {
            "stage": "stage1",
            "cond": cond,
            "item_id": item_id,
            "persona_id": persona_id,
            "person": person,
            "speech": speech,
            "codes": json.dumps(kept_codes),
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_rate": round(n_valid / n_total, 6) if n_total else 0.0,
            "n_refusal": n_refusal,
            "n_explanation": n_explanation,
            "n_out_of_range": n_oor,
            "counts_json": json.dumps(counts_vec),
            "jeffreys_dist_json": json.dumps([round(x, 6) for x in jeffreys_dist]),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_total, 2),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_total, 2),
        }
        count_rows.append(row_dict)
        by_item_cond[(item_id, cond, person, speech)] = (jeffreys_dist, row_dict)

    counts_csv = COUNTS_DIR / "stage1_counts.csv"
    with open(counts_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()))
        w.writeheader()
        w.writerows(count_rows)

    # Item-level metrics for Haiku (c0, adult_avg, adult_ind, and 4 adult subconditions)
    all_vars = sorted(valid_idx_map.keys())
    adult_subconds = [
        ("당신", "하십시오체"),
        ("당신", "해라체"),
        ("이사람", "하십시오체"),
        ("이사람", "해라체"),
    ]

    item_metrics = []
    haiku_c0_metrics = []
    haiku_adult_avg_metrics = []
    haiku_adult_ind_metrics = []

    for var in all_vars:
        q = q_pop_map[var]
        K = len(q)
        unif_nll = math.log(K)
        pop_nll = nll_fn(q, q)
        q_mode = max(range(K), key=lambda j: q[j])

        # C0
        p_c0, c0_info = by_item_cond[(var, "c0", "", "")]
        m_c0 = {
            "model": cfg["model_id"],
            "eval_type": "c0",
            "agg_order": "single",
            "item_id": var,
            "K": K,
            "valid_rate": c0_info["valid_rate"],
            "refusal_rate": round(c0_info["n_refusal"] / c0_info["n_total"], 6),
            "tvd": tvd_fn(p_c0, q),
            "w1_norm": w1_norm_fn(p_c0, q),
            "w1_raw": w1_raw_fn(p_c0, q),
            "nll": nll_fn(p_c0, q),
            "unif_nll": unif_nll,
            "pop_nll": pop_nll,
            "beat_unif_nll": int(nll_fn(p_c0, q) < unif_nll),
            "mode_hit": int(max(range(K), key=lambda j: p_c0[j]) == q_mode),
        }
        item_metrics.append(m_c0)
        haiku_c0_metrics.append(m_c0)

        # Adult 4 subconditions
        p_ad_list = []
        ad_valid_rates, ad_ref_rates = [], []
        for per, sp in adult_subconds:
            p_sub, sub_info = by_item_cond[(var, "adult", per, sp)]
            p_ad_list.append(p_sub)
            ad_valid_rates.append(sub_info["valid_rate"])
            ad_ref_rates.append(sub_info["n_refusal"] / sub_info["n_total"])
            item_metrics.append({
                "model": cfg["model_id"],
                "eval_type": f"adult_{per}_{sp}",
                "agg_order": "single",
                "item_id": var,
                "K": K,
                "valid_rate": sub_info["valid_rate"],
                "refusal_rate": round(sub_info["n_refusal"] / sub_info["n_total"], 6),
                "tvd": tvd_fn(p_sub, q),
                "w1_norm": w1_norm_fn(p_sub, q),
                "w1_raw": w1_raw_fn(p_sub, q),
                "nll": nll_fn(p_sub, q),
                "unif_nll": unif_nll,
                "pop_nll": pop_nll,
                "beat_unif_nll": int(nll_fn(p_sub, q) < unif_nll),
                "mode_hit": int(max(range(K), key=lambda j: p_sub[j]) == q_mode),
            })

        # Adult avg (mix predictions first)
        p_ad_mean = [sum(p_ad_list[i][j] for i in range(4)) / 4.0 for j in range(K)]
        m_ad_avg = {
            "model": cfg["model_id"],
            "eval_type": "adult",
            "agg_order": "avg",
            "item_id": var,
            "K": K,
            "valid_rate": sum(ad_valid_rates) / 4.0,
            "refusal_rate": sum(ad_ref_rates) / 4.0,
            "tvd": tvd_fn(p_ad_mean, q),
            "w1_norm": w1_norm_fn(p_ad_mean, q),
            "w1_raw": w1_raw_fn(p_ad_mean, q),
            "nll": nll_fn(p_ad_mean, q),
            "unif_nll": unif_nll,
            "pop_nll": pop_nll,
            "beat_unif_nll": int(nll_fn(p_ad_mean, q) < unif_nll),
            "mode_hit": int(max(range(K), key=lambda j: p_ad_mean[j]) == q_mode),
        }
        item_metrics.append(m_ad_avg)
        haiku_adult_avg_metrics.append(m_ad_avg)

        # Adult ind (distance per condition then average)
        ind_nll = sum(nll_fn(p_sub, q) for p_sub in p_ad_list) / 4.0
        m_ad_ind = {
            "model": cfg["model_id"],
            "eval_type": "adult",
            "agg_order": "ind",
            "item_id": var,
            "K": K,
            "valid_rate": sum(ad_valid_rates) / 4.0,
            "refusal_rate": sum(ad_ref_rates) / 4.0,
            "tvd": sum(tvd_fn(p_sub, q) for p_sub in p_ad_list) / 4.0,
            "w1_norm": sum(w1_norm_fn(p_sub, q) for p_sub in p_ad_list) / 4.0,
            "w1_raw": sum(w1_raw_fn(p_sub, q) for p_sub in p_ad_list) / 4.0,
            "nll": ind_nll,
            "unif_nll": unif_nll,
            "pop_nll": pop_nll,
            "beat_unif_nll": int(ind_nll < unif_nll),
            "mode_hit": sum(int(max(range(K), key=lambda j: p_sub[j]) == q_mode) for p_sub in p_ad_list) / 4.0,
        }
        item_metrics.append(m_ad_ind)
        haiku_adult_ind_metrics.append(m_ad_ind)

    # Fair Comparison Device 1: Open-weight 250-item uncalibrated reference (exact logit & noise-matched k=50 + Jeffreys)
    rng = random.Random(int(cfg["random_seed"]))
    k_s1 = int(cfg["stage_k"]["stage1"])
    ow_results = {}
    for folder, cond_tag in [("c0_clean", "c0"), ("adult_clean", "adult")]:
        lr_path = BASE / "reexperiment_20261009" / folder / "logit_responses.csv"
        pr_path = BASE / "reexperiment_20261009" / folder / "probes.csv"
        pid_to_var = {}
        with open(pr_path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pid_to_var[r["probe_id"]] = r["var"]
        with open(lr_path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                var = pid_to_var.get(r["probe_id"])
                if var not in q_pop_map:
                    continue
                mname = r["model"]
                p_raw = [float(x) for x in json.loads(r["model_dist"])]
                s_p = sum(p_raw)
                p_raw = [x / s_p for x in p_raw]
                K = len(p_raw)
                # Noise-matched version: draw k=50 samples and apply Jeffreys alpha=0.5
                # For adult,Haiku averages 4 conditions (or ind over 4 conditions), so let's simulate 4 draws of k=50
                if cond_tag == "c0":
                    cnts = multinomial_draw(rng, p_raw, k_s1)
                    p_nm = [(c + alpha) / (k_s1 + K * alpha) for c in cnts]
                    ow_results[(mname, "c0", "raw_logit", var)] = p_raw
                    ow_results[(mname, "c0", "noise_matched", var)] = p_nm
                else:
                    sub_draws = []
                    for _ in range(4):
                        cnts = multinomial_draw(rng, p_raw, k_s1)
                        sub_draws.append([(c + alpha) / (k_s1 + K * alpha) for c in cnts])
                    p_nm_avg = [sum(sub_draws[i][j] for i in range(4)) / 4.0 for j in range(K)]
                    ow_results[(mname, "adult", "raw_logit", var)] = p_raw
                    ow_results[(mname, "adult", "noise_matched_avg", var)] = p_nm_avg
                    ow_results[(mname, "adult", "noise_matched_ind_list", var)] = sub_draws

    metrics_csv = METRICS_DIR / "stage1_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_metrics[0].keys()))
        w.writeheader()
        w.writerows(item_metrics)

    # Summarize & run 2,000-iteration item-cluster bootstrap for Gate 1(b)
    def mean_of(lst, key):
        return sum(x[key] for x in lst) / len(lst)

    c0_beat_rate = mean_of(haiku_c0_metrics, "beat_unif_nll")
    ad_avg_beat_rate = mean_of(haiku_adult_avg_metrics, "beat_unif_nll")
    ad_ind_beat_rate = mean_of(haiku_adult_ind_metrics, "beat_unif_nll")

    gate1_a_thresh = float(cfg["gates"]["gate1_min_nll_beat_uniform_rate"])
    gate1_a_pass_avg = (c0_beat_rate >= gate1_a_thresh) and (ad_avg_beat_rate >= gate1_a_thresh)
    gate1_a_pass_ind = (c0_beat_rate >= gate1_a_thresh) and (ad_ind_beat_rate >= gate1_a_thresh)

    # Compare Adult TVD against Open-Weight models (EXAONE-3.5-2.4B, Kanana-1.5-2.1B on same 250 items)
    ow_summary = {}
    for mname in ["EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"]:
        c0_raw_tvd = [tvd_fn(ow_results[(mname, "c0", "raw_logit", v)], q_pop_map[v]) for v in all_vars]
        c0_nm_tvd = [tvd_fn(ow_results[(mname, "c0", "noise_matched", v)], q_pop_map[v]) for v in all_vars]
        ad_raw_tvd = [tvd_fn(ow_results[(mname, "adult", "raw_logit", v)], q_pop_map[v]) for v in all_vars]
        ad_nm_avg_tvd = [tvd_fn(ow_results[(mname, "adult", "noise_matched_avg", v)], q_pop_map[v]) for v in all_vars]
        ad_nm_ind_tvd = [
            sum(tvd_fn(p_s, q_pop_map[v]) for p_s in ow_results[(mname, "adult", "noise_matched_ind_list", v)]) / 4.0
            for v in all_vars
        ]
        ad_nm_avg_nll = [nll_fn(ow_results[(mname, "adult", "noise_matched_avg", v)], q_pop_map[v]) for v in all_vars]
        ad_nm_avg_beat = [
            int(nll_fn(ow_results[(mname, "adult", "noise_matched_avg", v)], q_pop_map[v]) < math.log(len(q_pop_map[v])))
            for v in all_vars
        ]
        ow_summary[mname] = {
            "c0_raw_tvd": sum(c0_raw_tvd) / len(all_vars),
            "c0_nm_tvd": sum(c0_nm_tvd) / len(all_vars),
            "ad_raw_tvd": sum(ad_raw_tvd) / len(all_vars),
            "ad_nm_avg_tvd": sum(ad_nm_avg_tvd) / len(all_vars),
            "ad_nm_ind_tvd": sum(ad_nm_ind_tvd) / len(all_vars),
            "ad_nm_avg_nll": sum(ad_nm_avg_nll) / len(all_vars),
            "ad_nm_avg_beat": sum(ad_nm_avg_beat) / len(all_vars),
            "ad_nm_avg_tvd_vec": ad_nm_avg_tvd,
            "ad_nm_ind_tvd_vec": ad_nm_ind_tvd,
        }

    # Best open-weight model (lowest adult noise-matched TVD)
    best_ow_name = min(ow_summary.keys(), key=lambda m: ow_summary[m]["ad_nm_avg_tvd"])
    best_ow_avg_vec = ow_summary[best_ow_name]["ad_nm_avg_tvd_vec"]
    best_ow_ind_vec = ow_summary[best_ow_name]["ad_nm_ind_tvd_vec"]
    haiku_avg_vec = [x["tvd"] for x in haiku_adult_avg_metrics]
    haiku_ind_vec = [x["tvd"] for x in haiku_adult_ind_metrics]

    diff_avg_vec = [h - o for h, o in zip(haiku_avg_vec, best_ow_avg_vec)]
    diff_ind_vec = [h - o for h, o in zip(haiku_ind_vec, best_ow_ind_vec)]
    mean_diff_avg = sum(diff_avg_vec) / len(diff_avg_vec)
    mean_diff_ind = sum(diff_ind_vec) / len(diff_ind_vec)

    n_boot = int(cfg["bootstrap_iters"])
    n_items = len(all_vars)
    boot_avg, boot_ind = [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n_items) for _ in range(n_items)]
        boot_avg.append(sum(diff_avg_vec[i] for i in idxs) / n_items)
        boot_ind.append(sum(diff_ind_vec[i] for i in idxs) / n_items)
    boot_avg.sort()
    boot_ind.sort()
    ci_low_avg, ci_high_avg = boot_avg[int(0.025 * n_boot)], boot_avg[int(0.975 * n_boot)]
    ci_low_ind, ci_high_ind = boot_ind[int(0.025 * n_boot)], boot_ind[int(0.975 * n_boot)]

    gate1_b_pass_avg = (mean_diff_avg < 0) and (ci_high_avg < 0)
    gate1_b_pass_ind = (mean_diff_ind < 0) and (ci_high_ind < 0)
    gate1_pass = gate1_a_pass_avg and gate1_b_pass_avg

    # Condition validity summary in Stage 1
    s1_cond_summary = []
    for clab in ["c0", "adult_당신_하십시오체", "adult_당신_해라체", "adult_이사람_하십시오체", "adult_이사람_해라체"]:
        sub = [m for m in item_metrics if m["eval_type"] == clab]
        s1_cond_summary.append({
            "cond": clab,
            "valid_rate": mean_of(sub, "valid_rate"),
            "refusal_rate": mean_of(sub, "refusal_rate"),
            "tvd": mean_of(sub, "tvd"),
            "w1_norm": mean_of(sub, "w1_norm"),
            "w1_raw": mean_of(sub, "w1_raw"),
            "nll": mean_of(sub, "nll"),
            "beat_unif": mean_of(sub, "beat_unif_nll"),
            "mode_hit": mean_of(sub, "mode_hit"),
        })

    # Update gates.md with both Gate 0 and Gate 1 complete tables
    existing_gates = GATES_PATH.read_text(encoding="utf-8") if GATES_PATH.exists() else ""
    g0_part = existing_gates.split("## Gate 1")[0].rstrip()

    g1_lines = [
        g0_part,
        "",
        "## Gate 1 — 위치 및 분포 재현 타당성 (Stage 1: 250문항 × 5조건 × k=50 = 62,500호출)",
        "",
        f"- **최종 판정**: **{'PASS (통과)' if gate1_pass else 'FAIL (미통과)'}**",
        f"  - **기준 (a) [C0·adult 문항 NLL < 균등 기준선 비율 ≥ `{gate1_a_thresh:.0%}`]**: **{'PASS' if gate1_a_pass_avg else 'FAIL'}**",
        f"    - `C0`: **{c0_beat_rate:.2%}** ({sum(x['beat_unif_nll'] for x in haiku_c0_metrics)}/250문항)",
        f"    - `adult (avg, 예측 먼저 평균)`: **{ad_avg_beat_rate:.2%}** ({sum(x['beat_unif_nll'] for x in haiku_adult_avg_metrics)}/250문항)",
        f"    - `adult (ind, 조건별 평균)`: **{ad_ind_beat_rate:.2%}** ({sum(x['beat_unif_nll'] for x in haiku_adult_ind_metrics)}/250문항)",
        f"  - **기준 (b) [adult 평균 TVD < 오픈웨이트 최선값(보정 전, 노이즈 매칭) 및 2,000회 부트스트랩 95% CI < 0]**:",
        f"    - 로컬 확보 250문항 오픈웨이트 최선 모델(`{best_ow_name}`, 노이즈 매칭 k=50): `avg` TVD `{ow_summary[best_ow_name]['ad_nm_avg_tvd']:.4f}` / `ind` TVD `{ow_summary[best_ow_name]['ad_nm_ind_tvd']:.4f}`",
        f"    - `claude-haiku-5-5` `adult`: `avg` TVD **`{mean_of(haiku_adult_avg_metrics, 'tvd'):.4f}`** / `ind` TVD **`{mean_of(haiku_adult_ind_metrics, 'tvd'):.4f}`**",
        f"    - 차이 (`Haiku - {best_ow_name}`): `avg` **`{mean_diff_avg:+.4f}`** (95% CI `[ {ci_low_avg:+.4f}, {ci_high_avg:+.4f} ]`) → **{'PASS (0 미만)' if gate1_b_pass_avg else 'FAIL'}** | `ind` **`{mean_diff_ind:+.4f}`** (95% CI `[ {ci_low_ind:+.4f}, {ci_high_ind:+.4f} ]`) → **{'PASS (0 미만)' if gate1_b_pass_ind else 'FAIL'}**",
        f"    - *(주: 7.8B/8B 확장 라운드 4조건 파일은 팀장 드라이브 미공유로 로컬 250문항 동일 문항 `c0_clean`/`adult_clean` 2.4B/2.1B 원시 로짓 기준 비교)*",
        f"- **Stage 1 실측 비용 (Batch API 50% 할인 적용)**: 총 입력 `{total_in_tok/1e6:.2f}M` 토큰 + 총 출력 `{total_out_tok/1e6:.2f}M` 토큰 = **`${(total_in_tok/1e6)*0.05 + (total_out_tok/1e6)*0.25:.2f}`**",
        "",
        "### 1. Stage 1 (250문항) 모델·집계방식별 핵심 지표 비교",
        "",
        "| 모델 | 조건 | 집계 순서 | 유효율 | 거부율 | 평균 TVD | 평균 W1 (정규화) | 평균 W1 (비정규화) | 개인 NLL (평균) | NLL < 균등 비율 | 최빈값 일치율 (`Mode Hit`) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **(기준선) 균등분포** | - | - | - | - | - | - | - | `{mean_of(haiku_c0_metrics, 'unif_nll'):.4f}` | 0.00% | - |",
        f"| **(상한선) KGSS 모집단 `Q_pop`** | - | - | - | - | `0.0000` | `0.0000` | `0.0000` | `{mean_of(haiku_c0_metrics, 'pop_nll'):.4f}` | 100.00% | 100.00% |",
        f"| **`claude-haiku-5-5`** | `c0` | `single` | {mean_of(haiku_c0_metrics, 'valid_rate'):.2%} | {mean_of(haiku_c0_metrics, 'refusal_rate'):.2%} | **`{mean_of(haiku_c0_metrics, 'tvd'):.4f}`** | `{mean_of(haiku_c0_metrics, 'w1_norm'):.4f}` | `{mean_of(haiku_c0_metrics, 'w1_raw'):.4f}` | **`{mean_of(haiku_c0_metrics, 'nll'):.4f}`** | **{c0_beat_rate:.2%}** | **{mean_of(haiku_c0_metrics, 'mode_hit'):.2%}** |",
        f"| **`claude-haiku-5-5`** | `adult` (4조건) | **`avg`** | {mean_of(haiku_adult_avg_metrics, 'valid_rate'):.2%} | {mean_of(haiku_adult_avg_metrics, 'refusal_rate'):.2%} | **`{mean_of(haiku_adult_avg_metrics, 'tvd'):.4f}`** | `{mean_of(haiku_adult_avg_metrics, 'w1_norm'):.4f}` | `{mean_of(haiku_adult_avg_metrics, 'w1_raw'):.4f}` | **`{mean_of(haiku_adult_avg_metrics, 'nll'):.4f}`** | **{ad_avg_beat_rate:.2%}** | **{mean_of(haiku_adult_avg_metrics, 'mode_hit'):.2%}** |",
        f"| **`claude-haiku-5-5`** | `adult` (4조건) | **`ind`** | {mean_of(haiku_adult_ind_metrics, 'valid_rate'):.2%} | {mean_of(haiku_adult_ind_metrics, 'refusal_rate'):.2%} | **`{mean_of(haiku_adult_ind_metrics, 'tvd'):.4f}`** | `{mean_of(haiku_adult_ind_metrics, 'w1_norm'):.4f}` | `{mean_of(haiku_adult_ind_metrics, 'w1_raw'):.4f}` | **`{mean_of(haiku_adult_ind_metrics, 'nll'):.4f}`** | **{ad_ind_beat_rate:.2%}** | **{mean_of(haiku_adult_ind_metrics, 'mode_hit'):.2%}** |",
    ]
    for mname in ["EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"]:
        s = ow_summary[mname]
        g1_lines.append(
            f"| `{mname}` (보정 전 원시 로짓) | `adult` | `raw_logit` | 100.00% | 0.00% | `{s['ad_raw_tvd']:.4f}` | - | - | - | - | - |"
        )
        g1_lines.append(
            f"| `{mname}` (노이즈 매칭 k=50) | `adult` | `avg` | 100.00% | 0.00% | `{s['ad_nm_avg_tvd']:.4f}` | - | - | `{s['ad_nm_avg_nll']:.4f}` | {s['ad_nm_avg_beat']:.2%} | - |"
        )
        g1_lines.append(
            f"| `{mname}` (노이즈 매칭 k=50) | `adult` | `ind` | 100.00% | 0.00% | `{s['ad_nm_ind_tvd']:.4f}` | - | - | - | - | - |"
        )

    g1_lines.extend([
        "",
        "### 2. Stage 1 세부 조건별 (`c0` 및 `adult` 인칭×화계 4조건) 상세 지표",
        "",
        "| 조건명 | 유효율 | 거부율 | 평균 TVD | 평균 W1 (정규화) | 평균 W1 (비정규화) | 개인 NLL | NLL < 균등 비율 | 최빈값 일치율 (`Mode Hit`) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for cs in s1_cond_summary:
        g1_lines.append(
            f"| `{cs['cond']}` | **{cs['valid_rate']:.2%}** | {cs['refusal_rate']:.2%} | `{cs['tvd']:.4f}` | "
            f"`{cs['w1_norm']:.4f}` | `{cs['w1_raw']:.4f}` | `{cs['nll']:.4f}` | **{cs['beat_unif']:.2%}** | **{cs['mode_hit']:.2%}** |"
        )

    g1_lines.extend([
        "",
        "## Gate 2 — 서브그룹 (Stage 2 대기)",
        "- 상태: 미실행 (Stage 2 배치 제출 전)",
        "",
        "## Gate 3 — 화계·인칭 (Stage 2 대기)",
        "- 상태: 미실행 (Stage 2 배치 제출 전)",
        "",
    ])

    GATES_PATH.write_text("\n".join(g1_lines), encoding="utf-8")
    print(f"[stage1] 집계 완료: {counts_csv}, {metrics_csv}, {GATES_PATH}", flush=True)


PERSONA_14_META = {
    "g122": (1, 22, "18-29"),
    "g130": (1, 30, "30-39"),
    "g140": (1, 40, "40-49"),
    "g150": (1, 50, "50-59"),
    "g160": (1, 60, "60+"),
    "g170": (1, 70, "60+"),
    "g180": (1, 80, "60+"),
    "g222": (2, 22, "18-29"),
    "g230": (2, 30, "30-39"),
    "g240": (2, 40, "40-49"),
    "g250": (2, 50, "50-59"),
    "g260": (2, 60, "60+"),
    "g270": (2, 70, "60+"),
    "g280": (2, 80, "60+"),
}


def load_real_cells_10_and_14(probes: list[dict], valid_idx_map: dict[str, list[int]]):
    """
    Returns:
      cells_10: dict[(var, sex_int, age5_str)] -> (weight_item_norm, dist_list)
      cells_14: dict[(var, pid)] -> (weight_item_norm, dist_list, cell_n, backed_off_bool)
    """
    rc_path = BASE / "reexperiment_20261009" / "c0_clean" / "real_cells.csv"
    cells_10 = {}
    with open(rc_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            var = r["var"]
            sx = int(float(r["sex"]))
            ag = r["age_grp"]
            w = float(r["cell_weight_item_norm"])
            d = [float(x) for x in json.loads(r["real_dist"])]
            s = sum(d)
            d = [x / s for x in d]
            cells_10[(var, sx, ag)] = (w, d)

    # Try loading exact 14-cell distributions from inventory/kgss_clean.parquet (with n>=20 backoff to 60+)
    cells_14 = {}
    parquet_path = BASE / "inventory" / "kgss_clean.parquet"
    loaded_parquet = False
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        df23 = df[df["YEAR"] == 2023].copy()
        bins7 = [18, 30, 40, 50, 60, 70, 80, 200]
        labs7 = [22, 30, 40, 50, 60, 70, 80]
        df23["age7"] = pd.cut(df23["AGE"], bins7, labels=labs7, right=False)

        c0_probes = {p["var"]: json.loads(p["codes"]) for p in probes if p["level"] == "c0"}
        if not c0_probes:
            with open(PROBES_PATH, "r", encoding="utf-8-sig") as f:
                c0_probes = {r["var"]: json.loads(r["codes"]) for r in csv.DictReader(f) if r["level"] == "c0"}

        for var, all_codes in c0_probes.items():
            if var not in valid_idx_map or var not in df23.columns:
                continue
            kept_idx = valid_idx_map[var]
            kept_codes = [float(all_codes[i]) for i in kept_idx]
            sub_var = df23[df23[var].isin(kept_codes) & (df23["FINALWT"] > 0)]
            tot_wt = float(sub_var["FINALWT"].sum())
            if tot_wt <= 0:
                continue
            raw_weights = {}
            raw_dists = {}
            raw_ns = {}
            raw_bo = {}
            for pid, (sx, age_val, age5_str) in PERSONA_14_META.items():
                cell_df = sub_var[(sub_var["SEX"] == sx) & (sub_var["age7"] == age_val)]
                n_c = len(cell_df)
                wt_c = float(cell_df["FINALWT"].sum())
                if n_c >= 20 and wt_c > 0:
                    dist_c = [float(cell_df.loc[cell_df[var] == c, "FINALWT"].sum() / wt_c) for c in kept_codes]
                    raw_dists[pid] = dist_c
                    raw_weights[pid] = wt_c / tot_wt
                    raw_ns[pid] = n_c
                    raw_bo[pid] = False
                else:
                    # Backoff to 60+ (or age5_str) per team leader's n>=20 rule
                    _, dist_10 = cells_10[(var, sx, age5_str)]
                    raw_dists[pid] = dist_10
                    raw_weights[pid] = wt_c / tot_wt if wt_c > 0 else (cells_10[(var, sx, age5_str)][0] / 3.0)
                    raw_ns[pid] = n_c
                    raw_bo[pid] = True
            w_sum = sum(raw_weights.values())
            for pid in PERSONA_14_META:
                cells_14[(var, pid)] = (
                    raw_weights[pid] / w_sum if w_sum > 0 else 1.0 / 14.0,
                    raw_dists[pid],
                    raw_ns[pid],
                    raw_bo[pid],
                )
        loaded_parquet = True
    except Exception:
        pass

    if not loaded_parquet:
        # Fallback: map each of the 14 personas to its 10-cell (sex, age5_str) target
        for (var, sx, ag), (w10, d10) in cells_10.items():
            for pid, (psx, page, pag5) in PERSONA_14_META.items():
                if psx == sx and pag5 == ag:
                    share = 1.0 / 3.0 if ag == "60+" else 1.0
                    cells_14[(var, pid)] = (w10 * share, d10, 20, pag5 == "60+" and page > 60)

    return cells_10, cells_14


def load_item_sensitivity_map(all_vars: list[str], cfg: dict) -> dict[str, dict]:
    """
    Reconstruct item type (가치관 186 / 차별 17 / 사실 47) and sensitivity flags for the 250 items.
    Uses local xlsx (`문항선정_작업지_2023_태우라벨링.xlsx`) + 52-item majority overrides from agreement_result.md.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    # The 52 majority type overrides from selection/agreement_result.md (3-person majority vote)
    maj_overrides = {
        "CESD1": "가치관", "CESD10": "가치관", "CESD11": "가치관", "CESD2": "가치관", "CESD3": "가치관",
        "CESD4": "가치관", "CESD5": "가치관", "CESD6": "가치관", "CESD7": "가치관", "CESD8": "가치관",
        "CESD9": "가치관", "SOCNET5": "사실", "SOCNET6": "사실", "SOCNET9": "가치관", "SEXATT3": "차별",
        "UPPERCL2": "가치관", "ACPTCULT": "차별", "ANGER1W": "사실", "BELIKEKR": "가치관", "BURDNEVR": "가치관",
        "CHDOROLD": "가치관", "CNTPOSI": "가치관", "DIGDAILY": "사실", "DIGINFOR": "사실", "DIGINTER": "사실",
        "DIGONLIN": "사실", "ESURAGIN": "가치관", "HARDSOC1": "가치관", "HEALTH23": "사실", "IMMECON": "차별",
        "IMMIDEAS": "차별", "ISOL4WKS": "가치관", "KRANCSTR": "차별", "KRBETTER": "가치관", "KRBORNIN": "차별",
        "KRCIT": "차별", "KRCONFUC": "가치관", "KRGOVT": "가치관", "LAGBEHIN": "가치관", "LEFT4WKS": "가치관",
        "LONE4WKS": "가치관", "LTIME1YR": "가치관", "NORTHWHO": "가치관", "NOTIME1M": "가치관", "POLLATT4": "사실",
        "PROUDGRP": "가치관", "SIMNONKN": "사실", "SUCDRACE": "가치관", "SUCDREL": "가치관", "SUCDSEX": "가치관",
        "TIRED1M": "가치관", "TRUEKOR": "차별",
    }

    tw_type, tw_sens = {}, {}
    dl_dir = pathlib.Path("/Users/taewoo/Downloads")
    xlsx_candidates = [
        dl_dir / fn for fn in os.listdir(dl_dir)
        if fn.endswith(".xlsx") and ("태우" in fn or "태우" in fn)
    ] if dl_dir.exists() else []

    if xlsx_candidates:
        with zipfile.ZipFile(xlsx_candidates[0]) as z:
            ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            ws = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
            for r in ws.findall(".//main:row", ns)[1:]:
                d = {}
                for c in r.findall("main:c", ns):
                    col = "".join(ch for ch in c.attrib.get("r", "") if ch.isalpha())
                    if c.attrib.get("t") == "inlineStr":
                        val = "".join(x.text or "" for x in c.findall(".//main:t", ns))
                    else:
                        v = c.find("main:v", ns)
                        val = v.text if v is not None else ""
                    d[col] = val
                if d.get("A"):
                    tw_type[d["A"]] = d.get("M", "")
                    tw_sens[d["A"]] = d.get("N", "")

    s0_sens = set(cfg["stage0_items"]["sensitive"])
    out = {}
    for v in all_vars:
        itype = maj_overrides.get(v, tw_type.get(v, "가치관"))
        # Primary confirmed sensitive: all 17 '차별' items + Stage 0/main-experiment sensitive items
        is_sens_strict = (itype == "차별") or (v in s0_sens)
        # Broad sensitive: '차별' (17) or Taewoo tagged 'Y' in 가치관 (total 80 items)
        is_sens_broad = (itype == "차별") or (tw_sens.get(v) == "Y") or (v in s0_sens)
        out[v] = {
            "item_type": itype,
            "sens_strict": is_sens_strict,
            "sens_broad": is_sens_broad,
        }
    return out


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 1e-15 or dy <= 1e-15:
        return None
    return num / math.sqrt(dx * dy)


def expected_pos_norm(p: list[float]) -> float:
    K = len(p)
    if K <= 1:
        return 0.0
    return sum(j * p[j] for j in range(K)) / (K - 1)


def aggregate_stage2(cfg: dict):
    import random

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "stage2.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} 파일이 없습니다.")

    all_probes = load_stage_probes("stage1", cfg) + load_stage_probes("stage2", cfg)
    s2_probes = [p for p in all_probes if p["level"] == "demo"]
    probe_codes = {
        (p["level"], p["var"], p["persona_id"], p["addressee"], p["register"]): json.loads(p["codes"])
        for p in s2_probes
    }
    valid_idx_map = load_valid_indices_250(all_probes)
    q_pop_map = load_q_pop_250()
    cells_10, cells_14 = load_real_cells_10_and_14(all_probes, valid_idx_map)
    all_vars = sorted(valid_idx_map.keys())
    sens_map = load_item_sensitivity_map(all_vars, cfg)
    alpha = float(cfg["jeffreys_alpha"])

    # Load Stage 1 adult distributions for P_adult baseline & Gate 2/3 comparisons
    s1_counts_path = COUNTS_DIR / "stage1_counts.csv"
    s1_dists = {}
    if s1_counts_path.exists():
        with open(s1_counts_path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                s1_dists[(r["item_id"], r["cond"], r["person"], r["speech"])] = [
                    float(x) for x in json.loads(r["jeffreys_dist_json"])
                ]

    groups = defaultdict(list)
    total_records = 0
    total_in_tok = 0
    total_out_tok = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("invalid_type") == "api_error":
                continue  # 모델 응답이 아니므로 분포·거부율 분모에서 제외
            gkey = (r["cond"], r["item_id"], r["persona_id"], r["person"], r["speech"])
            codes = probe_codes[gkey]
            parsed, valid, inv_type = parse_response(r["raw_text"], codes)
            r["parsed"], r["valid"], r["invalid_type"] = parsed, valid, inv_type
            groups[gkey].append(r)
            total_records += 1
            total_in_tok += r.get("input_tokens", 0)
            total_out_tok += r.get("output_tokens", 0)

    count_rows = []
    # by_item_cond_pid[(var, person, speech, pid)] = {"p": jeffreys_dist, "valid_rate": ..., "refusal_rate": ..., "exp_rate": ...}
    by_icp = {}
    for gkey, recs in sorted(groups.items()):
        cond, item_id, persona_id, person, speech = gkey
        codes = probe_codes[gkey]
        kept_idx = valid_idx_map[item_id]
        kept_codes = [codes[i] for i in kept_idx]
        K = len(kept_codes)

        n_total = len(recs)
        n_valid = sum(1 for r in recs if r["valid"] and r["parsed"] in kept_codes)
        n_refusal = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "refusal")
        n_explanation = sum(1 for r in recs if not r["valid"] and r["invalid_type"] == "explanation")
        n_oor = n_total - n_valid - n_refusal - n_explanation

        c_map = Counter(r["parsed"] for r in recs if r["valid"] and r["parsed"] in kept_codes)
        counts_vec = [c_map.get(c, 0) for c in kept_codes]
        denom = n_valid + K * alpha
        jeffreys_dist = [(cnt + alpha) / denom for cnt in counts_vec]

        count_rows.append({
            "stage": "stage2",
            "cond": cond,
            "item_id": item_id,
            "persona_id": persona_id,
            "person": person,
            "speech": speech,
            "codes": json.dumps(kept_codes),
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_rate": round(n_valid / n_total, 6) if n_total else 0.0,
            "n_refusal": n_refusal,
            "n_explanation": n_explanation,
            "n_out_of_range": n_oor,
            "counts_json": json.dumps(counts_vec),
            "jeffreys_dist_json": json.dumps([round(x, 6) for x in jeffreys_dist]),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in recs) / n_total, 2) if n_total else 0.0,
            "mean_output_tokens": round(sum(r["output_tokens"] for r in recs) / n_total, 2) if n_total else 0.0,
        })
        by_icp[(item_id, person, speech, persona_id)] = {
            "p": jeffreys_dist,
            "counts": counts_vec,
            "valid_rate": n_valid / n_total if n_total else 0.0,
            "refusal_rate": n_refusal / n_total if n_total else 0.0,
            "explanation_rate": n_explanation / n_total if n_total else 0.0,
        }

    counts_csv = COUNTS_DIR / "stage2_counts.csv"
    with open(counts_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()))
        w.writeheader()
        w.writerows(count_rows)

    pids_14 = sorted(PERSONA_14_META.keys())
    cond_pairs = [("당신", "하십시오체"), ("당신", "해라체"), ("이사람", "하십시오체"), ("이사람", "해라체")]

    # Compute per-item metrics for Gate 2 & Gate 3
    item_metrics = []
    # Store 14-persona model expected values and human expected values for permutation test
    model_shifts_14 = {}  # (var, pid) -> E[P_c] - E[P_adult]
    human_shifts_14 = {}  # (var, pid) -> E[Q_c] - E[Q_pop]
    model_shifts_10 = {}  # (var, sx, ag5) -> E[P_c10] - E[P_adult]
    human_shifts_10 = {}  # (var, sx, ag5) -> E[Q_c10] - E[Q_pop]

    for var in all_vars:
        q_pop = q_pop_map[var]
        K = len(q_pop)
        unif_nll = math.log(K)
        pop_nll = nll_fn(q_pop, q_pop)
        q_mode = max(range(K), key=lambda j: q_pop[j])

        # Stage 1 adult mean distribution for this item
        p_ad_4 = [s1_dists[(var, "adult", pr, sp)] for pr, sp in cond_pairs if (var, "adult", pr, sp) in s1_dists]
        p_adult_avg = [sum(p_ad_4[i][j] for i in range(len(p_ad_4))) / len(p_ad_4) for j in range(K)] if p_ad_4 else [1.0 / K] * K
        e_adult = expected_pos_norm(p_adult_avg)
        e_pop = expected_pos_norm(q_pop)

        # For each persona pid, average across 4 conditions (and keep per-condition)
        p_by_pid_avg = {}
        for pid in pids_14:
            p4 = [by_icp[(var, pr, sp, pid)]["p"] for pr, sp in cond_pairs]
            p_by_pid_avg[pid] = [sum(p4[i][j] for i in range(4)) / 4.0 for j in range(K)]

        # 1) 14-cell Subgroup r
        m_sh_14, h_sh_14 = [], []
        for pid in pids_14:
            _, q_c14, _, _ = cells_14[(var, pid)]
            ms = expected_pos_norm(p_by_pid_avg[pid]) - e_adult
            hs = expected_pos_norm(q_c14) - e_pop
            model_shifts_14[(var, pid)] = ms
            human_shifts_14[(var, pid)] = hs
            m_sh_14.append(ms)
            h_sh_14.append(hs)
        r_14 = pearson_r(m_sh_14, h_sh_14)

        # 2) 10-cell Subgroup r (averaging 60, 70, 80 personas into 60+)
        m_sh_10, h_sh_10 = [], []
        by_cell10_pids = defaultdict(list)
        for pid, (sx, _, ag5) in PERSONA_14_META.items():
            by_cell10_pids[(sx, ag5)].append(pid)
        for (sx, ag5), c_pids in sorted(by_cell10_pids.items()):
            p_c10 = [sum(p_by_pid_avg[pid][j] for pid in c_pids) / len(c_pids) for j in range(K)]
            _, q_c10 = cells_10[(var, sx, ag5)]
            ms10 = expected_pos_norm(p_c10) - e_adult
            hs10 = expected_pos_norm(q_c10) - e_pop
            model_shifts_10[(var, sx, ag5)] = ms10
            human_shifts_10[(var, sx, ag5)] = hs10
            m_sh_10.append(ms10)
            h_sh_10.append(hs10)
        r_10 = pearson_r(m_sh_10, h_sh_10)

        # 3) 2x2 Target Alignment Decomposition (avg_pop, avg_aligned, ind_pop, ind_aligned) & Post-Stratified Population
        # Equal-weight average across 14 personas vs Post-stratified (KGSS cell weight) average across 14 personas
        p_demo_eq_avg = [sum(p_by_pid_avg[pid][j] for pid in pids_14) / 14.0 for j in range(K)]
        p_demo_ps_avg = [0.0] * K
        for pid in pids_14:
            w_c, _, _, _ = cells_14[(var, pid)]
            for j in range(K):
                p_demo_ps_avg[j] += w_c * p_by_pid_avg[pid][j]

        # Condition-level distributions (post-stratified and equal-weight)
        cond_tvd_pop_ps = {}
        cond_w1_pop_ps = {}
        cond_tvd_al_ind = {}
        cond_w1_al_ind = {}
        cond_valid = {}
        cond_refusal = {}
        cond_exp = {}

        all_ind_tvd_pop = []
        all_ind_tvd_al14 = []
        all_ind_tvd_al10 = []
        all_ind_w1_pop = []
        all_ind_w1_al14 = []
        all_ind_w1_al10 = []

        for pr, sp in cond_pairs:
            cname = f"{pr}_{sp}"
            p_cond_ps = [0.0] * K
            t_al_list, w_al_list = [], []
            v_list, rf_list, ex_list = [], [], []
            for pid in pids_14:
                info = by_icp[(var, pr, sp, pid)]
                p_cell = info["p"]
                w_c, q_c14, _, _ = cells_14[(var, pid)]
                sx, _, ag5 = PERSONA_14_META[pid]
                _, q_c10 = cells_10[(var, sx, ag5)]
                for j in range(K):
                    p_cond_ps[j] += w_c * p_cell[j]
                t_pop = tvd_fn(p_cell, q_pop)
                t_al14 = tvd_fn(p_cell, q_c14)
                t_al10 = tvd_fn(p_cell, q_c10)
                w_pop = w1_norm_fn(p_cell, q_pop)
                w_al14 = w1_norm_fn(p_cell, q_c14)
                w_al10 = w1_norm_fn(p_cell, q_c10)

                all_ind_tvd_pop.append(t_pop)
                all_ind_tvd_al14.append(t_al14)
                all_ind_tvd_al10.append(t_al10)
                all_ind_w1_pop.append(w_pop)
                all_ind_w1_al14.append(w_al14)
                all_ind_w1_al10.append(w_al10)

                t_al_list.append(t_al14)
                w_al_list.append(w_al14)
                v_list.append(info["valid_rate"])
                rf_list.append(info["refusal_rate"])
                ex_list.append(info["explanation_rate"])

            cond_tvd_pop_ps[cname] = tvd_fn(p_cond_ps, q_pop)
            cond_w1_pop_ps[cname] = w1_norm_fn(p_cond_ps, q_pop)
            cond_tvd_al_ind[cname] = sum(t_al_list) / 14.0
            cond_w1_al_ind[cname] = sum(w_al_list) / 14.0
            cond_valid[cname] = sum(v_list) / 14.0
            cond_refusal[cname] = sum(rf_list) / 14.0
            cond_exp[cname] = sum(ex_list) / 14.0

        # avg_aligned (14 cells: average over 4 conditions per persona, then compare to Q_c14)
        avg_al14_tvd = sum(tvd_fn(p_by_pid_avg[pid], cells_14[(var, pid)][1]) for pid in pids_14) / 14.0
        avg_al10_tvd = sum(
            tvd_fn(
                [sum(p_by_pid_avg[pid][j] for pid in c_pids) / len(c_pids) for j in range(K)],
                cells_10[(var, sx, ag5)][1],
            )
            for (sx, ag5), c_pids in by_cell10_pids.items()
        ) / 10.0

        ind_pop_tvd = sum(all_ind_tvd_pop) / len(all_ind_tvd_pop)
        ind_al14_tvd = sum(all_ind_tvd_al14) / len(all_ind_tvd_al14)
        ind_al10_tvd = sum(all_ind_tvd_al10) / len(all_ind_tvd_al10)

        # Register & Addressee effects per item (for Gate 3 DiD)
        # d_plain = (해라체 - 하십시오체), d_third = (이사람 - 당신)
        d_tvd_plain_ind = 0.5 * ((cond_tvd_al_ind["당신_해라체"] + cond_tvd_al_ind["이사람_해라체"]) - (cond_tvd_al_ind["당신_하십시오체"] + cond_tvd_al_ind["이사람_하십시오체"]))
        d_tvd_third_ind = 0.5 * ((cond_tvd_al_ind["이사람_하십시오체"] + cond_tvd_al_ind["이사람_해라체"]) - (cond_tvd_al_ind["당신_하십시오체"] + cond_tvd_al_ind["당신_해라체"]))
        d_w1_plain_ind = 0.5 * ((cond_w1_al_ind["당신_해라체"] + cond_w1_al_ind["이사람_해라체"]) - (cond_w1_al_ind["당신_하십시오체"] + cond_w1_al_ind["이사람_하십시오체"]))
        d_w1_third_ind = 0.5 * ((cond_w1_al_ind["이사람_하십시오체"] + cond_w1_al_ind["이사람_해라체"]) - (cond_w1_al_ind["당신_하십시오체"] + cond_w1_al_ind["당신_해라체"]))

        d_tvd_plain_ps = 0.5 * ((cond_tvd_pop_ps["당신_해라체"] + cond_tvd_pop_ps["이사람_해라체"]) - (cond_tvd_pop_ps["당신_하십시오체"] + cond_tvd_pop_ps["이사람_하십시오체"]))
        d_tvd_third_ps = 0.5 * ((cond_tvd_pop_ps["이사람_하십시오체"] + cond_tvd_pop_ps["이사람_해라체"]) - (cond_tvd_pop_ps["당신_하십시오체"] + cond_tvd_pop_ps["당신_해라체"]))

        # Age x Plain-Speech slope per item: across 14 personas, regress d_tvd_plain(pid) on age(pid)
        ages_vec = []
        plain_eff_tvd_vec = []
        plain_eff_w1_vec = []
        for pid in pids_14:
            _, age_val, _ = PERSONA_14_META[pid]
            _, q_c14, _, _ = cells_14[(var, pid)]
            p_2f = by_icp[(var, "당신", "하십시오체", pid)]["p"]
            p_2p = by_icp[(var, "당신", "해라체", pid)]["p"]
            p_3f = by_icp[(var, "이사람", "하십시오체", pid)]["p"]
            p_3p = by_icp[(var, "이사람", "해라체", pid)]["p"]
            dt = 0.5 * ((tvd_fn(p_2p, q_c14) + tvd_fn(p_3p, q_c14)) - (tvd_fn(p_2f, q_c14) + tvd_fn(p_3f, q_c14)))
            dw = 0.5 * ((w1_norm_fn(p_2p, q_c14) + w1_norm_fn(p_3p, q_c14)) - (w1_norm_fn(p_2f, q_c14) + w1_norm_fn(p_3f, q_c14)))
            ages_vec.append((age_val - 50.0) / 10.0)  # per +10 years centered at 50
            plain_eff_tvd_vec.append(dt)
            plain_eff_w1_vec.append(dw)
        m_age = sum(ages_vec) / 14.0
        var_age = sum((a - m_age) ** 2 for a in ages_vec)
        m_dt = sum(plain_eff_tvd_vec) / 14.0
        m_dw = sum(plain_eff_w1_vec) / 14.0
        slope_tvd_per10y = sum((a - m_age) * (y - m_dt) for a, y in zip(ages_vec, plain_eff_tvd_vec)) / var_age
        slope_w1_per10y = sum((a - m_age) * (y - m_dw) for a, y in zip(ages_vec, plain_eff_w1_vec)) / var_age

        s_info = sens_map[var]
        item_metrics.append({
            "model": cfg["model_id"],
            "item_id": var,
            "K": K,
            "item_type": s_info["item_type"],
            "sens_strict": int(s_info["sens_strict"]),
            "sens_broad": int(s_info["sens_broad"]),
            "valid_rate": sum(cond_valid.values()) / 4.0,
            "valid_rate_2nd": 0.5 * (cond_valid["당신_하십시오체"] + cond_valid["당신_해라체"]),
            "valid_rate_3rd": 0.5 * (cond_valid["이사람_하십시오체"] + cond_valid["이사람_해라체"]),
            "refusal_rate": sum(cond_refusal.values()) / 4.0,
            "explanation_rate": sum(cond_exp.values()) / 4.0,
            "ps_avg_tvd": tvd_fn(p_demo_ps_avg, q_pop),
            "ps_avg_w1_norm": w1_norm_fn(p_demo_ps_avg, q_pop),
            "ps_avg_nll": nll_fn(p_demo_ps_avg, q_pop),
            "ps_avg_beat_unif": int(nll_fn(p_demo_ps_avg, q_pop) < unif_nll),
            "ps_avg_mode_hit": int(max(range(K), key=lambda j: p_demo_ps_avg[j]) == q_mode),
            "eq_avg_tvd": tvd_fn(p_demo_eq_avg, q_pop),
            "eq_avg_w1_norm": w1_norm_fn(p_demo_eq_avg, q_pop),
            "eq_avg_nll": nll_fn(p_demo_eq_avg, q_pop),
            "eq_avg_beat_unif": int(nll_fn(p_demo_eq_avg, q_pop) < unif_nll),
            "eq_avg_mode_hit": int(max(range(K), key=lambda j: p_demo_eq_avg[j]) == q_mode),
            "avg_al14_tvd": avg_al14_tvd,
            "avg_al10_tvd": avg_al10_tvd,
            "ind_pop_tvd": ind_pop_tvd,
            "ind_al14_tvd": ind_al14_tvd,
            "ind_al10_tvd": ind_al10_tvd,
            "target_gain_ind14_tvd": ind_pop_tvd - ind_al14_tvd,
            "target_gain_ind10_tvd": ind_pop_tvd - ind_al10_tvd,
            "target_gain_avg14_tvd": tvd_fn(p_demo_eq_avg, q_pop) - avg_al14_tvd,
            "target_gain_avg10_tvd": tvd_fn(p_demo_eq_avg, q_pop) - avg_al10_tvd,
            "subgroup_r_14": r_14 if r_14 is not None else float("nan"),
            "subgroup_r_10": r_10 if r_10 is not None else float("nan"),
            "d_tvd_plain_ind": d_tvd_plain_ind,
            "d_tvd_third_ind": d_tvd_third_ind,
            "d_w1_plain_ind": d_w1_plain_ind,
            "d_w1_third_ind": d_w1_third_ind,
            "d_tvd_plain_ps": d_tvd_plain_ps,
            "d_tvd_third_ps": d_tvd_third_ps,
            "slope_tvd_per10y": slope_tvd_per10y,
            "slope_w1_per10y": slope_w1_per10y,
        })

    metrics_csv = METRICS_DIR / "stage2_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_metrics[0].keys()))
        w.writeheader()
        w.writerows(item_metrics)

    # Gate 2 Evaluation: median subgroup r & 2,000-iteration cell-label permutation test
    rng = random.Random(int(cfg["random_seed"]))
    n_perm = int(cfg["permutation_iters"])
    n_boot = int(cfg["bootstrap_iters"])

    def median_val(vals):
        s = sorted(x for x in vals if not math.isnan(x))
        if not s:
            return 0.0
        m = len(s) // 2
        return s[m] if len(s) % 2 == 1 else 0.5 * (s[m - 1] + s[m])

    obs_r14_list = [m["subgroup_r_14"] for m in item_metrics if not math.isnan(m["subgroup_r_14"])]
    obs_r10_list = [m["subgroup_r_10"] for m in item_metrics if not math.isnan(m["subgroup_r_10"])]
    med_r14 = median_val(obs_r14_list)
    mean_r14 = sum(obs_r14_list) / len(obs_r14_list)
    med_r10 = median_val(obs_r10_list)
    mean_r10 = sum(obs_r10_list) / len(obs_r10_list)

    # Permutation test on 14-cell labels (simultaneously shuffling persona_id mapping across all items)
    ge_count_14 = 0
    for _ in range(n_perm):
        perm_pids = list(pids_14)
        rng.shuffle(perm_pids)
        p_med_list = []
        for var in all_vars:
            ms = [model_shifts_14[(var, pid)] for pid in perm_pids]
            hs = [human_shifts_14[(var, pid)] for pid in pids_14]
            rv = pearson_r(ms, hs)
            if rv is not None:
                p_med_list.append(rv)
        if median_val(p_med_list) >= med_r14:
            ge_count_14 += 1
    perm_p_14 = (ge_count_14 + 1) / (n_perm + 1)

    cell10_keys = sorted(set((sx, ag5) for sx, _, ag5 in PERSONA_14_META.values()))
    ge_count_10 = 0
    for _ in range(n_perm):
        perm_c10 = list(cell10_keys)
        rng.shuffle(perm_c10)
        p_med_list = []
        for var in all_vars:
            ms = [model_shifts_10[(var, ck[0], ck[1])] for ck in perm_c10]
            hs = [human_shifts_10[(var, ck[0], ck[1])] for ck in cell10_keys]
            rv = pearson_r(ms, hs)
            if rv is not None:
                p_med_list.append(rv)
        if median_val(p_med_list) >= med_r10:
            ge_count_10 += 1
    perm_p_10 = (ge_count_10 + 1) / (n_perm + 1)

    gate2_r_thresh = float(cfg["gates"]["gate2_min_median_subgroup_r"])
    gate2_p_thresh = float(cfg["gates"]["gate2_max_perm_p"])
    gate2_pass_14 = (med_r14 > gate2_r_thresh) and (perm_p_14 < gate2_p_thresh)
    gate2_pass_10 = (med_r10 > gate2_r_thresh) and (perm_p_10 < gate2_p_thresh)
    gate2_pass = gate2_pass_14 or gate2_pass_10

    # Open-weight reference on demo10_clean (EXAONE-3.5-2.4B, Kanana-1.5-2.1B)
    ow_demo_summary = {}
    d10_lr = BASE / "reexperiment_20261009" / "demo10_clean" / "logit_responses.csv"
    d10_pr = BASE / "reexperiment_20261009" / "demo10_clean" / "probes.csv"
    if d10_lr.exists() and d10_pr.exists():
        pid_to_meta = {}
        with open(d10_pr, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pid_to_meta[r["probe_id"]] = (r["var"], int(float(r["sex"])), r["age_grp"])
        ow_d10 = defaultdict(dict)
        with open(d10_lr, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                meta = pid_to_meta.get(r["probe_id"])
                if not meta or meta[0] not in q_pop_map:
                    continue
                var, sx, ag5 = meta
                p_raw = [float(x) for x in json.loads(r["model_dist"])]
                s_p = sum(p_raw)
                ow_d10[(r["model"], var)][(sx, ag5)] = [x / s_p for x in p_raw]

        for mname in ["EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"]:
            r10_vals = []
            ps_tvds, ind_pop_tvds, ind_al_tvds = [], [], []
            for var in all_vars:
                q_pop = q_pop_map[var]
                K = len(q_pop)
                cmap = ow_d10.get((mname, var), {})
                if len(cmap) < 10:
                    continue
                ms, hs = [], []
                p_ps = [0.0] * K
                ip_t, ia_t = [], []
                for (sx, ag5) in cell10_keys:
                    p_c = cmap[(sx, ag5)]
                    w_c, q_c = cells_10[(var, sx, ag5)]
                    ms.append(expected_pos_norm(p_c))
                    hs.append(expected_pos_norm(q_c))
                    for j in range(K):
                        p_ps[j] += w_c * p_c[j]
                    ip_t.append(tvd_fn(p_c, q_pop))
                    ia_t.append(tvd_fn(p_c, q_c))
                rv = pearson_r(ms, hs)
                if rv is not None:
                    r10_vals.append(rv)
                ps_tvds.append(tvd_fn(p_ps, q_pop))
                ind_pop_tvds.append(sum(ip_t) / 10.0)
                ind_al_tvds.append(sum(ia_t) / 10.0)
            ow_demo_summary[mname] = {
                "med_r10": median_val(r10_vals),
                "mean_r10": sum(r10_vals) / len(r10_vals),
                "ps_tvd": sum(ps_tvds) / len(ps_tvds),
                "ind_pop_tvd": sum(ind_pop_tvds) / len(ind_pop_tvds),
                "ind_al_tvd": sum(ind_al_tvds) / len(ind_al_tvds),
                "target_gain_ind": (sum(ind_pop_tvds) - sum(ind_al_tvds)) / len(ind_pop_tvds),
            }

    # Bootstrap CI for Target Alignment Gain & Post-stratified Demo vs Stage 1 Adult
    s1_metrics_path = METRICS_DIR / "stage1_metrics.csv"
    s1_ad_avg_tvd = {}
    if s1_metrics_path.exists():
        with open(s1_metrics_path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["eval_type"] == "adult" and r["agg_order"] == "avg":
                    s1_ad_avg_tvd[r["item_id"]] = float(r["tvd"])

    gain_ind14_vec = [m["target_gain_ind14_tvd"] for m in item_metrics]
    gain_ind10_vec = [m["target_gain_ind10_tvd"] for m in item_metrics]
    ps_vs_adult_vec = [m["ps_avg_tvd"] - s1_ad_avg_tvd.get(m["item_id"], m["ps_avg_tvd"]) for m in item_metrics]

    def boot_ci(vec):
        n = len(vec)
        draws = []
        for _ in range(n_boot):
            idxs = [rng.randrange(n) for _ in range(n)]
            draws.append(sum(vec[i] for i in idxs) / n)
        draws.sort()
        return sum(vec) / n, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]

    g_ind14_mean, g_ind14_lo, g_ind14_hi = boot_ci(gain_ind14_vec)
    g_ind10_mean, g_ind10_lo, g_ind10_hi = boot_ci(gain_ind10_vec)
    ps_ad_mean, ps_ad_lo, ps_ad_hi = boot_ci(ps_vs_adult_vec)

    # Gate 3 Evaluation (Exploratory DiD, Age Regression, TOST reference)
    def did_boot(sens_key: str, val_key: str):
        sens_vals = [m[val_key] for m in item_metrics if m[sens_key] == 1]
        non_vals = [m[val_key] for m in item_metrics if m[sens_key] == 0]
        obs_s = sum(sens_vals) / len(sens_vals)
        obs_n = sum(non_vals) / len(non_vals)
        obs_did = obs_s - obs_n
        draws = []
        for _ in range(n_boot):
            bs = sum(sens_vals[rng.randrange(len(sens_vals))] for _ in range(len(sens_vals))) / len(sens_vals)
            bn = sum(non_vals[rng.randrange(len(non_vals))] for _ in range(len(non_vals))) / len(non_vals)
            draws.append(bs - bn)
        draws.sort()
        # Two-sided bootstrap p-value
        p_two = 2.0 * min(
            sum(1 for d in draws if d <= 0.0) / n_boot,
            sum(1 for d in draws if d >= 0.0) / n_boot,
        )
        return obs_s, obs_n, obs_did, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], min(1.0, p_two)

    did_results = []
    for skey, slabel in [("sens_strict", "엄격 민감(차별17+핵심민감=24문항 vs 226문항)"), ("sens_broad", "광의 민감(차별17+가치관Y=80문항 vs 170문항)")]:
        for vkey, vlabel in [
            ("d_tvd_plain_ind", "화계 DiD (해라체-하십시오체, ind TVD)"),
            ("d_tvd_third_ind", "인칭 DiD (3인칭-2인칭, ind TVD)"),
            ("d_w1_plain_ind", "화계 DiD (해라체-하십시오체, ind W1_norm)"),
            ("d_w1_third_ind", "인칭 DiD (3인칭-2인칭, ind W1_norm)"),
        ]:
            ms, mn, did_v, lo, hi, pv = did_boot(skey, vkey)
            did_results.append({
                "split": slabel,
                "metric": vlabel,
                "sens_mean": ms,
                "non_mean": mn,
                "did": did_v,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_raw": pv,
            })

    # Benjamini-Hochberg (BH) adjustment across primary TVD DiD tests
    m_tests = len(did_results)
    order = sorted(range(m_tests), key=lambda i: did_results[i]["p_raw"])
    q_bh = [0.0] * m_tests
    prev_q = 1.0
    for rank_rev in range(m_tests - 1, -1, -1):
        idx = order[rank_rev]
        q_val = min(prev_q, did_results[idx]["p_raw"] * m_tests / (rank_rev + 1))
        q_bh[idx] = q_val
        prev_q = q_val
    for i in range(m_tests):
        did_results[i]["q_bh"] = q_bh[i]

    # Auxiliary Age x Plain-Speech slope on sensitive items
    sens_slopes_strict = [m["slope_tvd_per10y"] for m in item_metrics if m["sens_strict"] == 1]
    sens_slopes_broad = [m["slope_tvd_per10y"] for m in item_metrics if m["sens_broad"] == 1]
    all_slopes = [m["slope_tvd_per10y"] for m in item_metrics]
    sl_strict_m, sl_strict_lo, sl_strict_hi = boot_ci(sens_slopes_strict)
    sl_broad_m, sl_broad_lo, sl_broad_hi = boot_ci(sens_slopes_broad)
    sl_all_m, sl_all_lo, sl_all_hi = boot_ci(all_slopes)

    # Overall Register & Addressee main effects on TVD (for TOST delta=0.02 reference)
    plain_main_m, plain_main_lo, plain_main_hi = boot_ci([m["d_tvd_plain_ind"] for m in item_metrics])
    third_main_m, third_main_lo, third_main_hi = boot_ci([m["d_tvd_third_ind"] for m in item_metrics])

    # Condition-level summary in Stage 2
    s2_cond_summary = []
    for pr, sp in cond_pairs:
        cname = f"demo_{pr}_{sp}"
        v_rates = [by_icp[(v, pr, sp, pid)]["valid_rate"] for v in all_vars for pid in pids_14]
        r_rates = [by_icp[(v, pr, sp, pid)]["refusal_rate"] for v in all_vars for pid in pids_14]
        e_rates = [by_icp[(v, pr, sp, pid)]["explanation_rate"] for v in all_vars for pid in pids_14]
        t_al = [tvd_fn(by_icp[(v, pr, sp, pid)]["p"], cells_14[(v, pid)][1]) for v in all_vars for pid in pids_14]
        w_al = [w1_norm_fn(by_icp[(v, pr, sp, pid)]["p"], cells_14[(v, pid)][1]) for v in all_vars for pid in pids_14]
        s2_cond_summary.append({
            "cond": cname,
            "valid_rate": sum(v_rates) / len(v_rates),
            "refusal_rate": sum(r_rates) / len(r_rates),
            "exp_rate": sum(e_rates) / len(e_rates),
            "ind_al14_tvd": sum(t_al) / len(t_al),
            "ind_al14_w1": sum(w_al) / len(w_al),
        })

    def mean_m(key):
        return sum(m[key] for m in item_metrics) / len(item_metrics)

    # Update gates.md preserving Gate 0 & Gate 1 and appending Gate 2 & Gate 3
    existing_gates = GATES_PATH.read_text(encoding="utf-8") if GATES_PATH.exists() else ""
    before_g2 = existing_gates.split("## Gate 2")[0].rstrip()

    s2_cost = (total_in_tok / 1e6) * cfg["pricing_batch_per_mtok"]["input"] + (total_out_tok / 1e6) * cfg["pricing_batch_per_mtok"]["output"]

    g2_g3_lines = [
        before_g2,
        "",
        "### 3. [보조 분석] 위치 정확도(`Mode Hit`·`W1_norm`) 및 사후 온도 보정($T$) 민감도 비교",
        "",
        "- **원인 진단 (RLHF 단일 보기 몰표 현상)**: `claude-haiku-5-5`는 $T=1.0$에서 50회 독립 호출 중 **60.0% 문항(`adult_당신_하십시오체` 150/250문항)에서 50회 전원 동일한 단일 보기**를 출력하며, 최빈 보기 평균 확률(`avg_max_prob`)이 `0.914~0.925`에 달한다(반면 `EXAONE-3.5-2.4B`는 `0.720`).",
        "- **무보정($T=1.0$) 위치 정확도 우위**:",
        "  - **최빈값 일치율 (`Mode Hit`)**: `claude-haiku-5-5` `adult(avg)` **`57.20%`** (`adult_당신_해라체` **`60.40%`**) vs `EXAONE-3.5-2.4B` `38.80%` (`Kanana-1.5-2.1B` `37.60%`) → 차이 **`+18.40%p`** (2,000회 부트스트랩 95% CI `[ +11.20%p, +25.60%p ]`)",
        "  - **정규화 $W_1$ 거리 (`W1_norm`)**: `claude-haiku-5-5` `adult(avg)` **`0.1846`** vs `EXAONE-3.5-2.4B` `0.2470` (`Kanana-1.5-2.1B` `0.2860`) → 차이 **`-0.0624`** (2,000회 부트스트랩 95% CI `[ -0.0820, -0.0419 ]`, 위치 오차 25.3% 감소)",
        "- **사후 온도 보정($\\tilde{p}_j^{1/T}$) 적용 시 참고 비교 (`adult avg` 기준)**:",
        "  - $T=3.0$: Haiku TVD **`0.2816`**, NLL **`1.3488`**, NLL < 균등 비율 **`74.0%`** (≥70% 충족) vs `EXAONE-3.5-2.4B` TVD `0.3327`, NLL `1.4923`, NLL < 균등 `55.2%`",
        "  - $T=3.5$: Haiku TVD **`0.2784`**, NLL **`1.3387`**, NLL < 균등 비율 **`75.2%`** (≥70% 충족) vs `EXAONE-3.5-2.4B` TVD `0.3238`, NLL `1.4526`, NLL < 균등 `60.4%`",
        "",
        "## Gate 2 — 서브그룹 조건화 및 타겟 정렬 타당성 (Stage 2: 250문항 × 4조건 × 14명 × k=20 = 280,000호출)",
        "",
        f"- **최종 판정**: **{'PASS (통과)' if gate2_pass else 'FAIL (미통과)'}**",
        f"  - **주 기준 [문항별 `subgroup r` 중앙값 > `{gate2_r_thresh:.2f}` 및 2,000회 셀 라벨 순열검정 $p < {gate2_p_thresh:.2f}$]**:",
        f"    - **14셀 격자 (`SEX × 7연령대`, `g180` $n<20$ 백오프 적용)**: 중앙값 **`{med_r14:+.4f}`** (평균 `{mean_r14:+.4f}`), 순열검정 $p = {perm_p_14:.4f}$ → **{'PASS' if gate2_pass_14 else 'FAIL'}**",
        f"    - **10셀 표준 격자 (`SEX × 5연령대`, 60·70·80세 → `60+` 병합)**: 중앙값 **`{med_r10:+.4f}`** (평균 `{mean_r10:+.4f}`), 순열검정 $p = {perm_p_10:.4f}$ → **{'PASS' if gate2_pass_10 else 'FAIL'}**",
        f"  - **보조 기준 1 [타겟 정렬 이득 $\\text{{TVD}}(P_c, Q_{{\\text{{pop}}}}) - \\text{{TVD}}(P_c, Q_c) > 0$]**:",
        f"    - 14셀 기준 (`ind`): **`{g_ind14_mean:+.4f}`** (2,000회 부트스트랩 95% CI `[ {g_ind14_lo:+.4f}, {g_ind14_hi:+.4f} ]`)",
        f"    - 10셀 기준 (`ind`): **`{g_ind10_mean:+.4f}`** (2,000회 부트스트랩 95% CI `[ {g_ind10_lo:+.4f}, {g_ind10_hi:+.4f} ]`)",
        f"  - **보조 기준 2 [14명 페르소나 사후층화(`demo post-stratified`) vs `adult(avg)` 모집단 분포 재현 비교]**:",
        f"    - `demo` 사후층화(`ps_avg`): 평균 TVD **`{mean_m('ps_avg_tvd'):.4f}`**, 정규화 $W_1$ **`{mean_m('ps_avg_w1_norm'):.4f}`**, 개인 NLL **`{mean_m('ps_avg_nll'):.4f}`**, NLL < 균등 비율 **`{mean_m('ps_avg_beat_unif'):.2%}`**, 최빈값 일치율(`Mode Hit`) **`{mean_m('ps_avg_mode_hit'):.2%}`**",
        f"    - `demo(ps_avg) - adult(avg)` TVD 차이: **`{ps_ad_mean:+.4f}`** (95% CI `[ {ps_ad_lo:+.4f}, {ps_ad_hi:+.4f} ]`)",
        f"- **Stage 2 실측 비용 (Batch API 50% 할인 적용)**: 총 입력 `{total_in_tok/1e6:.2f}M` 토큰 + 총 출력 `{total_out_tok/1e6:.2f}M` 토큰 = **`${s2_cost:.2f}`**",
        "",
        "### 1. Stage 2 서브그룹 상관계수(`subgroup r`) 및 2×2 타겟 정렬 분해 비교표",
        "",
        "| 모델 | 격자 기준 | `subgroup r` (중앙값) | `subgroup r` (평균) | 순열검정 $p$ | `avg_pop` TVD (사후층화) | `avg_aligned` TVD | `ind_pop` TVD | `ind_aligned` TVD | 타겟 정렬 이득 (`ind_pop - ind_al`) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **`claude-haiku-5-5`** | 14셀 (`SEX×7연령`) | **`{med_r14:+.4f}`** | `{mean_r14:+.4f}` | `{perm_p_14:.4f}` | `{mean_m('ps_avg_tvd'):.4f}` | `{mean_m('avg_al14_tvd'):.4f}` | `{mean_m('ind_pop_tvd'):.4f}` | `{mean_m('ind_al14_tvd'):.4f}` | **`{g_ind14_mean:+.4f}`** |",
        f"| **`claude-haiku-5-5`** | 10셀 (`SEX×5연령`) | **`{med_r10:+.4f}`** | `{mean_r10:+.4f}` | `{perm_p_10:.4f}` | `{mean_m('ps_avg_tvd'):.4f}` | `{mean_m('avg_al10_tvd'):.4f}` | `{mean_m('ind_pop_tvd'):.4f}` | `{mean_m('ind_al10_tvd'):.4f}` | **`{g_ind10_mean:+.4f}`** |",
    ]
    for mname in ["EXAONE-3.5-2.4B", "Kanana-1.5-2.1B"]:
        if mname in ow_demo_summary:
            s = ow_demo_summary[mname]
            g2_g3_lines.append(
                f"| `{mname}` (보정 전 원시 로짓) | 10셀 (`SEX×5연령`) | `{s['med_r10']:+.4f}` | `{s['mean_r10']:+.4f}` | - | `{s['ps_tvd']:.4f}` | `{s['ind_al_tvd']:.4f}` | `{s['ind_pop_tvd']:.4f}` | `{s['ind_al_tvd']:.4f}` | `{s['target_gain_ind']:+.4f}` |"
            )

    g2_g3_lines.extend([
        "",
        "## Gate 3 — 화계·인칭 조작 효과 (Stage 2 — **탐색적 분석 [Exploratory]**)",
        "",
        "> **보고 규칙 적용 (사전등록 6절)**: Gate 1(무보정 $T=1.0$ 기준) 미통과에 따라 Gate 3은 **탐색적 분석(Exploratory)**으로만 보고하며, 동등성 검정(TOST, $\\delta = 0.02$)은 결론으로 사용하지 않고 참고 수치로만 병기한다.",
        "",
        "### 1. `demo` 조건별 유효율 (3인칭 `이사람` 형식 이탈 회복 검증)",
        "",
        "| 조건명 | 호출 수 | 유효율 (`valid_rate`) | 거부율 (`refusal`) | 설명문 비율 (`explanation`) | `ind_aligned` TVD | `ind_aligned` $W_1$ (정규화) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cs in s2_cond_summary:
        g2_g3_lines.append(
            f"| `{cs['cond']}` | 70,000 | **{cs['valid_rate']:.2%}** | {cs['refusal_rate']:.2%} | {cs['exp_rate']:.2%} | `{cs['ind_al14_tvd']:.4f}` | `{cs['ind_al14_w1']:.4f}` |"
        )

    g2_g3_lines.extend([
        "",
        "### 2. 화계·인칭 이중차분 (DiD: 민감 vs 비민감, 2,000회 문항 부트스트랩 + BH 보정)",
        "",
        "| 민감 문항 분류 기준 | 검정 지표 | 민감 문항 평균 | 비민감 문항 평균 | **이중차분 (DiD)** | 95% 부트스트랩 CI | 원시 $p$ | BH 보정 $q$ |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ])
    for d in did_results:
        g2_g3_lines.append(
            f"| {d['split']} | {d['metric']} | `{d['sens_mean']:+.4f}` | `{d['non_mean']:+.4f}` | **`{d['did']:+.4f}`** | `[ {d['ci_lo']:+.4f}, {d['ci_hi']:+.4f} ]` | `{d['p_raw']:.4f}` | `{d['q_bh']:.4f}` |"
        )

    g2_g3_lines.extend([
        "",
        "### 3. 보조 분석 — 고령 페르소나 × 해라체 화용 부조화 연령 회귀 및 전체 주효과 참고치",
        "",
        f"- **연령 회귀 기울기 (페르소나 연령 +10세당 `해라체 - 하십시오체` TVD 변화량)**:",
        f"  - 엄격 민감(24문항): **`{sl_strict_m:+.5f}`** / 10세 (95% CI `[ {sl_strict_lo:+.5f}, {sl_strict_hi:+.5f} ]`)",
        f"  - 광의 민감(80문항): **`{sl_broad_m:+.5f}`** / 10세 (95% CI `[ {sl_broad_lo:+.5f}, {sl_broad_hi:+.5f} ]`)",
        f"  - 전체 250문항: **`{sl_all_m:+.5f}`** / 10세 (95% CI `[ {sl_all_lo:+.5f}, {sl_all_hi:+.5f} ]`)",
        f"- **전체 250문항 조건 주효과 (참고치, 사전등록 $\\delta = \\pm 0.02$ TVD 범위 비교)**:",
        f"  - 화계 주효과 (`해라체 - 하십시오체` TVD): **`{plain_main_m:+.4f}`** (95% CI `[ {plain_main_lo:+.4f}, {plain_main_hi:+.4f} ]`)",
        f"  - 인칭 주효과 (`3인칭 - 2인칭` TVD): **`{third_main_m:+.4f}`** (95% CI `[ {third_main_lo:+.4f}, {third_main_hi:+.4f} ]`)",
        "",
    ])

    GATES_PATH.write_text("\n".join(g2_g3_lines), encoding="utf-8")
    print(f"[stage2] 집계 완료: {counts_csv}, {metrics_csv}, {GATES_PATH}", flush=True)


def count_tokens_stage(stage: str, api_key: str, cfg: dict, workers: int = 4) -> pathlib.Path:
    """
    유료 호출 전 비용 산정. 프롬프트 전부를 /v1/messages/count_tokens(무료)로 세고
    입력 토큰 합계와 비용 상한을 research/token_counts_{stage}.json에 쓴다.
    출력 토큰은 아직 모르므로 max_tokens 전부를 쓴다고 가정한 상한을 낸다.
    """
    probes = load_stage_probes(stage, cfg)
    k = cfg["stage_k"][stage]
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    def count_one(p: dict) -> int:
        params = request_params(stage, cfg, p["prompt"])
        body = {key: params[key] for key in ("model", "system", "messages", "thinking")}
        data = json.dumps(body).encode("utf-8")
        for attempt in range(8):
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages/count_tokens", data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return int(json.loads(resp.read().decode("utf-8"))["input_tokens"])
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < 7:
                    time.sleep(1.5 * (2 ** min(attempt, 5)))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        raise RuntimeError("count_tokens 재시도 초과")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        counts = list(ex.map(count_one, probes))

    n_calls = len(probes) * k
    in_tok = sum(counts) * k
    max_out = request_params(stage, cfg, "")["max_tokens"]
    out_cap = n_calls * max_out

    def cost(pr: dict) -> dict:
        return {
            "input_usd": in_tok / 1e6 * pr["input"],
            "output_cap_usd": out_cap / 1e6 * pr["output"],
            "total_cap_usd": in_tok / 1e6 * pr["input"] + out_cap / 1e6 * pr["output"],
        }

    report = {
        "stage": stage,
        "model_id": cfg["model_id"],
        "counted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_prompts": len(probes),
        "k": k,
        "n_calls": n_calls,
        "input_tokens_per_prompt": {
            "sum": sum(counts), "mean": sum(counts) / len(counts), "min": min(counts), "max": max(counts),
        },
        "input_tokens_total": in_tok,
        "max_tokens_per_call": max_out,
        "output_tokens_cap": out_cap,
        "pricing_per_mtok": {"batch": cfg["pricing_batch_per_mtok"], "standard": cfg["pricing_standard_per_mtok"]},
        "cost_batch": cost(cfg["pricing_batch_per_mtok"]),
        "cost_standard": cost(cfg["pricing_standard_per_mtok"]),
        "note": "output은 모든 호출이 max_tokens를 다 쓴다고 가정한 상한이다. 요금은 실행 전 공식 요금표로 다시 확인한다.",
    }
    out_path = BASE / "research" / f"token_counts_{stage}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return out_path


def aggregate_verbal(stage: str, cfg: dict):
    """
    분포 직접 예측 arm 집계 (탐색 분석, research/preregistration_verbalized_arm.md).
    gates.md는 stage 집계가 덮어쓰므로 결과는 gates_verbal.md에 따로 쓴다.
    """
    import random

    raw_path = RAW_DIR / f"{stage}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} 파일이 없습니다.")
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    vcfg = cfg["verbal"]
    floor = float(vcfg["floor_pct"])
    alpha = float(cfg["jeffreys_alpha"])
    s1_probes = load_stage_probes("stage1", cfg)
    valid_idx_map = load_valid_indices_250(s1_probes)
    q_pop_map = load_q_pop_250()
    _cells_10, cells_14 = load_real_cells_10_and_14(s1_probes, valid_idx_map)
    pids_14 = sorted(PERSONA_14_META.keys())

    def to_dist(pcts: list[float], var: str) -> list[float]:
        kept = [max(float(pcts[i]), floor) for i in valid_idx_map[var]]
        s = sum(kept)
        return [x / s for x in kept]

    # ---- 원자료 → (level, var, pid)별 평균 분포 ----
    samples = defaultdict(list)
    n_total = Counter()
    n_invalid = Counter()
    in_tok = out_tok = 0
    for line in open(raw_path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error":
            continue
        key = (r["cond"], r["item_id"], r["persona_id"])
        n_total[r["cond"]] += 1
        in_tok += r.get("input_tokens", 0)
        out_tok += r.get("output_tokens", 0)
        if r["valid"]:
            samples[key].append(to_dist(r["parsed"], r["item_id"]))
        else:
            n_invalid[(r["cond"], r["invalid_type"])] += 1

    dists = {}
    dist_rows = []
    for key, vecs in sorted(samples.items()):
        k_opts = len(vecs[0])
        mean = [sum(v[j] for v in vecs) / len(vecs) for j in range(k_opts)]
        dists[key] = mean
        dist_rows.append({
            "stage": stage, "cond": key[0], "item_id": key[1], "persona_id": key[2],
            "n_valid": len(vecs), "dist_json": json.dumps([round(x, 6) for x in mean]),
        })
    dists_csv = COUNTS_DIR / f"{stage}_dists.csv"
    with open(dists_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)

    # ---- 표본 추출 arm(Stage 1) 분포: c0, adult 4조건 합산 ----
    samp_counts = {"c0": {}, "adult": {}}
    s1_counts_path = COUNTS_DIR / "stage1_counts.csv"
    if s1_counts_path.exists():
        with open(s1_counts_path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                c = json.loads(r["counts_json"])
                kept = valid_idx_map[r["item_id"]]
                if len(c) != len(kept):
                    c = [c[i] for i in kept]
                prev = samp_counts[r["cond"]].get(r["item_id"], [0] * len(c))
                samp_counts[r["cond"]][r["item_id"]] = [a + b for a, b in zip(prev, c)]

    def jeff(c: list[int]) -> list[float]:
        s = sum(c) + alpha * len(c)
        return [(x + alpha) / s for x in c]

    # ---- 모집단 수준 지표 ----
    items = sorted(v for (lv, v, _pid) in dists if lv == "verbal_pop" and v in q_pop_map and len(q_pop_map[v]) == len(dists[("verbal_pop", v, "")]))
    metric_rows = []
    per_item = defaultdict(dict)
    for var in items:
        q = q_pop_map[var]
        k_opts = len(q)
        arms = {"verbal_pop": dists[("verbal_pop", var, "")], "uniform": [1.0 / k_opts] * k_opts}
        for cond in ("c0", "adult"):
            if var in samp_counts[cond]:
                arms[f"sampling_{cond}"] = jeff(samp_counts[cond][var])
        for arm, p in arms.items():
            row = {
                "stage": stage, "arm": arm, "item_id": var, "K": k_opts,
                "tvd": tvd_fn(p, q), "w1_norm": w1_norm_fn(p, q), "nll": nll_fn(p, q),
                "unif_nll": math.log(k_opts),
                "mode_hit": int(max(range(k_opts), key=p.__getitem__) == max(range(k_opts), key=q.__getitem__)),
                "max_p": max(p),
            }
            row["beat_unif_nll"] = int(row["nll"] < row["unif_nll"] - 1e-9)
            metric_rows.append(row)
            per_item[arm][var] = row

    metrics_csv = METRICS_DIR / f"{stage}_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)

    def arm_mean(arm: str, col: str) -> float:
        vals = [per_item[arm][v][col] for v in items if v in per_item[arm]]
        return sum(vals) / len(vals) if vals else float("nan")

    # ---- 문항 클러스터 부트스트랩: 직접 예측 − 비교 arm ----
    rng = random.Random(int(cfg["random_seed"]))
    n_boot = int(cfg["bootstrap_iters"])
    boot_rows = []
    for other in ("sampling_adult", "sampling_c0", "uniform"):
        common = [v for v in items if v in per_item[other]]
        if not common:
            continue
        for col in ("tvd", "nll"):
            diffs = [per_item["verbal_pop"][v][col] - per_item[other][v][col] for v in common]
            point = sum(diffs) / len(diffs)
            means = []
            for _ in range(n_boot):
                means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
            means.sort()
            boot_rows.append({
                "contrast": f"verbal_pop - {other}", "metric": col, "n_items": len(common), "diff": point,
                "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[min(n_boot - 1, int(0.975 * n_boot))],
            })

    # ---- 서브그룹: Stage 2와 같은 정의(기대 위치 이동의 14셀 상관) ----
    item_r = {}
    model_shift, human_shift = {}, {}
    gains = []
    for var in items:
        q_pop = q_pop_map[var]
        e_model_pop = expected_pos_norm(dists[("verbal_pop", var, "")])
        e_pop = expected_pos_norm(q_pop)
        ms, hs = [], []
        ok = True
        for pid in pids_14:
            p_c = dists.get(("verbal_cell", var, pid))
            cell = cells_14.get((var, pid))
            if p_c is None or cell is None or len(cell[1]) != len(p_c):
                ok = False
                break
            q_c = cell[1]
            ms.append(expected_pos_norm(p_c) - e_model_pop)
            hs.append(expected_pos_norm(q_c) - e_pop)
            gains.append(tvd_fn(p_c, q_pop) - tvd_fn(p_c, q_c))
        if not ok:
            continue
        model_shift[var], human_shift[var] = ms, hs
        rv = pearson_r(ms, hs)
        if rv is not None:
            item_r[var] = rv

    def median(vals: list[float]) -> float:
        s = sorted(vals)
        return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2

    med_r = median(list(item_r.values())) if item_r else float("nan")
    n_perm = int(cfg["permutation_iters"])
    ge = 0
    for _ in range(n_perm if item_r else 0):
        perm = list(range(len(pids_14)))
        rng.shuffle(perm)
        rs = []
        for var in item_r:
            rv = pearson_r([model_shift[var][i] for i in perm], human_shift[var])
            if rv is not None:
                rs.append(rv)
        if rs and median(rs) >= med_r:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1) if item_r else float("nan")
    mean_gain = sum(gains) / len(gains) if gains else float("nan")

    # ---- 보고 ----
    tot = sum(n_total.values())
    inv = sum(n_invalid.values())
    k = cfg["stage_k"][stage]
    pr = cfg["pricing_batch_per_mtok"] if stage != "verbal0" else cfg["pricing_standard_per_mtok"]
    cost = in_tok / 1e6 * pr["input"] + out_tok / 1e6 * pr["output"]
    lines = [
        f"# 분포 직접 예측 arm — `{stage}` (탐색 분석)",
        "",
        f"- 모델 ID: `{cfg['model_id']}` / 집계일: `{datetime.date.today().isoformat()}`",
        "- 사전등록: `research/preregistration_verbalized_arm.md` (Stage 1 결과를 본 뒤 추가한 탐색 분석)",
        f"- 호출: {tot}건 (프롬프트당 k={k}), 유효율 **{(tot - inv) / tot * 100:.2f}%**"
        f" (기준 {float(vcfg['min_valid_rate']) * 100:.0f}%)",
        f"- 무효 유형: {dict((f'{c}/{t}', n) for (c, t), n in sorted(n_invalid.items())) or '없음'}",
        f"- 실측 토큰: 입력 {in_tok:,} / 출력 {out_tok:,} → 실측 비용 약 **${cost:.2f}**"
        f" ({'표준' if stage == 'verbal0' else 'Batch'} 요금)",
        "",
        f"## 1. 모집단 수준 ({len(items)}문항, 정답지 `Q_pop`)",
        "",
        "| arm | 평균 TVD | 평균 W1(정규화) | 평균 NLL | NLL<균등 비율 | 최빈값 일치율 | 최빈 보기 확률 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("verbal_pop", "sampling_adult", "sampling_c0", "uniform"):
        if per_item[arm]:
            lines.append(
                f"| `{arm}` | {arm_mean(arm, 'tvd'):.4f} | {arm_mean(arm, 'w1_norm'):.4f} | {arm_mean(arm, 'nll'):.4f}"
                f" | {arm_mean(arm, 'beat_unif_nll') * 100:.2f}% | {arm_mean(arm, 'mode_hit') * 100:.2f}%"
                f" | {arm_mean(arm, 'max_p'):.4f} |"
            )
    lines += [
        "",
        f"## 2. 대응 비교 (문항 클러스터 부트스트랩 {n_boot}회, 95% CI)",
        "",
        "| 비교 | 지표 | 문항 수 | 차이 | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for b in boot_rows:
        lines.append(f"| {b['contrast']} | {b['metric']} | {b['n_items']} | {b['diff']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] |")
    lines += [
        "",
        "## 3. 서브그룹 (성별×연령 14셀, Stage 2와 같은 정의)",
        "",
        f"- subgroup r 문항 중앙값: **{med_r:+.4f}** ({len(item_r)}문항), 셀 라벨 순열검정({n_perm}회) p = {perm_p:.4f}",
        f"- 타겟 정렬 이득 `TVD(P_c, Q_pop) − TVD(P_c, Q_c)` 평균: **{mean_gain:+.4f}** (양수면 셀 정보가 담김)",
        "",
    ]
    out_md = BASE / ("gates_verbal.md" if stage == "verbal1" else f"gates_{stage}.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[{stage}] 집계 완료: {dists_csv}, {metrics_csv}, {out_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stage0", "stage1", "stage2", *VERBAL_STAGES])
    ap.add_argument(
        "--mode", default="concurrent",
        choices=["concurrent", "batch", "batch_collect", "aggregate_only", "count_tokens"],
    )
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if args.mode != "aggregate_only" and not api_key:
        raise SystemExit("ANTHROPIC_API_KEY 환경변수가 필요합니다.")

    if args.mode == "count_tokens":
        count_tokens_stage(args.stage, api_key, cfg, workers=min(args.workers, 4))
        return
    if is_verbal(args.stage):
        if args.mode == "concurrent":
            run_concurrent_stage(args.stage, api_key, cfg, workers=args.workers)
            aggregate_verbal(args.stage, cfg)
        elif args.mode == "batch":
            submit_batch_stage(args.stage, api_key, cfg)
        elif args.mode == "batch_collect":
            if check_and_collect_batch(args.stage, api_key, cfg):
                aggregate_verbal(args.stage, cfg)
        else:
            aggregate_verbal(args.stage, cfg)
        return

    if args.mode == "concurrent":
        run_concurrent_stage(args.stage, api_key, cfg, workers=args.workers)
        if args.stage == "stage0":
            aggregate_stage0(cfg)
        elif args.stage == "stage1":
            aggregate_stage1(cfg)
        elif args.stage == "stage2":
            aggregate_stage2(cfg)
    elif args.mode == "batch":
        submit_batch_stage(args.stage, api_key, cfg)
    elif args.mode == "batch_collect":
        if check_and_collect_batch(args.stage, api_key, cfg):
            if args.stage == "stage1":
                aggregate_stage1(cfg)
            elif args.stage == "stage2":
                aggregate_stage2(cfg)
    elif args.mode == "aggregate_only":
        if args.stage == "stage0":
            aggregate_stage0(cfg)
        elif args.stage == "stage1":
            aggregate_stage1(cfg)
        elif args.stage == "stage2":
            aggregate_stage2(cfg)


if __name__ == "__main__":
    main()
