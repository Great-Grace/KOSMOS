#!/usr/bin/env python3
"""
Haiku 5.5 파일럿 Stage 1 사후 진단 (탐색 분석, 새 API 호출 없음).

  A. 쏠림 진단: 모델 최빈 보기 확률 vs 인간, 한 보기에 전부 몰린 문항 비율,
     추정 분포를 사후 평탄화(p^(1/T))했을 때 NLL·TVD 변화. T는 표본 내에서 훑는다.
  B. 표본 수 충분성: 프롬프트당 50표본을 k개로 부분표집해 지표가 k에 따라 수렴하는지 본다.
  C. 표집 오차 한계: 모델 최빈 보기 확률의 Clopper-Pearson 95% 하한이
     그 보기의 인간 비율보다 높은 문항 비율. 높으면 표본을 늘려도 차이가 사라지지 않는다.

사전등록(research/preregistration_haiku_pilot.md)에 없는 분석이다. 결과는 "탐색"으로만 보고한다.

사용법:  python3 analysis_haiku_collapse.py
산출물:  metrics/stage1_collapse_diagnostic.csv, metrics/stage1_k_sufficiency.csv,
         metrics/stage1_sampling_bound.csv
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict

import kgss_haiku_pilot as pilot

T_GRID = [1, 2, 3, 4, 6, 8, 10, 15]
K_GRID = [5, 10, 20, 30, 50]
N_REPS = 200


def jeffreys(counts: list[int], alpha: float) -> list[float]:
    s = sum(counts) + alpha * len(counts)
    return [(c + alpha) / s for c in counts]


def flatten(p: list[float], t: float) -> list[float]:
    w = [x ** (1.0 / t) for x in p]
    s = sum(w)
    return [x / s for x in w]


def argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def binom_cdf(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))


def cp_lower(x: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson 양측 구간의 하한. x는 성공 수."""
    if n == 0 or x == 0:
        return 0.0
    a = (1 - conf) / 2
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # P(X >= x | p=mid) = 1 - cdf(x-1)
        if 1 - binom_cdf(x - 1, n, mid) < a:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_samples(cfg: dict):
    """(cond, item, person, speech) -> 유효 보기 인덱스 목록(무효는 None), 표본 순서 유지."""
    probes = pilot.load_stage_probes("stage1", cfg)
    valid_idx = pilot.load_valid_indices_250(probes)
    codes_by_key = {
        (p["level"], p["var"], p["addressee"], p["register"]): json.loads(p["codes"])
        for p in probes
    }
    samples = defaultdict(list)
    with open(pilot.RAW_DIR / "stage1.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("invalid_type") == "api_error":
                continue
            key = (r["cond"], r["item_id"], r["person"], r["speech"])
            codes = codes_by_key[key]
            kept = [codes[i] for i in valid_idx[r["item_id"]]]
            parsed, valid, _ = pilot.parse_response(r["raw_text"], codes)
            pos = kept.index(parsed) if (valid and parsed in kept) else None
            samples[key].append((r["sample_idx"], pos))
    out = {k: [pos for _, pos in sorted(v)] for k, v in samples.items()}
    n_opts = {var: len(idx) for var, idx in valid_idx.items()}
    return out, n_opts


def counts_from(positions: list[int | None], k_opts: int) -> list[int]:
    c = [0] * k_opts
    for pos in positions:
        if pos is not None:
            c[pos] += 1
    return c


def summarize(dists: dict[str, list[float]], q_pop: dict[str, list[float]]) -> dict:
    items = sorted(dists)
    n = len(items)
    nll = [pilot.nll_fn(dists[v], q_pop[v]) for v in items]
    unif = [math.log(len(q_pop[v])) for v in items]
    tvd = [pilot.tvd_fn(dists[v], q_pop[v]) for v in items]
    return {
        "n_items": n,
        "mean_tvd": sum(tvd) / n,
        "mean_nll": sum(nll) / n,
        "beat_unif_rate": sum(a < b - 1e-9 for a, b in zip(nll, unif)) / n,
        "mode_hit": sum(argmax(dists[v]) == argmax(q_pop[v]) for v in items) / n,
    }


def main():
    cfg = pilot.load_config()
    alpha = float(cfg["jeffreys_alpha"])
    rng = random.Random(int(cfg["random_seed"]))
    q_pop = pilot.load_q_pop_250()
    samples, n_opts = load_samples(cfg)

    # 평가 단위: c0(프롬프트 1개), adult(4조건 합산)
    pools = {"c0": defaultdict(list), "adult_pooled": defaultdict(list)}
    for (cond, item, _person, _speech), pos in samples.items():
        if item not in q_pop or len(q_pop[item]) != n_opts[item]:
            continue
        pools["c0" if cond == "c0" else "adult_pooled"][item].append(pos)

    pilot.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- A. 쏠림 진단 ----------
    rows_a = []
    for name, by_item in pools.items():
        counts = {v: counts_from([p for lst in ls for p in lst], n_opts[v]) for v, ls in by_item.items()}
        dists = {v: jeffreys(c, alpha) for v, c in counts.items()}
        items = sorted(dists)
        n = len(items)
        base = {
            "eval": name,
            "n_items": n,
            "mean_max_p_model": sum(max(dists[v]) for v in items) / n,
            "mean_max_p_human": sum(max(q_pop[v]) for v in items) / n,
            "one_option_item_rate": sum(sum(counts[v]) > 0 and max(counts[v]) == sum(counts[v]) for v in items) / n,
            "uniform_nll": sum(math.log(len(q_pop[v])) for v in items) / n,
            "uniform_tvd": sum(pilot.tvd_fn([1 / len(q_pop[v])] * len(q_pop[v]), q_pop[v]) for v in items) / n,
            "qpop_nll": sum(pilot.nll_fn(q_pop[v], q_pop[v]) for v in items) / n,
        }
        for t in T_GRID:
            s = summarize({v: flatten(dists[v], t) for v in items}, q_pop)
            rows_a.append({**base, "flatten_T": t, **{k: s[k] for k in ("mean_tvd", "mean_nll", "beat_unif_rate", "mode_hit")}})
    write_csv(pilot.METRICS_DIR / "stage1_collapse_diagnostic.csv", rows_a)

    # ---------- B. 표본 수 충분성 ----------
    rows_b = []
    for name, by_item in pools.items():
        for k in K_GRID:
            reps = defaultdict(list)
            for _ in range(N_REPS):
                dists = {}
                for v, lists in by_item.items():
                    sub = []
                    for lst in lists:  # 조건별로 k개씩 비복원 추출
                        sub.extend(rng.sample(lst, min(k, len(lst))))
                    dists[v] = jeffreys(counts_from(sub, n_opts[v]), alpha)
                s = summarize(dists, q_pop)
                for key in ("mean_tvd", "mean_nll", "beat_unif_rate", "mode_hit"):
                    reps[key].append(s[key])
            row = {"eval": name, "k_per_prompt": k, "n_prompts_pooled": 1 if name == "c0" else 4, "n_reps": N_REPS}
            for key, vals in reps.items():
                m = sum(vals) / len(vals)
                row[key] = m
                row[key + "_sd"] = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
            rows_b.append(row)
    write_csv(pilot.METRICS_DIR / "stage1_k_sufficiency.csv", rows_b)

    # ---------- C. 표집 오차 한계 ----------
    rows_c = []
    for name, by_item in pools.items():
        n_items = n_excess = 0
        gaps = []
        for v, lists in by_item.items():
            c = counts_from([p for lst in lists for p in lst], n_opts[v])
            n_valid = sum(c)
            if n_valid == 0:
                continue
            m = argmax(c)
            lower = cp_lower(c[m], n_valid)
            human = q_pop[v][m]
            n_items += 1
            n_excess += lower > human
            gaps.append(lower - human)
        gaps.sort()
        rows_c.append({
            "eval": name,
            "n_items": n_items,
            "items_mode_prob_lower_bound_above_human": n_excess,
            "rate": n_excess / n_items,
            "median_gap_lower_bound_minus_human": gaps[len(gaps) // 2],
        })
    write_csv(pilot.METRICS_DIR / "stage1_sampling_bound.csv", rows_c)

    for title, rows in (("A. 쏠림 진단", rows_a), ("B. 표본 수 충분성", rows_b), ("C. 표집 오차 한계", rows_c)):
        print(f"\n== {title} ==")
        for r in rows:
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})


def write_csv(path, rows: list[dict]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
