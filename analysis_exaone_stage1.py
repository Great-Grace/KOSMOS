#!/usr/bin/env python3
"""
K-EXAONE 2.5 Stage 1 집계 (탐색 분석, 새 API 호출 없음).

raw/exaone_stage1.jsonl을 읽어 Haiku 5.5 Stage 1(raw/stage1.jsonl)과 같은 지표로 비교한다.
- 표본 수를 맞춘다: 모든 프롬프트가 채운 표본 라운드 수 k만 쓰고, Haiku도 앞에서 k개만 쓴다.
- 실행 중에도 돌릴 수 있다. 덜 찬 라운드는 버린다.
- gates.md는 건드리지 않는다. 결과는 gates_exaone.md에 쓴다.

사용:  python3 analysis_exaone_stage1.py [--k N]
산출물: counts/exaone_stage1_counts.csv, metrics/exaone_stage1_metrics.csv, gates_exaone.md
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
from collections import Counter, defaultdict

import kgss_haiku_pilot as pilot
from analysis_haiku_collapse import argmax, counts_from, cp_lower, jeffreys

CONDS = [("당신", "하십시오체"), ("당신", "해라체"), ("이사람", "하십시오체"), ("이사람", "해라체")]


def load(path, cfg, probes, valid_idx):
    """(cond, item, person, speech) -> [(sample_idx, pos|None, invalid_type, rec)], sample_idx 순."""
    codes_by_key = {(p["level"], p["var"], p["addressee"], p["register"]): json.loads(p["codes"]) for p in probes}
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error":
            continue
        key = (r["cond"], r["item_id"], r["person"], r["speech"])
        codes = codes_by_key[key]
        kept = [codes[i] for i in valid_idx[r["item_id"]]]
        if r.get("invalid_type") == "length":
            pos, inv = None, "length"
        else:
            parsed, valid, inv = pilot.parse_response(r["raw_text"], codes)
            pos = kept.index(parsed) if (valid and parsed in kept) else None
            if valid and pos is None:
                inv = "dk_option"
        out[key].append((r["sample_idx"], pos, inv, r))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in out.items()}


def unit_metrics(counts, q_pop, alpha):
    """문항별 지표 dict."""
    rows = {}
    for var, c in counts.items():
        q = q_pop[var]
        p = jeffreys(c, alpha)
        nll = pilot.nll_fn(p, q)
        rows[var] = {
            "tvd": pilot.tvd_fn(p, q), "w1_norm": pilot.w1_norm_fn(p, q), "nll": nll,
            "beat_unif": int(nll < math.log(len(q)) - 1e-9), "mode_hit": int(argmax(p) == argmax(q)),
            "max_p": max(p), "one_option": int(sum(c) > 0 and max(c) == sum(c)), "n_valid": sum(c),
        }
    return rows


def mean(rows, col):
    vals = [r[col] for r in rows.values()]
    return sum(vals) / len(vals) if vals else float("nan")


def boot_ci(diffs, rng, n_boot):
    point = sum(diffs) / len(diffs)
    means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n_boot))
    return point, means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="쓸 표본 라운드 수 (기본: 모든 프롬프트가 채운 최대값)")
    args = ap.parse_args()

    cfg = pilot.load_config()
    alpha = float(cfg["jeffreys_alpha"])
    n_boot = int(cfg["bootstrap_iters"])
    rng = random.Random(int(cfg["random_seed"]))
    probes = pilot.load_stage_probes("stage1", cfg)
    valid_idx = pilot.load_valid_indices_250(probes)
    q_pop = pilot.load_q_pop_250()
    n_opts = {v: len(i) for v, i in valid_idx.items()}
    items = sorted(v for v in n_opts if v in q_pop and len(q_pop[v]) == n_opts[v])

    ex_all = load(pilot.RAW_DIR / "exaone_stage1.jsonl", cfg, probes, valid_idx)
    hk_all = load(pilot.RAW_DIR / "stage1.jsonl", cfg, probes, valid_idx)
    n_prompts = len(probes)
    per_prompt = [len(ex_all.get((p["level"], p["var"], p["addressee"], p["register"]), [])) for p in probes]
    k_full = min(per_prompt)
    k = args.k or k_full
    if k < 1:
        raise SystemExit(f"아직 모든 프롬프트가 1표본을 채우지 못했습니다 (최소 {k_full}, 받은 호출 {sum(per_prompt)}건).")
    total_rows = sum(per_prompt)

    def take(data, k):
        return {key: [s for s in v if s[0] < k] for key, v in data.items()}

    ex, hk = take(ex_all, k), take(hk_all, k)

    def build(data):
        """평가 단위별 문항 카운트와 유효율."""
        units = {"c0": {}, "adult_pooled": {}}
        valid = defaultdict(lambda: [0, 0])
        inv = Counter()
        for pr, sp in CONDS:
            units[f"adult_{pr}_{sp}"] = {}
        for (cond, var, pr, sp), samples in data.items():
            if var not in items:
                continue
            c = counts_from([s[1] for s in samples], n_opts[var])
            name = "c0" if cond == "c0" else f"adult_{pr}_{sp}"
            units[name][var] = c
            if cond != "c0":
                prev = units["adult_pooled"].get(var, [0] * n_opts[var])
                units["adult_pooled"][var] = [a + b for a, b in zip(prev, c)]
            for _, pos, iv, _r in samples:
                for nm in ([name] if cond == "c0" else [name, "adult_pooled"]):
                    valid[nm][0] += pos is not None
                    valid[nm][1] += 1
                if pos is None:
                    inv[(name, iv)] += 1
        return units, valid, inv

    ex_units, ex_valid, ex_inv = build(ex)
    hk_units, hk_valid, _ = build(hk)
    hk50_units, hk50_valid, _ = build(hk_all)

    ex_m = {u: unit_metrics(c, q_pop, alpha) for u, c in ex_units.items()}
    hk_m = {u: unit_metrics(c, q_pop, alpha) for u, c in hk_units.items()}
    hk50_m = {u: unit_metrics(c, q_pop, alpha) for u, c in hk50_units.items()}

    # ---- 파일 저장 ----
    pilot.COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    pilot.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pilot.COUNTS_DIR / "exaone_stage1_counts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "item_id", "k_per_prompt", "n_valid", "counts_json"])
        for u, by in ex_units.items():
            for var, c in sorted(by.items()):
                w.writerow([u, var, k, sum(c), json.dumps(c)])
    with open(pilot.METRICS_DIR / "exaone_stage1_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["tvd", "w1_norm", "nll", "beat_unif", "mode_hit", "max_p", "one_option", "n_valid"]
        w = csv.writer(f)
        w.writerow(["model", "k_per_prompt", "unit", "item_id"] + cols)
        for model, kk, m in (("k-exaone_v2.5", k, ex_m), ("claude-haiku-5-5", k, hk_m), ("claude-haiku-5-5", 50, hk50_m)):
            for u, by in m.items():
                for var, r in sorted(by.items()):
                    w.writerow([model, kk, u, var] + [r[c] for c in cols])

    # ---- 보고서 ----
    def vr(valid, u):
        a, b = valid[u]
        return a / b if b else float("nan")

    unif_tvd = sum(pilot.tvd_fn([1 / n_opts[v]] * n_opts[v], q_pop[v]) for v in items) / len(items)
    unif_nll = sum(math.log(n_opts[v]) for v in items) / len(items)
    qpop_nll = sum(pilot.nll_fn(q_pop[v], q_pop[v]) for v in items) / len(items)
    max_q = sum(max(q_pop[v]) for v in items) / len(items)

    lat = [s[3].get("latency_s", 0) for v in ex.values() for s in v]
    rtok = [s[3].get("reasoning_tokens", 0) or 0 for v in ex.values() for s in v]
    top_p = []
    for v in ex.values():
        for s in v:
            tl = s[3].get("answer_top_logprobs")
            if tl:
                top_p.append(math.exp(max(x[1] for x in tl)))

    L = [
        "# K-EXAONE 2.5 — Stage 1 설계 재현 (탐색 분석)",
        "",
        f"- 집계 시각: `{datetime.datetime.now().isoformat(timespec='minutes')}`",
        f"- 모델: `k-exaone_v2.5` (무료 API, `reasoning_effort=low`, temperature 1.0). 추론을 끌 수 없는 모델이다.",
        f"- 설계: Haiku Stage 1과 같은 {n_prompts}개 프롬프트(c0 250 + adult 1,000), 같은 system 문구.",
        f"- 사용 표본: 프롬프트당 **k={k}** (모든 프롬프트가 채운 라운드만 사용. 받은 호출 {total_rows:,}건, 프롬프트별 {min(per_prompt)}~{max(per_prompt)}건)",
        f"- 비교 대상 Haiku 5.5는 같은 k로 맞추려고 앞에서 {k}표본만 썼다. k=50 전체 수치는 참고로 함께 적는다.",
        "- 지위: 사전등록에 없는 탐색 분석이다. 두 모델은 추론 유무가 다르다(Haiku는 추론 끔).",
        f"- 호출당 평균: 지연 {sum(lat) / len(lat):.1f}초, 추론 토큰 {sum(rtok) / len(rtok):.0f}개",
        "",
        f"## 1. 분포 재현 ({len(items)}문항, 정답지 `Q_pop`)",
        "",
        "| 모델 | 평가 단위 | k | 유효율 | 평균 TVD | 평균 W1(정규화) | 평균 NLL | NLL<균등 | 최빈값 일치 | 최빈 보기 확률 | 한 보기 몰림 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(model, u, kk, m, valid):
        r = m[u]
        return (f"| {model} | `{u}` | {kk} | {vr(valid, u) * 100:.1f}% | {mean(r, 'tvd'):.4f} | {mean(r, 'w1_norm'):.4f}"
                f" | {mean(r, 'nll'):.4f} | {mean(r, 'beat_unif') * 100:.1f}% | {mean(r, 'mode_hit') * 100:.1f}%"
                f" | {mean(r, 'max_p'):.3f} | {mean(r, 'one_option') * 100:.1f}% |")

    for u in ("c0", "adult_pooled"):
        L.append(row("**K-EXAONE 2.5**", u, k, ex_m, ex_valid))
        L.append(row("Haiku 5.5", u, k, hk_m, hk_valid))
        L.append(row("Haiku 5.5 (참고)", u, 50, hk50_m, hk50_valid))
    L += [
        f"| 기준: 균등분포 | - | - | - | {unif_tvd:.4f} | - | {unif_nll:.4f} | - | - | - | - |",
        f"| 기준: `Q_pop` 자체 | - | - | - | 0.0000 | - | {qpop_nll:.4f} | - | 100% | {max_q:.3f} | - |",
        "",
        "- `adult_pooled`는 인칭×화계 4조건의 표본을 합친 것이다(문항당 k×4개).",
        "- 표본이 적을수록 Jeffreys 스무딩이 쏠림을 가려 TVD·NLL이 좋게 나온다. 그래서 같은 k끼리만 비교한다.",
        "",
        f"## 2. K-EXAONE 2.5 − Haiku 5.5 (같은 k={k}, 문항 대응, 부트스트랩 {n_boot}회 95% CI)",
        "",
        "| 평가 단위 | 지표 | 차이 | 95% CI |",
        "|---|---|---:|---|",
    ]
    for u in ("c0", "adult_pooled"):
        common = [v for v in items if v in ex_m[u] and v in hk_m[u]]
        for col in ("tvd", "nll", "mode_hit"):
            d, lo, hi = boot_ci([ex_m[u][v][col] - hk_m[u][v][col] for v in common], rng, n_boot)
            L.append(f"| `{u}` | {col} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    L += ["", "- TVD·NLL은 음수면 K-EXAONE 2.5가 인간 분포에 더 가깝다. mode_hit은 양수면 더 잘 맞춘다.", ""]

    L += [
        "## 3. 쏠림이 표집 오차로 설명되는가",
        "",
        "| 모델 | 평가 단위 | 유효 문항 | 최빈 보기 확률의 95% 하한 > 인간 비율 | 비율 |",
        "|---|---|---:|---:|---:|",
    ]
    for model, units in (("K-EXAONE 2.5", ex_units), ("Haiku 5.5", hk_units)):
        for u in ("c0", "adult_pooled"):
            n_it = n_ex = 0
            for var, c in units[u].items():
                if sum(c) == 0:
                    continue
                m_i = argmax(c)
                n_it += 1
                n_ex += cp_lower(c[m_i], sum(c)) > q_pop[var][m_i]
            L.append(f"| {model} | `{u}` | {n_it} | {n_ex} | {n_ex / n_it * 100:.1f}% |")
    if top_p:
        L += ["", f"- 참고: K-EXAONE 2.5가 추론을 마친 뒤 답 토큰에 준 확률은 평균 {sum(top_p) / len(top_p):.3f}이다"
                  f"({len(top_p):,}건). 한 번의 호출 안에서는 거의 결정돼 있고, 변동은 추론 과정에서 생긴다."]

    L += [
        "",
        "## 4. 인칭·화계 조건별 (K-EXAONE 2.5, adult)",
        "",
        "| 조건 | 유효율 | 평균 TVD | 평균 NLL | 최빈값 일치 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pr, sp in CONDS:
        u = f"adult_{pr}_{sp}"
        L.append(f"| `{pr}_{sp}` | {vr(ex_valid, u) * 100:.1f}% | {mean(ex_m[u], 'tvd'):.4f} | {mean(ex_m[u], 'nll'):.4f}"
                 f" | {mean(ex_m[u], 'mode_hit') * 100:.1f}% |")
    L += ["", "| 주효과 (TVD 차이, 문항 대응) | 차이 | 95% CI |", "|---|---:|---|"]
    for label, a, b in (
        ("화계: 해라체 − 하십시오체", [("당신", "해라체"), ("이사람", "해라체")], [("당신", "하십시오체"), ("이사람", "하십시오체")]),
        ("인칭: 이사람 − 당신", [("이사람", "하십시오체"), ("이사람", "해라체")], [("당신", "하십시오체"), ("당신", "해라체")]),
    ):
        diffs = []
        for v in items:
            try:
                ta = sum(ex_m[f"adult_{p}_{s}"][v]["tvd"] for p, s in a) / 2
                tb = sum(ex_m[f"adult_{p}_{s}"][v]["tvd"] for p, s in b) / 2
            except KeyError:
                continue
            diffs.append(ta - tb)
        d, lo, hi = boot_ci(diffs, rng, n_boot)
        L.append(f"| {label} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    L += [
        "",
        f"- 조건당 문항별 표본이 {k}개뿐이라 조건 간 TVD 차이는 노이즈가 크다. 탐색 참고치다.",
        f"- 무효 응답 유형: {dict((f'{u}/{t}', n) for (u, t), n in sorted(ex_inv.items())) or '없음'}",
        "",
    ]
    out = pilot.BASE / "gates_exaone.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
