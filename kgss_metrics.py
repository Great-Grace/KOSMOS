"""
KOSMOS 평가 파이프라인 검증 및 수정 지표 모듈 (kgss_metrics.py)

기존 kgss_c0_validity.py의 평가 산식을 보존하면서, 연구 질문(Estimand)에 부합하는
모집단 수준 집계 지표(Population Fidelity)와 조건부 지표(Conditional Fidelity)를
명확히 분리하여 계산한다.

핵심 구분:
1. Population Wasserstein (모집단 분포 재현성, RQ1/RQ2):
   A = W_1( sum_i w_i P_LLM(Y | X_i), P_KGSS(Y) )
2. Existing Wasserstein (개별 조건부 분포 대 전체 모집단 분포 거리의 평균):
   B = (1/N) * sum_i W_1( P_LLM(Y | X_i), P_KGSS(Y) )
3. Conditional Cell Wasserstein (셀별 조건부 분포 재현성, Demo/Full):
   C = sum_c w_c W_1( P_LLM(Y | c), P_KGSS(Y | c) )
4. Entropy 분해 (모드 붕괴 vs 조건부 확신도):
   H(sum_i w_i P_i) [모집단 엔트로피]
   = sum_i w_i H(P_i) [조건부 엔트로피] + I(Y; X) [페르소나 상호정보량]
5. Subgroup r (하위집단 편차 상관):
   - 기존: corr_v( mu_human(v,c) - mu_human(v), mu_model(v,c) ) [모델 측 미중심화]
   - 수정(Centered): corr_v( mu_human(v,c) - mu_human(v), mu_model(v,c) - mu_model(v) )
   - 문항 내(Within-item): corr_c( mu_human(v,c), mu_model(v,c) )
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wasserstein_distance

AGE_BINS_5 = [18, 30, 40, 50, 60, 200]
AGE_LABELS_5 = ["18-29", "30-39", "40-49", "50-59", "60+"]


def validate_prob_dist(p: Sequence[float], tol: float = 1e-5) -> np.ndarray:
    """확률분포 유효성(비음수, 합=1)을 검증하고 정규화된 float 배열을 반환한다."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("확률분포는 비어 있지 않은 1차원 벡터여야 합니다.")
    if np.any(~np.isfinite(arr)) or np.any(arr < -tol):
        raise ValueError(f"유효하지 않은 확률값이 포함되어 있습니다: {arr}")
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("확률분포의 합이 0 이하입니다.")
    return arr / total


def entropy_bits(p: Sequence[float]) -> float:
    """Shannon 엔트로피 (단위: bit)."""
    arr = validate_prob_dist(p, tol=1e-3)
    pos = arr[arr > 0]
    return float(-np.sum(pos * np.log2(pos)))


def wasserstein_1d(
    u_probs: Sequence[float],
    v_probs: Sequence[float],
    codes: Sequence[float] | None = None,
    use_code_positions: bool = False,
) -> float:
    """1차원 이산 확률분포 간 Wasserstein-1 거리.

    기본값(use_code_positions=False)은 기존 코드와 동일하게 정렬된 범주 순번
    0, 1, ..., K-1을 좌표로 사용한다. 비연속 선택지 코드의 실제 간격을 반영하려면
    use_code_positions=True와 codes를 함께 전달한다.
    """
    u = validate_prob_dist(u_probs, tol=1e-3)
    v = validate_prob_dist(v_probs, tol=1e-3)
    if len(u) != len(v):
        raise ValueError(f"분포 길이 불일치: {len(u)} != {len(v)}")
    n = len(u)
    if use_code_positions and codes is not None:
        support = np.asarray(codes, dtype=float)
        if len(support) != n:
            raise ValueError(f"선택지 코드 개수({len(support)})와 분포 길이({n}) 불일치")
    else:
        support = np.arange(n, dtype=float)
    return float(wasserstein_distance(support, support, u_weights=u, v_weights=v))


def tvd_distance(u_probs: Sequence[float], v_probs: Sequence[float]) -> float:
    """총변동거리(Total Variation Distance): 0.5 * sum_k |u_k - v_k|."""
    u = validate_prob_dist(u_probs, tol=1e-3)
    v = validate_prob_dist(v_probs, tol=1e-3)
    if len(u) != len(v):
        raise ValueError(f"분포 길이 불일치: {len(u)} != {len(v)}")
    return float(0.5 * np.sum(np.abs(u - v)))


def human_weighted_distribution(
    df: pd.DataFrame,
    var: str,
    codes: Sequence[float | int],
    weight_col: str = "FINALWT",
    min_n: int = 100,
    round_digits: int | None = None,
) -> list[float] | None:
    """문항별 실제 KGSS 응답자의 FINALWT 가중 응답분포 P_KGSS(Y=k)를 계산한다.

    결측(NaN), 비해당/무응답, codes에 포함되지 않은 값은 제외한다.
    유효 응답자 수가 min_n 미만이면 None을 반환한다.
    """
    if var not in df.columns or weight_col not in df.columns:
        return None
    float_codes = [float(c) for c in codes]
    s = df[[var, weight_col]].dropna()
    s = s[s[var].isin(float_codes) & (s[weight_col] > 0)]
    if len(s) < min_n:
        return None
    tot = float(s[weight_col].sum())
    if tot <= 0:
        return None
    dist = [float(s.loc[s[var] == c, weight_col].sum() / tot) for c in float_codes]
    if round_digits is not None:
        return [round(x, round_digits) for x in dist]
    return dist


def add_age_group(df: pd.DataFrame, age_col: str = "AGE", out_col: str = "AGE_GRP") -> pd.DataFrame:
    """KGSS 표준 5개 연령대(18-29, 30-39, 40-49, 50-59, 60+) 컬럼을 추가한다."""
    out = df.copy()
    out[out_col] = pd.cut(out[age_col], AGE_BINS_5, labels=AGE_LABELS_5, right=False)
    return out


def human_cell_distributions(
    df: pd.DataFrame,
    var: str,
    codes: Sequence[float | int],
    cell_cols: Sequence[str] = ("SEX", "AGE_GRP"),
    weight_col: str = "FINALWT",
    min_cell_n: int = 5,
) -> pd.DataFrame:
    """성별×연령 등 교차 셀별 실제 KGSS 가중 응답분포 P_KGSS(Y|c)와 셀 비중 P_KGSS(c)를 계산한다."""
    float_codes = [float(c) for c in codes]
    req_cols = [var, weight_col] + list(cell_cols)
    if any(c not in df.columns for c in req_cols):
        return pd.DataFrame()

    # 전체 조사 표본 기준 셀 가중치 (문항 무관 모집단 비중)
    base = df[list(cell_cols) + [weight_col]].dropna()
    base = base[base[weight_col] > 0]
    total_pop_wt = float(base[weight_col].sum()) if len(base) else 0.0
    pop_cell_wt = (
        base.groupby(list(cell_cols), observed=True)[weight_col].sum() / total_pop_wt
        if total_pop_wt > 0
        else pd.Series(dtype=float)
    )

    # 해당 문항 유효 응답자 기준
    s = df[req_cols].dropna()
    s = s[s[var].isin(float_codes) & (s[weight_col] > 0)]
    if s.empty:
        return pd.DataFrame()
    total_item_wt = float(s[weight_col].sum())
    if total_item_wt <= 0:
        return pd.DataFrame()

    grand_dist = np.array([s.loc[s[var] == c, weight_col].sum() / total_item_wt for c in float_codes])
    grand_mean = float(np.sum(np.array(float_codes) * grand_dist))

    rows = []
    for key, g in s.groupby(list(cell_cols), observed=True):
        wt_sum = float(g[weight_col].sum())
        if len(g) < min_cell_n or wt_sum <= 0:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        c_dist = [float(g.loc[g[var] == c, weight_col].sum() / wt_sum) for c in float_codes]
        c_mean = float(np.sum(np.array(float_codes) * np.array(c_dist)))
        row = {col: val for col, val in zip(cell_cols, key_tuple)}
        row.update({
            "var": var,
            "n_cell": int(len(g)),
            "cell_weight_item": float(wt_sum / total_item_wt),
            "cell_weight_pop": float(pop_cell_wt.get(key, wt_sum / total_item_wt)),
            "real_dist": json.dumps(c_dist),
            "real_mean": c_mean,
            "real_grand": grand_mean,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # 유효 셀들의 가중치 합이 1이 되도록 정규화한 열 제공
        s_item = out["cell_weight_item"].sum()
        out["cell_weight_item_norm"] = out["cell_weight_item"] / s_item if s_item > 0 else 1.0 / len(out)
        s_pop = out["cell_weight_pop"].sum()
        out["cell_weight_pop_norm"] = out["cell_weight_pop"] / s_pop if s_pop > 0 else 1.0 / len(out)
    return out


def aggregate_distributions(
    dists: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """조건부 예측분포 P_i들을 가중치 w_i로 집계하여 모집단 예측분포 sum_i w_i P_i를 반환한다."""
    if len(dists) == 0:
        raise ValueError("집계할 분포가 없습니다.")
    mat = np.vstack([validate_prob_dist(d, tol=1e-3) for d in dists])
    if weights is None:
        w = np.ones(len(mat), dtype=float) / len(mat)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(mat):
            raise ValueError(f"가중치 개수({len(w)})와 분포 개수({len(mat)}) 불일치")
        if np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("가중치는 비음수이고 합이 양수여야 합니다.")
        w = w / float(w.sum())
    agg = np.sum(w[:, None] * mat, axis=0)
    return validate_prob_dist(agg, tol=1e-5)


def aggregate_demo_cells(
    cell_df: pd.DataFrame,
    cell_weights_df: pd.DataFrame,
    dist_col: str = "model_dist",
    cell_cols: Sequence[str] = ("sex", "age_grp"),
    weight_col: str = "cell_weight_item_norm",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Demo(또는 셀 매핑된 Full) 조건의 셀별 예측분포를 KGSS 셀 가중치로 집계한다.

    한 셀에 여러 프롬프트(n_c개)가 있는 경우 셀 내 단순평균으로 P_LLM(Y|c)를 먼저 구한 뒤,
    KGSS 모집단 셀 비중 P_KGSS(c)를 곱해 P_LLM(Y) = sum_c P_KGSS(c) P_LLM(Y|c)를 산출한다.
    이는 각 프롬프트 행 i in c에 w_i = w_c / n_c 가중치를 부여해 행 단위로 집계한 것과
    수학적으로 완전히 동치이다(sum_{i in c} w_i = w_c).
    일부 셀이 결측인 경우 관측된 셀들의 가중치 합으로 재정규화한다.
    """
    records = []
    for key, g in cell_df.groupby(list(cell_cols), observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        dists = [json.loads(x) if isinstance(x, str) else x for x in g[dist_col]]
        cell_mean_dist = aggregate_distributions(dists)
        rec = {col: val for col, val in zip(cell_cols, key_tuple)}
        rec["cell_model_dist"] = cell_mean_dist
        rec["n_prompts"] = int(len(dists))
        records.append(rec)

    c_pred = pd.DataFrame(records)
    merged = c_pred.merge(cell_weights_df, on=list(cell_cols), how="inner")
    if merged.empty:
        # 매칭되는 셀 가중치가 없을 경우 관측된 셀의 동일 가중 평균으로 폴백
        c_pred["cell_weight_matched_norm"] = 1.0 / len(c_pred) if len(c_pred) > 0 else np.nan
        c_pred["row_weight_per_prompt"] = c_pred["cell_weight_matched_norm"] / c_pred["n_prompts"]
        agg = aggregate_distributions(c_pred["cell_model_dist"].tolist())
        return agg, c_pred

    w = merged[weight_col].to_numpy(dtype=float)
    w_norm = w / float(w.sum())
    merged = merged.copy()
    merged["cell_weight_matched_norm"] = w_norm
    merged["row_weight_per_prompt"] = merged["cell_weight_matched_norm"] / merged["n_prompts"].to_numpy(dtype=float)
    agg = aggregate_distributions(merged["cell_model_dist"].tolist(), weights=w_norm)
    return agg, merged


def build_cell_poststratified_row_weights(
    row_df: pd.DataFrame,
    merged_cells: pd.DataFrame,
    cell_cols: Sequence[str] = ("sex", "age_grp"),
    dist_col: str | None = "model_dist",
    expected_cell_agg: np.ndarray | None = None,
    tol: float = 1e-10,
) -> list[float]:
    """각 프롬프트 행 i에 셀 가중치 나누기 셀 내 인원수(w_i = w_c / n_c)를 할당하고 불변식을 검증한다.

    불변식(Invariant):
    1. 관측된 모든 셀 c에 대해 sum_{i in c} w_i == w_c_norm
    2. 행 가중 집계 sum_i w_i P_i == 셀 평균 선집계 sum_c w_c_norm P_bar_c
    """
    if merged_cells.empty or "row_weight_per_prompt" not in merged_cells.columns:
        return [1.0 / len(row_df)] * len(row_df)

    key_cols = list(cell_cols)
    w_per_row_map = {
        tuple(row[c] for c in key_cols): float(row["row_weight_per_prompt"])
        for _, row in merged_cells.iterrows()
    }
    cell_target_map = {
        tuple(row[c] for c in key_cols): float(row["cell_weight_matched_norm"])
        for _, row in merged_cells.iterrows()
    }

    raw_weights = []
    row_keys = []
    for _, r in row_df.iterrows():
        k = tuple(r[c] for c in key_cols)
        row_keys.append(k)
        if k not in w_per_row_map:
            raise ValueError(f"관측 행의 셀 키 {k}가 merged_cells에 존재하지 않습니다.")
        raw_weights.append(w_per_row_map[k])

    w_arr = np.asarray(raw_weights, dtype=float)
    w_arr = w_arr / float(w_arr.sum())

    # 불변식 1: 셀별 행 가중치 합 == 정규화된 셀 가중치 w_c_norm
    for k, target_wc in cell_target_map.items():
        cell_sum = float(np.sum([w for rk, w in zip(row_keys, w_arr) if rk == k]))
        if abs(cell_sum - target_wc) > tol:
            raise AssertionError(
                f"셀 가중치 불변식 위반 (cell={k}): sum_{{i in c}} w_i = {cell_sum:.12f} != w_c = {target_wc:.12f}"
            )

    # 불변식 2: 행 단위 가중합 분포 == 셀 평균 선집계 분포
    if dist_col is not None and dist_col in row_df.columns and expected_cell_agg is not None:
        row_dists = [json.loads(x) if isinstance(x, str) else x for x in row_df[dist_col]]
        row_agg = aggregate_distributions(row_dists, weights=w_arr)
        max_diff = float(np.max(np.abs(row_agg - expected_cell_agg)))
        if max_diff > tol:
            raise AssertionError(
                f"행 가중 집계와 셀 선집계 분포 불일치: max|diff| = {max_diff:.12e} > {tol}"
            )

    return w_arr.tolist()


def aggregate_full_personas(
    persona_dists: Sequence[Sequence[float]],
    already_pps_sampled: bool = True,
    raw_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Full 조건의 개인별 예측분포 P_LLM(Y|X_i)를 모집단 분포로 집계한다.

    주의 (이중 가중 방지):
    kgss_persona_bootstrap.py에서 이미 FINALWT에 비례하여 복원추출(PPS with replacement)한
    표본인 경우(already_pps_sampled=True), Hansen-Hurwitz 불편추정량은 단순 표본평균
    (1/N) * sum_i P_LLM(Y|X_i)이다. 여기에 FINALWT를 다시 곱하면 FINALWT^2에 비례하는
    이중 가중(double-weighting) 편향이 발생하므로 raw_weights를 적용하지 않는다.
    비가중 표본에서 추출된 경우(already_pps_sampled=False)에만 raw_weights를 적용한다.
    """
    if already_pps_sampled or raw_weights is None:
        return aggregate_distributions(persona_dists, weights=None)
    return aggregate_distributions(persona_dists, weights=raw_weights)


def evaluate_item_distributions(
    pred_dists: Sequence[Sequence[float]],
    gold_dist: Sequence[float],
    codes: Sequence[float | int] | None = None,
    weights: Sequence[float] | None = None,
    cell_model_dists: Sequence[Sequence[float]] | None = None,
    cell_human_dists: Sequence[Sequence[float]] | None = None,
    cell_weights: Sequence[float] | None = None,
    use_code_positions: bool = False,
) -> dict[str, float]:
    """단일 문항에 대해 기존 지표와 수정된 지표(가중/비가중 포함)를 모두 계산하여 반환한다."""
    gold = validate_prob_dist(gold_dist, tol=1e-3)
    preds = [validate_prob_dist(p, tol=1e-3) for p in pred_dists]
    n_opts = len(gold)

    # 1. 기존 방식: 개별 프롬프트별 W_1(P_i, Q) 및 TVD(P_i, Q) 계산 후 평균
    indiv_w = [
        wasserstein_1d(p, gold, codes=codes, use_code_positions=use_code_positions)
        for p in preds
    ]
    indiv_tvd = [tvd_distance(p, gold) for p in preds]
    existing_w = float(np.mean(indiv_w))
    existing_tvd = float(np.mean(indiv_tvd))

    # 가중치가 주어진 경우 정규화 (기본값은 동일 가중 1/N)
    unw_arr = np.ones(len(preds), dtype=float) / len(preds)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        w_arr = w_arr / float(w_arr.sum())
    else:
        w_arr = unw_arr.copy()
    weighted_mean_w = float(np.sum(w_arr * np.array(indiv_w)))
    weighted_mean_tvd = float(np.sum(w_arr * np.array(indiv_tvd)))

    # 2. 수정 방식: 모집단 예측분포 P_agg = sum_i w_i P_i 먼저 집계 후 거리 계산
    pop_pred = aggregate_distributions(preds, weights=w_arr)
    pop_pred_unw = aggregate_distributions(preds, weights=unw_arr)
    population_w = wasserstein_1d(
        pop_pred, gold, codes=codes, use_code_positions=use_code_positions
    )
    population_w_unw = wasserstein_1d(
        pop_pred_unw, gold, codes=codes, use_code_positions=use_code_positions
    )
    population_tvd = tvd_distance(pop_pred, gold)
    population_tvd_unw = tvd_distance(pop_pred_unw, gold)

    # 3. 기준선(Baselines)
    unif = np.ones(n_opts, dtype=float) / n_opts
    uniform_w = wasserstein_1d(unif, gold, codes=codes, use_code_positions=use_code_positions)
    uniform_tvd = tvd_distance(unif, gold)

    modal_1hot = np.zeros(n_opts, dtype=float)
    modal_1hot[int(np.argmax(gold))] = 1.0
    marginal_modal_1hot_w = wasserstein_1d(
        modal_1hot, gold, codes=codes, use_code_positions=use_code_positions
    )
    marginal_modal_1hot_tvd = tvd_distance(modal_1hot, gold)

    # 1D Wasserstein의 이론적 최적 점질량(1-hot) 위치는 최빈값(mode)이 아니라 중앙값(median)임
    cdf_gold = np.cumsum(gold)
    median_idx = int(np.searchsorted(cdf_gold, 0.5 - 1e-12))
    median_idx = min(max(median_idx, 0), n_opts - 1)
    median_1hot = np.zeros(n_opts, dtype=float)
    median_1hot[median_idx] = 1.0
    marginal_median_1hot_w = wasserstein_1d(
        median_1hot, gold, codes=codes, use_code_positions=use_code_positions
    )
    marginal_median_1hot_tvd = tvd_distance(median_1hot, gold)

    # 4. 엔트로피 분해: 조건부 엔트로피 E[H(P_i)] vs 모집단 엔트로피 H(E[P_i])
    indiv_ent = [entropy_bits(p) for p in preds]
    cond_entropy_unweighted = float(np.mean(indiv_ent))
    cond_entropy_weighted = float(np.sum(w_arr * np.array(indiv_ent)))
    pop_entropy = entropy_bits(pop_pred)
    pop_entropy_unweighted = entropy_bits(pop_pred_unw)
    gold_entropy = entropy_bits(gold)
    mutual_info_bits = float(max(0.0, pop_entropy - cond_entropy_weighted))
    mutual_info_unweighted_bits = float(max(0.0, pop_entropy_unweighted - cond_entropy_unweighted))

    out = {
        "n_options": float(n_opts),
        "n_probes": float(len(preds)),
        "existing_wasserstein": existing_w,
        "weighted_mean_wasserstein": weighted_mean_w,
        "population_wasserstein": population_w,
        "population_wasserstein_unweighted": population_w_unw,
        "wasserstein_jensen_gap": float(weighted_mean_w - population_w),
        "existing_minus_population_w": float(existing_w - population_w),
        "existing_tvd": existing_tvd,
        "weighted_mean_tvd": weighted_mean_tvd,
        "population_tvd": population_tvd,
        "population_tvd_unweighted": population_tvd_unw,
        "uniform_wasserstein": uniform_w,
        "uniform_tvd": uniform_tvd,
        "marginal_modal_1hot_wasserstein": marginal_modal_1hot_w,
        "marginal_modal_1hot_tvd": marginal_modal_1hot_tvd,
        "marginal_median_1hot_wasserstein": marginal_median_1hot_w,
        "marginal_median_1hot_tvd": marginal_median_1hot_tvd,
        "human_entropy_bits": gold_entropy,
        "model_conditional_entropy_bits": cond_entropy_unweighted,
        "model_conditional_entropy_weighted_bits": cond_entropy_weighted,
        "model_population_entropy_bits": pop_entropy,
        "model_population_entropy_unweighted_bits": pop_entropy_unweighted,
        "mutual_information_bits": mutual_info_bits,
        "mutual_information_unweighted_bits": mutual_info_unweighted_bits,
    }

    # 5. 조건부 셀 재현성 (Conditional Cell Fidelity) 및 셀 가중 모집단 집계
    if cell_model_dists is not None and cell_human_dists is not None and len(cell_model_dists) > 0:
        c_ws = [
            wasserstein_1d(pm, ph, codes=codes, use_code_positions=use_code_positions)
            for pm, ph in zip(cell_model_dists, cell_human_dists)
        ]
        c_tvds = [tvd_distance(pm, ph) for pm, ph in zip(cell_model_dists, cell_human_dists)]
        out["conditional_cell_wasserstein_unweighted"] = float(np.mean(c_ws))
        out["conditional_cell_tvd_unweighted"] = float(np.mean(c_tvds))
        if cell_weights is not None:
            cw = np.asarray(cell_weights, dtype=float)
            cw = cw / float(cw.sum())
            out["conditional_cell_wasserstein_weighted"] = float(np.sum(cw * np.array(c_ws)))
            out["conditional_cell_tvd_weighted"] = float(np.sum(cw * np.array(c_tvds)))
            cell_pop_pred = aggregate_distributions(cell_model_dists, weights=cw)
        else:
            out["conditional_cell_wasserstein_weighted"] = float(np.mean(c_ws))
            out["conditional_cell_tvd_weighted"] = float(np.mean(c_tvds))
            cell_pop_pred = aggregate_distributions(cell_model_dists)
        out["population_wasserstein_cell_weighted"] = wasserstein_1d(
            cell_pop_pred, gold, codes=codes, use_code_positions=use_code_positions
        )
        out["population_tvd_cell_weighted"] = tvd_distance(cell_pop_pred, gold)

    return out


def compute_subgroup_correlations(
    merged_cells: pd.DataFrame,
    cell_cols: Sequence[str] = ("sex", "age_grp"),
    min_items_per_cell: int = 5,
    min_cells_per_item: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
    cluster_col: str | None = "battery",
) -> dict[str, object]:
    """성별×연령 셀별 하위집단 상관(subgroup r)을 다각도로 분리·검증하여 계산한다.

    1. existing_cell_r:
       corr_v(real_mean - real_grand, model_pred)
       기존 kgss_c0_validity.py 산식. 인간 편차만 중심화하고 모델 기댓값은 중심화하지 않아
       문항 척도 분산 Var_v(model_grand)가 분모를 팽창시켜 오라클조차 r≈0.18로 감쇠됨.
    2. centered_cell_r (엄격 가중치·관측셀 일치 중심화):
       corr_v(real_mean - real_grand_matched, model_pred - model_grand_matched)
       동일한 관측 셀 집합 C_v와 동일한 정규화 가중치 w_{v,c}를 인간과 모델 양쪽에 똑같이 적용해
       문항 전체평균을 빼서 셀 편차끼리 상관시킴. 오라클(model_pred == real_mean)일 때 정확히 r = 1.0.
    3. centered_unweighted_cell_r:
       인간과 모델 양쪽 모두 관측 셀 단순평균(1/|C_v|)으로 중심화한 상관계수.
    4. scale_normalized_cell_r:
       문항별 척도 범위 차이(2점 척도 vs 7점 척도)의 영향을 제거하기 위해 각 문항의 편차를
       척도 폭(span = max(codes) - min(codes))으로 나눈 뒤 계산한 상관계수.
    5. within_item_r:
       각 문항 v 내부에서 셀들에 대해 corr_c(real_mean, model_pred)를 구한 뒤 문항 평균.
    6. 부트스트랩 95% CI (탐색적 불확실성 범위):
       - 문항(var) 단위 부트스트랩: 관측된 셀 평균이 주어졌을 때 문항 표집 변동을 반영.
       - 문항군(battery) 클러스터 부트스트랩: POSGEN1..10 등 동일 배터리 문항 묶음 간 의존성을 반영.
       (주의: 인간 응답자 및 페르소나 표본추출 불확실성까지 모두 반영한 모수적 추론이 아니므로
        탐색적 지표로 해석해야 하며, 평균 r을 제곱하여 설명분산 R^2로 해석해서는 안 됨.)
    """
    if merged_cells.empty:
        return {
            "existing_mean_r": np.nan,
            "centered_mean_r": np.nan,
            "centered_unweighted_mean_r": np.nan,
            "scale_normalized_mean_r": np.nan,
            "within_item_mean_r": np.nan,
            "centered_boot_ci_low": np.nan,
            "centered_boot_ci_high": np.nan,
            "centered_boot_p_two_sided": np.nan,
            "centered_battery_boot_ci_low": np.nan,
            "centered_battery_boot_ci_high": np.nan,
            "centered_battery_boot_p_two_sided": np.nan,
            "n_items": 0,
            "n_clusters": 0,
            "existing_cell_rs": {},
            "centered_cell_rs": {},
            "within_item_rs": {},
        }

    df = merged_cells.copy()
    df["real_dev_legacy"] = df["real_mean"] - df["real_grand"]

    # 관측된 동일 셀 집합에서 인간과 모델에 동일한 가중치(FINALWT 정규화 가중치)를 적용해 중심화
    if "cell_weight_item_norm" in df.columns:
        def _matched_w_means(g: pd.DataFrame) -> pd.Series:
            w = g["cell_weight_item_norm"].to_numpy(dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(g), dtype=float) / len(g)
            else:
                w = w / w.sum()
            return pd.Series({
                "real_grand_matched": float(np.sum(w * g["real_mean"].to_numpy(dtype=float))),
                "model_grand_matched": float(np.sum(w * g["model_pred"].to_numpy(dtype=float))),
            })
        gm = df.groupby("var", observed=True).apply(_matched_w_means, include_groups=False)
        df = df.merge(gm, on="var", how="left")
    else:
        df["real_grand_matched"] = df.groupby("var", observed=True)["real_mean"].transform("mean")
        df["model_grand_matched"] = df.groupby("var", observed=True)["model_pred"].transform("mean")

    df["real_dev"] = df["real_mean"] - df["real_grand_matched"]
    df["model_dev"] = df["model_pred"] - df["model_grand_matched"]

    # 비가중(1/|C_v|) 동일 중심화 편차
    df["real_dev_unw"] = df["real_mean"] - df.groupby("var", observed=True)["real_mean"].transform("mean")
    df["model_dev_unw"] = df["model_pred"] - df.groupby("var", observed=True)["model_pred"].transform("mean")

    # 척도 범위(span) 정규화 편차
    if "span" in df.columns:
        span_vals = np.maximum(df["span"].to_numpy(dtype=float), 1.0)
        df["real_dev_sn"] = df["real_dev"] / span_vals
        df["model_dev_sn"] = df["model_dev"] / span_vals
    else:
        df["real_dev_sn"] = df["real_dev"]
        df["model_dev_sn"] = df["model_dev"]

    existing_rs: dict[str, float] = {}
    centered_rs: dict[str, float] = {}
    centered_unw_rs: dict[str, float] = {}
    scale_norm_rs: dict[str, float] = {}

    for key, gc in df.groupby(list(cell_cols), observed=True):
        if len(gc) < min_items_per_cell:
            continue
        label = "_".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
        if gc["real_dev_legacy"].std(ddof=1) > 1e-12 and gc["model_pred"].std(ddof=1) > 1e-12:
            r_ex, _ = pearsonr(gc["real_dev_legacy"], gc["model_pred"])
            existing_rs[label] = float(r_ex)
        if gc["real_dev"].std(ddof=1) > 1e-12 and gc["model_dev"].std(ddof=1) > 1e-12:
            r_ce, _ = pearsonr(gc["real_dev"], gc["model_dev"])
            centered_rs[label] = float(r_ce)
        if gc["real_dev_unw"].std(ddof=1) > 1e-12 and gc["model_dev_unw"].std(ddof=1) > 1e-12:
            r_cu, _ = pearsonr(gc["real_dev_unw"], gc["model_dev_unw"])
            centered_unw_rs[label] = float(r_cu)
        if gc["real_dev_sn"].std(ddof=1) > 1e-12 and gc["model_dev_sn"].std(ddof=1) > 1e-12:
            r_sn, _ = pearsonr(gc["real_dev_sn"], gc["model_dev_sn"])
            scale_norm_rs[label] = float(r_sn)

    within_item_rs: dict[str, float] = {}
    for var, gv in df.groupby("var", observed=True):
        if len(gv) < min_cells_per_item:
            continue
        if gv["real_mean"].std(ddof=1) > 1e-12 and gv["model_pred"].std(ddof=1) > 1e-12:
            r_it, _ = pearsonr(gv["real_mean"], gv["model_pred"])
            within_item_rs[str(var)] = float(r_it)

    def _boot_mean_r_from_indices(R_mat: np.ndarray, M_mat: np.ndarray, idx: np.ndarray) -> float | None:
        Rb = R_mat[idx]
        Mb = M_mat[idx]
        c_rs = []
        for c_idx in range(R_mat.shape[1]):
            mask = np.isfinite(Rb[:, c_idx]) & np.isfinite(Mb[:, c_idx])
            if np.sum(mask) < min_items_per_cell:
                continue
            rc = Rb[mask, c_idx]
            mc = Mb[mask, c_idx]
            rc_c = rc - np.mean(rc)
            mc_c = mc - np.mean(mc)
            denom = float(np.sqrt(np.sum(rc_c ** 2) * np.sum(mc_c ** 2)))
            if denom > 1e-12:
                c_rs.append(float(np.sum(rc_c * mc_c) / denom))
        return float(np.mean(c_rs)) if c_rs else None

    # 문항 단위 및 배터리 클러스터 단위 부트스트랩
    ci_low, ci_high, boot_p = np.nan, np.nan, np.nan
    b_ci_low, b_ci_high, b_boot_p = np.nan, np.nan, np.nan
    n_items_total = int(df["var"].nunique())
    n_clusters_total = n_items_total

    if centered_rs and n_boot > 0:
        df["_cell_key"] = [
            "_".join(str(row[c]) for c in cell_cols) for _, row in df[list(cell_cols)].iterrows()
        ]
        piv_r = df.pivot_table(index="var", columns="_cell_key", values="real_dev", aggfunc="mean")
        piv_m = df.pivot_table(index="var", columns="_cell_key", values="model_dev", aggfunc="mean")
        if len(piv_r) >= min_items_per_cell:
            R_mat = piv_r.to_numpy(dtype=float)
            M_mat = piv_m.to_numpy(dtype=float)
            n_items = R_mat.shape[0]
            rng = np.random.default_rng(seed)
            boot_means = []
            for _ in range(n_boot):
                idx = rng.integers(0, n_items, size=n_items)
                val = _boot_mean_r_from_indices(R_mat, M_mat, idx)
                if val is not None:
                    boot_means.append(val)
            if boot_means:
                b_arr = np.asarray(boot_means, dtype=float)
                ci_low = float(np.percentile(b_arr, 2.5))
                ci_high = float(np.percentile(b_arr, 97.5))
                boot_p = float(2.0 * min(np.mean(b_arr <= 0.0), np.mean(b_arr >= 0.0)))

            # 배터리(문항 묶음) 클러스터 부트스트랩
            if cluster_col and cluster_col in df.columns:
                var_to_cluster = (
                    df[["var", cluster_col]]
                    .drop_duplicates(subset=["var"])
                    .set_index("var")[cluster_col]
                    .fillna("")
                    .astype(str)
                    .to_dict()
                )
                clusters: dict[str, list[int]] = {}
                for row_i, v_name in enumerate(piv_r.index):
                    raw_cl = var_to_cluster.get(v_name, "").strip()
                    cl_key = raw_cl if raw_cl and raw_cl.lower() != "nan" else f"__single__{v_name}"
                    clusters.setdefault(cl_key, []).append(row_i)
                cluster_keys = list(clusters.keys())
                n_clusters_total = len(cluster_keys)
                rng_cl = np.random.default_rng(seed)
                cl_boot_means = []
                for _ in range(n_boot):
                    sampled_cl = rng_cl.integers(0, n_clusters_total, size=n_clusters_total)
                    idx_list = []
                    for c_i in sampled_cl:
                        idx_list.extend(clusters[cluster_keys[int(c_i)]])
                    idx_cl = np.asarray(idx_list, dtype=int)
                    val_cl = _boot_mean_r_from_indices(R_mat, M_mat, idx_cl)
                    if val_cl is not None:
                        cl_boot_means.append(val_cl)
                if cl_boot_means:
                    cb_arr = np.asarray(cl_boot_means, dtype=float)
                    b_ci_low = float(np.percentile(cb_arr, 2.5))
                    b_ci_high = float(np.percentile(cb_arr, 97.5))
                    b_boot_p = float(2.0 * min(np.mean(cb_arr <= 0.0), np.mean(cb_arr >= 0.0)))

    return {
        "existing_mean_r": float(np.mean(list(existing_rs.values()))) if existing_rs else np.nan,
        "centered_mean_r": float(np.mean(list(centered_rs.values()))) if centered_rs else np.nan,
        "centered_unweighted_mean_r": float(np.mean(list(centered_unw_rs.values()))) if centered_unw_rs else np.nan,
        "scale_normalized_mean_r": float(np.mean(list(scale_norm_rs.values()))) if scale_norm_rs else np.nan,
        "within_item_mean_r": float(np.mean(list(within_item_rs.values()))) if within_item_rs else np.nan,
        "centered_boot_ci_low": ci_low,
        "centered_boot_ci_high": ci_high,
        "centered_boot_p_two_sided": boot_p,
        "centered_battery_boot_ci_low": b_ci_low,
        "centered_battery_boot_ci_high": b_ci_high,
        "centered_battery_boot_p_two_sided": b_boot_p,
        "n_items": n_items_total,
        "n_clusters": n_clusters_total,
        "existing_cell_rs": existing_rs,
        "centered_cell_rs": centered_rs,
        "within_item_rs": within_item_rs,
    }


def evaluate_personal_accuracy_paired(
    obs_df: pd.DataFrame,
    cond_a: str = "full",
    cond_b: str = "demo",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """개인 응답 예측 정확도(일반 정확도 및 셀 최빈 정규화 정확도)와 대응 차이 부트스트랩을 계산한다.

    입력 obs_df 필수 열:
    - level: 조건명 (예: 'c0', 'adult', 'demo', 'full')
    - persona_id: 페르소나 표본 ID (또는 RESPID)
    - var: 문항 변수명
    - is_correct: 모델 최빈 예측 == 실제 개인 응답 여부 (0 또는 1)
    - cell_modal: 해당 문항의 성별×연령 셀 최빈 정확도 (0 < cell_modal <= 1)
    - respid (선택): 중복 추출된 원 응답자 클러스터 ID
    """
    df = obs_df.copy()
    df["norm_acc"] = df["is_correct"].astype(float) / df["cell_modal"].astype(float)

    # 동일 (persona_id, var) 쌍으로 정렬하여 완전 대응(balanced panel) 구성
    piv_raw = df.pivot_table(index=["persona_id", "var"], columns="level", values="is_correct", aggfunc="mean")
    piv_norm = df.pivot_table(index=["persona_id", "var"], columns="level", values="norm_acc", aggfunc="mean")

    both_raw = piv_raw[[cond_a, cond_b]].dropna()
    both_norm = piv_norm[[cond_a, cond_b]].dropna()

    mean_raw_a = float(both_raw[cond_a].mean())
    mean_raw_b = float(both_raw[cond_b].mean())
    diff_raw = float((both_raw[cond_a] - both_raw[cond_b]).mean())

    mean_norm_a = float(both_norm[cond_a].mean())
    mean_norm_b = float(both_norm[cond_b].mean())
    diff_norm = float((both_norm[cond_a] - both_norm[cond_b]).mean())

    # 페르소나 단위 집계 후 부트스트랩 (일반 정확도 vs 정규화 정확도 분리)
    p_raw_diff = (both_raw[cond_a] - both_raw[cond_b]).groupby("persona_id").mean()
    p_norm_diff = (both_norm[cond_a] - both_norm[cond_b]).groupby("persona_id").mean()

    rng = np.random.default_rng(seed)
    n_p = len(p_raw_diff)
    boot_raw = np.empty(n_boot, dtype=float)
    boot_norm = np.empty(n_boot, dtype=float)
    raw_vals = p_raw_diff.to_numpy(dtype=float)
    norm_vals = p_norm_diff.to_numpy(dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        boot_raw[b] = float(raw_vals[idx].mean())
        boot_norm[b] = float(norm_vals[idx].mean())

    return {
        f"raw_acc_{cond_a}": mean_raw_a,
        f"raw_acc_{cond_b}": mean_raw_b,
        "raw_diff_mean": diff_raw,
        "raw_diff_linearity_error": abs(diff_raw - (mean_raw_a - mean_raw_b)),
        "raw_boot_ci_low": float(np.percentile(boot_raw, 2.5)),
        "raw_boot_ci_high": float(np.percentile(boot_raw, 97.5)),
        f"norm_acc_{cond_a}": mean_norm_a,
        f"norm_acc_{cond_b}": mean_norm_b,
        "norm_diff_mean": diff_norm,
        "norm_diff_linearity_error": abs(diff_norm - (mean_norm_a - mean_norm_b)),
        "norm_boot_ci_low": float(np.percentile(boot_norm, 2.5)),
        "norm_boot_ci_high": float(np.percentile(boot_norm, 97.5)),
    }
