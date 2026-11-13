"""
Post-hoc probability calibration, uniform mixing, stratified item splitting,
nested cross-validation, conditional shift fidelity, and stratified expanded
panel construction for the KOSMOS KGSS 2023 silicon sampling study.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from kgss_metrics import AGE_BINS_5, AGE_LABELS_5, entropy_bits, tvd_distance, validate_prob_dist, wasserstein_1d


PERSONA_DEPTH_CONFIGS: dict[str, dict[str, Any]] = {
    "C2": {
        "vars_exact": ["SEX", "AGE"],
        "vars_band": ["SEX", "AGE_GRP"],
        "role": "core",
        "depth": 2,
        "description": "성별 + 연령 (기본 2변수)",
    },
    "C3_M": {
        "vars_exact": ["SEX", "AGE", "MARITAL"],
        "vars_band": ["SEX", "AGE_GRP", "MARITAL"],
        "role": "core",
        "depth": 3,
        "description": "성별 + 연령 + 혼인상태 (높은 표본 커버리지 3변수)",
    },
    "C3_E": {
        "vars_exact": ["SEX", "AGE", "EDUC"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC"],
        "role": "core",
        "depth": 3,
        "description": "성별 + 연령 + 교육수준 (핵심 사회경제 3변수)",
    },
    "C3_R": {
        "vars_exact": ["SEX", "AGE", "REGION"],
        "vars_band": ["SEX", "AGE_GRP", "REGION"],
        "role": "support_candidate",
        "depth": 3,
        "description": "성별 + 연령 + 거주지역 (3변수 후보)",
    },
    "C3_U": {
        "vars_exact": ["SEX", "AGE", "URBAN"],
        "vars_band": ["SEX", "AGE_GRP", "URBAN"],
        "role": "support_candidate",
        "depth": 3,
        "description": "성별 + 연령 + 도시성 (3변수 후보)",
    },
    "C4_EM": {
        "vars_exact": ["SEX", "AGE", "EDUC", "MARITAL"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC", "MARITAL"],
        "role": "extension",
        "depth": 4,
        "description": "성별 + 연령 + 교육수준 + 혼인상태 (확장 비교 4변수)",
    },
    "C4_ER": {
        "vars_exact": ["SEX", "AGE", "EDUC", "REGION"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC", "REGION"],
        "role": "exploratory",
        "depth": 4,
        "description": "성별 + 연령 + 교육수준 + 거주지역 (희소성 비교용 4변수)",
    },
    "C4_EU": {
        "vars_exact": ["SEX", "AGE", "EDUC", "URBAN"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC", "URBAN"],
        "role": "support_candidate",
        "depth": 4,
        "description": "성별 + 연령 + 교육수준 + 도시성 (4변수 후보)",
    },
    "C4_MR": {
        "vars_exact": ["SEX", "AGE", "MARITAL", "REGION"],
        "vars_band": ["SEX", "AGE_GRP", "MARITAL", "REGION"],
        "role": "support_candidate",
        "depth": 4,
        "description": "성별 + 연령 + 혼인상태 + 거주지역 (4변수 후보)",
    },
    "C4_MU": {
        "vars_exact": ["SEX", "AGE", "MARITAL", "URBAN"],
        "vars_band": ["SEX", "AGE_GRP", "MARITAL", "URBAN"],
        "role": "support_candidate",
        "depth": 4,
        "description": "성별 + 연령 + 혼인상태 + 도시성 (4변수 후보)",
    },
    "C4_RU": {
        "vars_exact": ["SEX", "AGE", "REGION", "URBAN"],
        "vars_band": ["SEX", "AGE_GRP", "REGION", "URBAN"],
        "role": "support_candidate",
        "depth": 4,
        "description": "성별 + 연령 + 거주지역 + 도시성 (4변수 후보)",
    },
    "C5_EMR": {
        "vars_exact": ["SEX", "AGE", "EDUC", "MARITAL", "REGION"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC", "MARITAL", "REGION"],
        "role": "support_candidate",
        "depth": 5,
        "description": "성별 + 연령 + 교육 + 혼인 + 지역 (5변수 후보)",
    },
    "C6": {
        "vars_exact": ["SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN"],
        "vars_band": ["SEX", "AGE_GRP", "EDUC", "MARITAL", "REGION", "URBAN"],
        "role": "exploratory",
        "depth": 6,
        "description": "기존 Full 6변수 (성별+연령+교육+혼인+지역+도시성, 고희소성 기준)",
    },
}

DEFAULT_TEMPERATURES: tuple[float, ...] = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.5,
    4.0,
    5.0,
    7.5,
    10.0,
)

DEFAULT_MIXING_LAMBDAS: tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.7,
    1.0,
)

SUPPORT_THRESHOLDS: tuple[int, ...] = (5, 10, 20, 30)


def scale_probability_temperature(
    prob_dist: Sequence[float] | np.ndarray,
    temperature: float,
    *,
    epsilon: float | None = None,
) -> np.ndarray:
    """
    Apply post-hoc temperature scaling to a probability vector P = (p_1, ..., p_K):
        p_i^(T) = p_i^(1/T) / sum_j p_j^(1/T)

    Parameters
    ----------
    prob_dist : Sequence[float] | np.ndarray
        Input probability distribution (must be non-negative, finite, and sum to 1).
    temperature : float
        Temperature T > 0. T = 1.0 is the identity transform (when epsilon is None).
    epsilon : float | None
        If None, uses Original-Support Temperature Scaling (p_i = 0 remains 0).
        If > 0, applies additive smoothing (p_i + eps) / sum_j (p_j + eps) prior to scaling.
    """
    p = validate_prob_dist(prob_dist)
    t_val = float(temperature)
    if not math.isfinite(t_val) or t_val <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")

    if epsilon is not None:
        eps = float(epsilon)
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError(f"epsilon must be finite and non-negative, got {epsilon!r}")
        if eps > 0.0:
            p = p + eps
            p = p / float(p.sum())

    if epsilon is None and abs(t_val - 1.0) <= 1e-15:
        return p.copy()

    pos_mask = p > 0.0
    if not np.any(pos_mask):
        raise ValueError("Probability vector has no positive support.")

    scaled_log = np.full(len(p), -np.inf, dtype=float)
    scaled_log[pos_mask] = np.log(p[pos_mask]) / t_val
    max_log = float(np.max(scaled_log[pos_mask]))

    out = np.zeros(len(p), dtype=float)
    out[pos_mask] = np.exp(scaled_log[pos_mask] - max_log)
    total = float(out.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Temperature-scaled probability sum is invalid.")
    out = out / total
    return out


def mix_with_uniform(
    prob_dist: Sequence[float] | np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Apply linear uniform mixing to a probability vector P:
        P' = (1 - lam) * P + lam * U,  where U_k = 1 / K.
    """
    p = validate_prob_dist(prob_dist)
    lam_val = float(lam)
    if not math.isfinite(lam_val) or lam_val < -1e-12 or lam_val > 1.0 + 1e-12:
        raise ValueError(f"lam must be in [0, 1], got {lam!r}")
    lam_val = min(1.0, max(0.0, lam_val))
    if lam_val == 0.0:
        return p.copy()
    k = len(p)
    unif = np.full(k, 1.0 / float(k), dtype=float)
    if lam_val == 1.0:
        return unif
    out = (1.0 - lam_val) * p + lam_val * unif
    return out / float(out.sum())


def calibrate_probability_vector(
    prob_dist: Sequence[float] | np.ndarray,
    *,
    temperature: float = 1.0,
    mix_lambda: float = 0.0,
    epsilon: float | None = None,
) -> np.ndarray:
    """Apply post-hoc temperature scaling followed by optional uniform mixing."""
    p_temp = scale_probability_temperature(prob_dist, temperature, epsilon=epsilon)
    if float(mix_lambda) == 0.0:
        return p_temp
    return mix_with_uniform(p_temp, mix_lambda)


def calibrate_probability_matrix(
    prob_mat: np.ndarray,
    *,
    temperature: float = 1.0,
    mix_lambda: float = 0.0,
    epsilon: float | None = None,
) -> np.ndarray:
    """Vectorized post-hoc temperature scaling + uniform mixing over a 2D array (N, K)."""
    p = np.asarray(prob_mat, dtype=float)
    if p.ndim != 2:
        raise ValueError("prob_mat must be 2D (N, K).")
    t_val = float(temperature)
    if not math.isfinite(t_val) or t_val <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")

    if epsilon is not None and float(epsilon) > 0.0:
        eps = float(epsilon)
        p = p + eps
        p = p / p.sum(axis=1, keepdims=True)

    if epsilon is None and abs(t_val - 1.0) <= 1e-15:
        out = p.copy()
    else:
        pos_mask = p > 0.0
        scaled_log = np.full_like(p, -np.inf, dtype=float)
        scaled_log[pos_mask] = np.log(p[pos_mask]) / t_val
        row_max = np.max(scaled_log, axis=1, keepdims=True)
        out = np.zeros_like(p, dtype=float)
        out[pos_mask] = np.exp((scaled_log - row_max)[pos_mask])
        out = out / out.sum(axis=1, keepdims=True)

    lam_val = float(mix_lambda)
    if lam_val > 0.0:
        lam_val = min(1.0, max(0.0, lam_val))
        k = out.shape[1]
        out = (1.0 - lam_val) * out + (lam_val / float(k))
        out = out / out.sum(axis=1, keepdims=True)
    return out



def aggregate_calibrated_group(
    pred_dists: Sequence[Sequence[float] | np.ndarray],
    *,
    temperature: float = 1.0,
    mix_lambda: float = 0.0,
    epsilon: float | None = None,
    method: str = "individual",
    draw_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the calibrated group predictive distribution P_bar_{g, T}.

    Parameters
    ----------
    pred_dists : Sequence[Sequence[float] | np.ndarray]
        Probability vectors for each persona draw i in group g.
    method : str
        - "individual" (Method A, Primary):
            P_bar_{g, T} = sum_{i in g} w_i * Calib(P_i, T, mix_lambda, epsilon)
        - "group" (Method B, Secondary):
            P_bar_{g, T} = Calib(sum_{i in g} w_i * P_i, T, mix_lambda, epsilon)
    draw_weights : Sequence[float] | np.ndarray | None
        Optional inner weights w_i for draws in group g. If None, uses equal weight 1/m_g.
    """
    if not pred_dists:
        raise ValueError("pred_dists must be non-empty.")
    if method not in ("individual", "group"):
        raise ValueError(f"Unknown calibration aggregation method: {method!r}")

    validated = [validate_prob_dist(p) for p in pred_dists]
    m_g = len(validated)
    k = len(validated[0])
    if any(len(p) != k for p in validated):
        raise ValueError("All probability vectors in group must have the same length K.")

    if draw_weights is None:
        w = np.full(m_g, 1.0 / float(m_g), dtype=float)
    else:
        w_arr = np.asarray(draw_weights, dtype=float)
        if w_arr.ndim != 1 or len(w_arr) != m_g:
            raise ValueError("draw_weights length must match pred_dists length.")
        if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
            raise ValueError("draw_weights must be finite and non-negative.")
        w_sum = float(w_arr.sum())
        if w_sum <= 0.0:
            raise ValueError("draw_weights sum must be positive.")
        w = w_arr / w_sum

    mat = np.vstack(validated)
    if method == "individual":
        calib_rows = np.vstack(
            [
                calibrate_probability_vector(
                    row,
                    temperature=temperature,
                    mix_lambda=mix_lambda,
                    epsilon=epsilon,
                )
                for row in mat
            ]
        )
        pbar = np.dot(w, calib_rows)
        return pbar / float(pbar.sum())

    # method == "group"
    raw_pbar = np.dot(w, mat)
    raw_pbar = raw_pbar / float(raw_pbar.sum())
    return calibrate_probability_vector(
        raw_pbar,
        temperature=temperature,
        mix_lambda=mix_lambda,
        epsilon=epsilon,
    )


def build_persona_header(
    row: Mapping[str, Any],
    labels: Mapping[str, Mapping[str, str]],
    config_name: str,
    *,
    addressee: str = "당신",
    register: str = "formal",
) -> str:
    """
    Construct the Korean persona prompt header for any Persona Depth configuration
    (C2, C3_M, C3_E, C3_R, C3_U, C4_EM, C4_ER, C4_EU, C6), preserving exact syntactic
    parity with paired_demo40_clean (C2) and full40_clean (C6).
    """
    if config_name not in PERSONA_DEPTH_CONFIGS:
        raise ValueError(f"Unknown persona config: {config_name!r}")

    vars_exact = set(PERSONA_DEPTH_CONFIGS[config_name]["vars_exact"])
    age = int(float(row["AGE"]))
    sex_lab = str(labels["SEX"].get(str(float(row["SEX"])), row.get("SEX_label", ""))).strip()
    educ_lab = str(labels["EDUC"].get(str(float(row["EDUC"])), row.get("EDUC_label", ""))).strip()
    region_lab = str(labels["REGION"].get(str(float(row["REGION"])), row.get("REGION_label", ""))).strip()
    marital_lab = str(labels["MARITAL"].get(str(float(row["MARITAL"])), row.get("MARITAL_label", ""))).strip()
    urban_lab = str(labels["URBAN"].get(str(float(row["URBAN"])), row.get("URBAN_label", ""))).strip()

    is_formal = register == "formal"
    ending = "입니다" if is_formal else "이다"

    if addressee == "당신":
        head = f"당신은 {age}세 {sex_lab}{ending}."
    else:
        head = f"다음은 {age}세 {sex_lab}에 대한 설명{ending}."

    extra_vars = vars_exact - {"SEX", "AGE"}
    if not extra_vars:
        return head

    clauses: list[str] = []
    if "EDUC" in extra_vars:
        clauses.append(f"최종학력은 {educ_lab}")
    if "REGION" in extra_vars:
        clauses.append(f"{region_lab}에 거주")
    if "MARITAL" in extra_vars:
        clauses.append(f"{marital_lab}")

    tail = ""
    if clauses:
        if len(clauses) == 1:
            c0 = clauses[0]
            if c0.endswith("에 거주"):
                tail = f"{c0}합니다." if is_formal else f"{c0}한다."
            else:
                tail = f"{c0}{ending}."
        else:
            joined = ""
            for idx, c in enumerate(clauses):
                if idx == len(clauses) - 1:
                    if c.endswith("에 거주"):
                        joined += f"{c}합니다." if is_formal else f"{c}한다."
                    else:
                        joined += f"{c}{ending}."
                else:
                    if c.endswith("에 거주"):
                        joined += f"{c}하며 "
                    else:
                        joined += f"{c}이며 "
            tail = joined

    urban_suffix = f" ({urban_lab} 지역 거주)" if "URBAN" in extra_vars else ""
    if tail:
        return f"{head} {tail}{urban_suffix}"
    return f"{head}{urban_suffix}"


def extract_topic_prefix(var_name: str) -> str:
    """Extract thematic prefix (first 3 letters) for grouped cross-validation."""
    s = "".join(ch for ch in str(var_name).upper() if ch.isalpha())
    return s[:3] if len(s) >= 3 else s


def create_stratified_item_splits(
    items_df: pd.DataFrame,
    *,
    seed: int = 42,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Create deterministic 60/20/20 (Calibration / Validation / Test) splits and
    5-fold Nested CV assignments stratified by (scale_type, k_bucket), plus
    topic-prefix grouped 5-fold assignments.

    Expects `items_df` to contain at least columns: `var`, `scale_type`, `n_choices`.
    """
    if n_folds != 5:
        raise ValueError("n_folds must be 5 to match 60/20/20 Calibration/Validation/Test proportions.")

    df = items_df.sort_values("var").reset_index(drop=True).copy()

    def _k_bucket(k: int) -> str:
        k_int = int(k)
        if k_int <= 2:
            return "K2"
        if k_int <= 4:
            return "K3_4"
        if k_int == 5:
            return "K5"
        return "K6_plus"

    df["k_bucket"] = [
        _k_bucket(int(k)) for k in df["n_choices"]
    ]
    df["stratum"] = df["scale_type"].astype(str) + "__" + df["k_bucket"].astype(str)
    df["topic_prefix"] = [extract_topic_prefix(v) for v in df["var"]]

    rng = np.random.default_rng(int(seed))
    cv_folds = np.zeros(len(df), dtype=int)

    running_pos = 0
    for _, grp in df.groupby("stratum", sort=True):
        idxs = grp.index.to_numpy().copy()
        rng.shuffle(idxs)
        for idx in idxs:
            cv_folds[idx] = running_pos % n_folds
            running_pos += 1

    df["cv_fold"] = cv_folds
    # Fold 0, 1, 2 -> calibration (60%); Fold 3 -> validation (20%); Fold 4 -> test (20%)
    split_map = {0: "calibration", 1: "calibration", 2: "calibration", 3: "validation", 4: "test"}
    df["split_60_20_20"] = [split_map[int(f)] for f in df["cv_fold"]]

    # Also build topic-prefix grouped 5-fold assignment (so items sharing a prefix stay in the same fold)
    prefix_counts = df.groupby("topic_prefix")["var"].count().sort_values(ascending=False)
    fold_loads = [0] * n_folds
    prefix_to_fold: dict[str, int] = {}
    rng_grp = np.random.default_rng(int(seed) + 1000)
    # Shuffle prefixes within equal-count ties deterministically
    prefix_df = pd.DataFrame({"prefix": prefix_counts.index, "cnt": prefix_counts.to_numpy()})
    prefix_df["rand"] = rng_grp.random(len(prefix_df))
    prefix_df = prefix_df.sort_values(["cnt", "rand"], ascending=[False, True])
    for _, prow in prefix_df.iterrows():
        target_f = int(np.argmin(fold_loads))
        prefix_to_fold[str(prow["prefix"])] = target_f
        fold_loads[target_f] += int(prow["cnt"])

    df["group_cv_fold"] = [prefix_to_fold[str(p)] for p in df["topic_prefix"]]
    return df


def compute_conditional_shift_metrics(
    model_group_dists: Sequence[Sequence[float] | np.ndarray],
    human_group_dists: Sequence[Sequence[float] | np.ndarray],
    group_weights: Sequence[float] | np.ndarray,
    *,
    codes: Sequence[float] | None = None,
    w1_valid: bool = False,
) -> dict[str, float | None]:
    """
    Evaluate whether the model reproduces between-group conditional shifts
    Delta P_{g,h} = P_g - P_h relative to human shifts Delta Q_{g,h} = Q_g - Q_h.
    """
    n_g = len(model_group_dists)
    if n_g != len(human_group_dists) or n_g != len(group_weights):
        raise ValueError("Length mismatch among model_group_dists, human_group_dists, and group_weights.")
    if n_g < 2:
        return {
            "n_group_pairs": 0,
            "mean_human_pair_tvd": None,
            "mean_model_pair_tvd": None,
            "shift_magnitude_ratio": None,
            "shift_vector_cosine": None,
            "shift_vector_pearson_r": None,
            "option_shift_sign_acc": None,
            "expected_value_shift_pearson_r": None,
            "expected_value_shift_sign_acc": None,
        }

    p_mat = np.asarray(model_group_dists, dtype=float)
    q_mat = np.asarray(human_group_dists, dtype=float)
    w = np.asarray(group_weights, dtype=float)
    w = w / float(w.sum())

    idx_g, idx_h = np.triu_indices(n_g, k=1)
    pw_raw = w[idx_g] * w[idx_h]
    pos_pairs = pw_raw > 0.0
    if not np.any(pos_pairs):
        return {
            "n_group_pairs": 0,
            "mean_human_pair_tvd": None,
            "mean_model_pair_tvd": None,
            "shift_magnitude_ratio": None,
            "shift_vector_cosine": None,
            "shift_vector_pearson_r": None,
            "option_shift_sign_acc": None,
            "expected_value_shift_pearson_r": None,
            "expected_value_shift_sign_acc": None,
        }

    idx_g = idx_g[pos_pairs]
    idx_h = idx_h[pos_pairs]
    pw_arr = pw_raw[pos_pairs]
    pw_norm = pw_arr / float(pw_arr.sum())

    dp_mat = p_mat[idx_g] - p_mat[idx_h]
    dq_mat = q_mat[idx_g] - q_mat[idx_h]
    human_tvds = 0.5 * np.sum(np.abs(dq_mat), axis=1)
    model_tvds = 0.5 * np.sum(np.abs(dp_mat), axis=1)

    mean_h_tvd = float(np.dot(pw_norm, human_tvds))
    mean_m_tvd = float(np.dot(pw_norm, model_tvds))
    mag_ratio = (mean_m_tvd / mean_h_tvd) if mean_h_tvd > 1e-12 else None

    w_broadcast = np.repeat(pw_norm, dp_mat.shape[1])
    dp_flat = dp_mat.reshape(-1)
    dq_flat = dq_mat.reshape(-1)

    # Weighted cosine similarity across all pairwise option shifts
    dot_pq = float(np.sum(w_broadcast * dp_flat * dq_flat))
    norm_p = math.sqrt(float(np.sum(w_broadcast * (dp_flat ** 2))))
    norm_q = math.sqrt(float(np.sum(w_broadcast * (dq_flat ** 2))))
    cosine = (dot_pq / (norm_p * norm_q)) if (norm_p > 1e-12 and norm_q > 1e-12) else 0.0

    # Weighted Pearson r across pairwise option shifts
    mp = float(np.sum(w_broadcast * dp_flat))
    mq = float(np.sum(w_broadcast * dq_flat))
    vp = float(np.sum(w_broadcast * ((dp_flat - mp) ** 2)))
    vq = float(np.sum(w_broadcast * ((dq_flat - mq) ** 2)))
    cov_pq = float(np.sum(w_broadcast * (dp_flat - mp) * (dq_flat - mq)))
    pearson_r = (cov_pq / math.sqrt(vp * vq)) if (vp > 1e-12 and vq > 1e-12) else 0.0

    # Option shift sign agreement where |Delta Q| >= 0.02
    nontrivial = np.abs(dq_flat) >= 0.02
    if np.any(nontrivial):
        w_nt = w_broadcast[nontrivial]
        w_nt = w_nt / float(w_nt.sum())
        same_sign = (np.sign(dp_flat[nontrivial]) == np.sign(dq_flat[nontrivial])).astype(float)
        sign_acc = float(np.dot(w_nt, same_sign))
    else:
        sign_acc = None

    ev_r = None
    ev_sign_acc = None
    if w1_valid:
        k = p_mat.shape[1]
        score_vec = np.arange(1, k + 1, dtype=float)
        ev_p = p_mat @ score_vec
        ev_q = q_mat @ score_vec
        dep = ev_p[idx_g] - ev_p[idx_h]
        deq = ev_q[idx_g] - ev_q[idx_h]
        m_ep = float(np.dot(pw_norm, dep))
        m_eq = float(np.dot(pw_norm, deq))
        v_ep = float(np.dot(pw_norm, (dep - m_ep) ** 2))
        v_eq = float(np.dot(pw_norm, (deq - m_eq) ** 2))
        if v_ep > 1e-12 and v_eq > 1e-12:
            ev_r = float(np.dot(pw_norm, (dep - m_ep) * (deq - m_eq)) / math.sqrt(v_ep * v_eq))
        else:
            ev_r = 0.0
        nt_ev = np.abs(deq) >= 0.02
        if np.any(nt_ev):
            w_ev = pw_norm[nt_ev] / float(pw_norm[nt_ev].sum())
            ev_sign_acc = float(np.dot(w_ev, (np.sign(dep[nt_ev]) == np.sign(deq[nt_ev])).astype(float)))

    return {
        "n_group_pairs": int(len(pw_arr)),
        "mean_human_pair_tvd": mean_h_tvd,
        "mean_model_pair_tvd": mean_m_tvd,
        "shift_magnitude_ratio": mag_ratio,
        "shift_vector_cosine": cosine,
        "shift_vector_pearson_r": pearson_r,
        "option_shift_sign_acc": sign_acc,
        "expected_value_shift_pearson_r": ev_r,
        "expected_value_shift_sign_acc": ev_sign_acc,
    }


def build_stratified_expanded_panel(
    human_df: pd.DataFrame,
    panel40_df: pd.DataFrame,
    labels: Mapping[str, Mapping[str, str]],
    *,
    seed: int = 20261010,
) -> pd.DataFrame:
    """
    Construct a Stratified Without-Replacement Expanded Persona Panel from KGSS 2023
    for Experiment 2 (Expanded Population Experiment):
      - Strictly 0 duplicate respondents (unique RESPID).
      - Covers 100% of all C2, C3_M, C3_E, and C4_EM human groups that meet
        valid_n >= 5 AND effective_n >= 5 (including 100% of T>=10, T>=20, T>=30 groups).
      - Allocates multiple distinct-age respondents to larger groups (target m_s = 3 for
        C4_EM cells with effective_n >= 20, m_s = 2 for 5 <= effective_n < 20).
      - Computes exact post-stratified / Horvitz-Thompson weights `stratum_ht_weight`
        within every cell so that aggregating sampled personas to any coarser Fixed Target
        (C2, C3_M, C3_E, C4_EM) reproduces the true KGSS conditional composition P(z | g).
    """
    df = human_df.copy()
    if "AGE_GRP" not in df.columns:
        df["AGE_GRP"] = pd.cut(df["AGE"], bins=AGE_BINS_5, labels=AGE_LABELS_5, right=False).astype(str)

    valid_mask = (
        np.isfinite(df["FINALWT"].to_numpy(dtype=float))
        & (df["FINALWT"].to_numpy(dtype=float) > 0.0)
    )
    for col in ("SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN"):
        valid_mask &= df[col].notna() & (~df[col].isin([-1.0, -8.0, 88.0, 99.0]))
    elig_human = df[valid_mask].copy()
    elig_human["RESPID_INT"] = elig_human["RESPID"].astype(int)

    p40_respids = set(panel40_df["RESPID"].astype(int))
    rng = np.random.default_rng(int(seed))

    c4_cols = ["SEX", "AGE_GRP", "EDUC", "MARITAL"]
    c3m_cols = ["SEX", "AGE_GRP", "MARITAL"]
    c3e_cols = ["SEX", "AGE_GRP", "EDUC"]

    def _eligible_cell_keys(frame: pd.DataFrame, cols: list[str], threshold: int = 5) -> set[tuple]:
        out: set[tuple] = set()
        for k, grp in frame.groupby(cols, observed=True):
            w = grp["FINALWT"].to_numpy(dtype=float)
            neff = float((w.sum() ** 2) / np.dot(w, w)) if len(w) > 0 else 0.0
            if len(w) >= threshold and neff >= float(threshold):
                out.add(k if isinstance(k, tuple) else (k,))
        return out

    elig_c4 = _eligible_cell_keys(elig_human, c4_cols, 5)
    elig_c3m = _eligible_cell_keys(elig_human, c3m_cols, 5)
    elig_c3e = _eligible_cell_keys(elig_human, c3e_cols, 5)

    # Start with all 37 unique respondents from panel40 so expanded_panel is a strict superset
    chosen_respids: set[int] = set(p40_respids)

    c4_groups = []
    for cell_key, grp in elig_human.groupby(c4_cols, observed=True):
        if cell_key not in elig_c4:
            continue
        w = grp["FINALWT"].to_numpy(dtype=float)
        neff = float((w.sum() ** 2) / np.dot(w, w))
        c4_groups.append((-(1 if (len(w) >= 20 and neff >= 20.0) else 0), -neff, cell_key, grp, len(w), neff))
    c4_groups.sort(key=lambda x: (x[0], x[1]))

    for _, _, cell_key, grp, n_w, neff in c4_groups:
        target_m = 2 if (n_w >= 20 and neff >= 20.0) else 1
        already_in_cell = grp[grp["RESPID_INT"].isin(chosen_respids)]
        if len(already_in_cell) >= target_m:
            continue
        rem = target_m - len(already_in_cell)
        for _ in range(rem):
            cur_panel = elig_human[elig_human["RESPID_INT"].isin(chosen_respids)]
            c2_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"]))
            c3m_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["MARITAL"]))
            c3e_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["EDUC"]))
            c4_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["EDUC"], cur_panel["MARITAL"]))
            candidates = grp[~grp["RESPID_INT"].isin(chosen_respids)].copy()
            candidates["overlap_score"] = [
                1 * int((r.SEX, r.AGE) in c2_tuples)
                + 2 * int((r.SEX, r.AGE, r.MARITAL) in c3m_tuples)
                + 2 * int((r.SEX, r.AGE, r.EDUC) in c3e_tuples)
                + 3 * int((r.SEX, r.AGE, r.EDUC, r.MARITAL) in c4_tuples)
                for _, r in candidates.iterrows()
            ]
            candidates = candidates.sort_values(["overlap_score", "FINALWT"], ascending=[False, False])
            chosen_respids.add(int(candidates.iloc[0]["RESPID_INT"]))

    # Ensure 100% of T>=5 C3_M and C3_E cells are covered, with >=2 respondents in all T>=20 cells
    for cols, target_set in ((c3m_cols, elig_c3m), (c3e_cols, elig_c3e)):
        for cell_key in sorted(target_set):
            sub_g = elig_human.copy()
            for c, val in zip(cols, cell_key):
                sub_g = sub_g[sub_g[c] == val]
            w = sub_g["FINALWT"].to_numpy(dtype=float)
            neff = float((w.sum() ** 2) / np.dot(w, w))
            target_m = 2 if (len(w) >= 20 and neff >= 20.0) else 1
            already = sub_g[sub_g["RESPID_INT"].isin(chosen_respids)]
            if len(already) < target_m:
                need = target_m - len(already)
                for _ in range(need):
                    cur_panel = elig_human[elig_human["RESPID_INT"].isin(chosen_respids)]
                    c2_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"]))
                    c3m_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["MARITAL"]))
                    c3e_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["EDUC"]))
                    c4_tuples = set(zip(cur_panel["SEX"], cur_panel["AGE"], cur_panel["EDUC"], cur_panel["MARITAL"]))
                    rem_g = sub_g[~sub_g["RESPID_INT"].isin(chosen_respids)].copy()
                    rem_g["overlap_score"] = [
                        1 * int((r.SEX, r.AGE) in c2_tuples)
                        + 2 * int((r.SEX, r.AGE, r.MARITAL) in c3m_tuples)
                        + 2 * int((r.SEX, r.AGE, r.EDUC) in c3e_tuples)
                        + 3 * int((r.SEX, r.AGE, r.EDUC, r.MARITAL) in c4_tuples)
                        for _, r in rem_g.iterrows()
                    ]
                    rem_g = rem_g.sort_values(["overlap_score", "FINALWT"], ascending=[False, False])
                    chosen_respids.add(int(rem_g.iloc[0]["RESPID_INT"]))

    panel = elig_human[elig_human["RESPID_INT"].isin(chosen_respids)].sort_values("RESPID_INT").reset_index(drop=True).copy()
    panel.insert(0, "expanded_persona_id", range(len(panel)))
    panel["in_original_panel40"] = panel["RESPID_INT"].isin(p40_respids)

    # Compute exact C4_EM stratum Horvitz-Thompson / post-stratification weight:
    # Within each C4_EM cell s represented in the panel, respondents i in s receive
    # weight proportional to their FINALWT_i normalized to sum to the true KGSS cell weight W_KGSS(s).
    c4_pop_wt = elig_human.groupby(c4_cols, observed=True)["FINALWT"].sum().to_dict()
    c4_panel_wt = panel.groupby(c4_cols, observed=True)["FINALWT"].sum().to_dict()

    ht_weights: list[float] = []
    cell_sample_counts = panel.groupby(c4_cols, observed=True)["RESPID_INT"].count().to_dict()
    cell_pop_counts = elig_human.groupby(c4_cols, observed=True)["RESPID_INT"].count().to_dict()
    inclusion_probs: list[float] = []

    for _, r in panel.iterrows():
        ck = tuple(r[c] for c in c4_cols)
        pop_w = float(c4_pop_wt[ck])
        smp_w = float(c4_panel_wt[ck])
        ht_w = float(r["FINALWT"]) * (pop_w / smp_w) if smp_w > 0 else float(r["FINALWT"])
        ht_weights.append(ht_w)
        inclusion_probs.append(float(cell_sample_counts[ck]) / float(cell_pop_counts[ck]))

    panel["c4_stratum_ht_weight"] = ht_weights
    panel["c4_stratum_inclusion_prob"] = inclusion_probs

    # Attach human-readable labels for all 6 variables
    for v in ("SEX", "EDUC", "MARITAL", "REGION", "URBAN"):
        panel[f"{v}_label"] = [labels.get(v, {}).get(str(float(val)), "") for val in panel[v]]

    return panel


def compute_cell_poststrat_weights(
    panel_df: pd.DataFrame,
    human_df: pd.DataFrame,
    cell_cols: Sequence[str],
) -> np.ndarray:
    """
    Compute exact KGSS post-stratification weights for each row in `panel_df` with respect
    to stratum columns `cell_cols` (e.g. the union of condition band variables and target
    band variables).

    Within each observed cell `s` defined by `cell_cols`, panel respondents `i in s` receive
    weight proportional to `FINALWT_i` scaled so that the cell's total weight equals the
    true KGSS population weight `W_KGSS(s) = sum_{r in human_df, r in s} FINALWT_r`.
    When normalized within any coarser target group `g`, the weights of sub-cells `z` inside
    `g` equal the true KGSS conditional composition `P_KGSS(z | g) = W_KGSS(g, z) / W_KGSS(g)`.
    """
    cols = list(cell_cols)
    if not cols:
        w = panel_df["FINALWT"].to_numpy(dtype=float)
        return w / float(w.sum())

    h_df = human_df.copy()
    if "AGE_GRP" in cols and "AGE_GRP" not in h_df.columns:
        h_df["AGE_GRP"] = pd.cut(h_df["AGE"], bins=AGE_BINS_5, labels=AGE_LABELS_5, right=False).astype(str)

    p_df = panel_df.copy()
    if "AGE_GRP" in cols and "AGE_GRP" not in p_df.columns:
        p_df["AGE_GRP"] = pd.cut(p_df["AGE"], bins=AGE_BINS_5, labels=AGE_LABELS_5, right=False).astype(str)

    valid_mask = np.isfinite(h_df["FINALWT"].to_numpy(dtype=float)) & (h_df["FINALWT"].to_numpy(dtype=float) > 0.0)
    for c in cols:
        if c != "AGE_GRP":
            valid_mask &= h_df[c].notna() & (~h_df[c].isin([-1.0, -8.0, 88.0, 99.0]))
        else:
            valid_mask &= h_df["AGE"].notna() & (~h_df["AGE"].isin([-1.0, -8.0, 88.0, 99.0]))
    elig_h = h_df[valid_mask]

    pop_wt = elig_h.groupby(cols, observed=True)["FINALWT"].sum().to_dict()
    smp_wt = p_df.groupby(cols, observed=True)["FINALWT"].sum().to_dict()

    out = np.zeros(len(p_df), dtype=float)
    for pos, (_, row) in enumerate(p_df.iterrows()):
        key = tuple(row[c] for c in cols) if len(cols) > 1 else row[cols[0]]
        pw = float(pop_wt.get(key, row["FINALWT"]))
        sw = float(smp_wt.get(key, row["FINALWT"]))
        fwt = float(row["FINALWT"])
        out[pos] = fwt * (pw / sw) if sw > 0.0 else fwt

    return out

