"""Strict empirical KGSS references for the attributes actually in a persona.

The caller selects the survey year before calling this module. Every supplied
attribute is matched exactly; sparse groups are reported rather than widened.
References include the source respondent and are empirical benchmarks, not
holdout estimates of the conditional population distribution.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from kgss_metrics import entropy_bits, tvd_distance, validate_prob_dist, wasserstein_1d


MATCHING_KEYS = ("SEX", "AGE", "AGE_GRP", "EDUC", "MARITAL", "REGION", "URBAN")
AGE_GROUPS = ("18-29", "30-39", "40-49", "50-59", "60+")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value.")
    return number


def _validated_codes(codes: Sequence[float | int]) -> list[float]:
    values = [_finite_number(code, "Response code") for code in codes]
    if not values or len(values) != len(set(values)):
        raise ValueError("Response codes must be nonempty and unique.")
    return values


def build_matched_human_reference(
    human_df: pd.DataFrame,
    var: str,
    codes: Sequence[float | int],
    matching_conditions: Mapping[str, float | int | str],
    *,
    weight_col: str = "FINALWT",
    min_valid_n: int = 5,
    min_effective_n: float = 5.0,
) -> dict:
    """Build a weighted item distribution for an exact persona match.

    ``matched_n`` counts matching respondents before response or weight filters.
    ``valid_n`` excludes invalid item codes and nonfinite/nonpositive weights.
    A nonempty sparse group's empirical distribution is retained for diagnostics;
    only groups meeting both sample thresholds are eligible for primary metrics.
    An empty condition dictionary explicitly requests the full supplied frame.
    ``AGE_GRP`` can explicitly request one of the five declared age bands instead
    of exact ``AGE``. The caller constructs this column; the two conditions cannot
    be combined and the scorer never substitutes an age band for an exact age.
    """
    if not isinstance(matching_conditions, Mapping):
        raise ValueError("matching_conditions must be a dictionary-like mapping.")
    unknown = set(matching_conditions) - set(MATCHING_KEYS)
    if unknown:
        raise ValueError(f"Unknown matching fields: {sorted(unknown, key=str)}")
    if "AGE" in matching_conditions and "AGE_GRP" in matching_conditions:
        raise ValueError("AGE and AGE_GRP cannot both be matching conditions.")
    if isinstance(min_valid_n, (bool, np.bool_)) or not isinstance(min_valid_n, (int, np.integer)) or min_valid_n < 1:
        raise ValueError("min_valid_n must be a positive integer.")
    effective_threshold = _finite_number(min_effective_n, "min_effective_n")
    if effective_threshold < 0:
        raise ValueError("min_effective_n must be nonnegative.")
    code_values = _validated_codes(codes)
    matching_keys = [key for key in MATCHING_KEYS if key in matching_conditions]
    conditions = {}
    for key in matching_keys:
        if key == "AGE_GRP":
            value = matching_conditions[key]
            if not isinstance(value, str) or value not in AGE_GROUPS:
                raise ValueError(f"AGE_GRP must be one of {AGE_GROUPS}.")
            conditions[key] = str(value)
        else:
            conditions[key] = _finite_number(matching_conditions[key], f"Matching condition {key}")
    required_columns = [var, weight_col, *matching_keys]
    missing = [column for column in required_columns if column not in human_df.columns]
    if missing:
        raise ValueError(f"Missing human reference columns: {missing}")
    if not human_df.columns.is_unique:
        raise ValueError("Human reference column names must be unique.")

    mask = pd.Series(True, index=human_df.index)
    for key, value in conditions.items():
        column = human_df[key] if key == "AGE_GRP" else pd.to_numeric(human_df[key], errors="coerce")
        mask &= column.eq(value).fillna(False)
    matched = human_df.loc[mask]
    matched_n = int(len(matched))
    responses = pd.to_numeric(matched[var], errors="coerce")
    weights = pd.to_numeric(matched[weight_col], errors="coerce")
    valid_mask = responses.isin(code_values) & np.isfinite(weights) & weights.gt(0)
    valid_responses = responses.loc[valid_mask].to_numpy(dtype=float)
    valid_weights = weights.loc[valid_mask].to_numpy(dtype=float)
    valid_n = int(len(valid_weights))
    total_weight = 0.0
    effective_n = 0.0
    human_dist = None
    if valid_n:
        try:
            total_weight = float(math.fsum(valid_weights))
        except OverflowError as exc:
            raise ValueError("Total human reference weight exceeds finite numeric range.") from exc
        if not math.isfinite(total_weight):
            raise ValueError("Total human reference weight exceeds finite numeric range.")
        # Scaling preserves the distribution and effective n without squaring
        # large survey weights or underflowing uniformly small weights.
        scaled_weights = valid_weights / float(valid_weights.max())
        scaled_total = float(scaled_weights.sum())
        effective_n = float(scaled_total**2 / np.dot(scaled_weights, scaled_weights))
        human_dist = [
            float(scaled_weights[valid_responses == code].sum() / scaled_total)
            for code in code_values
        ]

    n_ok = valid_n >= int(min_valid_n)
    effective_ok = effective_n >= effective_threshold
    eligible = bool(valid_n > 0 and n_ok and effective_ok)
    if matched_n == 0:
        reason = "empty_match"
    elif valid_n == 0:
        reason = "no_valid_response"
    elif not n_ok and not effective_ok:
        reason = "insufficient_valid_n_and_effective_n"
    elif not n_ok:
        reason = "insufficient_valid_n"
    elif not effective_ok:
        reason = "insufficient_effective_n"
    else:
        reason = None

    age_policy = "exact_age" if "AGE" in conditions else "age_band" if "AGE_GRP" in conditions else "not_conditioned"
    return {
        "var": str(var),
        "codes": code_values,
        "matching_keys": matching_keys,
        "matching_conditions": conditions,
        "matched_n": matched_n,
        "valid_n": valid_n,
        "total_weight": total_weight,
        "effective_n": effective_n,
        "human_dist": human_dist,
        "eligible": eligible,
        "reason": reason,
        "metadata": {
            "matching_rule": "exact_attribute_equality",
            "age_policy": age_policy,
            "fallback": None,
            "year_filter": "caller_responsibility",
            "weight_col": str(weight_col),
            "min_valid_n": int(min_valid_n),
            "min_effective_n": effective_threshold,
            "threshold_source": "existing_cell_minimum_n_5_with_effective_n_requirement",
            "reference_type": "empirical_benchmark",
            "include_source_respondent": True,
            "holdout": False,
            "codes_order": "caller_supplied_category_order",
        },
    }


def evaluate_matched_distribution(
    pred_dist: Sequence[float],
    reference: Mapping,
    *,
    w1_valid: bool = True,
) -> dict:
    """Compare one model distribution with its matching human reference.

    Primary metrics are absent when the human group is ineligible. Diagnostic
    metrics preserve sparse empirical comparisons, including a single person.
    Nominal items must pass ``w1_valid=False``; their W1 values remain ``None``.
    W1 uses the supplied categories' ordinal positions and normalized W1 divides
    by K-1; the normalized value is undefined for a single-category item.
    """
    if not isinstance(reference, Mapping):
        raise ValueError("reference must be a dictionary-like mapping.")
    required = ("codes", "human_dist", "eligible", "reason")
    missing = [key for key in required if key not in reference]
    if missing:
        raise ValueError(f"Missing reference fields: {missing}")
    if not isinstance(w1_valid, (bool, np.bool_)):
        raise ValueError("w1_valid must be boolean.")
    if not isinstance(reference["eligible"], (bool, np.bool_)):
        raise ValueError("Reference eligibility must be boolean.")
    codes = _validated_codes(reference["codes"])
    prediction = validate_prob_dist(pred_dist)
    if len(prediction) != len(codes):
        raise ValueError("Prediction length must equal the number of response codes.")
    model_entropy = entropy_bits(prediction)
    diagnostic = None
    human_dist = reference["human_dist"]
    if human_dist is not None:
        human = validate_prob_dist(human_dist)
        if len(human) != len(codes):
            raise ValueError("Human distribution length must equal the number of response codes.")
        human_entropy = entropy_bits(human)
        w1 = wasserstein_1d(prediction, human) if w1_valid else None
        diagnostic = {
            "tvd": tvd_distance(prediction, human),
            "w1": w1,
            "normalized_w1": w1 / (len(codes) - 1) if w1 is not None and len(codes) > 1 else None,
            "human_entropy_bits": human_entropy,
            "model_entropy_bits": model_entropy,
            "entropy_gap_bits": model_entropy - human_entropy,
        }
    if reference["eligible"] and diagnostic is None:
        raise ValueError("An eligible reference must have a human distribution.")
    return {
        "eligible": bool(reference["eligible"]),
        "reason": reference["reason"],
        "primary": dict(diagnostic) if reference["eligible"] else None,
        "raw_diagnostic": diagnostic,
        "model_dist": prediction.tolist(),
        "model_entropy_bits": model_entropy,
    }


def check_group_support(
    valid_n: int,
    effective_n: float,
    *,
    matched_n: int | None = None,
    support_policy: str = "n_and_effective_n",
    threshold: int = 5,
) -> tuple[bool, str | None]:
    """Evaluate group support eligibility for a given support_policy and threshold T."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, np.integer)) or int(threshold) < 1:
        raise ValueError("threshold must be a positive integer.")
    t_val = int(threshold)
    v_n = int(valid_n)
    eff_n = float(effective_n)
    m_n = v_n if matched_n is None else int(matched_n)
    if support_policy not in ("n_only", "n_and_effective_n"):
        raise ValueError(f"Unknown support_policy: {support_policy!r}")
    if m_n == 0:
        return False, "empty_match"
    if v_n == 0:
        return False, "no_valid_response"
    n_ok = v_n >= t_val
    if support_policy == "n_only":
        return (True, None) if n_ok else (False, "insufficient_valid_n")
    eff_ok = eff_n >= float(t_val)
    if n_ok and eff_ok:
        return True, None
    if not n_ok and not eff_ok:
        return False, "insufficient_valid_n_and_effective_n"
    if not n_ok:
        return False, "insufficient_valid_n"
    return False, "insufficient_effective_n"


def evaluate_group_conditional_distribution(
    pred_dists: Sequence[Sequence[float]],
    reference: Mapping,
    *,
    w1_valid: bool = True,
    thresholds: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict:
    """Pre-aggregate model distributions within a human group c, then compute distances to Q(v, c).

    Each element of ``pred_dists`` is one original PPS draw in group ``c`` and receives
    equal inner weight ``1 / m_c``. Also computes the average individual row distance
    ``(1 / m_c) * sum_i d(P_i, Q(v, c))`` so pre-aggregation effects can be isolated
    against row-level diagnostics.
    """
    if not isinstance(reference, Mapping):
        raise ValueError("reference must be a dictionary-like mapping.")
    required = ("codes", "human_dist", "matched_n", "valid_n", "total_weight", "effective_n")
    missing = [key for key in required if key not in reference]
    if missing:
        raise ValueError(f"Missing reference fields: {missing}")
    if not isinstance(w1_valid, (bool, np.bool_)):
        raise ValueError("w1_valid must be boolean.")
    if not pred_dists:
        raise ValueError("pred_dists must contain at least one model draw.")
    codes = _validated_codes(reference["codes"])
    n_cats = len(codes)

    validated_draws = []
    draw_entropies = []
    for dist in pred_dists:
        p = validate_prob_dist(dist)
        if len(p) != n_cats:
            raise ValueError("Each prediction length must equal the number of response codes.")
        validated_draws.append(p)
        draw_entropies.append(entropy_bits(p))

    m_c = len(validated_draws)
    draws_matrix = np.vstack(validated_draws)
    pbar = validate_prob_dist(np.mean(draws_matrix, axis=0))
    model_group_entropy = entropy_bits(pbar)
    model_draw_cond_entropy = float(np.mean(draw_entropies))
    within_group_mi = float(max(0.0, model_group_entropy - model_draw_cond_entropy))

    matched_n = int(reference["matched_n"])
    valid_n = int(reference["valid_n"])
    total_weight = float(reference["total_weight"])
    effective_n = float(reference["effective_n"])
    human_dist = reference["human_dist"]

    metrics = {
        "group_tvd": None,
        "group_w1": None,
        "group_normalized_w1": None,
        "row_mean_tvd_in_group": None,
        "row_mean_w1_in_group": None,
        "row_mean_normalized_w1_in_group": None,
        "uniform_tvd": None,
        "uniform_w1": None,
        "uniform_normalized_w1": None,
        "human_group_entropy_bits": None,
        "model_group_entropy_bits": model_group_entropy,
        "group_entropy_gap_bits": None,
        "model_draw_cond_entropy_bits": model_draw_cond_entropy,
        "within_group_mi_bits": within_group_mi,
    }

    if human_dist is not None and valid_n > 0:
        human = validate_prob_dist(human_dist)
        if len(human) != n_cats:
            raise ValueError("Human distribution length must equal the number of response codes.")
        human_entropy = entropy_bits(human)
        g_tvd = tvd_distance(pbar, human)
        g_w1 = wasserstein_1d(pbar, human) if w1_valid else None
        g_nw1 = g_w1 / (n_cats - 1) if g_w1 is not None and n_cats > 1 else None

        row_tvds = [tvd_distance(p, human) for p in validated_draws]
        r_tvd = float(np.mean(row_tvds))
        if w1_valid:
            row_w1s = [wasserstein_1d(p, human) for p in validated_draws]
            r_w1 = float(np.mean(row_w1s))
            r_nw1 = r_w1 / (n_cats - 1) if n_cats > 1 else None
        else:
            r_w1 = None
            r_nw1 = None

        uniform = np.full(n_cats, 1.0 / n_cats, dtype=float)
        u_tvd = tvd_distance(uniform, human)
        u_w1 = wasserstein_1d(uniform, human) if w1_valid else None
        u_nw1 = u_w1 / (n_cats - 1) if u_w1 is not None and n_cats > 1 else None

        metrics.update({
            "group_tvd": g_tvd,
            "group_w1": g_w1,
            "group_normalized_w1": g_nw1,
            "row_mean_tvd_in_group": r_tvd,
            "row_mean_w1_in_group": r_w1,
            "row_mean_normalized_w1_in_group": r_nw1,
            "uniform_tvd": u_tvd,
            "uniform_w1": u_w1,
            "uniform_normalized_w1": u_nw1,
            "human_group_entropy_bits": human_entropy,
            "group_entropy_gap_bits": model_group_entropy - human_entropy,
        })

    support_by_threshold = {}
    for t_val in thresholds:
        el_n, rsn_n = check_group_support(
            valid_n, effective_n, matched_n=matched_n, support_policy="n_only", threshold=int(t_val)
        )
        el_ne, rsn_ne = check_group_support(
            valid_n, effective_n, matched_n=matched_n, support_policy="n_and_effective_n", threshold=int(t_val)
        )
        support_by_threshold[int(t_val)] = {
            "n_only": {"eligible": el_n, "reason": rsn_n},
            "n_and_effective_n": {"eligible": el_ne, "reason": rsn_ne},
        }

    return {
        "matched_n": matched_n,
        "valid_n": valid_n,
        "total_weight": total_weight,
        "effective_n": effective_n,
        "n_model_draws": m_c,
        "human_dist": list(human_dist) if human_dist is not None else None,
        "model_group_dist": pbar.tolist(),
        **metrics,
        "support_by_threshold": support_by_threshold,
    }


def aggregate_item_group_metrics(
    group_records: Sequence[Mapping],
    *,
    support_policy: str = "n_and_effective_n",
    threshold: int = 5,
    w1_valid: bool = True,
    item_kgss_total_weight: float | None = None,
) -> dict:
    """Aggregate group-level conditional distances across supported groups c for one item v.

    - Outer human group weights: ``alpha(v, c) = W(v, c) / sum_{c' in C_{v,T}} W(v, c')``
      where ``W(v, c)`` is the group's human ``FINALWT`` sum (applied once per unique group c).
    - Also computes equal-group-weighted ``1 / |C_{v,T}|`` and draw-proportional
      ``m_c / sum_{c'} m_{c'}`` aggregates for both pre-aggregated group distances
      and within-group row-mean distances.
    """
    if not group_records:
        raise ValueError("group_records must be non-empty.")
    seen_ids = set()
    for rec in group_records:
        gid = rec.get("reference_id") or json.dumps(rec.get("matching_conditions", {}), sort_keys=True)
        if gid in seen_ids:
            raise ValueError(f"Duplicate human group {gid!r} in item aggregation; groups must be unique.")
        seen_ids.add(gid)

    n_observed_groups = len(group_records)
    n_observed_draws = int(sum(int(r["n_model_draws"]) for r in group_records))

    included = []
    reason_counts: dict[str, int] = {}
    for rec in group_records:
        elig, reason = check_group_support(
            int(rec["valid_n"]),
            float(rec["effective_n"]),
            matched_n=int(rec["matched_n"]),
            support_policy=support_policy,
            threshold=int(threshold),
        )
        if elig:
            if rec.get("group_tvd") is None or float(rec["total_weight"]) <= 0:
                raise ValueError("Eligible group must have positive total_weight and non-null group_tvd.")
            included.append(rec)
        else:
            key = reason or "ineligible"
            reason_counts[key] = reason_counts.get(key, 0) + 1

    n_included_groups = len(included)
    n_excluded_groups = n_observed_groups - n_included_groups
    n_included_draws = int(sum(int(r["n_model_draws"]) for r in included))
    included_human_weight = float(math.fsum(float(r["total_weight"]) for r in included)) if included else 0.0
    coverage = (
        included_human_weight / float(item_kgss_total_weight)
        if item_kgss_total_weight is not None and float(item_kgss_total_weight) > 0
        else None
    )

    base_info = {
        "support_policy": support_policy,
        "threshold": int(threshold),
        "n_observed_groups": n_observed_groups,
        "n_included_groups": n_included_groups,
        "n_excluded_groups": n_excluded_groups,
        "n_observed_draws": n_observed_draws,
        "n_included_draws": n_included_draws,
        "included_human_weight_sum": included_human_weight,
        "item_kgss_total_weight": float(item_kgss_total_weight) if item_kgss_total_weight is not None else None,
        "human_weight_coverage_fraction": coverage,
        "excluded_reason_counts": reason_counts,
    }

    metric_fields = (
        "preagg_tvd_human_wt", "preagg_tvd_group_eq", "preagg_tvd_draw_wt",
        "row_tvd_human_wt", "row_tvd_group_eq", "row_tvd_draw_wt",
        "uniform_tvd_human_wt", "uniform_tvd_group_eq", "uniform_tvd_draw_wt",
        "preagg_w1_human_wt", "preagg_w1_group_eq", "preagg_w1_draw_wt",
        "row_w1_human_wt", "row_w1_group_eq", "row_w1_draw_wt",
        "uniform_w1_human_wt", "uniform_w1_group_eq", "uniform_w1_draw_wt",
        "preagg_normalized_w1_human_wt", "preagg_normalized_w1_group_eq", "preagg_normalized_w1_draw_wt",
        "row_normalized_w1_human_wt", "row_normalized_w1_group_eq", "row_normalized_w1_draw_wt",
        "uniform_normalized_w1_human_wt", "uniform_normalized_w1_group_eq", "uniform_normalized_w1_draw_wt",
        "human_group_entropy_human_wt", "model_group_entropy_human_wt",
        "group_entropy_gap_human_wt", "model_draw_cond_entropy_human_wt", "within_group_mi_human_wt",
        "human_group_entropy_group_eq", "model_group_entropy_group_eq",
        "group_entropy_gap_group_eq", "within_group_mi_group_eq",
    )

    if n_included_groups == 0:
        return {
            **base_info,
            "item_evaluable": False,
            "na_reason": "no_group_meets_threshold",
            "alpha_weights": [],
            **{k: None for k in metric_fields},
        }

    w_vec = np.array([float(r["total_weight"]) for r in included], dtype=float)
    alpha = w_vec / float(w_vec.sum())
    beta = np.full(n_included_groups, 1.0 / n_included_groups, dtype=float)
    m_vec = np.array([float(r["n_model_draws"]) for r in included], dtype=float)
    gamma = m_vec / float(m_vec.sum())

    def _agg(col: str) -> tuple[float, float, float]:
        vals = np.array([float(r[col]) for r in included], dtype=float)
        return float(np.dot(alpha, vals)), float(np.dot(beta, vals)), float(np.dot(gamma, vals))

    pt_h, pt_e, pt_d = _agg("group_tvd")
    rt_h, rt_e, rt_d = _agg("row_mean_tvd_in_group")
    ut_h, ut_e, ut_d = _agg("uniform_tvd")

    if w1_valid:
        pw_h, pw_e, pw_d = _agg("group_w1")
        rw_h, rw_e, rw_d = _agg("row_mean_w1_in_group")
        uw_h, uw_e, uw_d = _agg("uniform_w1")
        pnw_h, pnw_e, pnw_d = _agg("group_normalized_w1")
        rnw_h, rnw_e, rnw_d = _agg("row_mean_normalized_w1_in_group")
        unw_h, unw_e, unw_d = _agg("uniform_normalized_w1")
    else:
        pw_h = pw_e = pw_d = None
        rw_h = rw_e = rw_d = None
        uw_h = uw_e = uw_d = None
        pnw_h = pnw_e = pnw_d = None
        rnw_h = rnw_e = rnw_d = None
        unw_h = unw_e = unw_d = None

    he_h, he_e, _ = _agg("human_group_entropy_bits")
    me_h, me_e, _ = _agg("model_group_entropy_bits")
    ge_h, ge_e, _ = _agg("group_entropy_gap_bits")
    ce_h, _, _ = _agg("model_draw_cond_entropy_bits")
    mi_h, mi_e, _ = _agg("within_group_mi_bits")

    return {
        **base_info,
        "item_evaluable": True,
        "na_reason": None,
        "alpha_weights": alpha.tolist(),
        "preagg_tvd_human_wt": pt_h,
        "preagg_tvd_group_eq": pt_e,
        "preagg_tvd_draw_wt": pt_d,
        "row_tvd_human_wt": rt_h,
        "row_tvd_group_eq": rt_e,
        "row_tvd_draw_wt": rt_d,
        "uniform_tvd_human_wt": ut_h,
        "uniform_tvd_group_eq": ut_e,
        "uniform_tvd_draw_wt": ut_d,
        "preagg_w1_human_wt": pw_h,
        "preagg_w1_group_eq": pw_e,
        "preagg_w1_draw_wt": pw_d,
        "row_w1_human_wt": rw_h,
        "row_w1_group_eq": rw_e,
        "row_w1_draw_wt": rw_d,
        "uniform_w1_human_wt": uw_h,
        "uniform_w1_group_eq": uw_e,
        "uniform_w1_draw_wt": uw_d,
        "preagg_normalized_w1_human_wt": pnw_h,
        "preagg_normalized_w1_group_eq": pnw_e,
        "preagg_normalized_w1_draw_wt": pnw_d,
        "row_normalized_w1_human_wt": rnw_h,
        "row_normalized_w1_group_eq": rnw_e,
        "row_normalized_w1_draw_wt": rnw_d,
        "uniform_normalized_w1_human_wt": unw_h,
        "uniform_normalized_w1_group_eq": unw_e,
        "uniform_normalized_w1_draw_wt": unw_d,
        "human_group_entropy_human_wt": he_h,
        "model_group_entropy_human_wt": me_h,
        "group_entropy_gap_human_wt": ge_h,
        "model_draw_cond_entropy_human_wt": ce_h,
        "within_group_mi_human_wt": mi_h,
        "human_group_entropy_group_eq": he_e,
        "model_group_entropy_group_eq": me_e,
        "group_entropy_gap_group_eq": ge_e,
        "within_group_mi_group_eq": mi_e,
    }

