from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kgss_persona_matching import (
    aggregate_item_group_metrics,
    build_matched_human_reference,
    check_group_support,
    evaluate_group_conditional_distribution,
)


REPO = Path(__file__).resolve().parent.parent
GROUP_EVAL_DIR = REPO / "reexperiment_20261009" / "group_conditional_eval_20261010"
THRESH_DIR = REPO / "reexperiment_20261009" / "threshold_sensitivity_20261010"


class GroupConditionalSyntheticTests(unittest.TestCase):
    def test_preaggregation_vs_row_distance_counterexample(self):
        """Protocol Section 8 Check 5: [1,0] and [0,1] in same group vs human [0.5, 0.5]."""
        profile = {"SEX": 1.0, "AGE": 35.0, "EDUC": 5.0, "MARITAL": 1.0, "REGION": 1.0, "URBAN": 1.0}
        human_df = pd.DataFrame(
            [{**profile, "Q": 1.0, "FINALWT": 1.0} for _ in range(5)]
            + [{**profile, "Q": 2.0, "FINALWT": 1.0} for _ in range(5)]
        )
        ref = build_matched_human_reference(human_df, "Q", [1.0, 2.0], profile)
        ev = evaluate_group_conditional_distribution([[1.0, 0.0], [0.0, 1.0]], ref, w1_valid=True)

        self.assertEqual(ev["n_model_draws"], 2)
        self.assertTrue(np.allclose(ev["model_group_dist"], [0.5, 0.5]))
        self.assertTrue(np.allclose(ev["human_dist"], [0.5, 0.5]))
        self.assertAlmostEqual(ev["group_tvd"], 0.0, places=12)
        self.assertAlmostEqual(ev["group_w1"], 0.0, places=12)
        self.assertAlmostEqual(ev["row_mean_tvd_in_group"], 0.5, places=12)
        self.assertAlmostEqual(ev["row_mean_w1_in_group"], 0.5, places=12)
        self.assertAlmostEqual(ev["within_group_mi_bits"], 1.0, places=12)

    def test_additional_four_full_variables_each_change_reference_and_score(self):
        """Protocol Section 8 Check 3: EDUC, MARITAL, REGION, URBAN each change human group Q(v,c)."""
        base = {"SEX": 1.0, "AGE": 40.0, "EDUC": 4.0, "MARITAL": 1.0, "REGION": 1.0, "URBAN": 1.0}
        for key in ("EDUC", "MARITAL", "REGION", "URBAN"):
            alt = {**base, key: base[key] + 1.0}
            df = pd.DataFrame(
                [{**base, "Q": 1.0, "FINALWT": 1.0} for _ in range(5)]
                + [{**alt, "Q": 2.0, "FINALWT": 2.0} for _ in range(5)]
            )
            ref_base = build_matched_human_reference(df, "Q", [1.0, 2.0], base)
            ref_alt = build_matched_human_reference(df, "Q", [1.0, 2.0], alt)
            ev_base = evaluate_group_conditional_distribution([[0.9, 0.1]], ref_base, w1_valid=True)
            ev_alt = evaluate_group_conditional_distribution([[0.9, 0.1]], ref_alt, w1_valid=True)
            self.assertAlmostEqual(ev_base["group_tvd"], 0.1, places=12)
            self.assertAlmostEqual(ev_alt["group_tvd"], 0.9, places=12)

    def test_conditional_oracle_vs_population_marginal_copy_model(self):
        """Protocol Section 8 Check 4: Conditional oracle achieves 0; marginal copy achieves >0."""
        p1 = {"SEX": 1.0, "AGE": 35.0, "EDUC": 5.0, "MARITAL": 1.0, "REGION": 1.0, "URBAN": 1.0}
        p2 = {**p1, "EDUC": 3.0}
        df = pd.DataFrame(
            [{**p1, "Q": 1.0, "FINALWT": 1.0} for _ in range(5)]
            + [{**p2, "Q": 2.0, "FINALWT": 3.0} for _ in range(5)]
        )
        ref1 = {**build_matched_human_reference(df, "Q", [1.0, 2.0], p1), "reference_id": "g1"}
        ref2 = {**build_matched_human_reference(df, "Q", [1.0, 2.0], p2), "reference_id": "g2"}
        ref_pop = build_matched_human_reference(df, "Q", [1.0, 2.0], {})

        # Population marginal is [5/20, 15/20] = [0.25, 0.75]
        oracle_item = aggregate_item_group_metrics(
            [
                {**evaluate_group_conditional_distribution([[1.0, 0.0]], ref1), "reference_id": "g1"},
                {**evaluate_group_conditional_distribution([[0.0, 1.0]], ref2), "reference_id": "g2"},
            ],
            support_policy="n_and_effective_n",
            threshold=5,
        )
        popcopy_item = aggregate_item_group_metrics(
            [
                {**evaluate_group_conditional_distribution([[0.25, 0.75]], ref1), "reference_id": "g1"},
                {**evaluate_group_conditional_distribution([[0.25, 0.75]], ref2), "reference_id": "g2"},
            ],
            support_policy="n_and_effective_n",
            threshold=5,
        )
        popcopy_marginal = evaluate_group_conditional_distribution([[0.25, 0.75]], ref_pop)

        self.assertAlmostEqual(oracle_item["preagg_tvd_human_wt"], 0.0, places=12)
        self.assertAlmostEqual(oracle_item["preagg_w1_human_wt"], 0.0, places=12)
        self.assertAlmostEqual(popcopy_marginal["group_tvd"], 0.0, places=12)
        # Human weights: g1 has total_weight=5 (alpha=0.25, TVD=0.75), g2 has total_weight=15 (alpha=0.75, TVD=0.25) -> 0.375
        self.assertAlmostEqual(popcopy_item["preagg_tvd_human_wt"], 0.375, places=12)
        self.assertAlmostEqual(popcopy_item["preagg_tvd_group_eq"], 0.5, places=12)

    def test_support_rules_and_outer_human_weight_once_per_unique_group(self):
        """Protocol Section 8 Checks 6 & 7: n_only vs n_and_effective_n, N/A when no groups, and unique group alpha."""
        p1 = {"SEX": 1.0, "AGE": 30.0}
        p2 = {"SEX": 2.0, "AGE": 30.0}
        # Group 1 has 5 respondents with skewed weights [10, 1, 1, 1, 1] -> valid_n=5, eff_n = 196 / 104 = 1.8846 < 5
        # Group 2 has 5 respondents with equal weights [2, 2, 2, 2, 2] -> valid_n=5, eff_n = 5.0 >= 5
        df = pd.DataFrame(
            [{"SEX": 1.0, "AGE": 30.0, "Q": 1.0, "FINALWT": 10.0}]
            + [{"SEX": 1.0, "AGE": 30.0, "Q": 2.0, "FINALWT": 1.0} for _ in range(4)]
            + [{"SEX": 2.0, "AGE": 30.0, "Q": 1.0, "FINALWT": 2.0} for _ in range(5)]
        )
        ref1 = {**build_matched_human_reference(df, "Q", [1.0, 2.0], p1), "reference_id": "g1"}
        ref2 = {**build_matched_human_reference(df, "Q", [1.0, 2.0], p2), "reference_id": "g2"}

        self.assertEqual(
            check_group_support(
                ref1["valid_n"],
                ref1["effective_n"],
                matched_n=ref1["matched_n"],
                support_policy="n_only",
                threshold=5,
            ),
            (True, None),
        )
        self.assertEqual(
            check_group_support(
                ref1["valid_n"],
                ref1["effective_n"],
                matched_n=ref1["matched_n"],
                support_policy="n_and_effective_n",
                threshold=5,
            ),
            (False, "insufficient_effective_n"),
        )

        # Duplicate draws in group 1 (m_1 = 3 draws) vs single draw in group 2 (m_2 = 1 draw):
        # Outer human weight alpha must use W(v, g1)=14 once and W(v, g2)=10 once -> alpha = [14/24, 10/24]
        ev1 = {
            **evaluate_group_conditional_distribution([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], ref1),
            "reference_id": "g1",
        }
        ev2 = {
            **evaluate_group_conditional_distribution([[1.0, 0.0]], ref2),
            "reference_id": "g2",
        }
        agg_n_only = aggregate_item_group_metrics([ev1, ev2], support_policy="n_only", threshold=5)
        self.assertEqual(agg_n_only["n_included_groups"], 2)
        self.assertEqual(agg_n_only["n_included_draws"], 4)
        self.assertTrue(np.allclose(agg_n_only["alpha_weights"], [14.0 / 24.0, 10.0 / 24.0]))

        agg_n_eff = aggregate_item_group_metrics([ev1, ev2], support_policy="n_and_effective_n", threshold=5)
        self.assertEqual(agg_n_eff["n_included_groups"], 1)
        self.assertTrue(np.allclose(agg_n_eff["alpha_weights"], [1.0]))

        agg_t10 = aggregate_item_group_metrics([ev1, ev2], support_policy="n_only", threshold=10)
        self.assertFalse(agg_t10["item_evaluable"])
        self.assertEqual(agg_t10["na_reason"], "no_group_meets_threshold")
        self.assertIsNone(agg_t10["preagg_tvd_human_wt"])


class GroupConditionalRealOutputsTests(unittest.TestCase):
    def test_required_outputs_and_verification_json(self):
        required = (
            "group_item_metrics.csv",
            "item_metrics.csv",
            "threshold_summary.csv",
            "threshold_summary.json",
            "paired_differences.csv",
            "paired_differences.json",
            "decomposition.csv",
            "decomposition.json",
            "manifest.json",
            "verification.json",
            "GROUP_CONDITIONAL_REPORT.md",
        )
        for fname in required:
            self.assertTrue((GROUP_EVAL_DIR / fname).exists(), f"Missing {fname}")

        manifest = json.loads((GROUP_EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["exit_status"], 0)
        self.assertEqual(manifest["new_model_inference_calls"], 0)
        self.assertFalse(manifest["matching_conditions_relaxed"])

        ver = json.loads((GROUP_EVAL_DIR / "verification.json").read_text(encoding="utf-8"))
        for k, v in ver.items():
            if isinstance(v, dict) and "passed" in v:
                self.assertTrue(v["passed"], f"Verification {k} did not pass")

    def test_real_data_invariants_and_baseline_numbers(self):
        summary_df = pd.read_csv(GROUP_EVAL_DIR / "threshold_summary.csv")
        paired_df = pd.read_csv(GROUP_EVAL_DIR / "paired_differences.csv")
        decomp_df = pd.read_csv(GROUP_EVAL_DIR / "decomposition.csv")

        # 1. exact_age Full at T >= 5 has 0 evaluable items and NaN primary score (never 0.0)
        exact_t5 = summary_df[
            summary_df["target_mode"].eq("fixed_full_target")
            & summary_df["age_policy"].eq("exact_age")
            & summary_df["threshold"].ge(5)
        ]
        self.assertTrue((exact_t5["n_evaluable_items"] == 0).all())
        self.assertTrue(exact_t5["primary_group_human_weighted_mean"].isna().all())

        # 2. age_band T=5 operational baseline support counts and Full - Paired Demo differences
        p_ex = paired_df[
            paired_df["age_policy"].eq("age_band")
            & paired_df["support_policy"].eq("n_and_effective_n")
            & paired_df["threshold"].eq(5)
            & paired_df["model"].eq("EXAONE-3.5-2.4B")
            & paired_df["comparison"].eq("full40_clean-paired_demo40_clean")
            & paired_df["metric"].eq("tvd")
        ].iloc[0]
        self.assertEqual(int(p_ex["n_groups_supported_any_item"]), 10)
        self.assertEqual(int(p_ex["n_evaluable_items"]), 246)
        self.assertEqual(int(p_ex["n_included_group_item_pairs"]), 2441)
        self.assertEqual(int(p_ex["n_included_draw_item_pairs"]), 2923)
        self.assertAlmostEqual(float(p_ex["primary_group_human_weighted_diff"]), -0.034000, places=5)
        self.assertAlmostEqual(float(p_ex["row_conditional_diagnostic_pair_diff"]), -0.031855, places=5)

        p_ka = paired_df[
            paired_df["age_policy"].eq("age_band")
            & paired_df["support_policy"].eq("n_and_effective_n")
            & paired_df["threshold"].eq(5)
            & paired_df["model"].eq("Kanana-1.5-2.1B")
            & paired_df["comparison"].eq("full40_clean-paired_demo40_clean")
            & paired_df["metric"].eq("tvd")
        ].iloc[0]
        self.assertAlmostEqual(float(p_ka["primary_group_human_weighted_diff"]), -0.028395, places=5)
        self.assertAlmostEqual(float(p_ka["row_conditional_diagnostic_pair_diff"]), -0.028184, places=5)

        # 3. Decomposition telescope residual is ~0 across all evaluable rows
        eval_decomp = decomp_df[decomp_df["n_evaluable_items"] > 0]
        self.assertTrue((eval_decomp["telescope_residual_check"].abs() < 1e-12).all())

        # 4. Exact agreement of row_conditional_diagnostic_pair_mean with existing threshold_sensitivity_20261010
        old_ts = pd.read_csv(THRESH_DIR / "threshold_summary.csv")
        for _, r in old_ts.iterrows():
            sub = summary_df[
                summary_df["target_mode"].eq("fixed_full_target")
                & summary_df["age_policy"].eq(r["age_policy"])
                & summary_df["support_policy"].eq(r["support_policy"])
                & summary_df["threshold"].eq(int(r["threshold"]))
                & summary_df["model"].eq(r["model"])
                & summary_df["condition"].eq("full40_clean")
                & summary_df["metric"].eq(r["metric"])
            ].iloc[0]
            if pd.isna(r["full_mean_selected"]):
                self.assertTrue(pd.isna(sub["row_conditional_diagnostic_pair_mean"]))
            else:
                self.assertAlmostEqual(
                    float(r["full_mean_selected"]),
                    float(sub["row_conditional_diagnostic_pair_mean"]),
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
