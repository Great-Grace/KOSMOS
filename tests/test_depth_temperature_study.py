#!/usr/bin/env python3
"""
Unit and integration tests for `kgss_calibration.py` and the Phase 0–7
Persona Depth Reduction & Post-Hoc Temperature Calibration Study.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = REPO_ROOT / "reexperiment_20261009" / "depth_temperature_study_20261010"
sys.path.insert(0, str(REPO_ROOT))

from kgss_calibration import (
    PERSONA_DEPTH_CONFIGS,
    aggregate_calibrated_group,
    build_persona_header,
    calibrate_probability_matrix,
    calibrate_probability_vector,
    compute_cell_poststrat_weights,
    compute_conditional_shift_metrics,
    mix_with_uniform,
    scale_probability_temperature,
)
from kgss_metrics import entropy_bits
from kgss_persona_bootstrap import load_value_labels


class TestDepthTemperatureStudy(unittest.TestCase):
    def test_scale_probability_temperature_identity_and_limits(self) -> None:
        p = np.array([0.70, 0.20, 0.08, 0.02], dtype=float)
        p_t1 = scale_probability_temperature(p, 1.0)
        self.assertTrue(np.allclose(p_t1, p, atol=1e-12))
        self.assertAlmostEqual(float(p_t1.sum()), 1.0, places=12)

        # As T -> large, approaches uniform 1/K
        p_t1000 = scale_probability_temperature(p, 1000.0)
        self.assertTrue(np.allclose(p_t1000, np.full(4, 0.25), atol=1e-3))

        # Entropy strictly increases with T for non-uniform positive vectors
        t_grid = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        entropies = [entropy_bits(scale_probability_temperature(p, t)) for t in t_grid]
        for h1, h2 in zip(entropies[:-1], entropies[1:]):
            self.assertGreater(h2, h1)

        # Uniform 4-choice entropy in bits is 2.0
        nh = entropy_bits(np.full(4, 0.25))
        self.assertAlmostEqual(nh, 2.0, places=12)

    def test_scale_probability_temperature_zero_entries_and_epsilon(self) -> None:
        p_zero = np.array([0.85, 0.15, 0.0], dtype=float)
        p_scaled = scale_probability_temperature(p_zero, 2.0, epsilon=None)
        self.assertEqual(p_scaled[2], 0.0)
        self.assertAlmostEqual(float(p_scaled.sum()), 1.0, places=12)
        self.assertTrue(p_scaled[0] < 0.85 and p_scaled[1] > 0.15)

        p_eps = scale_probability_temperature(p_zero, 2.0, epsilon=1e-6)
        self.assertGreater(p_eps[2], 0.0)
        self.assertAlmostEqual(float(p_eps.sum()), 1.0, places=12)

    def test_vectorized_calibrate_probability_matrix_matches_single(self) -> None:
        rng = np.random.default_rng(123)
        mat = rng.dirichlet(alpha=[0.4, 1.2, 0.8, 2.0], size=25)
        for t in (0.75, 1.0, 1.75, 3.0):
            for lam in (0.0, 0.20):
                for eps in (None, 1e-6):
                    vec_out = calibrate_probability_matrix(mat, temperature=t, mix_lambda=lam, epsilon=eps)
                    row_out = np.vstack(
                        [
                            calibrate_probability_vector(row, temperature=t, mix_lambda=lam, epsilon=eps)
                            for row in mat
                        ]
                    )
                    self.assertTrue(np.allclose(vec_out, row_out, atol=1e-12))

    def test_mix_with_uniform(self) -> None:
        p = np.array([0.90, 0.10], dtype=float)
        self.assertTrue(np.allclose(mix_with_uniform(p, 0.0), p, atol=1e-12))
        self.assertTrue(np.allclose(mix_with_uniform(p, 1.0), np.array([0.5, 0.5]), atol=1e-12))
        self.assertTrue(np.allclose(mix_with_uniform(p, 0.5), np.array([0.7, 0.3]), atol=1e-12))

    def test_aggregate_calibrated_group_methods_and_weights(self) -> None:
        dists = [
            np.array([0.90, 0.08, 0.02]),
            np.array([0.10, 0.80, 0.10]),
        ]
        p_ind_t1 = aggregate_calibrated_group(dists, temperature=1.0, method="individual")
        p_grp_t1 = aggregate_calibrated_group(dists, temperature=1.0, method="group")
        self.assertTrue(np.allclose(p_ind_t1, p_grp_t1, atol=1e-12))
        self.assertTrue(np.allclose(p_ind_t1, np.array([0.50, 0.44, 0.06]), atol=1e-12))

        p_ind_t25 = aggregate_calibrated_group(dists, temperature=2.5, method="individual")
        p_grp_t25 = aggregate_calibrated_group(dists, temperature=2.5, method="group")
        self.assertAlmostEqual(float(p_ind_t25.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(p_grp_t25.sum()), 1.0, places=12)

        p_w = aggregate_calibrated_group(dists, temperature=1.0, method="individual", draw_weights=[3.0, 1.0])
        self.assertTrue(np.allclose(p_w, 0.75 * dists[0] + 0.25 * dists[1], atol=1e-12))

    def test_conditional_shift_metrics_properties(self) -> None:
        q_dists = [
            np.array([0.60, 0.30, 0.10]),
            np.array([0.20, 0.50, 0.30]),
            np.array([0.35, 0.25, 0.40]),
        ]
        weights = [0.5, 0.3, 0.2]

        m_perf = compute_conditional_shift_metrics(q_dists, q_dists, weights, w1_valid=True)
        self.assertAlmostEqual(m_perf["shift_magnitude_ratio"], 1.0, places=12)
        self.assertAlmostEqual(m_perf["shift_vector_cosine"], 1.0, places=12)
        self.assertAlmostEqual(m_perf["shift_vector_pearson_r"], 1.0, places=12)
        self.assertAlmostEqual(m_perf["option_shift_sign_acc"], 1.0, places=12)
        self.assertAlmostEqual(m_perf["expected_value_shift_pearson_r"], 1.0, places=12)

        const_dists = [np.array([0.40, 0.40, 0.20]) for _ in range(3)]
        m_const = compute_conditional_shift_metrics(const_dists, q_dists, weights, w1_valid=True)
        self.assertAlmostEqual(m_const["mean_model_pair_tvd"], 0.0, places=12)
        self.assertAlmostEqual(m_const["shift_magnitude_ratio"], 0.0, places=12)

    def test_persona_header_generation_and_splits(self) -> None:
        labels = load_value_labels(REPO_ROOT / "inventory")
        p40 = pd.read_csv(STUDY_DIR / "panel40_personas.csv")
        row0 = p40.iloc[0]

        for cfg in PERSONA_DEPTH_CONFIGS:
            hdr = build_persona_header(row0, labels, cfg)
            self.assertIsInstance(hdr, str)
            self.assertTrue(hdr.startswith("당신은 "))

        splits_df = pd.read_csv(STUDY_DIR / "item_splits_246.csv")
        self.assertEqual(len(splits_df), 246)
        counts = splits_df["split_60_20_20"].value_counts().to_dict()
        self.assertEqual(counts, {"calibration": 148, "validation": 49, "test": 49})
        self.assertEqual(set(splits_df["cv_fold"].unique()), {0, 1, 2, 3, 4})
        self.assertEqual(set(splits_df["group_cv_fold"].unique()), {0, 1, 2, 3, 4})

    def test_expanded_panel_no_duplicates_and_poststrat_weights(self) -> None:
        exp_df = pd.read_csv(STUDY_DIR / "expanded_persona_panel.csv")
        self.assertEqual(exp_df["RESPID"].nunique(), len(exp_df))
        self.assertGreaterEqual(len(exp_df), 80)

        p40 = pd.read_csv(STUDY_DIR / "panel40_personas.csv")
        self.assertTrue(set(p40["RESPID"].astype(int)).issubset(set(exp_df["RESPID"].astype(int))))

        human_all = pd.read_parquet(REPO_ROOT / "inventory" / "kgss_clean.parquet")
        human_df = human_all[human_all["YEAR"].eq(2023)].copy()
        w_post = compute_cell_poststrat_weights(exp_df, human_df, ["SEX", "AGE_GRP", "EDUC", "MARITAL"])
        self.assertEqual(len(w_post), len(exp_df))
        self.assertTrue(np.all(w_post > 0.0))

    def test_phase_5_to_7_deliverables_and_verification(self) -> None:
        verif_path = STUDY_DIR / "verification.json"
        manifest_path = STUDY_DIR / "manifest.json"
        if not verif_path.exists() or not manifest_path.exists():
            self.skipTest("verification.json or manifest.json not generated yet")
        verif = json.loads(verif_path.read_text(encoding="utf-8"))
        self.assertTrue(verif["all_required_deliverables_exist"])
        self.assertTrue(verif["preserved_prior_directories_intact"])
        self.assertTrue(verif["expanded_panel_zero_duplicates"])
        self.assertTrue(verif["item_splits_246_locked_60_20_20"])
        self.assertTrue(verif["hardware_parity_exaone_pass"])
        self.assertTrue(verif["hardware_parity_kanana_pass"])
        self.assertEqual(verif["n_figures_generated"], 8)

        df_depth = pd.read_csv(STUDY_DIR / "depth_comparison.csv")
        self.assertIn("exp1_panel40", set(df_depth["track"]))
        self.assertIn("exp2_expanded", set(df_depth["track"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

