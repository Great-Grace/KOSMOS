"""Independent real-data checks of the saved-response matching execution path."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestPersonaMatchedRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        results = root / "reexperiment_20261009/persona_matched_eval_20261010"
        if not (results / "manifest.json").exists():
            raise unittest.SkipTest("Run the saved-response persona matching scorer first.")
        cls.manifest = json.loads((results / "manifest.json").read_text())
        cls.rows = pd.read_csv(results / "persona_matched_rows.csv", low_memory=False)
        cls.fixed = pd.read_csv(results / "fixed_full_target_rows.csv",
                               usecols=["age_policy", "condition", "model", "var", "persona_id", "reference_id", "eligible"])
        cls.summary = pd.read_csv(results / "persona_matched_summary.csv")
        cls.human = pd.read_parquet(root / "inventory/kgss_clean.parquet")
        cls.human = cls.human[cls.human.YEAR.eq(cls.manifest["year"])].copy()
        cls.human["AGE_GRP"] = pd.cut(cls.human.AGE, cls.manifest["age_bands"]["bins"],
                                     labels=cls.manifest["age_bands"]["labels"], right=False)
        cls.probes = pd.read_csv(root / "reexperiment_20261009/full40_clean/probes.csv")

    def test_saved_full_rows_use_all_attributes_and_direct_human_counts(self):
        selected = self.rows[self.rows.condition.eq("full40_clean") & self.rows.include_in_clean_eval]
        samples = pd.concat([group.head(6) for _, group in selected.groupby(["age_policy", "eligible"])])
        probe_map = self.probes.set_index("probe_id")
        for _, row in samples.iterrows():
            conditions = json.loads(row.matching_conditions)
            age_key = "AGE" if row.age_policy == "exact_age" else "AGE_GRP"
            self.assertEqual(set(conditions), {"SEX", age_key, "EDUC", "MARITAL", "REGION", "URBAN"})
            matched = self.human.copy()
            for key, value in conditions.items():
                matched = matched[matched[key].eq(value)]
            self.assertEqual(len(matched), row.matched_n)
            codes = json.loads(probe_map.loc[row.probe_id, "codes"])
            valid = matched[matched[row['var']].isin(codes) & np.isfinite(matched.FINALWT) & matched.FINALWT.gt(0)]
            self.assertEqual(len(valid), row.valid_n)
            if len(valid):
                weights = valid.FINALWT.to_numpy()
                expected = np.array([weights[valid[row['var']].to_numpy() == code].sum() / weights.sum() for code in codes])
                np.testing.assert_allclose(json.loads(row.human_dist), expected, atol=1e-12)
                self.assertAlmostEqual(row.effective_n, weights.sum()**2 / np.square(weights).sum(), places=10)
                model = np.array(json.loads(row.model_dist))
                self.assertAlmostEqual(row.diagnostic_tvd, np.abs(model - expected).sum() / 2, places=12)
                if row.w1_valid_for_evaluation:
                    self.assertAlmostEqual(row.diagnostic_w1, np.abs(np.cumsum(model)[:-1] - np.cumsum(expected)[:-1]).sum(), places=12)

    def test_information_ablation_uses_identical_human_targets(self):
        grouped = self.fixed.groupby(["age_policy", "model", "var", "persona_id"])
        self.assertTrue(grouped.reference_id.nunique().eq(1).all())
        self.assertTrue(grouped.condition.nunique().eq(4).all())
        self.assertTrue(grouped.eligible.nunique().eq(1).all())

    def test_duplicate_pps_draws_do_not_increase_human_reference_counts(self):
        panel = self.probes.drop_duplicates("persona_id")
        duplicate_ids = panel[panel.respid.duplicated(keep=False)].persona_id
        self.assertGreater(len(duplicate_ids), 0)
        rows = self.rows[self.rows.condition.eq("full40_clean") & self.rows.persona_id.isin(duplicate_ids)]
        for _, group in rows.groupby(["age_policy", "model", "var", "reference_id"]):
            self.assertEqual(group.matched_n.nunique(), 1)
            self.assertEqual(group.valid_n.nunique(), 1)
            self.assertEqual(group.human_dist.nunique(dropna=False), 1)
        for _, row in rows[rows['var'].eq(rows['var'].iloc[0])].drop_duplicates("reference_id").iterrows():
            mask = pd.Series(True, index=self.human.index)
            for key, value in json.loads(row.matching_conditions).items():
                mask &= self.human[key].eq(value)
            self.assertEqual(mask.sum(), row.matched_n)

    def test_missing_support_is_exposed_in_full_scope_denominators(self):
        for _, row in self.summary.iterrows():
            selected = self.rows[self.rows.age_policy.eq(row.age_policy) & self.rows.model.eq(row.model) & self.rows.condition.eq(row.condition)]
            selected = selected[selected.include_in_clean_eval if row.scope == "clean_tvd" else selected.w1_valid_for_evaluation]
            self.assertEqual(len(selected), row.n_required_rows)
            self.assertEqual(selected.eligible.sum(), row.n_supported_rows)
            if row.n_supported_rows < row.n_required_rows:
                field = "full_scope_mean_tvd" if row.scope == "clean_tvd" else "full_scope_mean_w1"
                self.assertTrue(pd.isna(row[field]))
        self.assertEqual(self.manifest["new_model_inference_calls"], 0)
        self.assertIsNone(self.manifest["matching_fallback"])


if __name__ == "__main__":
    unittest.main()
