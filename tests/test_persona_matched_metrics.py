"""Counterexamples for strict persona-conditioned human reference scoring."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kgss_persona_matching import (
    build_matched_human_reference,
    evaluate_matched_distribution,
)


PROFILE = {
    "SEX": 1,
    "AGE": 40,
    "EDUC": 4,
    "MARITAL": 1,
    "REGION": 1,
    "URBAN": 1,
}


def respondents(profile, answer, n=5):
    return [{**profile, "Q": answer, "FINALWT": 1.0} for _ in range(n)]


class TestPersonaMatchedMetrics(unittest.TestCase):
    def test_full_reference_changes_for_each_additional_persona_attribute(self):
        for key in ("EDUC", "MARITAL", "REGION", "URBAN"):
            with self.subTest(key=key):
                other_profile = {**PROFILE, key: PROFILE[key] + 1}
                humans = pd.DataFrame(
                    respondents(PROFILE, 1) + respondents(other_profile, 2)
                )
                original = build_matched_human_reference(humans, "Q", [1, 2], PROFILE)
                other = build_matched_human_reference(humans, "Q", [1, 2], other_profile)
                self.assertTrue(original["eligible"])
                self.assertTrue(other["eligible"])
                self.assertEqual(original["valid_n"], 5)
                self.assertEqual(other["valid_n"], 5)
                np.testing.assert_allclose(original["human_dist"], [1.0, 0.0])
                np.testing.assert_allclose(other["human_dist"], [0.0, 1.0])
                self.assertEqual(original["matching_keys"], list(PROFILE))

    def test_demo_uses_sex_and_age_without_hidden_full_attribute_restrictions(self):
        other_profile = {**PROFILE, "EDUC": 2, "MARITAL": 3, "REGION": 2, "URBAN": 2}
        different_sex = {**PROFILE, "SEX": 2}
        humans = pd.DataFrame(
            respondents(PROFILE, 1)
            + respondents(other_profile, 2)
            + respondents(different_sex, 1)
        )
        reference = build_matched_human_reference(
            humans, "Q", [1, 2], {"SEX": 1, "AGE": 40}
        )
        self.assertEqual(reference["matching_keys"], ["SEX", "AGE"])
        self.assertEqual(reference["matched_n"], 10)
        np.testing.assert_allclose(reference["human_dist"], [0.5, 0.5])

    def test_empty_conditions_use_population_distribution(self):
        humans = pd.DataFrame(
            respondents(PROFILE, 1)
            + respondents({**PROFILE, "SEX": 2, "AGE": 70}, 2, n=15)
        )
        reference = build_matched_human_reference(humans, "Q", [1, 2], {})
        self.assertEqual(reference["matching_keys"], [])
        self.assertEqual(reference["matched_n"], 20)
        self.assertTrue(reference["eligible"])
        np.testing.assert_allclose(reference["human_dist"], [0.25, 0.75])

    def test_exact_age_does_not_fall_back_to_decade_or_nearby_respondents(self):
        humans = pd.DataFrame(
            respondents(PROFILE, 1)
            + respondents({**PROFILE, "AGE": 41}, 2, n=20)
        )
        exact = build_matched_human_reference(humans, "Q", [1, 2], PROFILE)
        self.assertEqual(exact["matched_n"], 5)
        np.testing.assert_allclose(exact["human_dist"], [1.0, 0.0])
        absent = build_matched_human_reference(
            humans, "Q", [1, 2], {**PROFILE, "AGE": 42}
        )
        self.assertEqual(absent["matched_n"], 0)
        self.assertEqual(absent["reason"], "empty_match")
        self.assertIsNone(absent["human_dist"])

    def test_explicit_age_band_is_separate_and_preserves_full_attribute_matching(self):
        rows = respondents(PROFILE, 1) + respondents({**PROFILE, "AGE": 41}, 2)
        for key in ("EDUC", "MARITAL", "REGION", "URBAN"):
            rows += respondents({**PROFILE, "AGE": 41, key: PROFILE[key] + 1}, 2)
        humans = pd.DataFrame(rows)
        humans["AGE_GRP"] = "40-49"
        exact = build_matched_human_reference(humans, "Q", [1, 2], PROFILE)
        band_conditions = {key: value for key, value in PROFILE.items() if key != "AGE"}
        band_conditions["AGE_GRP"] = "40-49"
        band = build_matched_human_reference(humans, "Q", [1, 2], band_conditions)
        self.assertEqual(exact["matched_n"], 5)
        self.assertEqual(band["matched_n"], 10)
        self.assertTrue(band["eligible"])
        self.assertIn("AGE_GRP", band["matching_keys"])
        self.assertNotIn("AGE", band["matching_keys"])
        np.testing.assert_allclose(exact["human_dist"], [1.0, 0.0])
        np.testing.assert_allclose(band["human_dist"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            build_matched_human_reference(humans, "Q", [1, 2], {**PROFILE, "AGE_GRP": "40-49"})

    def test_valid_answers_positive_final_weights_and_effective_sample_size(self):
        humans = pd.DataFrame([
            {**PROFILE, "Q": answer, "FINALWT": weight}
            for answer, weight in [
                (1, 1.0), (2, 3.0), (99, 10.0), (1, 0.0),
                (1, np.nan), (2, -1.0), (2, np.inf),
            ]
        ])
        reference = build_matched_human_reference(
            humans, "Q", [1, 2], PROFILE, min_valid_n=2, min_effective_n=1.5
        )
        self.assertEqual(reference["matched_n"], 7)
        self.assertEqual(reference["valid_n"], 2)
        self.assertAlmostEqual(reference["total_weight"], 4.0)
        self.assertAlmostEqual(reference["effective_n"], 1.6)
        np.testing.assert_allclose(reference["human_dist"], [0.25, 0.75])
        self.assertTrue(reference["eligible"])
        strict = build_matched_human_reference(
            humans, "Q", [1, 2], PROFILE, min_valid_n=2, min_effective_n=2.0
        )
        result = evaluate_matched_distribution([0.25, 0.75], strict)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "insufficient_effective_n")
        self.assertIsNone(result["primary"])
        self.assertAlmostEqual(result["raw_diagnostic"]["tvd"], 0.0)

    def test_sparse_or_empty_human_references_are_not_primary_results(self):
        cases = [
            (pd.DataFrame(respondents(PROFILE, 1, n=1)), PROFILE,
             "insufficient_valid_n_and_effective_n", True),
            (pd.DataFrame(respondents(PROFILE, 1)), {**PROFILE, "AGE": 42},
             "empty_match", False),
            (pd.DataFrame(respondents(PROFILE, 99)), PROFILE,
             "no_valid_response", False),
        ]
        for humans, conditions, reason, has_diagnostic in cases:
            with self.subTest(reason=reason):
                reference = build_matched_human_reference(humans, "Q", [1, 2], conditions)
                result = evaluate_matched_distribution([1.0, 0.0], reference)
                self.assertFalse(reference["eligible"])
                self.assertFalse(result["eligible"])
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["primary"])
                self.assertEqual(result["raw_diagnostic"] is not None, has_diagnostic)

    def test_nominal_items_report_tvd_without_numeric_code_wasserstein(self):
        humans = pd.DataFrame(respondents(PROFILE, 10))
        reference = build_matched_human_reference(humans, "Q", [10, 20], PROFILE)
        result = evaluate_matched_distribution([0.0, 1.0], reference, w1_valid=False)
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["primary"]["tvd"], 1.0)
        self.assertIsNone(result["primary"]["w1"])
        self.assertIsNone(result["primary"]["normalized_w1"])
        self.assertIsNone(result["raw_diagnostic"]["w1"])

    def test_conditional_oracle_scores_zero_but_population_copy_has_error(self):
        other_profile = {**PROFILE, "EDUC": 2}
        humans = pd.DataFrame(
            respondents(PROFILE, 1) + respondents(other_profile, 2)
        )
        for profile, prediction in [(PROFILE, [1.0, 0.0]), (other_profile, [0.0, 1.0])]:
            with self.subTest(profile=profile):
                reference = build_matched_human_reference(humans, "Q", [1, 2], profile)
                oracle = evaluate_matched_distribution(prediction, reference)["primary"]
                population_copy = evaluate_matched_distribution([0.5, 0.5], reference)["primary"]
                self.assertAlmostEqual(oracle["tvd"], 0.0)
                self.assertAlmostEqual(oracle["w1"], 0.0)
                self.assertAlmostEqual(population_copy["tvd"], 0.5)
                self.assertAlmostEqual(population_copy["w1"], 0.5)
        population = build_matched_human_reference(humans, "Q", [1, 2], {})
        marginal_score = evaluate_matched_distribution([0.5, 0.5], population)["primary"]
        self.assertAlmostEqual(marginal_score["tvd"], 0.0)
        self.assertAlmostEqual(marginal_score["w1"], 0.0)


if __name__ == "__main__":
    unittest.main()
