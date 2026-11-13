"""
Unit tests for the KOSMOS evaluation pipeline audit (TEST 1 through TEST 7 + additional metric checks).

Verifies:
- TEST 1: Identical distributions -> Wasserstein = 0
- TEST 2: Single-prompt condition (C0 / adult) -> Existing W1 == Population W1
- TEST 3: Male/Female 50:50 opposite conditional distributions counterexample ->
          Existing W1 = 0.5, Corrected Population W1 = 0.0, Conditional Cell W1 = 0.0
- TEST 4: Unconditional marginal copy (P_LLM(Y|c) = P_KGSS(Y)) under true polarizing subgroups ->
          Existing W1 = 0.0 (falsely rewards stereotype blindness),
          Population W1 = 0.0,
          Conditional Cell W1 = 0.5 (detects failure to capture subgroup heterogeneity)
- TEST 5: Non-uniform cell weights (e.g., 80:20) -> weighted vs. unweighted aggregation difference
- TEST 6: Probability simplex normalization (sum = 1, non-negative, invalid weights rejected)
- TEST 7: Missing cell / zero-sample cell handling -> automatic weight renormalization to 1.0
- Additional tests:
  - TEST 8: Subgroup r centering bug: Perfect Oracle achieves centered r = 1.0, while uncentered r < 1.0
  - TEST 9: Personal Accuracy paired difference linearity: mean(A - B) == mean(A) - mean(B)
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kgss_metrics import (
    aggregate_demo_cells,
    aggregate_distributions,
    aggregate_full_personas,
    compute_subgroup_correlations,
    evaluate_item_distributions,
    evaluate_personal_accuracy_paired,
    validate_prob_dist,
    wasserstein_1d,
)


class TestKosmosAuditMetrics(unittest.TestCase):
    def test_1_identical_distributions_zero_wasserstein(self):
        """TEST 1: 동일한 두 분포의 Wasserstein = 0."""
        p = np.array([0.15, 0.35, 0.30, 0.20])
        q = np.array([0.15, 0.35, 0.30, 0.20])
        w = wasserstein_1d(p, q, codes=[1, 2, 3, 4])
        self.assertAlmostEqual(w, 0.0, places=12)

    def test_2_single_prompt_condition_equivalence(self):
        """TEST 2: C0/adult 단일 분포에서 기존 방식과 수정 방식이 동일한 값 반환."""
        codes = [1, 2, 3, 4, 5]
        p_llm = [[0.10, 0.20, 0.40, 0.20, 0.10]]
        q_kgss = [0.25, 0.25, 0.20, 0.15, 0.15]

        res = evaluate_item_distributions(
            pred_dists=p_llm,
            gold_dist=q_kgss,
            codes=codes,
        )
        self.assertAlmostEqual(res["existing_wasserstein"], res["population_wasserstein"], places=12)
        self.assertAlmostEqual(res["weighted_mean_wasserstein"], res["population_wasserstein"], places=12)
        self.assertAlmostEqual(res["wasserstein_jensen_gap"], 0.0, places=12)
        self.assertAlmostEqual(
            res["model_conditional_entropy_bits"], res["model_population_entropy_bits"], places=12
        )

    def test_3_counterexample_perfect_conditional_reproduction(self):
        """
        TEST 3: 남녀 50:50 반례 검증.
        남성(50%): 찬성 100%, 반대 0% -> [1.0, 0.0]
        여성(50%): 찬성 0%, 반대 100% -> [0.0, 1.0]
        모집단 전체: [0.5, 0.5]
        모델이 남녀 조건부 분포를 완벽히 재현했을 때:
        - 기존 방식 B = (1/2)*W([1,0],[0.5,0.5]) + (1/2)*W([0,1],[0.5,0.5]) = 0.5
        - 수정 모집단 방식 A = W(0.5*[1,0] + 0.5*[0,1], [0.5,0.5]) = W([0.5,0.5], [0.5,0.5]) = 0.0
        - 조건부 방식 C = 0.5*W([1,0],[1,0]) + 0.5*W([0,1],[0,1]) = 0.0
        """
        codes = [1, 2]
        pred_dists = [
            [1.0, 0.0],  # 남성 예측
            [0.0, 1.0],  # 여성 예측
        ]
        gold_marginal = [0.5, 0.5]
        weights = [0.5, 0.5]
        cell_human_dists = [
            [1.0, 0.0],  # 남성 실제
            [0.0, 1.0],  # 여성 실제
        ]

        res = evaluate_item_distributions(
            pred_dists=pred_dists,
            gold_dist=gold_marginal,
            codes=codes,
            weights=weights,
            cell_model_dists=pred_dists,
            cell_human_dists=cell_human_dists,
            cell_weights=weights,
        )
        self.assertAlmostEqual(res["existing_wasserstein"], 0.5, places=12)
        self.assertAlmostEqual(res["population_wasserstein"], 0.0, places=12)
        self.assertAlmostEqual(res["conditional_cell_wasserstein_weighted"], 0.0, places=12)
        self.assertAlmostEqual(res["wasserstein_jensen_gap"], 0.5, places=12)
        # 개별 조건부 엔트로피는 0비트이지만 집계된 모집단 엔트로피는 1비트
        self.assertAlmostEqual(res["model_conditional_entropy_bits"], 0.0, places=12)
        self.assertAlmostEqual(res["model_population_entropy_bits"], 1.0, places=12)
        self.assertAlmostEqual(res["mutual_information_bits"], 1.0, places=12)

    def test_4_unconditional_copy_penalized_only_by_conditional_metric(self):
        """
        TEST 4: 모든 집단이 동일한 [0.5, 0.5]를 예측할 때:
        - 기존 방식 = 0.0 (집단별 차이를 전혀 못 잡아내도 0점 부여)
        - 수정 모집단 방식 = 0.0 (모집단 주변분포는 [0.5, 0.5]로 일치하므로 0점)
        - 조건부 방식 = 0.5 (실제 남녀 조건부 분포 [1,0], [0,1]과의 괴리를 정확히 포착)
        """
        codes = [1, 2]
        pred_dists = [
            [0.5, 0.5],  # 남성 예측 (차별화 실패)
            [0.5, 0.5],  # 여성 예측 (차별화 실패)
        ]
        gold_marginal = [0.5, 0.5]
        weights = [0.5, 0.5]
        cell_human_dists = [
            [1.0, 0.0],  # 남성 실제
            [0.0, 1.0],  # 여성 실제
        ]

        res = evaluate_item_distributions(
            pred_dists=pred_dists,
            gold_dist=gold_marginal,
            codes=codes,
            weights=weights,
            cell_model_dists=pred_dists,
            cell_human_dists=cell_human_dists,
            cell_weights=weights,
        )
        self.assertAlmostEqual(res["existing_wasserstein"], 0.0, places=12)
        self.assertAlmostEqual(res["population_wasserstein"], 0.0, places=12)
        self.assertAlmostEqual(res["conditional_cell_wasserstein_weighted"], 0.5, places=12)

    def test_5_nonuniform_cell_weights_80_20(self):
        """TEST 5: 가중치가 균등하지 않은 경우(예: 80:20) 가중 집계와 단순 평균의 차이 검증."""
        codes = [1, 2]
        cell_df = pd.DataFrame([
            {"sex": 1, "age_grp": "20s", "model_dist": [1.0, 0.0]},
            {"sex": 2, "age_grp": "20s", "model_dist": [0.0, 1.0]},
        ])
        cell_weights_df = pd.DataFrame([
            {"sex": 1, "age_grp": "20s", "cell_weight_item_norm": 0.8},
            {"sex": 2, "age_grp": "20s", "cell_weight_item_norm": 0.2},
        ])

        agg_weighted, _ = aggregate_demo_cells(cell_df, cell_weights_df)
        agg_unweighted = aggregate_distributions([[1.0, 0.0], [0.0, 1.0]], weights=None)

        np.testing.assert_allclose(agg_weighted, np.array([0.8, 0.2]), atol=1e-12)
        np.testing.assert_allclose(agg_unweighted, np.array([0.5, 0.5]), atol=1e-12)

        # 실제 모집단이 셀 비중(80:20)과 일치하는 [0.8, 0.2]일 때:
        gold_marginal = [0.8, 0.2]
        w_weighted = wasserstein_1d(agg_weighted, gold_marginal, codes=codes)
        w_unweighted = wasserstein_1d(agg_unweighted, gold_marginal, codes=codes)
        self.assertAlmostEqual(w_weighted, 0.0, places=12)
        self.assertAlmostEqual(w_unweighted, 0.3, places=12)

    def test_6_probability_normalization_and_validation(self):
        """TEST 6: 확률분포 합이 1인지(정규화 검증) 및 음수/영벡터 예외 처리 검증."""
        raw = [2.0, 3.0, 5.0]
        normed = validate_prob_dist(raw)
        self.assertAlmostEqual(float(normed.sum()), 1.0, places=12)
        self.assertTrue(np.all(normed >= 0.0))

        agg = aggregate_full_personas([[0.2, 0.8], [0.6, 0.4]], already_pps_sampled=True)
        self.assertAlmostEqual(float(agg.sum()), 1.0, places=12)
        np.testing.assert_allclose(agg, np.array([0.4, 0.6]), atol=1e-12)

        with self.assertRaises(ValueError):
            validate_prob_dist([-0.1, 1.1])
        with self.assertRaises(ValueError):
            validate_prob_dist([0.0, 0.0])

    def test_7_missing_cell_weight_renormalization(self):
        """TEST 7: 결측 셀 또는 표본 0인 셀이 있을 때 가중치 재정규화 및 안전한 예외 처리 검증."""
        cell_df = pd.DataFrame([
            {"sex": 1, "age_grp": "20s", "model_dist": [0.6, 0.4]},
            {"sex": 2, "age_grp": "20s", "model_dist": [0.2, 0.8]},
        ])
        # 3개 셀 중 1개 셀(sex=1, age_grp=60+)이 관측되지 않은 상황 -> 관측 셀 가중치(0.30:0.30 -> 0.5:0.5)로 재정규화
        cell_weights_df = pd.DataFrame([
            {"sex": 1, "age_grp": "20s", "cell_weight_item_norm": 0.30},
            {"sex": 2, "age_grp": "20s", "cell_weight_item_norm": 0.30},
            {"sex": 1, "age_grp": "60+", "cell_weight_item_norm": 0.40},
        ])
        agg, merged = aggregate_demo_cells(cell_df, cell_weights_df)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(float(agg.sum()), 1.0, places=12)
        np.testing.assert_allclose(agg, np.array([0.4, 0.6]), atol=1e-12)

        # 가중치 벡터 합이 0 이하인 경우 ValueError 발생
        with self.assertRaises(ValueError):
            aggregate_distributions([[0.6, 0.4], [0.2, 0.8]], weights=[0.0, 0.0])

    def test_8_subgroup_r_centering_bug_and_oracle(self):
        """
        TEST 8 (추가 검증): Subgroup r 중심화 누락 버그 증명.
        문항 간 척도 평균(1~5점 vs 1~2점)이 다를 때,
        모델이 실제 인간 셀 평균을 100% 완벽하게 맞혀도(model_pred == real_mean),
        기존 코드(real_dev vs uncentered model_pred)는 r < 1.0이 나오고,
        수정된 코드(real_dev vs centered model_dev)는 정확히 r = 1.0을 반환함을 검증.
        """
        rng = np.random.default_rng(42)
        rows = []
        cells = [(s, a) for s in (1, 2) for a in ("18-29", "30-39", "40-49", "50-59", "60+")]
        for v_idx in range(30):
            grand_mean = float(rng.uniform(1.2, 6.5))
            devs = rng.normal(0.0, 0.15, size=len(cells))
            devs = devs - devs.mean()
            for (sex, age_grp), d in zip(cells, devs):
                val = grand_mean + float(d)
                rows.append({
                    "var": f"V{v_idx:02d}",
                    "sex": sex,
                    "age_grp": age_grp,
                    "real_mean": val,
                    "real_grand": grand_mean,
                    "model_pred": val,  # 100% 완벽한 오라클 예측
                })
        df = pd.DataFrame(rows)
        res = compute_subgroup_correlations(df)
        self.assertAlmostEqual(res["centered_mean_r"], 1.0, places=12)
        self.assertAlmostEqual(res["within_item_mean_r"], 1.0, places=12)
        self.assertLess(res["existing_mean_r"], 0.35)

    def test_9_personal_accuracy_linearity_and_normalization(self):
        """
        TEST 9 (추가 검증): 개인 정확도 차이(mean(A-B) == mean(A) - mean(B)) 선형성 및
        원시 정확도(Raw Acc) vs 정규화 정확도(Normalized Acc) 분리 검증.
        """
        rng = np.random.default_rng(123)
        rows = []
        for pid in range(1, 21):
            for q_idx in range(10):
                cell_modal = float(rng.uniform(0.45, 0.75))
                rows.append({
                    "level": "full",
                    "persona_id": pid,
                    "var": f"Q{q_idx}",
                    "is_correct": int(rng.random() < 0.40),
                    "cell_modal": cell_modal,
                })
                rows.append({
                    "level": "demo",
                    "persona_id": pid,
                    "var": f"Q{q_idx}",
                    "is_correct": int(rng.random() < 0.33),
                    "cell_modal": cell_modal,
                })
        df = pd.DataFrame(rows)
        res = evaluate_personal_accuracy_paired(df, cond_a="full", cond_b="demo", n_boot=500, seed=42)
        self.assertAlmostEqual(res["raw_diff_linearity_error"], 0.0, places=12)
        self.assertAlmostEqual(res["norm_diff_linearity_error"], 0.0, places=12)
        self.assertNotAlmostEqual(res["raw_diff_mean"], res["norm_diff_mean"], places=3)

    def test_10_median_vs_mode_point_mass_wasserstein(self):
        """
        TEST 10 (2차 검증): 1D Wasserstein 최적 점질량(1-hot) 위치가 최빈값(mode)이 아니라
        중앙값(median)임을 이론적 반례 및 실제 KGSS 255개 문항 전체에서 검증.
        """
        # 1. 이론적 비대칭 분포: Q = [0.40, 0.25, 0.35] -> 최빈값=1(확률 0.40), 중앙값=2(누적 0.65 >= 0.5)
        q = [0.40, 0.25, 0.35]
        res = evaluate_item_distributions(pred_dists=[q], gold_dist=q, codes=[1, 2, 3])
        # Mode(1번) 원핫 W1 = 0*0.40 + 1*0.25 + 2*0.35 = 0.95
        # Median(2번) 원핫 W1 = 1*0.40 + 0*0.25 + 1*0.35 = 0.75
        self.assertAlmostEqual(res["marginal_modal_1hot_wasserstein"], 0.95, places=12)
        self.assertAlmostEqual(res["marginal_median_1hot_wasserstein"], 0.75, places=12)
        self.assertLess(res["marginal_median_1hot_wasserstein"], res["marginal_modal_1hot_wasserstein"])

        # 2. 실제 KGSS 2023 255개 문항 검증 (0.575823 < 0.590259)
        import json
        from pathlib import Path
        probes_path = Path("c0_check/probes.csv")
        if probes_path.exists():
            probes = pd.read_csv(probes_path)
            w_modes, w_meds, w_unifs = [], [], []
            for _, r in probes.iterrows():
                gold = json.loads(r["gold"])
                codes = json.loads(r["codes"])
                ev = evaluate_item_distributions(pred_dists=[gold], gold_dist=gold, codes=codes)
                self.assertLessEqual(
                    ev["marginal_median_1hot_wasserstein"],
                    ev["marginal_modal_1hot_wasserstein"] + 1e-12,
                )
                w_modes.append(ev["marginal_modal_1hot_wasserstein"])
                w_meds.append(ev["marginal_median_1hot_wasserstein"])
                w_unifs.append(ev["uniform_wasserstein"])
            self.assertAlmostEqual(float(np.mean(w_unifs)), 0.677218, places=5)
            self.assertAlmostEqual(float(np.mean(w_modes)), 0.590259, places=5)
            self.assertAlmostEqual(float(np.mean(w_meds)), 0.575823, places=5)

    def test_11_real_data_integration_audit(self):
        """
        TEST 11 (2차 검증 통합 테스트): 실제 KGSS 2023 참조표 및 로컬 EXAONE 추론 캐시 데이터에서
        원본 결과 파일 보존(SHA-256), 가중치 일치 중심화 subgroup r, 문항 부트스트랩 CI,
        동일 문항 교집합(6문항/125문항) Wasserstein 재계산 정합성을 통합 검증.
        """
        import hashlib
        import json
        from pathlib import Path

        # 1. 원본 GitHub c0_check*/c0_validity_result.md 보존 검증
        expected_hashes = {
            "c0_check/c0_validity_result.md": "77b47387ccfad681",
            "c0_check_adult/c0_validity_result.md": "ed929349d134b730",
            "c0_check_demo/c0_validity_result.md": "64f81ed43e3edcda",
            "c0_check_full/c0_validity_result.md": "54c426e67f73e03a",
        }
        for rel_path, prefix in expected_hashes.items():
            p = Path(rel_path)
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                self.assertEqual(h, prefix)

        # 2. 실제 paired_demo 125문항에서 가중치 일치 중심화 및 부트스트랩 CI 검증
        demo_dir = Path("audit_outputs/paired_demo_125items")
        if (demo_dir / "probes.csv").exists() and (demo_dir / "logit_responses.csv").exists():
            pr = pd.read_csv(demo_dir / "probes.csv")
            lg = pd.read_csv(demo_dir / "logit_responses.csv")[["probe_id", "model", "model_dist", "mass"]]
            rc = pd.read_csv(demo_dir / "real_cells.csv")
            d = pr.merge(lg, on="probe_id")
            d["model_pred"] = [
                float(np.sum(np.array(json.loads(c), dtype=float) * np.array(json.loads(dist), dtype=float)))
                for c, dist in zip(d["codes"], d["model_dist"])
            ]
            mc = d.groupby(["var", "sex", "age_grp"], as_index=False)["model_pred"].mean()
            merged = mc.merge(rc, on=["var", "sex", "age_grp"], how="inner")
            sub_r = compute_subgroup_correlations(merged, n_boot=2000, seed=42)
            self.assertAlmostEqual(sub_r["existing_mean_r"], 0.0010, places=3)
            self.assertAlmostEqual(sub_r["centered_mean_r"], 0.1833, places=3)
            self.assertGreater(sub_r["centered_boot_ci_low"], 0.10)
            self.assertLess(sub_r["centered_boot_ci_high"], 0.26)

    def test_12_unequal_cell_size_end_to_end_score_invariant(self):
        """
        TEST 12 (3차 검증 - End-to-End Caller 및 불균등 셀 인원수 불변식 테스트):
        셀별 프롬프트 수 n_c가 불균등(예: 셀 A는 1명, 셀 B는 9명)할 때,
        kgss_c0_validity.score() 실제 실행 경로가 각 행에 w_i = w_c / n_c를 부여하여
        독립적인 셀 평균 선집계(sum_c w_c P_bar_c)와 모든 지표(W1, TVD, 조건부 엔트로피,
        모집단 엔트로피, 상호정보량)에서 1e-12 이내로 정확히 일치하는지 검증.
        """
        import argparse
        import json
        import tempfile
        from pathlib import Path
        import kgss_c0_validity
        from kgss_metrics import build_cell_poststratified_row_weights, entropy_bits, tvd_distance

        # 1. 합성 불균등 셀(n_A=1, n_B=9)에서 score() End-to-End 실행 검증
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold = [0.60, 0.30, 0.10]
            codes = [1, 2, 3]
            # 셀 (1.0, '18-29'): w_A = 0.70, 프롬프트 1개 -> P_A = [0.80, 0.15, 0.05]
            # 셀 (2.0, '60+'):   w_B = 0.30, 프롬프트 9개 -> P_B 평균 = [0.10, 0.50, 0.40]
            p_a = [0.80, 0.15, 0.05]
            p_b_list = [
                [0.05, 0.55, 0.40], [0.15, 0.45, 0.40], [0.10, 0.50, 0.40],
                [0.08, 0.52, 0.40], [0.12, 0.48, 0.40], [0.10, 0.50, 0.40],
                [0.07, 0.53, 0.40], [0.13, 0.47, 0.40], [0.10, 0.50, 0.40],
            ]
            probes_rows = [{
                "probe_id": "DEMO_Q1_0", "var": "Q1", "sex": 1.0, "age_grp": "18-29",
                "prompt": "test", "codes": json.dumps(codes), "gold": json.dumps(gold),
            }]
            logit_rows = [{
                "probe_id": "DEMO_Q1_0", "model": "test_model",
                "model_dist": json.dumps(p_a), "mass": 0.999,
            }]
            for idx, pb in enumerate(p_b_list, start=1):
                probes_rows.append({
                    "probe_id": f"DEMO_Q1_{idx}", "var": "Q1", "sex": 2.0, "age_grp": "60+",
                    "prompt": "test", "codes": json.dumps(codes), "gold": json.dumps(gold),
                })
                logit_rows.append({
                    "probe_id": f"DEMO_Q1_{idx}", "model": "test_model",
                    "model_dist": json.dumps(pb), "mass": 0.999,
                })

            cells_rows = [
                {
                    "sex": 1.0, "age_grp": "18-29", "var": "Q1", "n_cell": 100,
                    "cell_weight_item": 0.70, "cell_weight_pop": 0.70,
                    "real_dist": json.dumps([0.75, 0.20, 0.05]),
                    "real_mean": 1.30, "real_grand": 1.50,
                    "cell_weight_item_norm": 0.70, "cell_weight_pop_norm": 0.70,
                },
                {
                    "sex": 2.0, "age_grp": "60+", "var": "Q1", "n_cell": 100,
                    "cell_weight_item": 0.30, "cell_weight_pop": 0.30,
                    "real_dist": json.dumps([0.25, 0.55, 0.20]),
                    "real_mean": 1.95, "real_grand": 1.50,
                    "cell_weight_item_norm": 0.30, "cell_weight_pop_norm": 0.30,
                },
            ]
            pd.DataFrame(probes_rows).to_csv(tmp / "probes.csv", index=False)
            pd.DataFrame(logit_rows).to_csv(tmp / "logit_responses.csv", index=False)
            pd.DataFrame(cells_rows).to_csv(tmp / "real_cells.csv", index=False)

            kgss_c0_validity.score(
                argparse.Namespace(out=str(tmp), inv="./inventory", use_code_positions=False, overwrite_legacy=False)
            )
            comp = pd.read_csv(tmp / "c0_validity_comparison.csv").iloc[0]

            # 독립적인 셀 평균 선집계 계산: P_pop = 0.70 * P_A + 0.30 * mean(P_B)
            p_b_mean = np.mean(np.array(p_b_list, dtype=float), axis=0)
            indep_pop = 0.70 * np.array(p_a, dtype=float) + 0.30 * p_b_mean
            indep_w1 = wasserstein_1d(indep_pop, gold)
            indep_tvd = tvd_distance(indep_pop, gold)
            indep_h_cond = 0.70 * entropy_bits(p_a) + 0.30 * float(np.mean([entropy_bits(pb) for pb in p_b_list]))
            indep_h_pop = entropy_bits(indep_pop)
            indep_mi = indep_h_pop - indep_h_cond

            self.assertAlmostEqual(float(comp["population_wasserstein"]), indep_w1, places=12)
            self.assertAlmostEqual(float(comp["population_tvd"]), indep_tvd, places=12)
            self.assertAlmostEqual(float(comp["model_conditional_entropy_weighted_bits"]), indep_h_cond, places=12)
            self.assertAlmostEqual(float(comp["model_population_entropy_bits"]), indep_h_pop, places=12)
            self.assertAlmostEqual(float(comp["mutual_information_bits"]), indep_mi, places=12)

            # 구버전 버그(셀 내 모든 행에 w_c를 그대로 부여: w_B_total = 9*0.30 = 2.70)와는 명확히 달라야 함
            buggy_pop = (0.70 * np.array(p_a) + 2.70 * p_b_mean) / (0.70 + 2.70)
            buggy_w1 = wasserstein_1d(buggy_pop, gold)
            self.assertGreater(abs(indep_w1 - buggy_w1), 0.15)

        # 2. 실제 paired_demo 125문항 및 공통 6문항에서 score() 출력과 독립 셀 선집계 일치 검증
        demo_comp_path = Path("audit_outputs/paired_demo_125items/c0_validity_comparison.csv")
        full_comp_path = Path("audit_outputs/full_6items/c0_validity_comparison.csv")
        if demo_comp_path.exists() and full_comp_path.exists():
            dm = pd.read_csv(demo_comp_path)
            fl = pd.read_csv(full_comp_path)
            # 모든 문항에서 행 가중 집계(population_wasserstein) == 셀 선집계(population_wasserstein_cell_weighted)
            max_diff_w1 = float(np.max(np.abs(dm["population_wasserstein"] - dm["population_wasserstein_cell_weighted"])))
            max_diff_tvd = float(np.max(np.abs(dm["population_tvd"] - dm["population_tvd_cell_weighted"])))
            self.assertLess(max_diff_w1, 1e-12)
            self.assertLess(max_diff_tvd, 1e-12)
            # 125문항 평균 = 0.6020, 공통 6문항 평균 = 0.2713 검증
            self.assertAlmostEqual(float(dm["population_wasserstein"].mean()), 0.6020, places=4)
            dm_6 = dm[dm["var"].isin(set(fl["var"]))]
            self.assertAlmostEqual(float(dm_6["population_wasserstein"].mean()), 0.2713, places=4)

    def test_13_nearly_normalized_distributions_are_exactly_normalized(self):
        from kgss_metrics import entropy_bits, tvd_distance
        p = np.array([0.25, 0.75])
        raw = p * 1.00005
        np.testing.assert_allclose(validate_prob_dist(raw, tol=1e-3), p, atol=1e-15)
        self.assertAlmostEqual(tvd_distance(raw, p), 0.0, places=14)
        self.assertAlmostEqual(entropy_bits(raw), entropy_bits(p), places=14)

    def test_14_persona_education_does_not_assert_graduation(self):
        from kgss_persona_bootstrap import persona_sentence
        labels = {
            "SEX": {"1.0": "남성"}, "EDUC": {"4.0": "대학교"},
            "REGION": {"1.0": "서울"}, "MARITAL": {"1.0": "미혼"},
            "URBAN": {"1.0": "도시"},
        }
        row = pd.Series({"AGE": 22, "SEX": 1, "EDUC": 4, "REGION": 1, "MARITAL": 1, "URBAN": 1})
        for addressee in ("당신", "이사람"):
            for register in ("formal", "plain"):
                sentence = persona_sentence(row, labels, addressee, register)
                self.assertIn("최종학력은 대학교", sentence)
                self.assertNotIn("졸업", sentence)

    @staticmethod
    def _write_score_fixture(tmp):
        import json
        pd.DataFrame([
            {"probe_id": "C0_BIN", "var": "BIN", "prompt": "binary", "codes": "[1, 2]", "gold": "[0.5, 0.5]"},
            {"probe_id": "C0_NOM", "var": "NOM", "prompt": "nominal", "codes": "[1, 2, 3]", "gold": "[0.2, 0.5, 0.3]"},
        ]).to_csv(tmp / "probes.csv", index=False)
        pd.DataFrame([
            {"probe_id": "C0_BIN", "model": "m", "model_dist": "[0.6, 0.4]", "mass": 1.0},
            {"probe_id": "C0_NOM", "model": "m", "model_dist": "[0.8, 0.1, 0.1]", "mass": 1.0},
        ]).to_csv(tmp / "logit_responses.csv", index=False)
        pd.DataFrame([
            {"var": v, "sex": 1, "age_grp": "18-29", "real_mean": 1.5, "real_grand": 1.5}
            for v in ("BIN", "NOM")
        ]).to_csv(tmp / "real_cells.csv", index=False)
        pd.DataFrame([
            {"var": "BIN", "scale_type": "binary", "include_in_clean_eval": True, "w1_valid_in_clean_eval": True},
            {"var": "NOM", "scale_type": "nominal_unordered", "include_in_clean_eval": True, "w1_valid_in_clean_eval": False},
        ]).to_csv(tmp / "classification.csv", index=False)

    def test_15_score_applies_classification_to_metrics_and_subgroups(self):
        import argparse
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import kgss_c0_validity
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_score_fixture(tmp)
            args = argparse.Namespace(out=str(tmp), inv=str(tmp), classification=str(tmp / "classification.csv"))
            with patch("kgss_metrics.compute_subgroup_correlations", wraps=compute_subgroup_correlations) as subgroup:
                with contextlib.redirect_stdout(io.StringIO()):
                    kgss_c0_validity.score(args)
                self.assertEqual(set(subgroup.call_args.args[0]["var"]), {"BIN"})
            metrics = pd.read_csv(tmp / "c0_validity_comparison.csv").set_index("var")
            self.assertTrue(np.isnan(metrics.loc["NOM", "population_wasserstein"]))
            self.assertGreater(metrics.loc["NOM", "diagnostic_population_wasserstein"], 0)
            self.assertGreater(metrics.loc["NOM", "population_tvd"], 0)
            self.assertAlmostEqual(metrics.loc["BIN", "population_wasserstein"], 0.1, places=12)
            self.assertIn("정제 이진/단조서열 1문항", (tmp / "c0_validity_recomputed.md").read_text())
            cl = pd.read_csv(tmp / "classification.csv")
            cl.iloc[:1].to_csv(tmp / "classification.csv", index=False)
            with self.assertRaisesRegex(ValueError, "분류표에 없는"):
                kgss_c0_validity.score(args)

    def test_16_score_rejects_missing_and_duplicate_logit_rows(self):
        import argparse
        import tempfile
        from pathlib import Path
        import kgss_c0_validity
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_score_fixture(tmp)
            args = argparse.Namespace(out=str(tmp), inv=str(tmp))
            logits = pd.read_csv(tmp / "logit_responses.csv")
            logits.iloc[:1].to_csv(tmp / "logit_responses.csv", index=False)
            with self.assertRaisesRegex(ValueError, "누락 1개"):
                kgss_c0_validity.score(args)
            pd.concat([logits, logits.iloc[:1]], ignore_index=True).to_csv(tmp / "logit_responses.csv", index=False)
            with self.assertRaisesRegex(ValueError, "중복"):
                kgss_c0_validity.score(args)

    def test_17_actual_clean_build_marks_provisional_stems(self):
        import argparse
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        import kgss_c0_validity
        required = [Path("inventory/kgss_clean.parquet"), Path("inventory/variables.csv"),
                    Path("inventory/wave_items.csv"), Path("selection/variables_worded.csv")]
        if any(not path.exists() for path in required):
            self.skipTest("실제 clean build 검증용 KGSS 참조 파일이 없습니다.")
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                kgss_c0_validity.build(argparse.Namespace(
                    inv="./inventory", out=tmpdir, year=2023,
                    worded="./selection/variables_worded.csv", level="c0", clean_prompts=True,
                ))
            probes = pd.read_csv(Path(tmpdir) / "probes.csv")
            self.assertNotIn("SAMPLEAB", set(probes["var"]))
            self.assertFalse(probes["prompt"].str.contains(r"문항:\s*(?:nan|None)(?:\n|$)", regex=True).any())
            fallback = probes[probes["prompt_stem_source"] == "label_fallback_provisional"]
            self.assertEqual(set(fallback["var"]), {"HATOPSEX", "HHSEX", "HHMARI", "HHAWAY", "HHEMPLY"})
            self.assertTrue(fallback["prompt_requires_review"].all())
            for _, row in probes.iterrows():
                codes, gold = json.loads(row["codes"]), json.loads(row["gold"])
                self.assertEqual(len(codes), len(gold))
                self.assertTrue(np.all(np.asarray(gold) >= 0))
                self.assertLess(abs(sum(gold) - 1), 1e-3)
                for line in row["prompt"].splitlines():
                    if line[:1].isdigit() and ". " in line:
                        self.assertFalse(kgss_c0_validity.is_leaked_missing_label(line.split(". ", 1)[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
