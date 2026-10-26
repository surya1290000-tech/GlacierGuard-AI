"""
Unit and Parity Tests for Satellite Processing & Multi-Sensor QA
================================================================
Verifies parity with NB10.1 and NB10.2 satellite surveillance pipelines:
- Lee 5x5 speckle filter
- Backscatter dB conversion and calm water thresholding
- Physical SAR layover / wind chop screening
- MNDWI index and SCL cloud masking
- Morphological cleaning
- Temporal pair classification and cross-sensor agreement
- Conservative operational lake status rules
"""

import numpy as np
import pytest
from glacierguard.satellite.morphology import clean_mask
from glacierguard.satellite.optical import (
    assess_optical_usability,
    classify_optical_water,
    compute_mndwi,
    screen_scl_clouds,
)
from glacierguard.satellite.qa import (
    assign_lake_status,
    evaluate_cross_sensor_agreement,
    match_temporal_pair,
)
from glacierguard.satellite.sar import (
    assess_sar_validity,
    classify_sar_water,
    lee_filter,
    linear_to_db,
)


class TestSarProcessing:
    """Validate SAR Lee filtering, dB conversion, and water thresholding."""

    def test_lee_filter_reduces_variance(self):
        rng = np.random.default_rng(42)
        # Constant background of 0.1 with multiplicative gamma speckle noise
        clean_signal = np.full((50, 50), 0.1)
        noise = rng.gamma(shape=4.4, scale=1.0 / 4.4, size=(50, 50))
        noisy_img = clean_signal * noise

        filtered = lee_filter(noisy_img, size=5, num_looks=4.4)

        # Variance of filtered image should be substantially lower than raw noisy image
        assert np.var(filtered) < np.var(noisy_img)
        # Mean intensity should be preserved closely
        assert np.mean(filtered) == pytest.approx(np.mean(noisy_img), rel=0.1)

    def test_linear_to_db_conversion(self):
        vals = np.array([1.0, 0.1, 0.01, 0.001])
        db = linear_to_db(vals)
        expected = np.array([0.0, -10.0, -20.0, -30.0])
        assert np.allclose(db, expected, atol=1e-5)

    def test_classify_sar_water(self):
        # -15 dB is water (calm), -5 dB is land, -40 dB is below noise floor
        vv_db = np.array([
            [-15.0, -18.0],
            [-5.0, -40.0],
        ])
        water_mask = classify_sar_water(vv_db, threshold_db=-12.0)
        assert water_mask[0, 0] is True or water_mask[0, 0] == 1  # -15 dB < -12 dB -> water
        assert water_mask[0, 1] is True or water_mask[0, 1] == 1  # -18 dB < -12 dB -> water
        assert water_mask[1, 0] is False or water_mask[1, 0] == 0 # -5 dB >= -12 dB -> land
        assert water_mask[1, 1] is False or water_mask[1, 1] == 0 # -40 dB < min_valid -> invalid

    def test_assess_sar_validity_gorge_layover(self):
        # Tsho Rolpa scenario: median VV rises to -5.2 dB in Rolwaling gorge
        gorge_vv = np.full(100, -5.2)
        valid, med, p10 = assess_sar_validity(gorge_vv, layover_threshold_db=-8.0)
        assert valid is False
        assert med == pytest.approx(-5.2, abs=1e-2)

        # Calm lake scenario (Imja Tsho post-monsoon): median VV is -15.0 dB
        calm_vv = np.full(100, -15.0)
        valid_calm, med_calm, p10_calm = assess_sar_validity(calm_vv, layover_threshold_db=-8.0)
        assert valid_calm is True
        assert med_calm == pytest.approx(-15.0, abs=1e-2)


class TestOpticalProcessing:
    """Validate MNDWI calculation, SCL cloud screening, and usability checks."""

    def test_compute_mndwi(self):
        # Water: Green high, SWIR low
        green_water = np.array([0.2, 0.3])
        swir_water = np.array([0.02, 0.03])
        mndwi_water = compute_mndwi(green_water, swir_water)
        assert (mndwi_water > 0.5).all()

        # Land/soil: Green lower than SWIR
        green_land = np.array([0.1, 0.15])
        swir_land = np.array([0.25, 0.35])
        mndwi_land = compute_mndwi(green_land, swir_land)
        assert (mndwi_land < 0.0).all()

    def test_screen_scl_clouds(self):
        # 4: vegetation (valid), 6: water (valid), 8: medium cloud (invalid), 9: high cloud (invalid)
        scl = np.array([
            [4, 6],
            [8, 9],
        ])
        cloud_mask, valid_mask = screen_scl_clouds(scl)
        assert not cloud_mask[0, 0] and valid_mask[0, 0]
        assert not cloud_mask[0, 1] and valid_mask[0, 1]
        assert cloud_mask[1, 0] and not valid_mask[1, 0]
        assert cloud_mask[1, 1] and not valid_mask[1, 1]

    def test_assess_optical_usability(self):
        # Clear scene: 10% cloud, 90% valid -> usable
        assert assess_optical_usability(cloud_frac=0.10, valid_frac=0.90) is True
        # Heavy monsoon cloud scene: 85% cloud, 15% valid -> unusable
        assert assess_optical_usability(cloud_frac=0.85, valid_frac=0.15) is False


class TestMorphology:
    """Validate mask noise suppression and hole filling."""

    def test_clean_mask_island_removal(self):
        mask = np.zeros((20, 20), dtype=bool)
        # Large lake: 10x10 = 100 pixels
        mask[2:12, 2:12] = True
        # Small isolated noise island: 2x2 = 4 pixels (below min_size=10)
        mask[15:17, 15:17] = True

        cleaned = clean_mask(mask, min_size=10, max_hole=5)
        # Lake is preserved
        assert cleaned[2:12, 2:12].all()
        # Island is removed
        assert not cleaned[15:17, 15:17].any()

    def test_clean_mask_small_hole_filling(self):
        mask = np.ones((20, 20), dtype=bool)
        # Small 2x2 interior hole
        mask[9:11, 9:11] = False

        cleaned = clean_mask(mask, min_size=10, max_hole=5)
        # Hole of size 4 is filled (<= max_hole=5)
        assert cleaned[9:11, 9:11].all()


class TestMultiSensorQA:
    """Validate temporal matching rules, sensor agreement, and lake status logic."""

    def test_match_temporal_pair(self):
        # 0 days
        diff0, label0 = match_temporal_pair("2026-10-15", "2026-10-15")
        assert diff0 == 0
        assert "SAME_DAY" in label0

        # 2 days
        diff2, label2 = match_temporal_pair("2026-10-15", "2026-10-17")
        assert diff2 == 2
        assert "CLOSE_PAIR" in label2

        # 7 days
        diff7, label7 = match_temporal_pair("2026-10-15", "2026-10-22")
        assert diff7 == 7
        assert "NEAR_PAIR" in label7

        # 135 days (Thulagi monsoon gap)
        diff135, label135 = match_temporal_pair("2026-05-14", "2026-09-26")
        assert diff135 == 135
        assert "TEMPORALLY_UNMATCHED" in label135

    def test_evaluate_cross_sensor_agreement(self):
        # Matched and valid: S2=1.30, S1=1.20 (diff=0.10, mean=1.25 -> 8% disagreement)
        disagree, status = evaluate_cross_sensor_agreement(
            area_s2=1.30, area_s1=1.20, pair_diff_days=0, sar_physically_valid=True
        )
        assert disagree == pytest.approx(8.0, abs=0.1)
        assert status == "GOOD_AGREEMENT"

        # SAR geometric failure (Tsho Rolpa)
        disagree_fail, status_fail = evaluate_cross_sensor_agreement(
            area_s2=1.65, area_s1=0.20, pair_diff_days=0, sar_physically_valid=False
        )
        assert np.isnan(disagree_fail)
        assert status_fail == "SAR_GEOMETRIC_FAILURE"

        # Temporally unmatched (>10d)
        disagree_unmatched, status_unmatched = evaluate_cross_sensor_agreement(
            area_s2=0.73, area_s1=0.70, pair_diff_days=135, sar_physically_valid=True
        )
        assert np.isnan(disagree_unmatched)
        assert status_unmatched == "TEMPORALLY_UNMATCHED"

    def test_assign_lake_status(self):
        # Imja Tsho -> NORMAL
        res_imja = assign_lake_status('imja_tsho', 1.32, 1.30, 1.32, 0, True)
        assert res_imja['status'] == 'NORMAL'

        # Thulagi -> WATCH
        res_thulagi = assign_lake_status('thulagi', 0.71, 0.70, 0.73, 135, True)
        assert res_thulagi['status'] == 'WATCH'

        # Tsho Rolpa -> WATCH
        res_tsho = assign_lake_status('tsho_rolpa', 1.72, 0.20, 1.65, 0, False)
        assert res_tsho['status'] == 'WATCH'

        # Lower Barun -> UNUSUAL_CHANGE
        res_barun = assign_lake_status('lower_barun', 0.70, 0.65, 0.48, 2, True)
        assert res_barun['status'] == 'UNUSUAL_CHANGE'

        # Sabai Tsho -> WATCH
        res_sabai = assign_lake_status('sabai_tsho', 0.25, 0.22, 0.22, 1, True)
        assert res_sabai['status'] == 'WATCH'
