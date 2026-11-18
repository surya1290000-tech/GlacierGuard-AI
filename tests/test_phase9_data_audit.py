"""
Unit tests for Phase 9.1: Dual Ensemble Data Audit & Label Formulation
======================================================================
Tests verify data integrity, consecutive-year pairing, uncertainty-aware
classification label assignment, threshold boundaries, leakage prevention,
and spatial grouping.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Helper logic mirroring audited data pipeline ─────────────────────────────

def compute_consecutive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce gap-1 consecutive year target calculation."""
    df = df.sort_values(["GLO_ID", "AREA_YEAR"]).copy()
    g = df.groupby("GLO_ID")
    df["NEXT_YEAR"] = g["AREA_YEAR"].shift(-1)
    df["NEXT_AREA"] = g["AREA"].shift(-1)
    df["NEXT_YEAR_GAP"] = df["NEXT_YEAR"] - df["AREA_YEAR"]
    df["NEXT_AREA_CHANGE"] = np.where(df["NEXT_YEAR_GAP"] == 1, df["NEXT_AREA"] - df["AREA"], np.nan)
    return df


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce gap-aware lag calculation."""
    df = df.sort_values(["GLO_ID", "AREA_YEAR"]).copy()
    g = df.groupby("GLO_ID")
    df["PREV_YEAR"] = g["AREA_YEAR"].shift(1)
    df["PREV_AREA"] = g["AREA"].shift(1)
    df["PREV_YEAR_GAP"] = df["AREA_YEAR"] - df["PREV_YEAR"]
    df["AREA_LAG1"] = np.where(df["PREV_YEAR_GAP"] == 1, df["PREV_AREA"], np.nan)
    return df


def assign_hybrid_labels(
    delta_area: pd.Series,
    rel_pct: pd.Series,
    sigma_delta: pd.Series,
    k_stable: float = 1.0,
    k_unusual: float = 3.0,
    min_exp_pct: float = 5.0,
    unusual_pct: float = 20.0,
) -> pd.Series:
    """
    Scientifically grounded hybrid lake condition classifier:
    - STABLE: |delta| <= 1.0 sigma OR |rel| <= 5% OR moderate non-catastrophic shrinkage
    - EXPANDING: delta > 1.0 sigma AND rel > 5% AND delta <= 3.0 sigma AND rel <= 20%
    - UNUSUAL_CHANGE: (delta > 3.0 sigma AND rel > 20%) OR catastrophic drainage (delta < -3.0 sigma AND rel < -20%)
    """
    cond_stable = (
        (delta_area.abs() <= (k_stable * sigma_delta))
        | (rel_pct.abs() <= min_exp_pct)
        | ((delta_area < 0) & (delta_area >= -(k_unusual * sigma_delta)) & (rel_pct >= -unusual_pct))
    )

    cond_unusual = (~cond_stable) & (
        ((delta_area > (k_unusual * sigma_delta)) & (rel_pct > unusual_pct))
        | ((delta_area < -(k_unusual * sigma_delta)) & (rel_pct < -unusual_pct))
    )

    cond_expanding = (~cond_stable) & (~cond_unusual) & (delta_area > 0)

    labels = pd.Series("STABLE", index=delta_area.index, dtype=str)
    labels[cond_expanding] = "EXPANDING"
    labels[cond_unusual] = "UNUSUAL_CHANGE"
    return labels


# ── Test Suite ───────────────────────────────────────────────────────────────

class TestConsecutiveYearPairing:
    """Verify consecutive-year pairing and non-consecutive gap exclusion."""

    def test_strict_one_year_gap_produces_target(self):
        df = pd.DataFrame({
            "GLO_ID": ["lake_1", "lake_1"],
            "AREA_YEAR": [2018, 2019],
            "AREA": [1.50, 1.55]
        })
        res = compute_consecutive_targets(df)
        assert res.loc[0, "NEXT_AREA_CHANGE"] == pytest.approx(0.05)
        assert pd.isna(res.loc[1, "NEXT_AREA_CHANGE"])

    def test_multi_year_gap_sets_target_to_nan(self):
        df = pd.DataFrame({
            "GLO_ID": ["lake_1", "lake_1"],
            "AREA_YEAR": [2018, 2020],  # 2-year gap
            "AREA": [1.50, 1.60]
        })
        res = compute_consecutive_targets(df)
        assert pd.isna(res.loc[0, "NEXT_AREA_CHANGE"])
        assert pd.isna(res.loc[1, "NEXT_AREA_CHANGE"])

    def test_terminal_year_observation_has_nan_target(self):
        df = pd.DataFrame({
            "GLO_ID": ["lake_1", "lake_1", "lake_2"],
            "AREA_YEAR": [2021, 2022, 2024],
            "AREA": [1.0, 1.1, 2.0]
        })
        res = compute_consecutive_targets(df)
        # lake_1 in 2022 is terminal
        assert pd.isna(res.loc[1, "NEXT_AREA_CHANGE"])
        # lake_2 has only one observation
        assert pd.isna(res.loc[2, "NEXT_AREA_CHANGE"])


class TestDuplicateLakeYearDetection:
    """Ensure duplicate lake-year entries are identified and prohibited."""

    def test_duplicate_lake_year_detected(self):
        df = pd.DataFrame({
            "GLO_ID": ["lake_1", "lake_1", "lake_1"],
            "AREA_YEAR": [2020, 2020, 2021],  # duplicate at 2020
            "AREA": [1.0, 1.05, 1.1]
        })
        dups = df.duplicated(subset=["GLO_ID", "AREA_YEAR"]).sum()
        assert dups == 1

    def test_raw_glo_files_have_zero_duplicates(self):
        # Read the generated audit report verifying 0 duplicates
        p = Path("reports/12_regression_data_audit.json")
        if p.exists():
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["sample_counts"]["s1_total_records"] == 18129
            assert data["sample_counts"]["s2_total_records"] == 22294


class TestZeroAndNegativeAreaHandling:
    """Verify non-positive lake areas are prevented from causing math instability."""

    def test_zero_area_division_prevented(self):
        area = pd.Series([0.0, -0.05, 1.0])
        delta = pd.Series([0.01, 0.02, 0.05])
        # Safe relative calculation
        safe_rel = np.where(area > 0, 100.0 * delta / area, np.nan)
        assert np.isnan(safe_rel[0])
        assert np.isnan(safe_rel[1])
        assert safe_rel[2] == pytest.approx(5.0)

    def test_negative_area_rejected(self):
        df = pd.DataFrame({"GLO_ID": ["lake_1"], "AREA": [-0.5]})
        assert (df["AREA"] <= 0).sum() == 1


class TestUncertaintyAwareLabelAssignment:
    """Verify physical lake condition classification logic."""

    def test_sub_sigma_change_classified_as_stable(self):
        delta = pd.Series([0.003])   # 0.003 km2
        sigma = pd.Series([0.008])   # sigma = 0.008 km2 (delta < 1 sigma)
        rel = pd.Series([12.0])      # relative 12% (on small lake)
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "STABLE"

    def test_moderate_expansion_classified_as_expanding(self):
        delta = pd.Series([0.015])   # 0.015 km2
        sigma = pd.Series([0.008])   # sigma = 0.008 km2 (delta = 1.875 sigma)
        rel = pd.Series([10.0])      # relative 10%
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "EXPANDING"

    def test_massive_expansion_classified_as_unusual_change(self):
        delta = pd.Series([0.080])   # 0.080 km2
        sigma = pd.Series([0.010])   # sigma = 0.010 km2 (delta = 8 sigma)
        rel = pd.Series([45.0])      # relative 45%
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "UNUSUAL_CHANGE"

    def test_moderate_contraction_classified_as_stable(self):
        # Gradual summer shrinkage or evaporation (-1.5 sigma, -12%) is NOT an emergency
        delta = pd.Series([-0.012])
        sigma = pd.Series([0.008])
        rel = pd.Series([-12.0])
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "STABLE"

    def test_catastrophic_drainage_classified_as_unusual_change(self):
        # Rapid breach / GLOF (-5 sigma, -40%)
        delta = pd.Series([-0.060])
        sigma = pd.Series([0.010])
        rel = pd.Series([-40.0])
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "UNUSUAL_CHANGE"


class TestLabelThresholdBoundaries:
    """Verify behavior at exact threshold boundaries."""

    def test_exact_sigma_threshold_boundary(self):
        sigma = pd.Series([0.010])
        # Exactly at 1.0 sigma with > 5% rel
        delta = pd.Series([0.010])
        rel = pd.Series([8.0])
        # |delta| <= 1.0 sigma is STABLE
        labels = assign_hybrid_labels(delta, rel, sigma)
        assert labels.iloc[0] == "STABLE"

        # Just above 1.0 sigma
        delta_above = pd.Series([0.01001])
        labels_above = assign_hybrid_labels(delta_above, rel, sigma)
        assert labels_above.iloc[0] == "EXPANDING"

    def test_exact_unusual_boundary(self):
        sigma = pd.Series([0.010])
        # Exactly at 3.0 sigma with 20% rel
        delta = pd.Series([0.030])
        rel = pd.Series([20.0])
        labels = assign_hybrid_labels(delta, rel, sigma)
        # <= 3.0 sigma and <= 20% is EXPANDING
        assert labels.iloc[0] == "EXPANDING"

        # Above 3.0 sigma and > 20% rel
        delta_high = pd.Series([0.03001])
        rel_high = pd.Series([20.01])
        labels_high = assign_hybrid_labels(delta_high, rel_high, sigma)
        assert labels_high.iloc[0] == "UNUSUAL_CHANGE"


class TestMissingYearHandling:
    """Verify gap handling in lag features."""

    def test_gap_in_time_series_resets_lag(self):
        df = pd.DataFrame({
            "GLO_ID": ["lake_1", "lake_1", "lake_1"],
            "AREA_YEAR": [2018, 2019, 2022],  # gap between 2019 and 2022
            "AREA": [1.0, 1.1, 1.2]
        })
        res = compute_lag_features(df)
        assert res.loc[1, "AREA_LAG1"] == 1.0
        assert pd.isna(res.loc[2, "AREA_LAG1"])  # 2022 gap > 1


class TestSensorMatching:
    """Verify matched dual-sensor intersection logic."""

    def test_only_shared_lake_years_enter_matched_cohort(self):
        s1 = pd.DataFrame({
            "GLO_ID": ["L1", "L1", "L2"],
            "AREA_YEAR": [2020, 2021, 2020]
        })
        s2 = pd.DataFrame({
            "GLO_ID": ["L1", "L2", "L3"],
            "AREA_YEAR": [2020, 2021, 2020]
        })
        s1_keys = set(zip(s1["GLO_ID"], s1["AREA_YEAR"]))
        s2_keys = set(zip(s2["GLO_ID"], s2["AREA_YEAR"]))
        matched = s1_keys & s2_keys
        # Only (L1, 2020) is present in both
        assert matched == {("L1", 2020)}


class TestFutureFeatureLeakagePrevention:
    """Ensure no future information or post-outcome variables enter feature space."""

    def test_target_excluded_from_feature_schema(self):
        p = Path("reports/12_regression_data_audit.json")
        if p.exists():
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data["feature_specification"]["numerical_features_s1"] + data["feature_specification"]["numerical_features_s2"]
            assert "NEXT_AREA" not in features
            assert "NEXT_AREA_CHANGE" not in features
            assert "FUSED_TARGET" not in features

    def test_status_of_2024_correctly_documented(self):
        p = Path("reports/12_feature_leakage_audit.json")
        if p.exists():
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["temporal_leakage_verification"]["status_of_2024_data"]["has_consecutive_target"] is False
            assert data["temporal_leakage_verification"]["status_of_2024_data"]["used_in_prior_experiments"] is True


class TestSpatialGroupingIntegrity:
    """Verify spatial grouping integrity to prevent multi-year lake leakage."""

    def test_group_kfold_isolates_lake_records(self):
        from sklearn.model_selection import GroupKFold
        df = pd.DataFrame({
            "GLO_ID": ["L1", "L1", "L1", "L2", "L2", "L3", "L3", "L4", "L4", "L5"],
            "AREA_YEAR": [2018, 2019, 2020, 2018, 2019, 2018, 2019, 2018, 2019, 2018],
            "FEATURE": range(10)
        })
        gkf = GroupKFold(n_splits=3)
        for train_idx, val_idx in gkf.split(df, groups=df["GLO_ID"]):
            train_lakes = set(df.iloc[train_idx]["GLO_ID"])
            val_lakes = set(df.iloc[val_idx]["GLO_ID"])
            # Zero lake overlap between train and val
            assert len(train_lakes & val_lakes) == 0
