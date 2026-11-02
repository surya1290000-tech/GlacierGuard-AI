"""
Unit tests for Phase 9.1.4: Canonical Classification Label Reconciliation
==========================================================================
Tests verify deterministic row-level label assignment, 4-state taxonomy,
ambiguous state isolation, cross-artifact count reconciliation, temporal
splits, GroupKFold lake leakage prevention, operational heuristic isolation,
and frozen baseline model byte-for-byte integrity.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from glacierguard.classification_labels import (
    CLASS_STABLE,
    CLASS_EXPANDING,
    CLASS_UNUSUAL_CHANGE,
    CLASS_AMBIGUOUS,
    PHYSICAL_CLASSES,
    ALL_LABEL_STATES,
    OPERATIONAL_HEURISTIC_STATES,
    AMBIG_CAT_A_SMALL_LAKE_POS,
    AMBIG_CAT_B_LARGE_LAKE_POS,
    AMBIG_CAT_C_SMALL_LAKE_NEG,
    AMBIG_CAT_D_LARGE_LAKE_NEG,
    compute_combined_sigma_delta,
    classify_observation_condition,
    classify_ambiguous_category,
)

FROZEN_XGBOOST_PATH = Path("reports/models/09_final_fused_xgboost.pkl")
EXPECTED_FROZEN_SHA256 = "5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f"


class TestCanonicalLabelFormulation:
    """Verify physical and mathematical semantics of canonical label assignment."""

    def test_noise_floor_sub_sigma_is_stable(self):
        delta = pd.Series([0.004])   # 0.004 km2
        sigma = pd.Series([0.008])   # 0.008 km2 (delta = 0.5 sigma)
        rel = pd.Series([15.0])      # relative 15% (small lake denominator)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_STABLE

    def test_sub_5pct_rel_is_stable(self):
        delta = pd.Series([0.015])   # 0.015 km2
        sigma = pd.Series([0.008])   # sigma = 0.008 km2 (delta = 1.875 sigma)
        rel = pd.Series([3.5])       # relative 3.5% (large lake denominator)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_STABLE

    def test_moderate_shrinkage_is_stable(self):
        delta = pd.Series([-0.018])  # -0.018 km2
        sigma = pd.Series([0.008])   # sigma = 0.008 km2 (delta = -2.25 sigma)
        rel = pd.Series([-14.0])     # relative -14% (moderate contraction)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_STABLE

    def test_moderate_expansion_clearing_noise_is_expanding(self):
        delta = pd.Series([0.016])   # 0.016 km2
        sigma = pd.Series([0.008])   # sigma = 0.008 km2 (delta = 2.0 sigma)
        rel = pd.Series([12.0])      # relative 12% (> 5% and <= 20%)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_EXPANDING

    def test_exceptional_growth_is_unusual_change(self):
        delta = pd.Series([0.040])   # 0.040 km2
        sigma = pd.Series([0.010])   # sigma = 0.010 km2 (delta = 4.0 sigma)
        rel = pd.Series([35.0])      # relative 35% (> 20%)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_UNUSUAL_CHANGE

    def test_catastrophic_drainage_is_unusual_change(self):
        delta = pd.Series([-0.050])  # -0.050 km2
        sigma = pd.Series([0.010])   # sigma = 0.010 km2 (delta = -5.0 sigma)
        rel = pd.Series([-30.0])     # relative -30% (<-20%)
        label = classify_observation_condition(delta, rel, sigma)
        assert label.iloc[0] == CLASS_UNUSUAL_CHANGE

    def test_exact_threshold_boundaries(self):
        sigma = pd.Series([0.010])

        # Exactly at 1.0 sigma boundary
        delta_at_1s = pd.Series([0.010])
        rel_med = pd.Series([8.0])
        assert classify_observation_condition(delta_at_1s, rel_med, sigma).iloc[0] == CLASS_STABLE

        # Just above 1.0 sigma boundary
        delta_above_1s = pd.Series([0.01001])
        assert classify_observation_condition(delta_above_1s, rel_med, sigma).iloc[0] == CLASS_EXPANDING

        # Exactly at 3.0 sigma and 20% boundary
        delta_at_3s = pd.Series([0.030])
        rel_at_20 = pd.Series([20.0])
        assert classify_observation_condition(delta_at_3s, rel_at_20, sigma).iloc[0] == CLASS_EXPANDING

        # Just above 3.0 sigma and >20% boundary
        delta_above_3s = pd.Series([0.03001])
        rel_above_20 = pd.Series([20.01])
        assert classify_observation_condition(delta_above_3s, rel_above_20, sigma).iloc[0] == CLASS_UNUSUAL_CHANGE


class TestAmbiguousStateHandling:
    """Verify explicit isolation and taxonomy of ambiguous observations."""

    def test_category_a_small_lake_high_rel_sub_3sigma(self):
        # Small lake: 25% relative growth, but delta = 1.875 sigma (<= 3.0 sigma noise)
        delta = pd.Series([0.015])
        sigma = pd.Series([0.008])
        rel = pd.Series([25.0])
        label = classify_observation_condition(delta, rel, sigma)
        reason = classify_ambiguous_category(delta, rel, sigma, label)
        assert label.iloc[0] == CLASS_AMBIGUOUS
        assert reason.iloc[0] == AMBIG_CAT_A_SMALL_LAKE_POS

    def test_category_b_large_lake_high_sigma_sub_20pct(self):
        # Large lake: 3.5 sigma growth, but relative growth is only 15% (<= 20%)
        delta = pd.Series([0.035])
        sigma = pd.Series([0.010])
        rel = pd.Series([15.0])
        label = classify_observation_condition(delta, rel, sigma)
        reason = classify_ambiguous_category(delta, rel, sigma, label)
        assert label.iloc[0] == CLASS_AMBIGUOUS
        assert reason.iloc[0] == AMBIG_CAT_B_LARGE_LAKE_POS

    def test_category_c_small_lake_shrinkage_high_rel_sub_3sigma(self):
        # Small lake: -25% shrinkage, but delta = -2.0 sigma (>= -3.0 sigma noise)
        delta = pd.Series([-0.016])
        sigma = pd.Series([0.008])
        rel = pd.Series([-25.0])
        label = classify_observation_condition(delta, rel, sigma)
        reason = classify_ambiguous_category(delta, rel, sigma, label)
        assert label.iloc[0] == CLASS_AMBIGUOUS
        assert reason.iloc[0] == AMBIG_CAT_C_SMALL_LAKE_NEG

    def test_category_d_large_lake_shrinkage_high_sigma_sub_20pct(self):
        # Large lake: -3.5 sigma shrinkage, but relative shrinkage is only -15% (>= -20%)
        delta = pd.Series([-0.035])
        sigma = pd.Series([0.010])
        rel = pd.Series([-15.0])
        label = classify_observation_condition(delta, rel, sigma)
        reason = classify_ambiguous_category(delta, rel, sigma, label)
        assert label.iloc[0] == CLASS_AMBIGUOUS
        assert reason.iloc[0] == AMBIG_CAT_D_LARGE_LAKE_NEG

    def test_ambiguous_never_forced_into_stable_or_expanding(self):
        deltas = pd.Series([0.015, 0.035, -0.016, -0.035])
        sigmas = pd.Series([0.008, 0.010, 0.008, 0.010])
        rels = pd.Series([25.0, 15.0, -25.0, -15.0])
        labels = classify_observation_condition(deltas, rels, sigmas)
        for val in labels:
            assert val == CLASS_AMBIGUOUS
            assert val != CLASS_STABLE
            assert val != CLASS_EXPANDING
            assert val != CLASS_UNUSUAL_CHANGE


class TestCrossArtifactCountReconciliation:
    """Verify that class counts reconcile exactly across every Phase 12 artifact."""

    @pytest.fixture
    def canonical_audit(self):
        path = Path("reports/12_canonical_label_audit.json")
        assert path.exists(), "reports/12_canonical_label_audit.json missing"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def canonical_dist_csv(self):
        path = Path("reports/12_canonical_class_distribution.csv")
        assert path.exists(), "reports/12_canonical_class_distribution.csv missing"
        return pd.read_csv(path)

    @pytest.fixture
    def observation_labels_df(self):
        path = Path("reports/12_canonical_observation_labels.parquet")
        assert path.exists(), "reports/12_canonical_observation_labels.parquet missing"
        return pd.read_parquet(path)

    @pytest.fixture
    def label_sensitivity_csv(self):
        path = Path("reports/12_label_sensitivity.csv")
        assert path.exists(), "reports/12_label_sensitivity.csv missing"
        return pd.read_csv(path)

    def test_overall_cohort_reconciliation(self, canonical_audit, canonical_dist_csv, observation_labels_df, label_sensitivity_csv):
        # 1. Total cohort size
        assert len(observation_labels_df) == 11849
        all_row = canonical_dist_csv[canonical_dist_csv["Partition"] == "All Matched (2017-2023)"].iloc[0]
        assert all_row["Total_N"] == 11849

        # 2. Reconcile STABLE
        assert (observation_labels_df["CANONICAL_LABEL"] == CLASS_STABLE).sum() == 10760
        assert all_row["STABLE_N"] == 10760
        assert canonical_audit["canonical_partition_counts"]["All Matched (2017-2023)"]["STABLE_N"] == 10760

        # 3. Reconcile EXPANDING
        assert (observation_labels_df["CANONICAL_LABEL"] == CLASS_EXPANDING).sum() == 77
        assert all_row["EXPANDING_N"] == 77
        assert canonical_audit["canonical_partition_counts"]["All Matched (2017-2023)"]["EXPANDING_N"] == 77

        # 4. Reconcile UNUSUAL_CHANGE
        assert (observation_labels_df["CANONICAL_LABEL"] == CLASS_UNUSUAL_CHANGE).sum() == 115
        assert all_row["UNUSUAL_CHANGE_N"] == 115
        assert canonical_audit["canonical_partition_counts"]["All Matched (2017-2023)"]["UNUSUAL_CHANGE_N"] == 115

        # 5. Reconcile AMBIGUOUS
        assert (observation_labels_df["CANONICAL_LABEL"] == CLASS_AMBIGUOUS).sum() == 897
        assert all_row["AMBIGUOUS_N"] == 897
        assert canonical_audit["canonical_partition_counts"]["All Matched (2017-2023)"]["AMBIGUOUS_N"] == 897

        # 6. Reconcile Row 8 in label_sensitivity.csv
        sens_row8 = label_sensitivity_csv[label_sensitivity_csv["Schema"] == "Hybrid_1s_3s_20pct"].iloc[0]
        assert sens_row8["STABLE_N"] == 10760
        assert sens_row8["EXPANDING_N"] == 77
        assert sens_row8["UNUSUAL_N"] == 115
        assert sens_row8["Ambiguous_N"] == 897

    def test_ambiguous_category_counts_reconciled(self, canonical_audit, observation_labels_df):
        ambig_df = observation_labels_df[observation_labels_df["CANONICAL_LABEL"] == CLASS_AMBIGUOUS]
        vc = ambig_df["AMBIGUOUS_REASON"].value_counts()
        cats = canonical_audit["ambiguous_observations_breakdown"]["categories"]
        assert vc[AMBIG_CAT_A_SMALL_LAKE_POS] == cats["category_a"]["count"] == 497
        assert vc[AMBIG_CAT_B_LARGE_LAKE_POS] == cats["category_b"]["count"] == 3
        assert vc[AMBIG_CAT_C_SMALL_LAKE_NEG] == cats["category_c"]["count"] == 393
        assert vc[AMBIG_CAT_D_LARGE_LAKE_NEG] == cats["category_d"]["count"] == 4
        assert len(ambig_df) == 497 + 3 + 393 + 4 == 897


class TestTemporalSplitCounts:
    """Verify deterministic partition counts across Train, Val, and Test."""

    @pytest.fixture
    def canonical_dist_csv(self):
        return pd.read_csv("reports/12_canonical_class_distribution.csv")

    def test_train_partition_counts(self, canonical_dist_csv):
        tr = canonical_dist_csv[canonical_dist_csv["Partition"] == "Train (<=2021)"].iloc[0]
        assert tr["Total_N"] == 8527
        assert tr["STABLE_N"] == 7744
        assert tr["EXPANDING_N"] == 47
        assert tr["UNUSUAL_CHANGE_N"] == 89
        assert tr["AMBIGUOUS_N"] == 647
        assert tr["Unambiguous_N"] == 7880
        assert tr["STABLE_N"] + tr["EXPANDING_N"] + tr["UNUSUAL_CHANGE_N"] + tr["AMBIGUOUS_N"] == 8527

    def test_val_partition_counts(self, canonical_dist_csv):
        va = canonical_dist_csv[canonical_dist_csv["Partition"] == "Val (2022)"].iloc[0]
        assert va["Total_N"] == 1580
        assert va["STABLE_N"] == 1440
        assert va["EXPANDING_N"] == 11
        assert va["UNUSUAL_CHANGE_N"] == 9
        assert va["AMBIGUOUS_N"] == 120
        assert va["Unambiguous_N"] == 1460
        assert va["STABLE_N"] + va["EXPANDING_N"] + va["UNUSUAL_CHANGE_N"] + va["AMBIGUOUS_N"] == 1580

    def test_test_partition_counts(self, canonical_dist_csv):
        te = canonical_dist_csv[canonical_dist_csv["Partition"] == "Test (2023 - Reused Benchmark)"].iloc[0]
        assert te["Total_N"] == 1742
        assert te["STABLE_N"] == 1576
        assert te["EXPANDING_N"] == 19
        assert te["UNUSUAL_CHANGE_N"] == 17
        assert te["AMBIGUOUS_N"] == 130
        assert te["Unambiguous_N"] == 1612
        assert te["STABLE_N"] + te["EXPANDING_N"] + te["UNUSUAL_CHANGE_N"] + te["AMBIGUOUS_N"] == 1742


class TestLakeLevelGroupingAndLeakagePrevention:
    """Ensure spatial grouping prevents multi-year lake leakage and data integrity rules hold."""

    def test_zero_duplicate_transitions_in_matched_cohort(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        dups = df_labels.duplicated(subset=["GLO_ID", "AREA_YEAR"]).sum()
        assert dups == 0

    def test_group_kfold_leakage_safety(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        # Subsample lakes for quick CV verification
        unique_lakes = df_labels["GLO_ID"].unique()[:100]
        sub = df_labels[df_labels["GLO_ID"].isin(unique_lakes)].copy()

        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(sub, groups=sub["GLO_ID"]):
            train_lakes = set(sub.iloc[train_idx]["GLO_ID"])
            val_lakes = set(sub.iloc[val_idx]["GLO_ID"])
            # Zero lake overlap
            assert len(train_lakes & val_lakes) == 0

    def test_target_excluded_from_predictor_schema(self):
        with open("reports/12_regression_data_audit.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = data["feature_specification"]["numerical_features_s1"] + data["feature_specification"]["numerical_features_s2"]
        assert "NEXT_AREA_CHANGE" not in feats
        assert "FUSED_TARGET" not in feats
        assert "CANONICAL_LABEL" not in feats

    def test_2023_and_2024_status_integrity(self):
        with open("reports/12_feature_leakage_audit.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        status_2024 = data["temporal_leakage_verification"]["status_of_2024_data"]
        assert status_2024["has_consecutive_target"] is False
        assert status_2024["used_in_prior_experiments"] is True


class TestOperationalStatusSeparation:
    """Verify that heuristic RiskEngine statuses are strictly separated from ML labels."""

    def test_operational_statuses_distinct_from_physical_classes(self):
        # Physical classes are strictly STABLE, EXPANDING, UNUSUAL_CHANGE
        assert set(PHYSICAL_CLASSES) == {"STABLE", "EXPANDING", "UNUSUAL_CHANGE"}
        # Operational heuristic states contain alert/quality tags
        assert "WATCH" in OPERATIONAL_HEURISTIC_STATES
        assert "INSUFFICIENT_DATA" in OPERATIONAL_HEURISTIC_STATES
        assert "STALE" in OPERATIONAL_HEURISTIC_STATES
        assert "WATCH" not in PHYSICAL_CLASSES
        assert "INSUFFICIENT_DATA" not in PHYSICAL_CLASSES
        assert "STALE" not in PHYSICAL_CLASSES


class TestFrozenModelIntegrity:
    """Verify byte-for-byte immutability of the frozen baseline XGBoost model."""

    def test_frozen_xgboost_hash_unmodified(self):
        assert FROZEN_XGBOOST_PATH.exists(), f"Frozen model {FROZEN_XGBOOST_PATH} is missing!"
        hasher = hashlib.sha256()
        with open(FROZEN_XGBOOST_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        assert actual_hash == EXPECTED_FROZEN_SHA256, (
            f"FROZEN XGBOOST MODEL MODIFIED!\n"
            f"Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"Actual:   {actual_hash}"
        )
