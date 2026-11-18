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


class TestPhysicalLabelVocabularySeparation:
    """Verify PHYSICAL_CLASSES and OPERATIONAL_HEURISTIC_STATES are strictly disjoint
    and RiskEngine user-facing behavior is preserved."""

    def test_physical_and_operational_sets_are_disjoint(self):
        physical_set = set(PHYSICAL_CLASSES)
        operational_set = set(OPERATIONAL_HEURISTIC_STATES)
        overlap = physical_set & operational_set
        assert overlap == set(), (
            f"PHYSICAL_CLASSES and OPERATIONAL_HEURISTIC_STATES share strings: {overlap!r}. "
            "Physical labels must never appear as operational status codes."
        )

    def test_all_label_states_disjoint_from_operational_states(self):
        assert set(ALL_LABEL_STATES).isdisjoint(set(OPERATIONAL_HEURISTIC_STATES))

    def test_unusual_change_not_in_operational_states(self):
        assert CLASS_UNUSUAL_CHANGE not in OPERATIONAL_HEURISTIC_STATES

    def test_stable_not_in_operational_states(self):
        assert CLASS_STABLE not in OPERATIONAL_HEURISTIC_STATES

    def test_expanding_not_in_operational_states(self):
        assert CLASS_EXPANDING not in OPERATIONAL_HEURISTIC_STATES

    def test_operational_states_contain_expected_tiers(self):
        for tier in ("WATCH", "ELEVATED_ALERT", "ELEVATED ALERT", "STALE", "STALE_DATA", "INSUFFICIENT_DATA"):
            assert tier in OPERATIONAL_HEURISTIC_STATES

    def test_risk_engine_user_facing_compatibility_preserved(self):
        from glacierguard.risk_engine import map_status_to_user_facing

        assert map_status_to_user_facing("ELEVATED_ALERT", staleness_days=5) == "ELEVATED ALERT"
        assert map_status_to_user_facing("UNUSUAL_CHANGE", staleness_days=5) == "ELEVATED ALERT"
        assert map_status_to_user_facing("WATCH", staleness_days=5) == "WATCH"
        assert map_status_to_user_facing("NORMAL", staleness_days=5) == "SAFE"
        assert map_status_to_user_facing("NORMAL", staleness_days=95) == "STALE DATA"
        assert map_status_to_user_facing("INSUFFICIENT_DATA", staleness_days=None) == "INSUFFICIENT DATA"


class TestBoundaryConditions:
    """Exact boundary tests at 1x/3x U_RMS, +5%, +20%, -5%, -20%, -1x/-3x U_RMS."""

    def _classify_and_reason(self, delta, rel, u_rms):
        d = pd.Series([delta])
        r = pd.Series([rel])
        u = pd.Series([u_rms])
        lab = classify_observation_condition(d, r, u)
        reason = classify_ambiguous_category(d, r, u, lab)
        return lab.iloc[0], reason.iloc[0]

    def test_positive_1x_urms_boundary(self):
        # Exactly +1.0x U_RMS -> STABLE (even if rel > 5% or > 20%)
        assert self._classify_and_reason(0.010, 15.0, 0.010) == (CLASS_STABLE, None)
        assert self._classify_and_reason(0.010, 25.0, 0.010) == (CLASS_STABLE, None)
        # Immediately above +1.0x U_RMS with 5% < rel <= 20% -> EXPANDING
        assert self._classify_and_reason(0.010001, 12.0, 0.010) == (CLASS_EXPANDING, None)
        # Immediately above +1.0x U_RMS with rel > 20% -> AMBIGUOUS (Cat A)
        assert self._classify_and_reason(0.010001, 25.0, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_A_SMALL_LAKE_POS)

    def test_positive_3x_urms_boundary(self):
        # Exactly +3.0x U_RMS and rel == +20.0% -> EXPANDING
        assert self._classify_and_reason(0.030, 20.0, 0.010) == (CLASS_EXPANDING, None)
        # Exactly +3.0x U_RMS and rel > +20.0% -> AMBIGUOUS (Cat A)
        assert self._classify_and_reason(0.030, 20.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_A_SMALL_LAKE_POS)
        # Immediately above +3.0x U_RMS and rel == +20.0% -> AMBIGUOUS (Cat B)
        assert self._classify_and_reason(0.030001, 20.0, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_B_LARGE_LAKE_POS)
        # Immediately above +3.0x U_RMS and rel > +20.0% -> UNUSUAL_CHANGE
        assert self._classify_and_reason(0.030001, 20.01, 0.010) == (CLASS_UNUSUAL_CHANGE, None)

    def test_positive_5pct_and_20pct_boundaries(self):
        # Exactly +5.0% -> STABLE (even if delta > 3.0x U_RMS)
        assert self._classify_and_reason(0.016, 5.0, 0.008) == (CLASS_STABLE, None)
        assert self._classify_and_reason(0.040, 5.0, 0.010) == (CLASS_STABLE, None)
        # Immediately above +5.0% at 2x U_RMS -> EXPANDING; at 4x U_RMS -> AMBIGUOUS (Cat B)
        assert self._classify_and_reason(0.016, 5.01, 0.008) == (CLASS_EXPANDING, None)
        assert self._classify_and_reason(0.040, 5.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_B_LARGE_LAKE_POS)
        # Exactly +20.0% at 2x U_RMS -> EXPANDING; immediately above +20.0% -> AMBIGUOUS (Cat A)
        assert self._classify_and_reason(0.020, 20.0, 0.010) == (CLASS_EXPANDING, None)
        assert self._classify_and_reason(0.020, 20.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_A_SMALL_LAKE_POS)

    def test_negative_1x_and_3x_urms_boundaries(self):
        # Exactly -1.0x U_RMS -> STABLE (even if rel < -20%)
        assert self._classify_and_reason(-0.010, -8.0, 0.010) == (CLASS_STABLE, None)
        assert self._classify_and_reason(-0.010, -25.0, 0.010) == (CLASS_STABLE, None)
        # Immediately below -1.0x U_RMS with rel < -20% -> AMBIGUOUS (Cat C)
        assert self._classify_and_reason(-0.010001, -25.0, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_C_SMALL_LAKE_NEG)
        # Exactly -3.0x U_RMS and rel == -20.0% -> STABLE (moderate contraction convention)
        assert self._classify_and_reason(-0.030, -20.0, 0.010) == (CLASS_STABLE, None)
        # Exactly -3.0x U_RMS and rel < -20.0% -> AMBIGUOUS (Cat C)
        assert self._classify_and_reason(-0.030, -20.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_C_SMALL_LAKE_NEG)
        # Immediately below -3.0x U_RMS and rel == -20.0% -> AMBIGUOUS (Cat D)
        assert self._classify_and_reason(-0.030001, -20.0, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_D_LARGE_LAKE_NEG)
        # Immediately below -3.0x U_RMS and rel < -20.0% -> UNUSUAL_CHANGE
        assert self._classify_and_reason(-0.030001, -20.01, 0.010) == (CLASS_UNUSUAL_CHANGE, None)

    def test_negative_5pct_and_20pct_boundaries(self):
        # Exactly -5.0% -> STABLE (even if delta < -3.0x U_RMS)
        assert self._classify_and_reason(-0.040, -5.0, 0.010) == (CLASS_STABLE, None)
        # Immediately below -5.0% at -4x U_RMS -> AMBIGUOUS (Cat D); at -2x U_RMS -> STABLE
        assert self._classify_and_reason(-0.040, -5.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_D_LARGE_LAKE_NEG)
        assert self._classify_and_reason(-0.020, -5.01, 0.010) == (CLASS_STABLE, None)
        # Exactly -20.0% at -2x U_RMS -> STABLE; at -4x U_RMS -> AMBIGUOUS (Cat D)
        assert self._classify_and_reason(-0.020, -20.0, 0.010) == (CLASS_STABLE, None)
        assert self._classify_and_reason(-0.040, -20.0, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_D_LARGE_LAKE_NEG)
        # Immediately below -20.0% at -2x U_RMS -> AMBIGUOUS (Cat C)
        assert self._classify_and_reason(-0.020, -20.01, 0.010) == (CLASS_AMBIGUOUS, AMBIG_CAT_C_SMALL_LAKE_NEG)


class TestAmbiguousDiagnosticExclusivity:
    """Every ambiguous observation gets exactly one diagnostic category and
    no non-ambiguous observation receives an ambiguous reason."""

    def test_all_ambiguous_in_parquet_have_exactly_one_valid_category(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        ambig = df_labels[df_labels["CANONICAL_LABEL"] == CLASS_AMBIGUOUS]
        valid_cats = {
            AMBIG_CAT_A_SMALL_LAKE_POS,
            AMBIG_CAT_B_LARGE_LAKE_POS,
            AMBIG_CAT_C_SMALL_LAKE_NEG,
            AMBIG_CAT_D_LARGE_LAKE_NEG,
        }
        assert len(ambig) == 897
        assert ambig["AMBIGUOUS_REASON"].isna().sum() == 0
        assert set(ambig["AMBIGUOUS_REASON"].unique()).issubset(valid_cats)

        vc = ambig["AMBIGUOUS_REASON"].value_counts()
        assert vc.get(AMBIG_CAT_A_SMALL_LAKE_POS, 0) == 497
        assert vc.get(AMBIG_CAT_B_LARGE_LAKE_POS, 0) == 3
        assert vc.get(AMBIG_CAT_C_SMALL_LAKE_NEG, 0) == 393
        assert vc.get(AMBIG_CAT_D_LARGE_LAKE_NEG, 0) == 4
        assert vc.sum() == 897

    def test_no_non_ambiguous_observation_has_ambiguous_reason(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        non_ambig = df_labels[df_labels["CANONICAL_LABEL"] != CLASS_AMBIGUOUS]
        assert len(non_ambig) == 10952
        assert non_ambig["AMBIGUOUS_REASON"].isna().all(), (
            "Non-ambiguous observations must NEVER receive an AMBIGUOUS_REASON."
        )

    def test_residual_category_predicates_mutually_exclusive_on_full_cohort(self):
        """Explicitly verify zero overlap across Category A-D boolean masks on all 11,849 rows."""
        df = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        d = df["FUSED_TARGET"]
        rel = df["RELATIVE_CHANGE_PCT_FUSED"]
        u = df["FUSED_SIGMA_DELTA"]
        amb = df["CANONICAL_LABEL"] == CLASS_AMBIGUOUS

        cat_a = amb & (d > u) & (d <= 3.0 * u) & (rel > 20.0)
        cat_b = amb & (d > 3.0 * u) & (rel > 5.0) & (rel <= 20.0)
        cat_c = amb & (d >= -3.0 * u) & (d < -u) & (rel < -20.0)
        cat_d = amb & (d < -3.0 * u) & (rel >= -20.0) & (rel < -5.0)

        hits = cat_a.astype(int) + cat_b.astype(int) + cat_c.astype(int) + cat_d.astype(int)
        assert (hits[amb] == 1).all(), "Every ambiguous observation must match exactly one residual category"
        assert (hits[~amb] == 0).all(), "No non-ambiguous observation may match any residual category"


class TestExcludedPredictorsAndTrainingWeights:
    """Verify target-derived feature exclusion and training-only class weight calculation."""

    def test_excluded_predictors_absent_from_regression_audit(self):
        from glacierguard.classification_labels import EXCLUDED_TARGET_DERIVED_COLUMNS

        required_excluded = {
            "FUSED_TARGET",
            "CANONICAL_LABEL",
            "RELATIVE_CHANGE_PCT_FUSED",
            "FUSED_SIGMA_DELTA",
            "S1_NEXT_AREA_CHANGE",
            "S2_NEXT_AREA_CHANGE",
            "S1_NEXT_UNCERTAINTY",
            "S2_NEXT_UNCERTAINTY",
        }
        assert required_excluded.issubset(set(EXCLUDED_TARGET_DERIVED_COLUMNS))

        audit_path = Path("reports/12_regression_data_audit.json")
        assert audit_path.exists(), "reports/12_regression_data_audit.json missing"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = (
            data.get("feature_specification", {}).get("numerical_features_s1", [])
            + data.get("feature_specification", {}).get("numerical_features_s2", [])
            + data.get("feature_specification", {}).get("categorical_features", [])
        )
        for col in EXCLUDED_TARGET_DERIVED_COLUMNS:
            assert col not in feats, f"Target-derived column {col!r} leaked into predictor schema"

    def test_training_only_class_weights(self):
        from glacierguard.classification_labels import compute_training_class_weights

        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        train_all = df_labels[df_labels["TEMPORAL_SPLIT"] == "TRAIN"]["CANONICAL_LABEL"]
        # Must reject AMBIGUOUS labels
        with pytest.raises(ValueError):
            compute_training_class_weights(train_all)

        train_unambig = train_all[train_all.isin(PHYSICAL_CLASSES)]
        assert len(train_unambig) == 7880
        weights = compute_training_class_weights(train_unambig)
        assert weights[CLASS_STABLE] == pytest.approx(7880 / (3 * 7744), rel=1e-5)
        assert weights[CLASS_EXPANDING] == pytest.approx(7880 / (3 * 47), rel=1e-5)
        assert weights[CLASS_UNUSUAL_CHANGE] == pytest.approx(7880 / (3 * 89), rel=1e-5)


class TestNoMissingOrDuplicateTransitions:
    """Verify GLO_ID + AREA_YEAR uniqueness and single valid label per observation."""

    def test_no_duplicate_glo_id_year_pairs(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        assert df_labels.duplicated(subset=["GLO_ID", "AREA_YEAR"]).sum() == 0

    def test_no_missing_identifiers_or_labels(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        assert df_labels["GLO_ID"].isna().sum() == 0
        assert df_labels["AREA_YEAR"].isna().sum() == 0
        assert df_labels["CANONICAL_LABEL"].isna().sum() == 0

    def test_exactly_one_valid_label_per_observation(self):
        df_labels = pd.read_parquet("reports/12_canonical_observation_labels.parquet")
        assert len(df_labels) == 11849
        assert set(df_labels["CANONICAL_LABEL"].unique()) == set(ALL_LABEL_STATES)
        unambig = df_labels[df_labels["CANONICAL_LABEL"] != CLASS_AMBIGUOUS]
        assert set(unambig["CANONICAL_LABEL"].unique()) == set(PHYSICAL_CLASSES)
        assert len(unambig) == 10952


class TestUncertaintySemanticsEvidenceNote:
    """Verify uncertainty semantics evidence note and Rawlins et al. (2026) documentation."""

    def test_uncertainty_semantics_note_exists_and_documents_rawlins_formula(self):
        note_path = Path("reports/12_uncertainty_semantics_note.md")
        assert note_path.exists(), "reports/12_uncertainty_semantics_note.md missing"
        note = note_path.read_text(encoding="utf-8")
        assert "10.5194/essd-18-5143-2026" in note
        assert "10.5281/zenodo.17802333" in note
        assert "0.0488" in note
        assert "0.537" in note
        assert "0.5489" in note
        assert "U_{\\text{RMS}}" in note

    def test_classification_labels_docstring_documents_empirical_scale(self):
        from glacierguard import classification_labels

        docstring = (classification_labels.__doc__ or "").lower()
        assert "rawlins" in docstring
        assert "u_rms" in docstring
        assert "not a one-sigma" in docstring
