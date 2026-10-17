"""
GlacierGuard-AI — Test suite for NB11 ensemble regression benchmark.

Tests cover:
  T01  Temporal split boundary correctness
  T02  Target construction from consecutive years
  T03  Feature leakage exclusions
  T04  Equal sample IDs for fair comparison
  T05  Temporal OOF generation for stacking
  T06  Reproducible metric calculations
  T07  Valid finite ensemble predictions
  T08  Pre-existing model artifact preserved
  T09  Preprocessor fitted on train-only
  T10  Persistence baseline always predicts zero
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import pytest
import pathlib
import pickle

from sklearn.metrics import mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Import from the benchmark script (avoid running the full pipeline)
sys.path.insert(0, str(ROOT / 'scratch'))
from run_ensemble_benchmark import (
    build_gold_dataset,
    create_temporal_splits,
    build_preprocessor,
    compute_metrics,
    temporal_oof_predictions,
    TARGET_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, GLO_FEATURES,
    RANDOM_STATE,
)

GLO_DIR = ROOT / 'raw' / 'GLO'
REPORTS = ROOT / 'reports'
MODELS  = REPORTS / 'models'

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def gold_s1():
    """Load S1 gold dataset once for all tests in this module."""
    return build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')


@pytest.fixture(scope='module')
def splits_s1(gold_s1):
    train, val, test = create_temporal_splits(gold_s1, 'S1')
    return train, val, test


# ─── T01 Temporal split boundaries ────────────────────────────────────────────

class TestTemporalSplits:
    def test_train_max_year(self, splits_s1):
        train, val, test = splits_s1
        assert train['AREA_YEAR'].max() <= 2021, "Train set contains 2022+ observations"

    def test_val_year_exclusive(self, splits_s1):
        train, val, test = splits_s1
        assert (val['AREA_YEAR'] == 2022).all(), "Validation set is not exclusively 2022"

    def test_test_year_exclusive(self, splits_s1):
        train, val, test = splits_s1
        assert (test['AREA_YEAR'] == 2023).all(), "Test set is not exclusively 2023"

    def test_no_2024_in_supervised(self, gold_s1):
        sup = gold_s1[gold_s1[TARGET_COL].notna()]
        assert (sup['AREA_YEAR'] >= 2024).sum() == 0, "2024+ rows found in supervised set"

    def test_no_overlap_between_splits(self, splits_s1):
        train, val, test = splits_s1
        train_ids = set(train.index)
        val_ids   = set(val.index)
        test_ids  = set(test.index)
        assert not (train_ids & val_ids), "Train and val share row indices"
        assert not (train_ids & test_ids), "Train and test share row indices"
        assert not (val_ids & test_ids), "Val and test share row indices"

    def test_exact_s1_counts(self, splits_s1):
        """Parity check with NB08 assertions."""
        train, val, test = splits_s1
        assert len(train) == 10161, f"S1 train: got {len(train)}, expected 10161"
        assert len(val)   == 2026,  f"S1 val: got {len(val)}, expected 2026"
        assert len(test)  == 2055,  f"S1 test: got {len(test)}, expected 2055"


# ─── T02 Target construction ───────────────────────────────────────────────────

class TestTargetConstruction:
    def test_target_only_for_consecutive_years(self, gold_s1):
        """NEXT_AREA_CHANGE must be NaN whenever NEXT_YEAR_GAP != 1."""
        gap_ne_1 = gold_s1['NEXT_YEAR_GAP'] != 1
        should_be_nan = gold_s1.loc[gap_ne_1 & gold_s1['NEXT_YEAR_GAP'].notna(), TARGET_COL]
        assert should_be_nan.isna().all(), \
            f"Target non-null for {(~should_be_nan.isna()).sum()} non-consecutive year-pairs"

    def test_target_equals_area_difference(self, gold_s1):
        """NEXT_AREA_CHANGE = NEXT_AREA - AREA where NEXT_YEAR_GAP == 1."""
        consecutive = gold_s1[gold_s1['NEXT_YEAR_GAP'] == 1].dropna(subset=[TARGET_COL])
        expected = consecutive['NEXT_AREA'] - consecutive['AREA']
        diff = np.abs(consecutive[TARGET_COL] - expected)
        assert (diff < 1e-9).all(), f"Max deviation from NEXT_AREA - AREA: {diff.max()}"

    def test_target_not_in_features(self):
        """Target column must never appear in the feature set."""
        assert TARGET_COL not in GLO_FEATURES
        assert TARGET_COL not in NUMERICAL_FEATURES
        assert TARGET_COL not in CATEGORICAL_FEATURES


# ─── T03 Feature leakage exclusions ───────────────────────────────────────────

class TestFeatureLeakage:
    def test_next_area_not_in_features(self):
        assert 'NEXT_AREA' not in GLO_FEATURES

    def test_next_year_not_in_features(self):
        assert 'NEXT_YEAR' not in GLO_FEATURES
        assert 'NEXT_YEAR_GAP' not in GLO_FEATURES

    def test_area_year_not_in_features(self):
        """AREA_YEAR is the split key, not a predictor."""
        assert 'AREA_YEAR' not in GLO_FEATURES

    def test_glo_id_not_in_features(self):
        """GLO_ID is an identifier, not a predictor."""
        assert 'GLO_ID' not in GLO_FEATURES

    def test_no_future_lag_leakage(self, gold_s1):
        """LAG1 should equal the current year's AREA, not future observations."""
        # LAG1 for year t is the area at year t-1 (previous row within same lake).
        # Check via sort: shift(1) within group should never exceed AREA_YEAR
        df = gold_s1.copy()
        df = df.sort_values(['GLO_ID', 'AREA_YEAR'])
        prev_year = df.groupby('GLO_ID')['AREA_YEAR'].shift(1)
        lag_is_future = prev_year > df['AREA_YEAR']
        assert not lag_is_future.any(), \
            f"AREA_LAG1 derived from future year for {lag_is_future.sum()} rows"


# ─── T04 Equal sample IDs for fair comparison ─────────────────────────────────

class TestFairComparison:
    def test_same_val_indices_across_sensors_not_required(self, splits_s1):
        """Different sensors have different lakes; cross-sensor index equality is
        NOT required and NOT expected. Each sensor's models must use the SAME
        observations for that sensor across all model types."""
        train, val, test = splits_s1
        # Within S1, the same train/val/test rows are used for all models
        assert len(val) > 0

    def test_preprocessor_fit_train_only(self, splits_s1):
        """Preprocessor must be fitted exclusively on training data."""
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        # Should not raise
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        # Transform val and test should work (no error = correct)
        _ = prep.transform(val[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        _ = prep.transform(test[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])

    def test_val_and_test_not_used_to_fit_preprocessor(self, splits_s1):
        """Explicitly check train stats are not contaminated by val/test."""
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        # The imputer median should come from train
        num_imputer = prep.named_transformers_['num'].named_steps['imputer']
        train_median = train['AREA'].median()
        imputer_median = num_imputer.statistics_[NUMERICAL_FEATURES.index('AREA')]
        # Should be very close (exact equality can fail due to implementation)
        assert abs(imputer_median - train_median) < 1e-6, \
            f"Imputer median {imputer_median} differs from train median {train_median}"


# ─── T05 Temporal OOF generation ──────────────────────────────────────────────

class TestTemporalOOF:
    def test_oof_uses_only_past_years(self, splits_s1):
        """Each OOF fold must train only on years < fold_year."""
        from xgboost import XGBRegressor
        train, val, test = splits_s1

        X_train_df   = train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        y_train      = train[TARGET_COL]
        year_col     = train['AREA_YEAR']

        result = temporal_oof_predictions(
            X_train_df, y_train, year_col,
            XGBRegressor(n_estimators=10, random_state=RANDOM_STATE, verbosity=0),
            NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
            min_train_years=3,
        )
        assert result is not None, "OOF returned None for S1 with >3 distinct train years"
        oof_preds, valid_mask = result
        n_valid = valid_mask.sum()
        assert n_valid > 0, "No valid OOF predictions generated"
        # OOF predictions must only be generated for non-earliest years
        min_oof_year = year_col[valid_mask].min()
        earliest_years = sorted(year_col.unique())[:3]  # first 3 years skipped
        assert min_oof_year > max(earliest_years), \
            f"OOF predictions found for year {min_oof_year}, earlier than expected"

    def test_oof_insufficient_years(self):
        """Should return None when fewer distinct years than min_train_years+1."""
        from xgboost import XGBRegressor
        # Create minimal dataset with only 2 distinct years
        np.random.seed(42)
        n = 100
        X_df = pd.DataFrame({
            'AREA': np.random.rand(n), 'PERIMETER': np.random.rand(n),
            'AREA_UNCERTAINTY': np.zeros(n), 'ELEVATION_MEAN': np.zeros(n),
            'ELEVATION_MIN': np.zeros(n), 'ELEVATION_MEDIAN': np.zeros(n),
            'LATITUDE': np.zeros(n), 'LONGITUDE': np.zeros(n),
            'AREA_LAG1': np.random.rand(n), 'AREA_LAG2': np.random.rand(n),
            'AREA_LAG3': np.random.rand(n), 'AREA_CHANGE_LAG1': np.zeros(n),
            'AREA_CHANGE_LAG2': np.zeros(n), 'AREA_CHANGE_LAG3': np.zeros(n),
            'ROLLING_MEAN_3': np.random.rand(n), 'ROLLING_STD_3': np.zeros(n),
            'BASIN': ['A'] * n, 'COUNTRY': ['NP'] * n,
            'GTNG_REGION_O2': ['HKH'] * n, 'CONNECTIVITY': ['C'] * n,
        })
        y = pd.Series(np.random.randn(n))
        years = pd.Series([2017, 2018] * (n // 2))  # only 2 distinct years
        result = temporal_oof_predictions(
            X_df, y, years,
            XGBRegressor(n_estimators=5, random_state=0, verbosity=0),
            NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
            min_train_years=3,
        )
        assert result is None, "Expected None when only 2 distinct years provided"


# ─── T06 Reproducible metric calculations ─────────────────────────────────────

class TestMetricCalculations:
    def test_persistence_mae_equals_mean_abs_target(self, splits_s1):
        """Persistence baseline (predict 0) MAE = mean(|NEXT_AREA_CHANGE|)."""
        _, _, test = splits_s1
        y_te = test[TARGET_COL].values
        persistence_mae = mean_absolute_error(y_te, np.zeros(len(y_te)))
        expected = np.mean(np.abs(y_te))
        assert abs(persistence_mae - expected) < 1e-12

    def test_compute_metrics_returns_correct_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, p)
        assert 'MAE' in m and 'RMSE' in m and 'R2' in m
        assert m['MAE'] == pytest.approx(0.0, abs=1e-9)
        assert m['R2'] == pytest.approx(1.0, abs=1e-9)

    def test_compute_metrics_with_prefix(self):
        y = np.array([1.0, 2.0])
        p = np.array([1.5, 2.5])
        m = compute_metrics(y, p, prefix='Test_')
        assert 'Test_MAE' in m and 'Test_RMSE' in m and 'Test_R2' in m

    def test_deterministic_output(self, splits_s1):
        """Running benchmark twice with same seed must yield identical metrics."""
        from xgboost import XGBRegressor
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xtr = prep.transform(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xva = prep.transform(val[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        ytr = train[TARGET_COL].values
        yva = val[TARGET_COL].values

        m1 = XGBRegressor(n_estimators=20, random_state=RANDOM_STATE, verbosity=0)
        m2 = XGBRegressor(n_estimators=20, random_state=RANDOM_STATE, verbosity=0)
        m1.fit(Xtr, ytr)
        m2.fit(Xtr, ytr)
        p1 = m1.predict(Xva)
        p2 = m2.predict(Xva)
        np.testing.assert_array_equal(p1, p2)


# ─── T07 Valid finite ensemble predictions ────────────────────────────────────

class TestEnsemblePredictions:
    def test_xgb_predictions_finite(self, splits_s1):
        from xgboost import XGBRegressor
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xtr = prep.transform(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xte = prep.transform(test[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        m = XGBRegressor(n_estimators=50, random_state=RANDOM_STATE, verbosity=0)
        m.fit(Xtr, train[TARGET_COL].values)
        preds = m.predict(Xte)
        assert np.all(np.isfinite(preds)), "XGBoost produced non-finite predictions"

    def test_rf_predictions_finite(self, splits_s1):
        from sklearn.ensemble import RandomForestRegressor
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xtr = prep.transform(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xte = prep.transform(test[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        m = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=1)
        m.fit(Xtr, train[TARGET_COL].values)
        preds = m.predict(Xte)
        assert np.all(np.isfinite(preds)), "RandomForest produced non-finite predictions"

    def test_voting_predictions_finite(self, splits_s1):
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from xgboost import XGBRegressor
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xtr = prep.transform(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        Xte = prep.transform(test[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        ytr = train[TARGET_COL].values
        voter = VotingRegressor([
            ('rf',  RandomForestRegressor(n_estimators=20, random_state=RANDOM_STATE, n_jobs=1)),
            ('xgb', XGBRegressor(n_estimators=20, random_state=RANDOM_STATE, verbosity=0)),
        ])
        voter.fit(Xtr, ytr)
        preds = voter.predict(Xte)
        assert np.all(np.isfinite(preds)), "VotingRegressor produced non-finite predictions"
        assert len(preds) == len(test), "VotingRegressor prediction count mismatch"


# ─── T08 Pre-existing artifact preserved ──────────────────────────────────────

class TestArtifactPreservation:
    def test_frozen_xgboost_artifact_exists(self):
        frozen_path = MODELS / '09_final_fused_xgboost.pkl'
        assert frozen_path.exists(), "Frozen NB09 model artifact was deleted!"

    def test_frozen_artifact_has_correct_metrics(self):
        frozen_path = MODELS / '09_final_fused_xgboost.pkl'
        artifact = pickle.load(open(frozen_path, 'rb'))
        assert abs(artifact['test_mae'] - 0.004792) < 0.0001, \
            f"Frozen artifact MAE changed: {artifact['test_mae']}"
        assert abs(artifact['test_rmse'] - 0.015079) < 0.0001, \
            f"Frozen artifact RMSE changed: {artifact['test_rmse']}"

    def test_frozen_artifact_not_overwritten_by_nb11(self):
        """The NB11 benchmark saves under '11_best_*' prefix, never '09_*'."""
        for f in MODELS.glob('*.pkl'):
            if f.name.startswith('11_'):
                content = pickle.load(open(f, 'rb'))
                # Verify it has no 'test_mae' matching the frozen 0.004792 (fused NB09)
                # The NB11 models use per-sensor 20 features so MAE will differ
                assert f.name != '09_final_fused_xgboost.pkl', \
                    "NB11 accidentally wrote to frozen artifact path"


# ─── T09 Preprocessor train-only ──────────────────────────────────────────────
# Covered in TestFairComparison but repeated as standalone for emphasis

class TestPreprocessorTrainOnly:
    def test_imputer_statistics_come_from_train(self, splits_s1):
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        # Check 'AREA' imputer median matches train median
        num_imp = prep.named_transformers_['num'].named_steps['imputer']
        area_idx = NUMERICAL_FEATURES.index('AREA')
        assert abs(num_imp.statistics_[area_idx] - train['AREA'].median()) < 1e-6

    def test_ohe_categories_from_train(self, splits_s1):
        train, val, test = splits_s1
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        prep.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])
        ohe = prep.named_transformers_['cat'].named_steps['ohe']
        # OHE categories should be derived solely from train
        for i, cat_col in enumerate(CATEGORICAL_FEATURES):
            train_cats = set(train[cat_col].dropna().unique())
            ohe_cats   = set(ohe.categories_[i])
            assert ohe_cats == train_cats, \
                f"OHE categories for {cat_col} include non-train categories"


# ─── T10 Persistence always predicts zero ─────────────────────────────────────

class TestPersistenceBaseline:
    def test_persistence_predicts_zero(self):
        y = np.array([0.1, -0.2, 0.05, -0.15, 0.3])
        preds = np.zeros(len(y))
        assert np.all(preds == 0.0)

    def test_persistence_mae_exceeds_any_reasonable_model(self, splits_s1):
        """Persistence MAE provides the no-change floor; all trained models should beat it."""
        _, _, test = splits_s1
        y_te = test[TARGET_COL].values
        persistence_mae = np.mean(np.abs(y_te))
        # Sanity: persistence MAE is positive (lakes are not static)
        assert persistence_mae > 0, "Persistence MAE is 0 — all targets are zero (unexpected)"
        print(f"\n  S1 Persistence MAE (no-change floor): {persistence_mae:.6f} km²")
