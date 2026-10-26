"""
Unit and Parity Tests for Feature Engineering & Splitting
=========================================================
Verifies parity with NB03, NB08, and NB11 data pipelines.
"""

import numpy as np
import pandas as pd
import pytest
from glacierguard.features.constants import (
    CATEGORICAL_FEATURES,
    GLO_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
)
from glacierguard.features.pipeline import (
    build_gold_dataset,
    build_preprocessor,
    create_temporal_splits,
)


@pytest.fixture
def synthetic_glo_records():
    """Create deterministic multi-year observations for two lakes."""
    rows = []
    # Lake 1: 5 consecutive years (2018-2022), area grows by 0.1 km² each year
    for i, yr in enumerate(range(2018, 2023)):
        rows.append({
            'GLO_ID': 'GLO_TEST_001',
            'AREA_YEAR': yr,
            'AREA': 1.0 + i * 0.1,
            'PERIMETER': 4.0 + i * 0.2,
            'AREA_UNCERTAINTY': 0.01,
            'ELEVATION_MEAN': 5000.0,
            'ELEVATION_MIN': 4950.0,
            'ELEVATION_MEDIAN': 5000.0,
            'LATITUDE': 28.0,
            'LONGITUDE': 85.0,
            'BASIN': 'DUDH_KOSHI',
            'COUNTRY': 'Nepal',
            'GTNG_REGION_O2': 'Central_Himalaya',
            'CONNECTIVITY': 'PROGLACIAL',
        })
    # Lake 2: years with a gap (2018, 2020, 2021)
    rows.append({
        'GLO_ID': 'GLO_TEST_002',
        'AREA_YEAR': 2018,
        'AREA': 0.5,
        'PERIMETER': 2.5,
        'AREA_UNCERTAINTY': 0.005,
        'ELEVATION_MEAN': 4800.0,
        'ELEVATION_MIN': 4750.0,
        'ELEVATION_MEDIAN': 4800.0,
        'LATITUDE': 27.5,
        'LONGITUDE': 86.0,
        'BASIN': 'MARSYANGDI',
        'COUNTRY': 'Nepal',
        'GTNG_REGION_O2': 'Central_Himalaya',
        'CONNECTIVITY': 'UNCONNECTED',
    })
    rows.append({
        'GLO_ID': 'GLO_TEST_002',
        'AREA_YEAR': 2020,
        'AREA': 0.55,
        'PERIMETER': 2.6,
        'AREA_UNCERTAINTY': 0.005,
        'ELEVATION_MEAN': 4800.0,
        'ELEVATION_MIN': 4750.0,
        'ELEVATION_MEDIAN': 4800.0,
        'LATITUDE': 27.5,
        'LONGITUDE': 86.0,
        'BASIN': 'MARSYANGDI',
        'COUNTRY': 'Nepal',
        'GTNG_REGION_O2': 'Central_Himalaya',
        'CONNECTIVITY': 'UNCONNECTED',
    })
    rows.append({
        'GLO_ID': 'GLO_TEST_002',
        'AREA_YEAR': 2021,
        'AREA': 0.58,
        'PERIMETER': 2.7,
        'AREA_UNCERTAINTY': 0.005,
        'ELEVATION_MEAN': 4800.0,
        'ELEVATION_MIN': 4750.0,
        'ELEVATION_MEDIAN': 4800.0,
        'LATITUDE': 27.5,
        'LONGITUDE': 86.0,
        'BASIN': 'MARSYANGDI',
        'COUNTRY': 'Nepal',
        'GTNG_REGION_O2': 'Central_Himalaya',
        'CONNECTIVITY': 'UNCONNECTED',
    })
    return pd.DataFrame(rows)


class TestFeatureConstants:
    """Validate feature contract definitions."""

    def test_feature_counts(self):
        assert len(CATEGORICAL_FEATURES) == 4
        assert len(NUMERICAL_FEATURES) == 16
        assert len(GLO_FEATURES) == 20
        assert TARGET_COL == 'NEXT_AREA_CHANGE'
        assert TARGET_COL not in GLO_FEATURES


class TestFeaturePipeline:
    """Validate gold dataset builder and temporal lag logic."""

    def test_build_gold_dataset_lags_and_target(self, synthetic_glo_records):
        df = build_gold_dataset(synthetic_glo_records)

        # Check required columns exist
        for col in GLO_FEATURES:
            assert col in df.columns, f"Missing feature: {col}"
        assert TARGET_COL in df.columns

        # Lake 1, index 0 (2018): no previous year, lag1 is NaN
        lk1 = df[df['GLO_ID'] == 'GLO_TEST_001'].reset_index(drop=True)
        assert np.isnan(lk1.loc[0, 'AREA_LAG1'])
        assert np.isnan(lk1.loc[0, 'AREA_CHANGE_LAG1'])
        assert lk1.loc[0, TARGET_COL] == pytest.approx(0.1, abs=1e-6)

        # Lake 1, index 1 (2019): previous year was 2018, area=1.0
        assert lk1.loc[1, 'AREA_LAG1'] == pytest.approx(1.0, abs=1e-6)
        assert lk1.loc[1, 'AREA_CHANGE_LAG1'] == pytest.approx(0.1, abs=1e-6)
        assert lk1.loc[1, TARGET_COL] == pytest.approx(0.1, abs=1e-6)

        # Lake 1, index 3 (2021): 3 lags available, ROLLING_MEAN_3 present
        assert lk1.loc[3, 'AREA_LAG1'] == pytest.approx(1.2, abs=1e-6)
        assert lk1.loc[3, 'AREA_LAG2'] == pytest.approx(1.1, abs=1e-6)
        assert lk1.loc[3, 'AREA_LAG3'] == pytest.approx(1.0, abs=1e-6)
        expected_roll = (1.3 + 1.2 + 1.1) / 3.0
        assert lk1.loc[3, 'ROLLING_MEAN_3'] == pytest.approx(expected_roll, abs=1e-6)

        # Lake 1, index 4 (2022): last year, NEXT_YEAR is NaN -> NEXT_AREA_CHANGE is NaN
        assert np.isnan(lk1.loc[4, TARGET_COL])

        # Lake 2, check gap handling: 2018 to 2020 has 2-year gap -> TARGET for 2018 is NaN
        lk2 = df[df['GLO_ID'] == 'GLO_TEST_002'].reset_index(drop=True)
        assert np.isnan(lk2.loc[0, TARGET_COL])  # gap is 2, not 1
        # Lake 2, 2020 to 2021 has 1-year gap -> TARGET for 2020 is 0.58 - 0.55 = 0.03
        assert lk2.loc[1, TARGET_COL] == pytest.approx(0.03, abs=1e-6)

    def test_column_renaming_fallback(self):
        """Verify fallback from LONGITUD and GTNG_REGION."""
        raw = pd.DataFrame([{
            'GLO_ID': 'GLO_001',
            'AREA_YEAR': 2020,
            'AREA': 1.0,
            'PERIMETER': 4.0,
            'AREA_UNCERTAINTY': 0.01,
            'ELEVATION_MEAN': 5000.0,
            'ELEVATION_MIN': 4950.0,
            'ELEVATION_MEDIAN': 5000.0,
            'LATITUDE': 28.0,
            'LONGITUD': 85.0,  # Old schema name
            'BASIN': 'DUDH_KOSHI',
            'COUNTRY': 'Nepal',
            'GTNG_REGION': 'Central_Himalaya',  # Old schema name
            'CONNECTIVITY': 'PROGLACIAL',
        }])
        df = build_gold_dataset(raw)
        assert 'LONGITUDE' in df.columns
        assert 'GTNG_REGION_O2' in df.columns


class TestTemporalSplits:
    """Validate temporal splitting rules and leakage prevention."""

    def test_create_temporal_splits_boundaries(self):
        records = []
        for yr in range(2017, 2025):
            records.append({
                'GLO_ID': 'GLO_SPLIT_001',
                'AREA_YEAR': yr,
                'NEXT_AREA_CHANGE': 0.05 if yr < 2024 else np.nan,
                'AREA': 1.0 + (yr - 2017) * 0.05,
            })
        df = pd.DataFrame(records)

        train, val, test = create_temporal_splits(
            df, train_max_year=2021, val_year=2022, test_year=2023
        )

        assert train['AREA_YEAR'].max() <= 2021
        assert (val['AREA_YEAR'] == 2022).all()
        assert (test['AREA_YEAR'] == 2023).all()
        assert len(train) == 5  # 2017, 2018, 2019, 2020, 2021
        assert len(val) == 1    # 2022
        assert len(test) == 1   # 2023


class TestPreprocessor:
    """Validate ColumnTransformer imputer and one-hot encoding."""

    def test_preprocessor_fitting(self):
        train_df = pd.DataFrame({
            'AREA': [1.0, 2.0, np.nan],
            'PERIMETER': [4.0, np.nan, 5.0],
            'BASIN': ['A', 'B', 'A'],
            'COUNTRY': ['Nepal', 'Nepal', 'India'],
        })
        num_cols = ['AREA', 'PERIMETER']
        cat_cols = ['BASIN', 'COUNTRY']

        preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
        preprocessor.fit(train_df)

        trans = preprocessor.transform(train_df)
        assert not np.isnan(trans).any(), "Preprocessor output must have no NaNs"
        assert trans.shape[0] == 3
