"""
Feature Pipeline and Temporal Split Engine
==========================================
Reproduces the exact data preparation, lag generation, temporal splitting,
and preprocessor construction from NB03, NB08, NB09, and NB11.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from glacierguard.features.constants import (
    CATEGORICAL_FEATURES,
    GLO_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    TEST_YEAR,
    TRAIN_MAX_YEAR,
    VAL_YEAR,
)


def build_gold_dataset(gpkg_or_df: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Build the gold feature dataset from a GLO GeoPackage file or DataFrame.

    Implements:
    - GeoPackage ingestion and geometry detachment
    - Column normalization (e.g. LONGITUD -> LONGITUDE, GTNG_REGION -> GTNG_REGION_O2)
    - Consecutive year-pair temporal lagging within each lake:
      - AREA_LAG1, AREA_LAG2, AREA_LAG3
      - AREA_CHANGE_LAG1, AREA_CHANGE_LAG2, AREA_CHANGE_LAG3
      - ROLLING_MEAN_3, ROLLING_STD_3
    - Target construction: NEXT_AREA_CHANGE = NEXT_AREA - AREA (strictly consecutive)

    Parameters
    ----------
    gpkg_or_df : Union[str, Path, pd.DataFrame]
        Path to a GLO GeoPackage (.gpkg) file, or an already loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with temporal lag features and consecutive target.
    """
    if isinstance(gpkg_or_df, (str, Path)):
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_or_df)
        df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    else:
        df = gpkg_or_df.copy()
        if 'geometry' in df.columns:
            df = df.drop(columns='geometry')

    # Normalize column names across GLO versions
    if 'LONGITUD' in df.columns and 'LONGITUDE' not in df.columns:
        df = df.rename(columns={'LONGITUD': 'LONGITUDE'})
    if 'GTNG_REGION' in df.columns and 'GTNG_REGION_O2' not in df.columns:
        df = df.rename(columns={'GTNG_REGION': 'GTNG_REGION_O2'})

    required = [
        'GLO_ID', 'AREA_YEAR', 'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
        'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
        'LATITUDE', 'LONGITUDE', 'BASIN', 'COUNTRY',
        'GTNG_REGION_O2', 'CONNECTIVITY',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    df = df[required].copy()
    df = df.sort_values(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)

    # Grouped temporal shifts
    grp = df.groupby('GLO_ID')
    df['PREV_YEAR'] = grp['AREA_YEAR'].shift(1)
    df['PREV_AREA'] = grp['AREA'].shift(1)
    df['NEXT_YEAR'] = grp['AREA_YEAR'].shift(-1)
    df['NEXT_AREA'] = grp['AREA'].shift(-1)
    df['NEXT_YEAR_GAP'] = df['NEXT_YEAR'] - df['AREA_YEAR']
    df['PREV_YEAR_GAP'] = df['AREA_YEAR'] - df['PREV_YEAR']

    # Lag area features
    df['AREA_LAG1'] = grp['AREA'].shift(1)
    df['AREA_LAG2'] = grp['AREA'].shift(2)
    df['AREA_LAG3'] = grp['AREA'].shift(3)

    # Lag delta features (only when observations are strictly consecutive 1-year pairs)
    df['AREA_CHANGE_LAG1'] = np.where(
        df['AREA_LAG1'].notna() & (df['PREV_YEAR_GAP'] == 1),
        df['AREA'] - df['PREV_AREA'],
        np.nan,
    )
    df['AREA_CHANGE_LAG2'] = np.where(
        df['AREA_LAG2'].notna() & df['AREA_LAG1'].notna(),
        df['AREA_LAG1'] - df['AREA_LAG2'],
        np.nan,
    )
    df['AREA_CHANGE_LAG3'] = np.where(
        df['AREA_LAG3'].notna() & df['AREA_LAG2'].notna(),
        df['AREA_LAG2'] - df['AREA_LAG3'],
        np.nan,
    )

    # 3-year rolling statistics
    df['ROLLING_MEAN_3'] = np.where(
        df['AREA_LAG3'].notna(),
        (df['AREA'] + df['AREA_LAG1'] + df['AREA_LAG2']) / 3.0,
        np.nan,
    )
    df['ROLLING_STD_3'] = np.where(
        df['AREA_LAG3'].notna(),
        df[['AREA', 'AREA_LAG1', 'AREA_LAG2']].std(axis=1, ddof=0),
        np.nan,
    )

    # Prediction target: NEXT_AREA_CHANGE (strictly 1-year gap)
    df[TARGET_COL] = np.where(
        df['NEXT_YEAR_GAP'] == 1,
        df['NEXT_AREA'] - df['AREA'],
        np.nan,
    )

    return df


def create_temporal_splits(
    df: pd.DataFrame,
    train_max_year: int = TRAIN_MAX_YEAR,
    val_year: int = VAL_YEAR,
    test_year: int = TEST_YEAR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological temporal split isolating train, validation, and test cohorts.

    Enforces strict leakage boundaries:
    - Train: observations with AREA_YEAR <= train_max_year
    - Validation: observations with AREA_YEAR == val_year
    - Test: observations with AREA_YEAR == test_year
    - Excludes observations with missing targets or future years (> test_year)
    """
    supervised = df[df[TARGET_COL].notna()].copy()

    train = supervised[supervised['AREA_YEAR'] <= train_max_year].copy()
    val = supervised[supervised['AREA_YEAR'] == val_year].copy()
    test = supervised[supervised['AREA_YEAR'] == test_year].copy()

    total_split_rows = len(train) + len(val) + len(test)
    if total_split_rows != len(supervised):
        raise ValueError(
            f"Split row sum ({total_split_rows}) != supervised total ({len(supervised)})."
        )

    if not train.empty and train['AREA_YEAR'].max() > train_max_year:
        raise ValueError(f"Train contains years > {train_max_year}")
    if not val.empty and not (val['AREA_YEAR'] == val_year).all():
        raise ValueError(f"Val contains years other than {val_year}")
    if not test.empty and not (test['AREA_YEAR'] == test_year).all():
        raise ValueError(f"Test contains years other than {test_year}")

    return train, val, test


def build_preprocessor(
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a train-only fitted ColumnTransformer matching NB08/NB09 contract.

    - Numerical pipeline: SimpleImputer(strategy='median')
    - Categorical pipeline: SimpleImputer(strategy='most_frequent') +
      OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    if num_cols is None:
        num_cols = NUMERICAL_FEATURES
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])
