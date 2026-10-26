"""
Feature Definitions and Constants
=================================
Reproduces the exact schema and feature contracts established in NB03/NB08/NB09.
"""

# Prediction target: annual lake area change in km²
TARGET_COL = 'NEXT_AREA_CHANGE'

# Categorical geospatial attributes
CATEGORICAL_FEATURES = [
    'BASIN',
    'COUNTRY',
    'GTNG_REGION_O2',
    'CONNECTIVITY',
]

# Single-sensor tabular numerical features (16 features)
NUMERICAL_FEATURES = [
    'AREA',
    'PERIMETER',
    'AREA_UNCERTAINTY',
    'ELEVATION_MEAN',
    'ELEVATION_MIN',
    'ELEVATION_MEDIAN',
    'LATITUDE',
    'LONGITUDE',
    'AREA_LAG1',
    'AREA_LAG2',
    'AREA_LAG3',
    'AREA_CHANGE_LAG1',
    'AREA_CHANGE_LAG2',
    'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3',
    'ROLLING_STD_3',
]

# Complete single-sensor GLO feature set (20 features: 4 categorical + 16 numerical)
GLO_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# S2 optical prefix for fused sensor tables
S2_PREFIX = 'S2_'

# Dual-sensor fused numerical features (32 features: 16 S1 + 16 S2)
FUSED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + [f'{S2_PREFIX}{col}' for col in NUMERICAL_FEATURES]

# Complete dual-sensor fused feature set (36 features: 4 categorical + 32 numerical)
FUSED_FEATURES = CATEGORICAL_FEATURES + FUSED_NUMERICAL_FEATURES

# Benchmark temporal splits (frozen protocol)
TRAIN_MAX_YEAR = 2021
VAL_YEAR = 2022
TEST_YEAR = 2023

# Default random seed across all deterministic routines
RANDOM_STATE = 42
