"""
Audit NB08 reproducibility with the exact build_preprocessor pipeline from NB08.
"""
import sys
import pathlib
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

# 1. Setup paths
ROOT = pathlib.Path('.').resolve()
GLO_DIR = ROOT / 'raw' / 'GLO'
ERA5_DIR = ROOT / 'raw' / 'ERA5'
CHIRPS_DIR = ROOT / 'raw' / 'CHIRPS'
RGI_DIR = ROOT / 'raw' / 'RGI'

RANDOM_STATE = 42

def build_gold_dataset(gpkg_path):
    df = gpd.read_file(str(gpkg_path)).drop(columns='geometry')
    df['AREA_YEAR'] = df['AREA_YEAR'].astype(int)
    df = df.sort_values(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)

    g = df.groupby('GLO_ID')
    df['PREV_YEAR']     = g['AREA_YEAR'].shift(1)
    df['PREV_AREA']     = g['AREA'].shift(1)
    df['PREV_YEAR_GAP'] = df['AREA_YEAR'] - df['PREV_YEAR']
    df['PREV_YEAR_2']   = g['AREA_YEAR'].shift(2)
    df['PREV_AREA_2']   = g['AREA'].shift(2)
    df['PREV_YEAR_3']   = g['AREA_YEAR'].shift(3)
    df['PREV_AREA_3']   = g['AREA'].shift(3)
    df['PREV_YEAR_4']   = g['AREA_YEAR'].shift(4)
    df['PREV_AREA_4']   = g['AREA'].shift(4)
    df['NEXT_YEAR']     = g['AREA_YEAR'].shift(-1)
    df['NEXT_AREA']     = g['AREA'].shift(-1)
    df['NEXT_YEAR_GAP'] = df['NEXT_YEAR'] - df['AREA_YEAR']

    # Gap-enforced area lags
    df['AREA_LAG1'] = np.where(df['PREV_YEAR_GAP'] == 1, df['PREV_AREA'], np.nan)
    df['AREA_LAG2'] = np.where(
        (df['PREV_YEAR_GAP'] == 1) & (df['AREA_YEAR'] - df['PREV_YEAR_2'] == 2),
        df['PREV_AREA_2'], np.nan)
    df['AREA_LAG3'] = np.where(
        (df['PREV_YEAR_GAP'] == 1) & (df['AREA_YEAR'] - df['PREV_YEAR_3'] == 3),
        df['PREV_AREA_3'], np.nan)

    df['AREA_CHANGE_LAG1'] = np.where(df['PREV_YEAR_GAP'] == 1,
                                       df['AREA'] - df['PREV_AREA'], np.nan)
    df['AREA_CHANGE_LAG2'] = np.where(
        df['AREA_LAG2'].notna() & df['AREA_LAG1'].notna(),
        df['AREA_LAG1'] - df['AREA_LAG2'], np.nan)
    df['AREA_CHANGE_LAG3'] = np.where(
        df['AREA_LAG3'].notna() & df['AREA_LAG2'].notna(),
        df['AREA_LAG2'] - df['AREA_LAG3'], np.nan)

    df['ROLLING_MEAN_3'] = np.where(
        df['AREA_LAG3'].notna(),
        (df['AREA'] + df['AREA_LAG1'] + df['AREA_LAG2']) / 3, np.nan)
    df['ROLLING_STD_3']  = np.where(
        df['AREA_LAG3'].notna(),
        df[['AREA', 'AREA_LAG1', 'AREA_LAG2']].std(axis=1, ddof=0), np.nan)

    # Target
    df['NEXT_AREA_CHANGE'] = np.where(
        df['NEXT_YEAR_GAP'] == 1,
        df['NEXT_AREA'] - df['AREA'], np.nan)

    return df

print("Building GLO Gold datasets...")
gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')

era5_all = pd.read_csv(ERA5_DIR / 'era5_features.csv')
ERA5_COLS_PRIMARY = [
    'ERA5_TEMP_MEAN', 'ERA5_TEMP_WARM', 'ERA5_PRECIP_TOTAL',
    'ERA5_PRECIP_MONSOON', 'ERA5_SNOW_MEAN', 'ERA5_PDD',
    'ERA5_TEMP_ANOM', 'ERA5_PRECIP_ANOM',
]
era5_feats = era5_all[['GLO_ID', 'AREA_YEAR'] + ERA5_COLS_PRIMARY].copy()

chirps_all = pd.read_csv(CHIRPS_DIR / 'chirps_features.csv')
CHIRPS_COLS = ['CHIRPS_TP_TOTAL', 'CHIRPS_TP_JJAS', 'CHIRPS_TP_ANOM']
chirps_feats = chirps_all[['GLO_ID', 'AREA_YEAR'] + CHIRPS_COLS].copy()

rgi_feats = pd.read_csv(RGI_DIR / 'lake_glacier_distances.csv')
RGI_COLS = ['DIST_NEAREST_GLACIER_KM']

TARGET_COL = 'NEXT_AREA_CHANGE'
CATEGORICAL_FEATURES = ['BASIN', 'COUNTRY', 'GTNG_REGION_O2', 'CONNECTIVITY']
NUMERICAL_FEATURES   = [
    'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
    'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
    'LATITUDE', 'LONGITUDE',
    'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
    'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3',
]

def create_temporal_splits(df, label):
    sup   = df[df[TARGET_COL].notna()].copy()
    train = sup[sup['AREA_YEAR'] <= 2021].copy()
    val   = sup[sup['AREA_YEAR'] == 2022].copy()
    test  = sup[sup['AREA_YEAR'] == 2023].copy()
    return train, val, test

s1_train, s1_val, s1_test = create_temporal_splits(gold_s1, 'Sentinel-1')
s2_train, s2_val, s2_test = create_temporal_splits(gold_s2, 'Sentinel-2')

def build_augmented_split(glo_split, era5_df, chirps_df, rgi_df):
    df = glo_split.copy()
    df = df.merge(era5_df,   on=['GLO_ID', 'AREA_YEAR'], how='left')
    df = df.merge(chirps_df, on=['GLO_ID', 'AREA_YEAR'], how='left')
    df = df.merge(rgi_df,    on='GLO_ID',               how='left')
    df['ELEVATION_RANGE'] = (df['ELEVATION_MEAN'] - df['ELEVATION_MIN']).clip(lower=0.0)
    return df

aug_s1_train = build_augmented_split(s1_train, era5_feats, chirps_feats, rgi_feats)
aug_s1_val   = build_augmented_split(s1_val,   era5_feats, chirps_feats, rgi_feats)
aug_s1_test  = build_augmented_split(s1_test,  era5_feats, chirps_feats, rgi_feats)
aug_s2_train = build_augmented_split(s2_train, era5_feats, chirps_feats, rgi_feats)
aug_s2_val   = build_augmented_split(s2_val,   era5_feats, chirps_feats, rgi_feats)
aug_s2_test  = build_augmented_split(s2_test,  era5_feats, chirps_feats, rgi_feats)

def match_sensors(s1_df, s2_df):
    s1_idx = set(zip(s1_df['GLO_ID'], s1_df['AREA_YEAR']))
    s2_idx = set(zip(s2_df['GLO_ID'], s2_df['AREA_YEAR']))
    matched_idx = s1_idx & s2_idx

    s1_matched = s1_df[s1_df.apply(lambda r: (r['GLO_ID'], r['AREA_YEAR']) in matched_idx, axis=1)].copy()
    s2_matched = s2_df[s2_df.apply(lambda r: (r['GLO_ID'], r['AREA_YEAR']) in matched_idx, axis=1)].copy()
    return s1_matched.reset_index(drop=True), s2_matched.reset_index(drop=True)

m_s1_tr, m_s2_tr = match_sensors(aug_s1_train, aug_s2_train)
m_s1_va, m_s2_va = match_sensors(aug_s1_val,   aug_s2_val)
m_s1_te, m_s2_te = match_sensors(aug_s1_test,  aug_s2_test)

print(f"Matched rows: Train={len(m_s1_tr)}, Val={len(m_s1_va)}, Test={len(m_s1_te)}")

def build_fused_dataset(ms1, ms2, num_cols, cat_cols):
    shared_keys  = ['GLO_ID', 'AREA_YEAR']
    s1_feat = ms1[shared_keys + cat_cols + num_cols + [TARGET_COL]].copy()
    s2_feat = ms2[shared_keys + num_cols + [TARGET_COL]].copy()

    s2_num_renamed = {c: f'S2_{c}' for c in num_cols}
    s2_feat = s2_feat.rename(columns=s2_num_renamed)
    s2_num_cols_new = [f'S2_{c}' for c in num_cols]

    fused = s1_feat.merge(
        s2_feat[shared_keys + s2_num_cols_new + [TARGET_COL]].rename(columns={TARGET_COL: 'S2_TARGET'}),
        on=shared_keys
    )
    fused['FUSED_TARGET'] = (fused[TARGET_COL] + fused['S2_TARGET']) / 2
    return fused, num_cols + s2_num_cols_new, cat_cols

fused_tr, fused_num_cols, fused_cat_cols = build_fused_dataset(m_s1_tr, m_s2_tr, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
fused_va, _, _ = build_fused_dataset(m_s1_va, m_s2_va, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
fused_te, _, _ = build_fused_dataset(m_s1_te, m_s2_te, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

# EXACT build_preprocessor from NB08
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])

# Reproduce Cell 40 exactly
results = []
for config_name, tr, va, te, num_c, cat_c, tgt in [
    ('S1_matched',  m_s1_tr, m_s1_va, m_s1_te, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL),
    ('S2_matched',  m_s2_tr, m_s2_va, m_s2_te, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL),
    ('S1+S2_fused', fused_tr, fused_va, fused_te, fused_num_cols, fused_cat_cols, 'FUSED_TARGET'),
]:
    prep_f = build_preprocessor(num_c, cat_c)
    prep_f.fit(tr[cat_c + num_c])
    Xt_f_tr = prep_f.transform(tr[cat_c + num_c])
    Xt_f_va = prep_f.transform(va[cat_c + num_c])
    Xt_f_te = prep_f.transform(te[cat_c + num_c])
    y_f_tr, y_f_va, y_f_te = tr[tgt], va[tgt], te[tgt]

    for model_name, est in [
        ('HGB', HistGradientBoostingRegressor(random_state=RANDOM_STATE, **{'max_iter':100,'max_depth':4,'min_samples_leaf':30,'learning_rate':0.05})),
        ('XGBoost', xgb.XGBRegressor(random_state=RANDOM_STATE, tree_method='hist', verbosity=0, n_jobs=-1, **{'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 1.0})),
    ]:
        est.fit(Xt_f_tr, y_f_tr)
        results.append({
            'Config': config_name, 'Model': model_name,
            'N_train': len(y_f_tr), 'N_val': len(y_f_va), 'N_test': len(y_f_te),
            'Val_MAE':  mean_absolute_error(y_f_va, est.predict(Xt_f_va)),
            'Val_RMSE': root_mean_squared_error(y_f_va, est.predict(Xt_f_va)),
            'Test_MAE': mean_absolute_error(y_f_te, est.predict(Xt_f_te)),
            'Test_RMSE': root_mean_squared_error(y_f_te, est.predict(Xt_f_te)),
        })

df_res = pd.DataFrame(results)
print("\nExact Reproduction Results:")
print(df_res.to_string(index=False))

fused_row = df_res[(df_res['Config']=='S1+S2_fused') & (df_res['Model']=='XGBoost')].iloc[0]
print(f"\nFused XGBoost:")
print(f"  Val MAE  : {fused_row['Val_MAE']:.6f} (Expected: 0.004917)")
print(f"  Test MAE : {fused_row['Test_MAE']:.6f} (Expected: 0.004792)")
print(f"  Test RMSE: {fused_row['Test_RMSE']:.6f} (Expected: 0.015079)")

assert abs(fused_row['Val_MAE'] - 0.004917) < 1e-5, f"Val MAE discrepancy: {fused_row['Val_MAE']}"
assert abs(fused_row['Test_MAE'] - 0.004792) < 1e-5, f"Test MAE discrepancy: {fused_row['Test_MAE']}"
assert abs(fused_row['Test_RMSE'] - 0.015079) < 1e-5, f"Test RMSE discrepancy: {fused_row['Test_RMSE']}"
print("\nALL REPRODUCIBILITY CHECKS MATCH TO < 1e-5 TOLERANCE!")
