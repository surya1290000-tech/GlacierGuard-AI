"""
GlacierGuard-AI — Notebook 06 Execution Script
06_model_development — Advanced Model Development

Runs all sections sequentially to verify correctness before notebook construction.
Saves all figures and CSV reports to reports/ and reports/figures/.
"""

# ===========================================================================
# §2  IMPORTS & CONFIGURATION
# ===========================================================================
import os, sys, pathlib, warnings, time, itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = pathlib.Path('reports/figures')
REPORTS_DIR = pathlib.Path('reports')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 65)
print('GLACIERGUARD-AI — 06 ADVANCED MODEL DEVELOPMENT')
print('=' * 65)
print(f'Python       : {sys.version.split()[0]}')
print(f'scikit-learn : {sklearn.__version__}')
print(f'pandas       : {pd.__version__}')
print(f'numpy        : {np.__version__}')
print(f'Figures dir  : {FIGURES_DIR.resolve()}')
print(f'Reports dir  : {REPORTS_DIR.resolve()}')
print(f'Random seed  : {RANDOM_STATE}')

# ===========================================================================
# §3  GOLD DATASET RECONSTRUCTION
# ===========================================================================
print('\n[§3] Reconstructing Gold Datasets...')

GLO_DIR_candidates = [
    pathlib.Path('raw/GLO'),
    pathlib.Path('../raw/GLO'),
    pathlib.Path('./raw/GLO'),
]
GLO_DIR = None
for p in GLO_DIR_candidates:
    if p.exists() and (p / 'S1_20172024_NTB_GLOID_v1.02.gpkg').exists():
        GLO_DIR = p.resolve()
        break
if GLO_DIR is None:
    raise FileNotFoundError('Cannot locate annual GLO GeoPackage files.')
print(f'  GLO directory : {GLO_DIR}')

def build_gold_dataset(gpkg_path):
    """Faithful replication of NB04 build_gold_dataset() — do not alter."""
    df = gpd.read_file(str(gpkg_path)).drop(columns='geometry')
    df['AREA_YEAR'] = df['AREA_YEAR'].astype(int)
    df = df.sort_values(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)

    g = df.groupby('GLO_ID')
    df['PREV_YEAR']   = g['AREA_YEAR'].shift(1)
    df['PREV_AREA']   = g['AREA'].shift(1)
    df['PREV_YEAR_GAP'] = df['AREA_YEAR'] - df['PREV_YEAR']
    df['PREV_YEAR_2'] = g['AREA_YEAR'].shift(2)
    df['PREV_AREA_2'] = g['AREA'].shift(2)
    df['PREV_YEAR_3'] = g['AREA_YEAR'].shift(3)
    df['PREV_AREA_3'] = g['AREA'].shift(3)
    df['PREV_YEAR_4'] = g['AREA_YEAR'].shift(4)
    df['PREV_AREA_4'] = g['AREA'].shift(4)
    df['NEXT_YEAR']   = g['AREA_YEAR'].shift(-1)
    df['NEXT_AREA']   = g['AREA'].shift(-1)
    df['NEXT_YEAR_GAP'] = df['NEXT_YEAR'] - df['AREA_YEAR']

    df['AREA_LAG1'] = np.where(df['PREV_YEAR_GAP'] == 1, df['PREV_AREA'], np.nan)
    df['AREA_LAG2'] = np.where(
        (df['PREV_YEAR_GAP'] == 1) & ((df['PREV_YEAR'] - df['PREV_YEAR_2']) == 1),
        df['PREV_AREA_2'], np.nan)
    df['AREA_LAG3'] = np.where(
        (df['PREV_YEAR_GAP'] == 1) &
        ((df['PREV_YEAR'] - df['PREV_YEAR_2']) == 1) &
        ((df['PREV_YEAR_2'] - df['PREV_YEAR_3']) == 1),
        df['PREV_AREA_3'], np.nan)
    df['AREA_LAG4'] = np.where(
        (df['PREV_YEAR_GAP'] == 1) &
        ((df['PREV_YEAR'] - df['PREV_YEAR_2']) == 1) &
        ((df['PREV_YEAR_2'] - df['PREV_YEAR_3']) == 1) &
        ((df['PREV_YEAR_3'] - df['PREV_YEAR_4']) == 1),
        df['PREV_AREA_4'], np.nan)

    df['AREA_CHANGE']      = np.where(df['PREV_YEAR_GAP'] == 1,
                                       df['AREA'] - df['PREV_AREA'], np.nan)
    df['AREA_CHANGE_LAG1'] = np.where(
        df['AREA_LAG1'].notna() & df['AREA_LAG2'].notna(),
        df['AREA_LAG1'] - df['AREA_LAG2'], np.nan)
    df['AREA_CHANGE_LAG2'] = np.where(
        df['AREA_LAG2'].notna() & df['AREA_LAG3'].notna(),
        df['AREA_LAG2'] - df['AREA_LAG3'], np.nan)
    df['AREA_CHANGE_LAG3'] = np.where(
        df['AREA_LAG3'].notna() & df['AREA_LAG4'].notna(),
        df['AREA_LAG3'] - df['AREA_LAG4'], np.nan)

    c_df = df[['AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3']]
    df['ROLLING_MEAN_3'] = c_df.mean(axis=1, skipna=True)
    df['ROLLING_STD_3']  = c_df.std(axis=1, skipna=True, ddof=1)

    df['NEXT_AREA_CHANGE'] = np.where(
        df['NEXT_YEAR_GAP'] == 1, df['NEXT_AREA'] - df['AREA'], np.nan)
    df['IS_TS_OUTLIER'] = df['TS_OUTLIER'].astype(str).str.upper().isin(['TRUE', '1'])

    gold_cols = [
        'GLO_ID', 'AREA_YEAR', 'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
        'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
        'CONNECTIVITY', 'LATITUDE', 'LONGITUDE', 'BASIN', 'COUNTRY', 'GTNG_REGION_O2',
        'TS_OUTLIER', 'IS_TS_OUTLIER',
        'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
        'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
        'ROLLING_MEAN_3', 'ROLLING_STD_3',
        'AREA_CHANGE',          # persistence benchmark only
        'NEXT_AREA_CHANGE'
    ]
    return df[gold_cols].copy()

t0 = time.time()
gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')
print(f'  Load time : {time.time()-t0:.1f}s')

# ===========================================================================
# §4  DATASET INTEGRITY ASSERTIONS
# ===========================================================================
print('\n[§4] Running integrity assertions...')

assert len(gold_s1) == 18129, f'S1 row mismatch: {len(gold_s1)}'
assert len(gold_s2) == 22294, f'S2 row mismatch: {len(gold_s2)}'
assert gold_s1['NEXT_AREA_CHANGE'].notna().sum() == 14242, 'S1 target mismatch'
assert gold_s2['NEXT_AREA_CHANGE'].notna().sum() == 16610, 'S2 target mismatch'

PROHIBITED_LEAKAGE = [
    'NEXT_AREA', 'NEXT_YEAR', 'NEXT_YEAR_GAP',
    'EXPANSION_RATE', 'EXPANSION_RATE_SIG', 'EXPANSION_UNCERTAINTY',
    'START_DATE', 'END_DATE', 'REF_MSTAT', 'S2_MSTAT', 'DATA_SOURCE', 'AREA_LAG4'
]
for lbl, df in [('S1', gold_s1), ('S2', gold_s2)]:
    leaked = [c for c in PROHIBITED_LEAKAGE if c in df.columns]
    assert len(leaked) == 0, f'Leakage in {lbl}: {leaked}'

print('  All integrity assertions passed.')

# ===========================================================================
# §5  FROZEN TEMPORAL SPLIT
# ===========================================================================
print('\n[§5] Constructing temporal splits...')

TARGET_COL = 'NEXT_AREA_CHANGE'

def create_temporal_splits(df, label):
    sup   = df[df[TARGET_COL].notna()].copy()
    train = sup[sup['AREA_YEAR'] <= 2021].copy()
    val   = sup[sup['AREA_YEAR'] == 2022].copy()
    test  = sup[sup['AREA_YEAR'] == 2023].copy()
    assert len(train) + len(val) + len(test) == len(sup)
    assert train['AREA_YEAR'].max() <= 2021
    assert (val['AREA_YEAR']  == 2022).all()
    assert (test['AREA_YEAR'] == 2023).all()
    assert (sup['AREA_YEAR']  >= 2024).sum() == 0
    return train, val, test

s1_train, s1_val, s1_test = create_temporal_splits(gold_s1, 'S1')
s2_train, s2_val, s2_test = create_temporal_splits(gold_s2, 'S2')

# Hard count assertions from NB04
assert len(s1_train) == 10161 and len(s1_val) == 2026 and len(s1_test) == 2055
assert len(s2_train) == 11858 and len(s2_val) == 2245 and len(s2_test) == 2507
print(f'  S1: train={len(s1_train):,}  val={len(s1_val):,}  test={len(s1_test):,}')
print(f'  S2: train={len(s2_train):,}  val={len(s2_val):,}  test={len(s2_test):,}')
print('  All temporal split assertions passed.')

# ===========================================================================
# §6  FEATURE MATRIX CONSTRUCTION
# ===========================================================================
print('\n[§6] Building feature matrices...')

CATEGORICAL_FEATURES = ['BASIN', 'COUNTRY', 'GTNG_REGION_O2', 'CONNECTIVITY']
NUMERICAL_FEATURES   = [
    'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
    'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
    'LATITUDE', 'LONGITUDE',
    'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
    'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3'
]
MODEL_FEATURE_VECTOR = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
assert len(MODEL_FEATURE_VECTOR) == 20
assert TARGET_COL not in MODEL_FEATURE_VECTOR
assert 'GLO_ID'         not in MODEL_FEATURE_VECTOR
assert 'AREA_YEAR'      not in MODEL_FEATURE_VECTOR
assert 'IS_TS_OUTLIER'  not in MODEL_FEATURE_VECTOR
assert 'AREA_CHANGE'    not in MODEL_FEATURE_VECTOR

X_s1_train, y_s1_train = s1_train[MODEL_FEATURE_VECTOR], s1_train[TARGET_COL]
X_s1_val,   y_s1_val   = s1_val[MODEL_FEATURE_VECTOR],   s1_val[TARGET_COL]
X_s1_test,  y_s1_test  = s1_test[MODEL_FEATURE_VECTOR],  s1_test[TARGET_COL]
X_s2_train, y_s2_train = s2_train[MODEL_FEATURE_VECTOR], s2_train[TARGET_COL]
X_s2_val,   y_s2_val   = s2_val[MODEL_FEATURE_VECTOR],   s2_val[TARGET_COL]
X_s2_test,  y_s2_test  = s2_test[MODEL_FEATURE_VECTOR],  s2_test[TARGET_COL]

for split_X, split_y in [(X_s1_train, y_s1_train), (X_s1_val, y_s1_val),
                          (X_s1_test, y_s1_test),   (X_s2_train, y_s2_train),
                          (X_s2_val, y_s2_val),     (X_s2_test, y_s2_test)]:
    assert split_X.shape[1] == 20
    assert TARGET_COL not in split_X.columns
    assert not split_y.isna().any()
print('  Feature matrices verified: 20 features, no NaN in targets.')

# ===========================================================================
# §7  TRAIN-ONLY PREPROCESSING
# ===========================================================================
print('\n[§7] Fitting preprocessing pipelines on training data only...')

def build_preprocessor():
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUMERICAL_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ])

# Fit once per sensor; reused by ALL models for that sensor
prep_s1 = build_preprocessor().fit(X_s1_train)
prep_s2 = build_preprocessor().fit(X_s2_train)

Xt_s1_train = prep_s1.transform(X_s1_train)
Xt_s1_val   = prep_s1.transform(X_s1_val)
Xt_s1_test  = prep_s1.transform(X_s1_test)
Xt_s2_train = prep_s2.transform(X_s2_train)
Xt_s2_val   = prep_s2.transform(X_s2_val)
Xt_s2_test  = prep_s2.transform(X_s2_test)

print(f'  S1 transformed shape: train={Xt_s1_train.shape}, val={Xt_s1_val.shape}, test={Xt_s1_test.shape}')
print(f'  S2 transformed shape: train={Xt_s2_train.shape}, val={Xt_s2_val.shape}, test={Xt_s2_test.shape}')

# ===========================================================================
# §8  BASELINE REFERENCE MODELS
# ===========================================================================
print('\n[§8] Evaluating baselines...')

results_records = []

def record(sensor, model, split, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    results_records.append({
        'Sensor': sensor, 'Model': model, 'Split': split,
        'N': len(y_true), 'MAE': mae, 'RMSE': rmse, 'R2': r2
    })
    return mae, rmse, r2

for sensor, y_tr, y_va, y_te, train_df, val_df, test_df in [
    ('Sentinel-1', y_s1_train, y_s1_val, y_s1_test, s1_train, s1_val, s1_test),
    ('Sentinel-2', y_s2_train, y_s2_val, y_s2_test, s2_train, s2_val, s2_test),
]:
    tr_mean = float(y_tr.mean())

    # Mean baseline
    for split, yt in [('Train', y_tr), ('Validation', y_va), ('Test', y_te)]:
        record(sensor, 'Mean Baseline', split, yt, np.full(len(yt), tr_mean))

    # Zero-change baseline
    for split, yt in [('Train', y_tr), ('Validation', y_va), ('Test', y_te)]:
        record(sensor, 'Zero-Change Baseline', split, yt, np.zeros(len(yt)))

print('  Baselines recorded.')

# ===========================================================================
# §9  CANDIDATE ADVANCED MODELS — FIXED PARAMS (NB04 equivalents)
# ===========================================================================
print('\n[§9] Training candidate models with NB04 hyperparameters...')

# Store trained raw estimators (already trained on transformed data)
trained_estimators = {}

fixed_model_specs = {
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingRegressor(
        max_iter=100, max_depth=6, min_samples_leaf=20,
        random_state=RANDOM_STATE),
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1),
}

for sensor, Xtr, ytr, Xva, yva, Xte, yte in [
    ('Sentinel-1', Xt_s1_train, y_s1_train, Xt_s1_val, y_s1_val, Xt_s1_test, y_s1_test),
    ('Sentinel-2', Xt_s2_train, y_s2_train, Xt_s2_val, y_s2_val, Xt_s2_test, y_s2_test),
]:
    for m_name, m_obj in fixed_model_specs.items():
        import copy
        est = copy.deepcopy(m_obj)
        t0 = time.time()
        est.fit(Xtr, ytr)
        elapsed = time.time() - t0

        key = f'{sensor}_{m_name}'
        trained_estimators[key] = est

        record(sensor, m_name, 'Train', ytr, est.predict(Xtr))
        record(sensor, m_name, 'Validation', yva, est.predict(Xva))
        record(sensor, m_name, 'Test', yte, est.predict(Xte))
        print(f'  {sensor:12s} | {m_name:25s} | {elapsed:.1f}s')

# ===========================================================================
# §10  EXPANDING-WINDOW TEMPORAL CV — HGB HYPERPARAMETER TUNING
# ===========================================================================
print('\n[§10] Expanding-window CV hyperparameter tuning...')

HGB_GRID = {
    'max_iter':         [100, 200, 300],
    'max_depth':        [4, 6, 8],
    'min_samples_leaf': [10, 20, 30],
    'learning_rate':    [0.05, 0.1],
}
CV_FOLD_YEARS = [(2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021)]

n_combos = 1
for v in HGB_GRID.values():
    n_combos *= len(v)
n_folds = len(CV_FOLD_YEARS)
total_fits = n_combos * n_folds * 2  # × 2 sensors

print(f'  HGB grid combinations : {n_combos}')
print(f'  Expanding-window folds: {n_folds}  '
      f'(train_end=[2017,2018,2019,2020], val_year=[2018,2019,2020,2021])')
print(f'  Total model fits      : {n_combos} combos × {n_folds} folds × 2 sensors = {total_fits}')

def expand_cv_score(Xtr_full, ytr_full, train_years, param_dict):
    """Single expanding-window CV MAE across all folds for given params."""
    fold_maes = []
    for train_end, val_year in CV_FOLD_YEARS:
        fold_mask_train = train_years <= train_end
        fold_mask_val   = train_years == val_year
        if fold_mask_val.sum() == 0:
            continue
        est = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **param_dict)
        est.fit(Xtr_full[fold_mask_train], ytr_full[fold_mask_train])
        y_pred = est.predict(Xtr_full[fold_mask_val])
        fold_maes.append(mean_absolute_error(ytr_full[fold_mask_val], y_pred))
    return float(np.mean(fold_maes))

param_keys = list(HGB_GRID.keys())
param_values = list(HGB_GRID.values())
all_combos = list(itertools.product(*param_values))

best_params = {}
cv_timing_start = time.time()

for sensor, Xtr, ytr, train_years_series in [
    ('Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values),
    ('Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values),
]:
    sensor_t0 = time.time()
    best_mae, best_combo = np.inf, None
    cv_detail = []
    for combo in all_combos:
        pd = dict(zip(param_keys, combo))
        cv_mae = expand_cv_score(Xtr, ytr, train_years_series, pd)
        cv_detail.append({'params': combo, 'cv_mae': cv_mae})
        if cv_mae < best_mae:
            best_mae, best_combo = cv_mae, combo
    elapsed = time.time() - sensor_t0
    best_params[sensor] = dict(zip(param_keys, best_combo))
    print(f'  {sensor}: best CV-MAE={best_mae:.6f}  '
          f'params={best_params[sensor]}  [{elapsed:.1f}s]')

total_cv_time = time.time() - cv_timing_start
print(f'  Total tuning time: {total_cv_time:.1f}s')

# Train tuned HGB on FULL training set (2017–2021)
print('\n[§10b] Training tuned HGB on full 2017-2021 training data...')
for sensor, Xtr, ytr, Xva, yva, Xte, yte in [
    ('Sentinel-1', Xt_s1_train, y_s1_train, Xt_s1_val, y_s1_val, Xt_s1_test, y_s1_test),
    ('Sentinel-2', Xt_s2_train, y_s2_train, Xt_s2_val, y_s2_val, Xt_s2_test, y_s2_test),
]:
    est = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, **best_params[sensor])
    est.fit(Xtr, ytr)
    trained_estimators[f'{sensor}_Tuned HGB'] = est
    record(sensor, 'Tuned HGB', 'Train',      ytr, est.predict(Xtr))
    record(sensor, 'Tuned HGB', 'Validation', yva, est.predict(Xva))
    record(sensor, 'Tuned HGB', 'Test',       yte, est.predict(Xte))
    val_mae = mean_absolute_error(yva, est.predict(Xva))
    print(f'  {sensor}: val MAE={val_mae:.6f}')

# ===========================================================================
# §11  VALIDATION COMPARISON TABLE
# ===========================================================================
print('\n[§11] Validation comparison...')

res_df = pd.DataFrame(results_records)
res_df = res_df.sort_values(['Sensor', 'Model', 'Split']).reset_index(drop=True)

val_table = res_df[res_df['Split'] == 'Validation'][
    ['Sensor', 'Model', 'MAE', 'RMSE', 'R2']
].sort_values(['Sensor', 'MAE'])

print('\nVALIDATION COMPARISON TABLE (sorted by MAE):')
print(val_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# ===========================================================================
# §12  MODEL SELECTION CRITERIA
# ===========================================================================
print('\n[§12] Applying model selection criteria...')

MODEL_SELECTION_CRITERIA = """
Model selection is based on VALIDATION performance (2022 period) only.
Selection is made before examining the frozen test set.

Criteria (in order of priority):
  1. Lowest validation MAE (primary physical metric)
  2. Among near-equivalent MAE (<2% relative difference): prefer lower RMSE
  3. Among near-equivalent MAE/RMSE: prefer simpler model (parsimony)
  4. Robustness: train/val MAE gap < 3× (no severe overfitting)
"""
print(MODEL_SELECTION_CRITERIA)

selected_models = {}
for sensor in ['Sentinel-1', 'Sentinel-2']:
    val_sub = val_table[val_table['Sensor'] == sensor].copy()
    best_model = val_sub.iloc[0]['Model']
    best_mae   = val_sub.iloc[0]['MAE']
    # Check train/val gap for robustness
    tr_mae_row = res_df[(res_df['Sensor'] == sensor) &
                        (res_df['Model']  == best_model) &
                        (res_df['Split']  == 'Train')]
    tr_mae = tr_mae_row['MAE'].values[0] if len(tr_mae_row) > 0 else np.nan
    gap = best_mae / tr_mae if tr_mae > 0 else np.nan
    selected_models[sensor] = best_model
    print(f'  {sensor}: SELECTED = {best_model}  '
          f'(val MAE={best_mae:.6f}, train MAE={tr_mae:.6f}, val/train ratio={gap:.2f}x)')

# ===========================================================================
# §13  FROZEN TEST SET EVALUATION
# ===========================================================================
print('\n[§13] Frozen test set evaluation (evaluated ONCE)...')

test_table = res_df[res_df['Split'] == 'Test'][
    ['Sensor', 'Model', 'MAE', 'RMSE', 'R2']
].sort_values(['Sensor', 'MAE'])
print('\nFROZEN TEST SET RESULTS:')
print(test_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# ===========================================================================
# MASTER RESULTS TABLE
# ===========================================================================
print('\n[MASTER] Full model comparison:')
master = []
for sensor in ['Sentinel-1', 'Sentinel-2']:
    for model in res_df['Model'].unique():
        sub = res_df[(res_df['Sensor'] == sensor) & (res_df['Model'] == model)]
        row = {'Sensor': sensor, 'Model': model}
        for split in ['Validation', 'Test']:
            s = sub[sub['Split'] == split]
            if len(s):
                row[f'{split} MAE']  = s['MAE'].values[0]
                row[f'{split} RMSE'] = s['RMSE'].values[0]
                row[f'{split} R2']   = s['R2'].values[0]
            else:
                row[f'{split} MAE'] = row[f'{split} RMSE'] = row[f'{split} R2'] = np.nan
        master.append(row)

master_df = pd.DataFrame(master)
master_df = master_df.sort_values(['Sensor', 'Validation MAE'])
print(master_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# Save master CSV
master_df.to_csv(REPORTS_DIR / '06_model_comparison.csv', index=False)
print(f'  Saved: reports/06_model_comparison.csv')

# ===========================================================================
# §14  LAKE-SIZE STRATIFIED EVALUATION
# ===========================================================================
print('\n[§14] Lake-size stratified evaluation...')

size_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    # TRAINING-DERIVED thresholds
    q33 = train_df['AREA'].quantile(0.333)
    q67 = train_df['AREA'].quantile(0.667)
    print(f'  {sensor}: AREA q33={q33:.5f} km², q67={q67:.5f} km²  [train-derived]')

    def size_tier(area, q33=q33, q67=q67):
        if area <= q33:   return 'Small'
        elif area <= q67: return 'Medium'
        else:             return 'Large'

    for split_lbl, split_df, Xt in [
        ('Validation', val_df, prep.transform(val_df[MODEL_FEATURE_VECTOR])),
        ('Test',       test_df, prep.transform(test_df[MODEL_FEATURE_VECTOR])),
    ]:
        y_true = split_df[TARGET_COL].values
        area_vals = split_df['AREA'].values
        tiers = np.array([size_tier(a) for a in area_vals])

        for model_name in ['Mean Baseline', 'Zero-Change Baseline', 'Ridge',
                           'Random Forest', 'HistGradientBoosting',
                           'ExtraTrees', 'Tuned HGB']:
            key = f'{sensor}_{model_name}'
            if key in trained_estimators:
                y_pred = trained_estimators[key].predict(Xt)
            elif model_name == 'Mean Baseline':
                y_pred = np.full(len(y_true), train_df[TARGET_COL].mean())
            elif model_name == 'Zero-Change Baseline':
                y_pred = np.zeros(len(y_true))
            else:
                continue

            for tier in ['Small', 'Medium', 'Large']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                mae  = mean_absolute_error(y_true[mask], y_pred[mask])
                rmse = root_mean_squared_error(y_true[mask], y_pred[mask])
                size_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Size Tier': tier, 'N': int(mask.sum()),
                    'MAE': mae, 'RMSE': rmse
                })

size_df = pd.DataFrame(size_records)
# Show selected model only for conciseness
print('\nSIZE-STRATIFIED (Selected models, Validation):')
for sensor, sel in selected_models.items():
    sub = size_df[(size_df['Sensor'] == sensor) &
                  (size_df['Model']  == sel) &
                  (size_df['Split']  == 'Validation')]
    if len(sub):
        print(f'\n  {sensor} | {sel}:')
        print(sub[['Size Tier', 'N', 'MAE', 'RMSE']].to_string(index=False,
              float_format=lambda x: f'{x:.6f}'))

size_df.to_csv(REPORTS_DIR / '06_size_stratified.csv', index=False)
print(f'\n  Saved: reports/06_size_stratified.csv')

# ===========================================================================
# §15  UNCERTAINTY-STRATIFIED EVALUATION
# ===========================================================================
print('\n[§15] Uncertainty-stratified evaluation...')

unc_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    # TRAINING-DERIVED uncertainty thresholds
    uq33 = train_df['AREA_UNCERTAINTY'].quantile(0.333)
    uq67 = train_df['AREA_UNCERTAINTY'].quantile(0.667)
    print(f'  {sensor}: AREA_UNCERTAINTY q33={uq33:.5f}, q67={uq67:.5f}  [train-derived]')

    def unc_tier(u, uq33=uq33, uq67=uq67):
        if u <= uq33:   return 'Low'
        elif u <= uq67: return 'Medium'
        else:           return 'High'

    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        y_true   = split_df[TARGET_COL].values
        unc_vals = split_df['AREA_UNCERTAINTY'].values
        tiers    = np.array([unc_tier(u) for u in unc_vals])
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])

        for model_name in ['Mean Baseline', 'Zero-Change Baseline', 'Ridge',
                           'Random Forest', 'HistGradientBoosting',
                           'ExtraTrees', 'Tuned HGB']:
            key = f'{sensor}_{model_name}'
            if key in trained_estimators:
                y_pred = trained_estimators[key].predict(Xt)
            elif model_name == 'Mean Baseline':
                y_pred = np.full(len(y_true), train_df[TARGET_COL].mean())
            elif model_name == 'Zero-Change Baseline':
                y_pred = np.zeros(len(y_true))
            else:
                continue

            for tier in ['Low', 'Medium', 'High']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                mae  = mean_absolute_error(y_true[mask], y_pred[mask])
                rmse = root_mean_squared_error(y_true[mask], y_pred[mask])
                unc_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Uncertainty Tier': tier, 'N': int(mask.sum()),
                    'MAE': mae, 'RMSE': rmse
                })

unc_df = pd.DataFrame(unc_records)
unc_df.to_csv(REPORTS_DIR / '06_uncertainty_stratified.csv', index=False)
print(f'  Saved: reports/06_uncertainty_stratified.csv')

# ===========================================================================
# §16  ERROR ANALYSIS
# ===========================================================================
print('\n[§16] Error analysis...')

error_by_year_records = []

for sensor, val_df, test_df, prep in [
    ('Sentinel-1', s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_val, s2_test, prep_s2),
]:
    sel = selected_models[sensor]
    key = f'{sensor}_{sel}'

    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])
        if key in trained_estimators:
            y_pred = trained_estimators[key].predict(Xt)
        else:
            y_pred = np.full(len(split_df), split_df[TARGET_COL].mean())
        y_true = split_df[TARGET_COL].values
        residuals = y_true - y_pred

        # By year
        for yr, idx in split_df.groupby('AREA_YEAR').groups.items():
            sub_true = y_true[split_df.index.get_indexer(idx)]
            sub_pred = y_pred[split_df.index.get_indexer(idx)]
            error_by_year_records.append({
                'Sensor': sensor, 'Model': sel, 'Split': split_lbl,
                'AREA_YEAR': yr,
                'N': len(idx),
                'MAE': mean_absolute_error(sub_true, sub_pred),
                'RMSE': root_mean_squared_error(sub_true, sub_pred),
                'Mean Residual': float(np.mean(sub_true - sub_pred))
            })

        # Residual distribution stats
        print(f'  {sensor} | {sel} | {split_lbl}:')
        print(f'    Residual mean ={np.mean(residuals):+.6f}  '
              f'std={np.std(residuals):.6f}  '
              f'skew={float(pd.Series(residuals).skew()):.3f}')
        top5_abs_err = sorted(zip(np.abs(residuals), y_true, y_pred),
                               reverse=True)[:5]
        print(f'    Top-5 absolute errors: '
              f'{[f"{e:.4f}" for e, *_ in top5_abs_err]}')

error_year_df = pd.DataFrame(error_by_year_records)
error_year_df.to_csv(REPORTS_DIR / '06_error_by_year.csv', index=False)
print(f'  Saved: reports/06_error_by_year.csv')

# ===========================================================================
# §17  FEATURE IMPORTANCE
# ===========================================================================
print('\n[§17] Computing permutation feature importances...')

# Get OHE feature names
def get_feature_names(prep):
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = list(prep.named_transformers_['cat']['ohe'].get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names

fi_records = []

for sensor, prep, Xva, yva in [
    ('Sentinel-1', prep_s1, Xt_s1_val, y_s1_val),
    ('Sentinel-2', prep_s2, Xt_s2_val, y_s2_val),
]:
    sel   = selected_models[sensor]
    key   = f'{sensor}_{sel}'
    est   = trained_estimators.get(key)
    if est is None:
        continue

    feat_names = get_feature_names(prep)
    t0 = time.time()
    pi = permutation_importance(est, Xva, yva,
                                n_repeats=10, random_state=RANDOM_STATE,
                                n_jobs=-1, scoring='neg_mean_absolute_error')
    print(f'  {sensor}: permutation importance done in {time.time()-t0:.1f}s')

    for i, name in enumerate(feat_names):
        fi_records.append({
            'Sensor': sensor, 'Feature': name,
            'Importance Mean': -pi.importances_mean[i],   # negate neg_MAE
            'Importance Std':   pi.importances_std[i]
        })

fi_df = pd.DataFrame(fi_records).sort_values(
    ['Sensor', 'Importance Mean'], ascending=[True, False])
fi_df.to_csv(REPORTS_DIR / '06_feature_importance.csv', index=False)
print(f'  Saved: reports/06_feature_importance.csv')

# Print top 10 per sensor
for sensor in ['Sentinel-1', 'Sentinel-2']:
    sub = fi_df[fi_df['Sensor'] == sensor].head(10)
    print(f'\n  {sensor} — Top 10 features by permutation importance:')
    for _, row in sub.iterrows():
        print(f'    {row["Feature"]:<35} {row["Importance Mean"]:+.6f} ± {row["Importance Std"]:.6f}')

# ===========================================================================
# §18  ROBUSTNESS DIAGNOSTICS
# ===========================================================================
print('\n[§18] Robustness diagnostics (train/val/test gap)...')

for sensor in ['Sentinel-1', 'Sentinel-2']:
    print(f'\n  {sensor}:')
    sub = res_df[res_df['Sensor'] == sensor].pivot(
        index='Model', columns='Split', values='MAE')
    for col in ['Train', 'Validation', 'Test']:
        if col not in sub.columns:
            sub[col] = np.nan
    sub = sub[['Train', 'Validation', 'Test']].copy()
    sub['Val/Train'] = (sub['Validation'] / sub['Train']).round(2)
    sub['Test/Val']  = (sub['Test'] / sub['Validation']).round(2)
    print(sub.to_string(float_format=lambda x: f'{x:.6f}'))

# ===========================================================================
# FIGURES
# ===========================================================================
print('\n[Figures] Generating all figures...')

DARK_BG    = '#0f1117'
PANEL_BG   = '#1a1d2e'
S1_COLOR   = '#00d4ff'
S2_COLOR   = '#a78bfa'
ACCENT     = '#ffcc00'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2d3e'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    '#3a3d4e',
    'axes.labelcolor':   TEXT_COLOR,
    'xtick.color':       TEXT_COLOR,
    'ytick.color':       TEXT_COLOR,
    'text.color':        TEXT_COLOR,
    'legend.facecolor':  PANEL_BG,
    'legend.edgecolor':  '#3a3d4e',
    'grid.color':        GRID_COLOR,
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

# ── Figure 1: Model Comparison Bar Chart ─────────────────────────────────────
print('  Generating 06_model_comparison.png...')
model_order = ['Mean Baseline', 'Zero-Change Baseline', 'Ridge',
               'Random Forest', 'HistGradientBoosting', 'ExtraTrees', 'Tuned HGB']

val_mae_s1 = [res_df[(res_df['Sensor']=='Sentinel-1') & (res_df['Model']==m) &
                     (res_df['Split']=='Validation')]['MAE'].values
              for m in model_order]
val_mae_s2 = [res_df[(res_df['Sensor']=='Sentinel-2') & (res_df['Model']==m) &
                     (res_df['Split']=='Validation')]['MAE'].values
              for m in model_order]
val_mae_s1 = [v[0] if len(v) else np.nan for v in val_mae_s1]
val_mae_s2 = [v[0] if len(v) else np.nan for v in val_mae_s2]

x = np.arange(len(model_order))
w = 0.38
fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
bars1 = ax.bar(x - w/2, val_mae_s1, w, color=S1_COLOR, alpha=0.85, label='Sentinel-1 (SAR)')
bars2 = ax.bar(x + w/2, val_mae_s2, w, color=S2_COLOR, alpha=0.85, label='Sentinel-2 (Optical)')
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ', '\n') for m in model_order], fontsize=8.5)
ax.set_ylabel('Validation MAE (km²)', fontsize=11)
ax.set_title('GlacierGuard-AI — Model Comparison: Validation MAE by Model & Sensor',
             fontsize=12, pad=12, color=TEXT_COLOR)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.5)
# Annotate bars
for bar in bars1:
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{bar.get_height():.5f}', ha='center', va='bottom', fontsize=6.5, color=S1_COLOR)
for bar in bars2:
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{bar.get_height():.5f}', ha='center', va='bottom', fontsize=6.5, color=S2_COLOR)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_model_comparison.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Figure 2 & 3: Residual diagnostics ───────────────────────────────────────
for sensor, val_df, prep, color, prefix in [
    ('Sentinel-1', s1_val, prep_s1, S1_COLOR, 's1'),
    ('Sentinel-2', s2_val, prep_s2, S2_COLOR, 's2'),
]:
    print(f'  Generating 06_residuals_{prefix}.png...')
    sel = selected_models[sensor]
    key = f'{sensor}_{sel}'
    est = trained_estimators.get(key)
    Xt = prep.transform(val_df[MODEL_FEATURE_VECTOR])
    y_true = val_df[TARGET_COL].values
    y_pred = est.predict(Xt) if est else np.full(len(y_true), y_true.mean())
    resid  = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f'{sensor} | {sel} — Residual Diagnostics (Validation 2022)',
                 color=TEXT_COLOR, fontsize=12, y=1.01)

    # (a) Residual distribution
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.hist(resid, bins=80, color=color, alpha=0.8, edgecolor='none')
    ax.axvline(0, color=ACCENT, linestyle='--', lw=1.5, label='Zero')
    ax.axvline(np.mean(resid), color='#ff7675', linestyle=':', lw=1.5,
               label=f'Mean={np.mean(resid):.4f}')
    ax.set_xlabel('Residual (km²)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Residual Distribution', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)

    # (b) Actual vs Predicted
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max()) * 1.05
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color=color, edgecolors='none')
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='Perfect')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual ΔA (km²)', fontsize=10)
    ax.set_ylabel('Predicted ΔA (km²)', fontsize=10)
    ax.set_title('Actual vs Predicted', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)

    # (c) Residual vs Predicted
    ax = axes[2]
    ax.set_facecolor(PANEL_BG)
    ax.scatter(y_pred, resid, alpha=0.2, s=8, color=color, edgecolors='none')
    ax.axhline(0, color=ACCENT, linestyle='--', lw=1.5)
    ax.set_xlabel('Predicted ΔA (km²)', fontsize=10)
    ax.set_ylabel('Residual (km²)', fontsize=10)
    ax.set_title('Residual vs Predicted', fontsize=10, color=TEXT_COLOR)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'06_residuals_{prefix}.png',
                dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()

# ── Figure 4: Size-stratified error ──────────────────────────────────────────
print('  Generating 06_size_stratified_error.png...')
size_val = size_df[size_df['Split'] == 'Validation']
sensors_  = size_val['Sensor'].unique()
models_to_plot = ['Mean Baseline', 'Zero-Change Baseline', 'Ridge',
                  'HistGradientBoosting', 'ExtraTrees', 'Tuned HGB']

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Size-Stratified Validation MAE (Training-Derived Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
colors_map = plt.cm.Set2(np.linspace(0, 1, len(models_to_plot)))

for ax, sensor, sen_color in zip(axes, ['Sentinel-1', 'Sentinel-2'], [S1_COLOR, S2_COLOR]):
    ax.set_facecolor(PANEL_BG)
    sub = size_val[size_val['Sensor'] == sensor]
    x_cats = ['Small', 'Medium', 'Large']
    n_models = sum(1 for m in models_to_plot if m in sub['Model'].values)
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    i = 0
    for m, c in zip(models_to_plot, colors_map):
        sub_m = sub[sub['Model'] == m]
        if len(sub_m) == 0:
            continue
        maes = [sub_m[sub_m['Size Tier'] == t]['MAE'].values
                for t in x_cats]
        maes = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats)) + offsets[i], maes, width*0.9,
               color=c, alpha=0.85, label=m)
        i += 1
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(x_cats)
    ax.set_xlabel('Lake Size Tier', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_size_stratified_error.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Figure 5: Uncertainty-stratified error ────────────────────────────────────
print('  Generating 06_uncertainty_stratified_error.png...')
unc_val = unc_df[unc_df['Split'] == 'Validation']

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Uncertainty-Stratified Validation MAE (Training-Derived Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)

for ax, sensor in zip(axes, ['Sentinel-1', 'Sentinel-2']):
    ax.set_facecolor(PANEL_BG)
    sub = unc_val[unc_val['Sensor'] == sensor]
    x_cats = ['Low', 'Medium', 'High']
    n_models = sum(1 for m in models_to_plot if m in sub['Model'].values)
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    i = 0
    for m, c in zip(models_to_plot, colors_map):
        sub_m = sub[sub['Model'] == m]
        if len(sub_m) == 0:
            continue
        maes = [sub_m[sub_m['Uncertainty Tier'] == t]['MAE'].values for t in x_cats]
        maes = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats)) + offsets[i], maes, width*0.9,
               color=c, alpha=0.85, label=m)
        i += 1
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(x_cats)
    ax.set_xlabel('Area Uncertainty Tier (σ_A)', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_uncertainty_stratified_error.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Figure 6 & 7: Feature Importance ────────────────────────────────────────
for sensor, prefix, color in [
    ('Sentinel-1', 's1', S1_COLOR),
    ('Sentinel-2', 's2', S2_COLOR),
]:
    print(f'  Generating 06_feature_importance_{prefix}.png...')
    sub = fi_df[fi_df['Sensor'] == sensor].head(20).sort_values('Importance Mean')
    if len(sub) == 0:
        continue
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub['Importance Mean'], xerr=sub['Importance Std'],
            color=color, alpha=0.85, height=0.7, error_kw={'ecolor': ACCENT, 'capsize': 3})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub['Feature'], fontsize=8.5)
    ax.set_xlabel('Permutation Importance (MAE decrease, km²)', fontsize=10)
    ax.set_title(f'{sensor} — Top-20 Feature Importance\n(Permutation, Validation 2022)',
                 fontsize=11, color=TEXT_COLOR, pad=10)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.axvline(0, color='#ff7675', linestyle='--', lw=1.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'06_feature_importance_{prefix}.png',
                dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()

# ── Figure 8: Error by year ───────────────────────────────────────────────────
print('  Generating 06_error_by_year.png...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('Selected Model — MAE by Observation Year (Val + Test)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
for ax, sensor, color in zip(axes, ['Sentinel-1', 'Sentinel-2'], [S1_COLOR, S2_COLOR]):
    ax.set_facecolor(PANEL_BG)
    sel = selected_models[sensor]
    sub = error_year_df[(error_year_df['Sensor'] == sensor) &
                        (error_year_df['Model']  == sel)]
    for split_lbl, ls in [('Validation', '-'), ('Test', '--')]:
        ss = sub[sub['Split'] == split_lbl].sort_values('AREA_YEAR')
        if len(ss):
            ax.plot(ss['AREA_YEAR'], ss['MAE'], marker='o', ls=ls,
                    color=color, label=split_lbl, linewidth=2)
    ax.set_xlabel('AREA_YEAR (current observation year)', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(f'{sensor} | {sel}', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_error_by_year.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Figure 9: CV tuning curves ───────────────────────────────────────────────
print('  Generating 06_tuning_cv_curves.png...')
# Rebuild top-5 param combos per sensor for display
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('HGB Expanding-Window CV Tuning — Fold MAE by Learning Rate',
             color=TEXT_COLOR, fontsize=12, y=1.01)

for ax, sensor, Xtr, ytr, train_years_series, color in [
    ('Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values, S1_COLOR),
    ('Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values, S2_COLOR),
]:
    # Sample representative param combos (fix mi=200, md=6, msl=20, vary lr)
    ref_combos = [(200, 6, 20, lr) for lr in HGB_GRID['learning_rate']]
    ax_ = axes[['Sentinel-1', 'Sentinel-2'].index(sensor)]
    ax_.set_facecolor(PANEL_BG)
    for combo in ref_combos:
        pd_ = dict(zip(param_keys, combo))
        fold_maes = []
        for train_end, val_year in CV_FOLD_YEARS:
            fm_tr = train_years_series <= train_end
            fm_va = train_years_series == val_year
            if fm_va.sum() == 0:
                continue
            est_ = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **pd_)
            est_.fit(Xtr[fm_tr], ytr[fm_tr])
            fold_maes.append(mean_absolute_error(ytr[fm_va], est_.predict(Xtr[fm_va])))
        fold_years = [y for _, y in CV_FOLD_YEARS if (train_years_series == y).sum() > 0]
        ax_.plot(fold_years, fold_maes, marker='o', label=f'lr={pd_["learning_rate"]}',
                 linewidth=2)
    ax_.set_xlabel('Validation Year (fold)', fontsize=10)
    ax_.set_ylabel('Fold MAE (km²)', fontsize=10)
    ax_.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax_.legend(fontsize=9)
    ax_.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_tuning_cv_curves.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

print('\n  All figures saved.')

# ===========================================================================
# §19  RESEARCH INTERPRETATION (summary)
# ===========================================================================
print('\n[§19] Research Interpretation:')
print('=' * 65)
for sensor in ['Sentinel-1', 'Sentinel-2']:
    sel = selected_models[sensor]
    val_row  = res_df[(res_df['Sensor']==sensor) & (res_df['Model']==sel) &
                      (res_df['Split']=='Validation')].iloc[0]
    test_row = res_df[(res_df['Sensor']==sensor) & (res_df['Model']==sel) &
                      (res_df['Split']=='Test')].iloc[0]
    base_val = res_df[(res_df['Sensor']==sensor) & (res_df['Model']=='Mean Baseline') &
                      (res_df['Split']=='Validation')].iloc[0]
    delta_mae = val_row['MAE'] - base_val['MAE']
    pct_improv = (delta_mae / base_val['MAE']) * 100

    print(f'\n  {sensor}:')
    print(f'    Selected model   : {sel}')
    print(f'    Val  MAE/RMSE/R² : {val_row["MAE"]:.6f} / {val_row["RMSE"]:.6f} / {val_row["R2"]:.4f}')
    print(f'    Test MAE/RMSE/R² : {test_row["MAE"]:.6f} / {test_row["RMSE"]:.6f} / {test_row["R2"]:.4f}')
    print(f'    vs Mean Baseline : ΔMAE = {delta_mae:+.6f} ({pct_improv:+.2f}%)')
    print(f'    Best CV params   : {best_params.get(sensor, "N/A")}')

# ===========================================================================
# §20  FINAL ARTIFACT
# ===========================================================================
print('\n[§20] Final artifacts:')
print(f'  reports/06_model_comparison.csv')
print(f'  reports/06_size_stratified.csv')
print(f'  reports/06_uncertainty_stratified.csv')
print(f'  reports/06_feature_importance.csv')
print(f'  reports/06_error_by_year.csv')
figs = sorted(FIGURES_DIR.glob('06_*.png'))
for f in figs:
    print(f'  reports/figures/{f.name}')

print('\n' + '=' * 65)
print('EXECUTION COMPLETE')
print('=' * 65)
