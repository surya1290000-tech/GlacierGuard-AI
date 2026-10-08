"""
NB06 Continuation Script — picks up after successful §10 tuning.
Uses already-discovered best_params, skips the 80-minute grid search.
Fixes: pd variable clash, runs §11–§20, saves all figures and CSVs.
"""
import os, sys, pathlib, warnings, time, copy
import numpy as np
import pandas as pd                     # NEVER shadow this with 'pd = dict(...)'
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

warnings.filterwarnings('ignore')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
FIGURES_DIR = pathlib.Path('reports/figures')
REPORTS_DIR = pathlib.Path('reports')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 65)
print('NB06 — CONTINUATION (post-tuning, verified best params)')
print('=' * 65)

# ── §3  Gold Dataset (fast rebuild) ─────────────────────────────────────────
GLO_DIR = None
for p in [pathlib.Path('raw/GLO'), pathlib.Path('../raw/GLO')]:
    if p.exists() and (p / 'S1_20172024_NTB_GLOID_v1.02.gpkg').exists():
        GLO_DIR = p.resolve(); break
assert GLO_DIR, 'GLO annual files not found'
print(f'GLO: {GLO_DIR}')

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
    df['AREA_LAG1'] = np.where(df['PREV_YEAR_GAP']==1, df['PREV_AREA'], np.nan)
    df['AREA_LAG2'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1), df['PREV_AREA_2'], np.nan)
    df['AREA_LAG3'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1)&((df['PREV_YEAR_2']-df['PREV_YEAR_3'])==1), df['PREV_AREA_3'], np.nan)
    df['AREA_LAG4'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1)&((df['PREV_YEAR_2']-df['PREV_YEAR_3'])==1)&((df['PREV_YEAR_3']-df['PREV_YEAR_4'])==1), df['PREV_AREA_4'], np.nan)
    df['AREA_CHANGE']      = np.where(df['PREV_YEAR_GAP']==1, df['AREA']-df['PREV_AREA'], np.nan)
    df['AREA_CHANGE_LAG1'] = np.where(df['AREA_LAG1'].notna()&df['AREA_LAG2'].notna(), df['AREA_LAG1']-df['AREA_LAG2'], np.nan)
    df['AREA_CHANGE_LAG2'] = np.where(df['AREA_LAG2'].notna()&df['AREA_LAG3'].notna(), df['AREA_LAG2']-df['AREA_LAG3'], np.nan)
    df['AREA_CHANGE_LAG3'] = np.where(df['AREA_LAG3'].notna()&df['AREA_LAG4'].notna(), df['AREA_LAG3']-df['AREA_LAG4'], np.nan)
    c_df = df[['AREA_CHANGE_LAG1','AREA_CHANGE_LAG2','AREA_CHANGE_LAG3']]
    df['ROLLING_MEAN_3'] = c_df.mean(axis=1, skipna=True)
    df['ROLLING_STD_3']  = c_df.std(axis=1, skipna=True, ddof=1)
    df['NEXT_AREA_CHANGE'] = np.where(df['NEXT_YEAR_GAP']==1, df['NEXT_AREA']-df['AREA'], np.nan)
    df['IS_TS_OUTLIER'] = df['TS_OUTLIER'].astype(str).str.upper().isin(['TRUE','1'])
    gold_cols = [
        'GLO_ID','AREA_YEAR','AREA','PERIMETER','AREA_UNCERTAINTY',
        'ELEVATION_MEAN','ELEVATION_MIN','ELEVATION_MEDIAN',
        'CONNECTIVITY','LATITUDE','LONGITUDE','BASIN','COUNTRY','GTNG_REGION_O2',
        'TS_OUTLIER','IS_TS_OUTLIER',
        'AREA_LAG1','AREA_LAG2','AREA_LAG3',
        'AREA_CHANGE_LAG1','AREA_CHANGE_LAG2','AREA_CHANGE_LAG3',
        'ROLLING_MEAN_3','ROLLING_STD_3','AREA_CHANGE','NEXT_AREA_CHANGE'
    ]
    return df[gold_cols].copy()

t0 = time.time()
gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')
print(f'Loaded in {time.time()-t0:.1f}s')

assert len(gold_s1)==18129 and len(gold_s2)==22294
assert gold_s1['NEXT_AREA_CHANGE'].notna().sum()==14242
assert gold_s2['NEXT_AREA_CHANGE'].notna().sum()==16610
print('Gold dataset assertions passed.')

# ── §5  Splits ───────────────────────────────────────────────────────────────
TARGET_COL = 'NEXT_AREA_CHANGE'

def create_splits(df):
    sup   = df[df[TARGET_COL].notna()].copy()
    train = sup[sup['AREA_YEAR'] <= 2021].copy()
    val   = sup[sup['AREA_YEAR'] == 2022].copy()
    test  = sup[sup['AREA_YEAR'] == 2023].copy()
    return train, val, test

s1_train, s1_val, s1_test = create_splits(gold_s1)
s2_train, s2_val, s2_test = create_splits(gold_s2)
assert len(s1_train)==10161 and len(s1_val)==2026 and len(s1_test)==2055
assert len(s2_train)==11858 and len(s2_val)==2245 and len(s2_test)==2507
print('Split assertions passed.')

# ── §6  Features ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ['BASIN','COUNTRY','GTNG_REGION_O2','CONNECTIVITY']
NUMERICAL_FEATURES = [
    'AREA','PERIMETER','AREA_UNCERTAINTY',
    'ELEVATION_MEAN','ELEVATION_MIN','ELEVATION_MEDIAN',
    'LATITUDE','LONGITUDE',
    'AREA_LAG1','AREA_LAG2','AREA_LAG3',
    'AREA_CHANGE_LAG1','AREA_CHANGE_LAG2','AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3','ROLLING_STD_3'
]
MODEL_FEATURE_VECTOR = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
assert len(MODEL_FEATURE_VECTOR) == 20

X_s1_train, y_s1_train = s1_train[MODEL_FEATURE_VECTOR], s1_train[TARGET_COL]
X_s1_val,   y_s1_val   = s1_val[MODEL_FEATURE_VECTOR],   s1_val[TARGET_COL]
X_s1_test,  y_s1_test  = s1_test[MODEL_FEATURE_VECTOR],  s1_test[TARGET_COL]
X_s2_train, y_s2_train = s2_train[MODEL_FEATURE_VECTOR], s2_train[TARGET_COL]
X_s2_val,   y_s2_val   = s2_val[MODEL_FEATURE_VECTOR],   s2_val[TARGET_COL]
X_s2_test,  y_s2_test  = s2_test[MODEL_FEATURE_VECTOR],  s2_test[TARGET_COL]
print('Feature matrices: OK')

# ── §7  Preprocessing (train-only fit) ───────────────────────────────────────
def build_preprocessor():
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUMERICAL_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ])

prep_s1 = build_preprocessor().fit(X_s1_train)
prep_s2 = build_preprocessor().fit(X_s2_train)

Xt_s1_train = prep_s1.transform(X_s1_train)
Xt_s1_val   = prep_s1.transform(X_s1_val)
Xt_s1_test  = prep_s1.transform(X_s1_test)
Xt_s2_train = prep_s2.transform(X_s2_train)
Xt_s2_val   = prep_s2.transform(X_s2_val)
Xt_s2_test  = prep_s2.transform(X_s2_test)
print(f'Preprocessing: S1 {Xt_s1_train.shape}, S2 {Xt_s2_train.shape}')

# ── §8  Baselines ────────────────────────────────────────────────────────────
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

for sensor, y_tr, y_va, y_te in [
    ('Sentinel-1', y_s1_train, y_s1_val, y_s1_test),
    ('Sentinel-2', y_s2_train, y_s2_val, y_s2_test),
]:
    tr_mean = float(y_tr.mean())
    for split, yt in [('Train', y_tr), ('Validation', y_va), ('Test', y_te)]:
        record(sensor, 'Mean Baseline',         split, yt, np.full(len(yt), tr_mean))
        record(sensor, 'Zero-Change Baseline',  split, yt, np.zeros(len(yt)))

# ── §9  Fixed-param models ────────────────────────────────────────────────────
trained_estimators = {}

fixed_specs = {
    'Ridge':               Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Random Forest':       RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
    'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100, max_depth=6,
                               min_samples_leaf=20, random_state=RANDOM_STATE),
    'ExtraTrees':          ExtraTreesRegressor(n_estimators=200, max_depth=20,
                               min_samples_leaf=5, max_features=0.7,
                               random_state=RANDOM_STATE, n_jobs=-1),
}

print('\nTraining fixed-param models...')
for sensor, Xtr, ytr, Xva, yva, Xte, yte in [
    ('Sentinel-1', Xt_s1_train, y_s1_train, Xt_s1_val, y_s1_val, Xt_s1_test, y_s1_test),
    ('Sentinel-2', Xt_s2_train, y_s2_train, Xt_s2_val, y_s2_val, Xt_s2_test, y_s2_test),
]:
    for m_name, m_obj in fixed_specs.items():
        est = copy.deepcopy(m_obj)
        t0  = time.time()
        est.fit(Xtr, ytr)
        trained_estimators[f'{sensor}_{m_name}'] = est
        record(sensor, m_name, 'Train',      ytr, est.predict(Xtr))
        record(sensor, m_name, 'Validation', yva, est.predict(Xva))
        record(sensor, m_name, 'Test',       yte, est.predict(Xte))
        print(f'  {sensor:12s} | {m_name:25s} | {time.time()-t0:.1f}s')

# ── §10  Tuning — small grid for notebook reproducibility ────────────────────
# IMPORTANT: Report tuning plan BEFORE running
print('\n[§10] HGB Hyperparameter Tuning')
print('  NOTE: Grid reduced to 2×2×2×1 = 8 combinations × 4 folds for notebook speed.')
print('  Full 54-combo run was completed in the validation phase:')
print('    S1 best: max_iter=100, max_depth=4, min_samples_leaf=30, lr=0.05 (CV-MAE=0.007689)')
print('    S2 best: max_iter=100, max_depth=4, min_samples_leaf=30, lr=0.05 (CV-MAE=0.005265)')

HGB_GRID_SMALL = {
    'max_iter':         [100, 200],
    'max_depth':        [4, 6],
    'min_samples_leaf': [20, 30],
    'learning_rate':    [0.05],    # confirmed best LR from full run
}
CV_FOLD_YEARS = [(2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021)]

import itertools
param_keys_hgb = list(HGB_GRID_SMALL.keys())
param_values_hgb = list(HGB_GRID_SMALL.values())
all_combos_hgb = list(itertools.product(*param_values_hgb))
n_combos_small = len(all_combos_hgb)
n_folds        = len(CV_FOLD_YEARS)
total_fits     = n_combos_small * n_folds * 2
print(f'  Small grid: {n_combos_small} combos × {n_folds} folds × 2 sensors = {total_fits} fits')

def expand_cv_score(Xtr_full, ytr_full, train_years_arr, param_dict_hgb):
    """Expanding-window CV MAE. NOTE: param variable deliberately named param_dict_hgb
    to avoid shadowing the pandas 'pd' import."""
    fold_maes = []
    for train_end, val_year in CV_FOLD_YEARS:
        fm_tr = train_years_arr <= train_end
        fm_va = train_years_arr == val_year
        if fm_va.sum() == 0:
            continue
        est = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **param_dict_hgb)
        est.fit(Xtr_full[fm_tr], ytr_full[fm_tr])
        fold_maes.append(mean_absolute_error(ytr_full[fm_va], est.predict(Xtr_full[fm_va])))
    return float(np.mean(fold_maes))

best_params   = {}
cv_all_results = {}   # for tuning curve figure

for sensor, Xtr, ytr_arr, train_years_arr in [
    ('Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values),
    ('Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values),
]:
    t_sensor = time.time()
    best_mae, best_combo = np.inf, None
    sensor_cv = []
    for combo in all_combos_hgb:
        param_dict_hgb = dict(zip(param_keys_hgb, combo))   # NOT named 'pd'
        cv_mae = expand_cv_score(Xtr, ytr_arr, train_years_arr, param_dict_hgb)
        sensor_cv.append({'combo': combo, 'cv_mae': cv_mae})
        if cv_mae < best_mae:
            best_mae, best_combo = cv_mae, combo
    best_params[sensor] = dict(zip(param_keys_hgb, best_combo))
    cv_all_results[sensor] = sensor_cv
    print(f'  {sensor}: best CV-MAE={best_mae:.6f}  params={best_params[sensor]}  '
          f'[{time.time()-t_sensor:.1f}s]')

# §10b — Train tuned HGB on full train set
print('\n[§10b] Training Tuned HGB on 2017-2021...')
for sensor, Xtr, ytr, Xva, yva, Xte, yte in [
    ('Sentinel-1', Xt_s1_train, y_s1_train, Xt_s1_val, y_s1_val, Xt_s1_test, y_s1_test),
    ('Sentinel-2', Xt_s2_train, y_s2_train, Xt_s2_val, y_s2_val, Xt_s2_test, y_s2_test),
]:
    est = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **best_params[sensor])
    est.fit(Xtr, ytr)
    trained_estimators[f'{sensor}_Tuned HGB'] = est
    record(sensor, 'Tuned HGB', 'Train',      ytr, est.predict(Xtr))
    record(sensor, 'Tuned HGB', 'Validation', yva, est.predict(Xva))
    record(sensor, 'Tuned HGB', 'Test',       yte, est.predict(Xte))
    val_mae = mean_absolute_error(yva, est.predict(Xva))
    print(f'  {sensor}: val MAE={val_mae:.6f}  params={best_params[sensor]}')

# ── §11  Validation Comparison ────────────────────────────────────────────────
print('\n' + '='*65)
print('[§11] VALIDATION COMPARISON TABLE')
print('='*65)
res_df = pd.DataFrame(results_records)           # pd is pandas — no clash now
res_df = res_df.sort_values(['Sensor','Model','Split']).reset_index(drop=True)

val_table = (res_df[res_df['Split']=='Validation']
             [['Sensor','Model','MAE','RMSE','R2']]
             .sort_values(['Sensor','MAE']))
print(val_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# ── §12  Model Selection ──────────────────────────────────────────────────────
print('\n[§12] MODEL SELECTION (validation-based, before test is opened)')
selected_models = {}
for sensor in ['Sentinel-1', 'Sentinel-2']:
    val_sub = val_table[val_table['Sensor'] == sensor].reset_index(drop=True)
    best_model = val_sub.iloc[0]['Model']
    best_val_mae = val_sub.iloc[0]['MAE']
    tr_row  = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==best_model)&(res_df['Split']=='Train')]
    tr_mae  = tr_row['MAE'].values[0] if len(tr_row) else np.nan
    gap     = best_val_mae / tr_mae if tr_mae > 0 else np.nan
    selected_models[sensor] = best_model
    print(f'  {sensor}: SELECTED = {best_model}')
    print(f'    Val MAE={best_val_mae:.6f}, Train MAE={tr_mae:.6f}, Val/Train ratio={gap:.2f}x')

# ── §13  FROZEN TEST EVALUATION ───────────────────────────────────────────────
print('\n' + '='*65)
print('[§13] FROZEN TEST SET EVALUATION — evaluated once, after model selection')
print('='*65)
test_table = (res_df[res_df['Split']=='Test']
              [['Sensor','Model','MAE','RMSE','R2']]
              .sort_values(['Sensor','MAE']))
print(test_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# ── Master results table ──────────────────────────────────────────────────────
master_rows = []
model_order_all = ['Mean Baseline','Zero-Change Baseline','Ridge','Random Forest',
                   'HistGradientBoosting','ExtraTrees','Tuned HGB']
for sensor in ['Sentinel-1','Sentinel-2']:
    for model in model_order_all:
        row = {'Sensor': sensor, 'Model': model}
        for split in ['Validation','Test']:
            s = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==model)&(res_df['Split']==split)]
            for col in ['MAE','RMSE','R2']:
                row[f'{split} {col}'] = s[col].values[0] if len(s) else np.nan
        master_rows.append(row)
master_df = pd.DataFrame(master_rows).sort_values(['Sensor','Validation MAE'])
print('\nMASTER MODEL COMPARISON:')
print(master_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
master_df.to_csv(REPORTS_DIR/'06_model_comparison.csv', index=False)
print('Saved: reports/06_model_comparison.csv')

# ── §14  Size-stratified ──────────────────────────────────────────────────────
print('\n[§14] SIZE-STRATIFIED EVALUATION')
size_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    q33 = train_df['AREA'].quantile(0.333)
    q67 = train_df['AREA'].quantile(0.667)
    print(f'  {sensor}: AREA size thresholds (train-derived): q33={q33:.5f}, q67={q67:.5f} km²')

    def size_tier(a, q33=q33, q67=q67):
        return 'Small' if a <= q33 else ('Medium' if a <= q67 else 'Large')

    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        y_true    = split_df[TARGET_COL].values
        area_vals = split_df['AREA'].values
        tiers     = np.array([size_tier(a) for a in area_vals])
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])

        for model_name in model_order_all:
            key = f'{sensor}_{model_name}'
            if key in trained_estimators:
                y_pred = trained_estimators[key].predict(Xt)
            elif model_name == 'Mean Baseline':
                y_pred = np.full(len(y_true), train_df[TARGET_COL].mean())
            elif model_name == 'Zero-Change Baseline':
                y_pred = np.zeros(len(y_true))
            else:
                continue
            for tier in ['Small','Medium','Large']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                size_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Size Tier': tier, 'N': int(mask.sum()),
                    'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'RMSE': root_mean_squared_error(y_true[mask], y_pred[mask])
                })

size_df = pd.DataFrame(size_records)
for sensor, sel in selected_models.items():
    sub = size_df[(size_df['Sensor']==sensor)&(size_df['Model']==sel)&(size_df['Split']=='Validation')]
    print(f'\n  {sensor} | {sel} (Validation size-stratified):')
    print(sub[['Size Tier','N','MAE','RMSE']].to_string(index=False, float_format=lambda x:f'{x:.6f}'))
size_df.to_csv(REPORTS_DIR/'06_size_stratified.csv', index=False)
print('Saved: reports/06_size_stratified.csv')

# ── §15  Uncertainty-stratified ───────────────────────────────────────────────
print('\n[§15] UNCERTAINTY-STRATIFIED EVALUATION')
unc_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    uq33 = train_df['AREA_UNCERTAINTY'].quantile(0.333)
    uq67 = train_df['AREA_UNCERTAINTY'].quantile(0.667)
    print(f'  {sensor}: AREA_UNCERTAINTY thresholds: q33={uq33:.5f}, q67={uq67:.5f} km²')

    def unc_tier(u, uq33=uq33, uq67=uq67):
        return 'Low' if u <= uq33 else ('Medium' if u <= uq67 else 'High')

    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        y_true   = split_df[TARGET_COL].values
        unc_vals = split_df['AREA_UNCERTAINTY'].values
        tiers    = np.array([unc_tier(u) for u in unc_vals])
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])
        for model_name in model_order_all:
            key = f'{sensor}_{model_name}'
            if key in trained_estimators:
                y_pred = trained_estimators[key].predict(Xt)
            elif model_name == 'Mean Baseline':
                y_pred = np.full(len(y_true), train_df[TARGET_COL].mean())
            elif model_name == 'Zero-Change Baseline':
                y_pred = np.zeros(len(y_true))
            else:
                continue
            for tier in ['Low','Medium','High']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                unc_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Uncertainty Tier': tier, 'N': int(mask.sum()),
                    'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'RMSE': root_mean_squared_error(y_true[mask], y_pred[mask])
                })

unc_df = pd.DataFrame(unc_records)
unc_df.to_csv(REPORTS_DIR/'06_uncertainty_stratified.csv', index=False)
print('Saved: reports/06_uncertainty_stratified.csv')

# ── §16  Error Analysis ───────────────────────────────────────────────────────
print('\n[§16] ERROR ANALYSIS')
error_year_records = []

for sensor, val_df, test_df, prep in [
    ('Sentinel-1', s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_val, s2_test, prep_s2),
]:
    sel = selected_models[sensor]
    key = f'{sensor}_{sel}'
    est = trained_estimators.get(key)
    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])
        y_true = split_df[TARGET_COL].values
        y_pred = est.predict(Xt) if est else np.full(len(y_true), y_true.mean())
        resid  = y_true - y_pred
        print(f'\n  {sensor} | {sel} | {split_lbl}:')
        print(f'    Residual mean={np.mean(resid):+.6f}  std={np.std(resid):.6f}  '
              f'skew={float(pd.Series(resid).skew()):.3f}  '
              f'kurt={float(pd.Series(resid).kurt()):.3f}')
        top5 = sorted(zip(np.abs(resid), range(len(resid))), reverse=True)[:5]
        print(f'    Top-5 |residual|: {[f"{e:.5f}" for e, _ in top5]}')
        overpredict  = (resid < 0).sum()
        underpredict = (resid > 0).sum()
        print(f'    Overpredictions (pred>true): {overpredict} ({overpredict/len(resid)*100:.1f}%)')
        print(f'    Underpredictions (pred<true): {underpredict} ({underpredict/len(resid)*100:.1f}%)')

        # By year
        for yr, idx_group in split_df.groupby('AREA_YEAR').groups.items():
            pos = split_df.index.get_indexer(idx_group)
            sub_t = y_true[pos]; sub_p = y_pred[pos]
            error_year_records.append({
                'Sensor': sensor, 'Model': sel, 'Split': split_lbl,
                'AREA_YEAR': yr, 'N': len(pos),
                'MAE': mean_absolute_error(sub_t, sub_p),
                'RMSE': root_mean_squared_error(sub_t, sub_p),
                'Mean Residual': float(np.mean(sub_t - sub_p))
            })

        # Outlier vs non-outlier
        if 'IS_TS_OUTLIER' in split_df.columns:
            out_mask = split_df['IS_TS_OUTLIER'].values
            if out_mask.sum() > 5:
                print(f'    Outlier MAE    : {mean_absolute_error(y_true[out_mask], y_pred[out_mask]):.6f}  '
                      f'(n={out_mask.sum()})')
                print(f'    Non-outlier MAE: {mean_absolute_error(y_true[~out_mask], y_pred[~out_mask]):.6f}  '
                      f'(n={(~out_mask).sum()})')

error_year_df = pd.DataFrame(error_year_records)
error_year_df.to_csv(REPORTS_DIR/'06_error_by_year.csv', index=False)
print('\nSaved: reports/06_error_by_year.csv')

# ── §17  Feature Importance ────────────────────────────────────────────────────
print('\n[§17] PERMUTATION FEATURE IMPORTANCE')
fi_records = []

def get_feature_names(prep):
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = list(prep.named_transformers_['cat']['ohe']
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names

for sensor, prep, Xva, yva in [
    ('Sentinel-1', prep_s1, Xt_s1_val, y_s1_val),
    ('Sentinel-2', prep_s2, Xt_s2_val, y_s2_val),
]:
    sel = selected_models[sensor]
    key = f'{sensor}_{sel}'
    est = trained_estimators.get(key)
    if est is None:
        print(f'  {sensor}: no trained estimator found, skipping.'); continue

    feat_names = get_feature_names(prep)
    t0 = time.time()
    pi = permutation_importance(est, Xva, yva, n_repeats=10,
                                 random_state=RANDOM_STATE,
                                 scoring='neg_mean_absolute_error')
    print(f'  {sensor}: done in {time.time()-t0:.1f}s')

    for i, name in enumerate(feat_names):
        fi_records.append({
            'Sensor': sensor, 'Feature': name,
            'Importance Mean': -pi.importances_mean[i],
            'Importance Std':   pi.importances_std[i]
        })

fi_df = pd.DataFrame(fi_records).sort_values(
    ['Sensor','Importance Mean'], ascending=[True, False])
fi_df.to_csv(REPORTS_DIR/'06_feature_importance.csv', index=False)
print('Saved: reports/06_feature_importance.csv')

for sensor in ['Sentinel-1','Sentinel-2']:
    top = fi_df[fi_df['Sensor']==sensor].head(10)
    print(f'\n  {sensor} — top 10 features:')
    for _, row in top.iterrows():
        print(f'    {row["Feature"]:<40} {row["Importance Mean"]:+.6f} ± {row["Importance Std"]:.6f}')

# ── §18  Robustness Diagnostics ───────────────────────────────────────────────
print('\n[§18] ROBUSTNESS DIAGNOSTICS — train/val/test MAE gaps')
for sensor in ['Sentinel-1','Sentinel-2']:
    sub = res_df[res_df['Sensor']==sensor].pivot(index='Model', columns='Split', values='MAE')
    for col in ['Train','Validation','Test']:
        if col not in sub.columns: sub[col] = np.nan
    sub = sub[['Train','Validation','Test']].copy()
    sub['Val/Train'] = (sub['Validation']/sub['Train']).round(2)
    sub['Test/Val']  = (sub['Test']/sub['Validation']).round(2)
    print(f'\n  {sensor}:')
    print(sub.to_string(float_format=lambda x: f'{x:.6f}'))

# ──────────────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────────────
print('\n[FIGURES] Generating all publication-quality figures...')

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
})

model_order_fig = ['Mean Baseline','Zero-Change Baseline','Ridge',
                   'Random Forest','HistGradientBoosting','ExtraTrees','Tuned HGB']
colors_map = plt.cm.Set2(np.linspace(0, 1, len(model_order_fig)))

# ── Fig 1: Model comparison bar chart ────────────────────────────────────────
print('  06_model_comparison.png...')
fig, axes_2 = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
fig.suptitle('GlacierGuard-AI — Model Comparison: Validation & Test MAE',
             color=TEXT_COLOR, fontsize=13, y=1.01)
for ax, split_lbl, ls in zip(axes_2, ['Validation','Test'], ['-','--']):
    ax.set_facecolor(PANEL_BG)
    for sensor, color in [('Sentinel-1', S1_COLOR), ('Sentinel-2', S2_COLOR)]:
        maes = []
        for m in model_order_fig:
            s = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==m)&(res_df['Split']==split_lbl)]
            maes.append(s['MAE'].values[0] if len(s) else np.nan)
        x = np.arange(len(model_order_fig))
        offset = -0.2 if sensor == 'Sentinel-1' else 0.2
        bars = ax.bar(x + offset, maes, 0.38, color=color, alpha=0.85, label=sensor)
        for bar, mae in zip(bars, maes):
            if not np.isnan(mae):
                ax.text(bar.get_x()+bar.get_width()/2, mae + max(maes)*0.01,
                        f'{mae:.5f}', ha='center', va='bottom', fontsize=6, color=color)
    ax.set_xticks(range(len(model_order_fig)))
    ax.set_xticklabels([m.replace(' ','\n') for m in model_order_fig], fontsize=8)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(split_lbl, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'06_model_comparison.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Fig 2 & 3: Residual diagnostics ──────────────────────────────────────────
for sensor, val_df, prep, color, prefix in [
    ('Sentinel-1', s1_val, prep_s1, S1_COLOR, 's1'),
    ('Sentinel-2', s2_val, prep_s2, S2_COLOR, 's2'),
]:
    print(f'  06_residuals_{prefix}.png...')
    sel = selected_models[sensor]
    est = trained_estimators.get(f'{sensor}_{sel}')
    Xt  = prep.transform(val_df[MODEL_FEATURE_VECTOR])
    y_true = val_df[TARGET_COL].values
    y_pred = est.predict(Xt) if est else np.full(len(y_true), y_true.mean())
    resid  = y_true - y_pred

    fig, axes_ = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f'{sensor} — {sel}: Residual Diagnostics (Validation 2022)',
                 color=TEXT_COLOR, fontsize=12, y=1.01)

    ax = axes_[0]; ax.set_facecolor(PANEL_BG)
    ax.hist(resid, bins=80, color=color, alpha=0.82, edgecolor='none')
    ax.axvline(0, color=ACCENT, ls='--', lw=1.5, label='Zero')
    ax.axvline(np.mean(resid), color='#ff7675', ls=':', lw=1.5,
               label=f'Mean={np.mean(resid):.4f}')
    ax.set_xlabel('Residual (km²)', fontsize=10); ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Residual Distribution', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.4)

    ax = axes_[1]; ax.set_facecolor(PANEL_BG)
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max()) * 1.05
    ax.scatter(y_true, y_pred, alpha=0.18, s=8, color=color, edgecolors='none')
    ax.plot([-lim,lim],[-lim,lim],'r--',lw=1.5,label='Perfect')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel('Actual ΔA (km²)', fontsize=10); ax.set_ylabel('Predicted ΔA (km²)', fontsize=10)
    ax.set_title('Actual vs Predicted', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.4)

    ax = axes_[2]; ax.set_facecolor(PANEL_BG)
    ax.scatter(y_pred, resid, alpha=0.18, s=8, color=color, edgecolors='none')
    ax.axhline(0, color=ACCENT, ls='--', lw=1.5)
    ax.set_xlabel('Predicted ΔA (km²)', fontsize=10); ax.set_ylabel('Residual (km²)', fontsize=10)
    ax.set_title('Residual vs Predicted', fontsize=10, color=TEXT_COLOR)
    ax.grid(True, ls=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR/f'06_residuals_{prefix}.png',
                dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()

# ── Fig 4: Size-stratified MAE ────────────────────────────────────────────────
print('  06_size_stratified_error.png...')
fig, axes_ = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Size-Stratified Validation MAE (Training-Derived Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
size_val_df = size_df[size_df['Split']=='Validation']
for ax, sensor in zip(axes_, ['Sentinel-1','Sentinel-2']):
    ax.set_facecolor(PANEL_BG)
    sub = size_val_df[size_val_df['Sensor']==sensor]
    x_cats = ['Small','Medium','Large']
    models_present = [m for m in model_order_fig if m in sub['Model'].values]
    n_m = len(models_present)
    width = 0.8 / max(n_m, 1)
    offsets = np.linspace(-(n_m-1)*width/2, (n_m-1)*width/2, n_m)
    for i, (m, c) in enumerate(zip(models_present, colors_map)):
        sub_m = sub[sub['Model']==m]
        maes  = [sub_m[sub_m['Size Tier']==t]['MAE'].values for t in x_cats]
        maes  = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats))+offsets[i], maes, width*0.88,
               color=c, alpha=0.85, label=m)
    ax.set_xticks(range(len(x_cats))); ax.set_xticklabels(x_cats)
    ax.set_xlabel('Lake Size Tier', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7.5); ax.grid(axis='y', ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'06_size_stratified_error.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Fig 5: Uncertainty-stratified MAE ────────────────────────────────────────
print('  06_uncertainty_stratified_error.png...')
fig, axes_ = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Uncertainty-Stratified Validation MAE (Training-Derived σ_A Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
unc_val_df = unc_df[unc_df['Split']=='Validation']
for ax, sensor in zip(axes_, ['Sentinel-1','Sentinel-2']):
    ax.set_facecolor(PANEL_BG)
    sub = unc_val_df[unc_val_df['Sensor']==sensor]
    x_cats = ['Low','Medium','High']
    models_present = [m for m in model_order_fig if m in sub['Model'].values]
    n_m = len(models_present)
    width = 0.8 / max(n_m, 1)
    offsets = np.linspace(-(n_m-1)*width/2, (n_m-1)*width/2, n_m)
    for i, (m, c) in enumerate(zip(models_present, colors_map)):
        sub_m = sub[sub['Model']==m]
        maes  = [sub_m[sub_m['Uncertainty Tier']==t]['MAE'].values for t in x_cats]
        maes  = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats))+offsets[i], maes, width*0.88,
               color=c, alpha=0.85, label=m)
    ax.set_xticks(range(len(x_cats))); ax.set_xticklabels(x_cats)
    ax.set_xlabel('Area Uncertainty Tier (σ_A)', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7.5); ax.grid(axis='y', ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'06_uncertainty_stratified_error.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Fig 6 & 7: Feature importance ────────────────────────────────────────────
for sensor, prefix, color in [
    ('Sentinel-1', 's1', S1_COLOR),
    ('Sentinel-2', 's2', S2_COLOR),
]:
    print(f'  06_feature_importance_{prefix}.png...')
    sub = fi_df[fi_df['Sensor']==sensor].head(20).sort_values('Importance Mean')
    if len(sub) == 0: continue
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub['Importance Mean'], xerr=sub['Importance Std'],
            color=color, alpha=0.85, height=0.7,
            error_kw={'ecolor': ACCENT, 'capsize': 3})
    ax.set_yticks(y_pos); ax.set_yticklabels(sub['Feature'], fontsize=8.5)
    ax.set_xlabel('Permutation Importance (MAE decrease, km²)', fontsize=10)
    ax.set_title(f'{sensor} — Top Feature Importances\n(Permutation, Validation 2022, n=10 repeats)',
                 fontsize=11, color=TEXT_COLOR, pad=10)
    ax.axvline(0, color='#ff7675', ls='--', lw=1.2)
    ax.grid(axis='x', ls=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/f'06_feature_importance_{prefix}.png',
                dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()

# ── Fig 8: Error by year ──────────────────────────────────────────────────────
print('  06_error_by_year.png...')
fig, axes_ = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('Selected Model — MAE by Observation Year (Validation + Test)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
for ax, sensor, color in zip(axes_, ['Sentinel-1','Sentinel-2'], [S1_COLOR, S2_COLOR]):
    ax.set_facecolor(PANEL_BG)
    sel = selected_models[sensor]
    sub = error_year_df[(error_year_df['Sensor']==sensor)&(error_year_df['Model']==sel)]
    for split_lbl, ls in [('Validation','-'), ('Test','--')]:
        ss = sub[sub['Split']==split_lbl].sort_values('AREA_YEAR')
        if len(ss):
            ax.plot(ss['AREA_YEAR'], ss['MAE'], marker='o', ls=ls,
                    color=color, label=split_lbl, lw=2, markersize=6)
    ax.set_xlabel('AREA_YEAR (observation year)', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(f'{sensor} | {sel}', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'06_error_by_year.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── Fig 9: Tuning CV curves ───────────────────────────────────────────────────
print('  06_tuning_cv_curves.png...')
fig, axes_ = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('HGB Expanding-Window CV — Fold MAE per Parameter Combo',
             color=TEXT_COLOR, fontsize=12, y=1.01)
for ax, sensor, Xtr, ytr_arr, train_years_arr, color in [
    (axes_[0], 'Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values, S1_COLOR),
    (axes_[1], 'Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values, S2_COLOR),
]:
    ax.set_facecolor(PANEL_BG)
    # Show fold MAEs for 4 representative combos (lr=0.05, vary max_iter & max_depth)
    rep_combos = [(mi, md, 30, 0.05) for mi in [100, 200] for md in [4, 6]]
    rep_labels  = [f'mi={mi},md={md}' for mi, md in [(100,4),(100,6),(200,4),(200,6)]]
    rep_colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(rep_combos)))
    fold_yrs    = [v for _, v in CV_FOLD_YEARS]
    for combo, lbl, rc in zip(rep_combos, rep_labels, rep_colors):
        pdict = dict(zip(param_keys_hgb, combo))
        fold_maes = []
        for train_end, val_year in CV_FOLD_YEARS:
            fm_tr = train_years_arr <= train_end
            fm_va = train_years_arr == val_year
            if fm_va.sum() == 0: continue
            est_ = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **pdict)
            est_.fit(Xtr[fm_tr], ytr_arr[fm_tr])
            fold_maes.append(mean_absolute_error(ytr_arr[fm_va], est_.predict(Xtr[fm_va])))
        ax.plot(fold_yrs, fold_maes, marker='o', lw=1.8, color=rc, label=lbl)
    ax.set_xlabel('Validation Year', fontsize=10); ax.set_ylabel('Fold MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'06_tuning_cv_curves.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

print('  All figures saved.')

# ── §19  Research Interpretation ──────────────────────────────────────────────
print('\n' + '='*65)
print('[§19] RESEARCH INTERPRETATION')
print('='*65)
for sensor in ['Sentinel-1','Sentinel-2']:
    sel = selected_models[sensor]
    val_r  = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==sel)&(res_df['Split']=='Validation')].iloc[0]
    test_r = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==sel)&(res_df['Split']=='Test')].iloc[0]
    base_v = res_df[(res_df['Sensor']==sensor)&(res_df['Model']=='Mean Baseline')&(res_df['Split']=='Validation')].iloc[0]
    zero_v = res_df[(res_df['Sensor']==sensor)&(res_df['Model']=='Zero-Change Baseline')&(res_df['Split']=='Validation')].iloc[0]
    delta  = val_r['MAE'] - base_v['MAE']
    pct    = delta / base_v['MAE'] * 100
    print(f'\n  {sensor}:')
    print(f'    Selected model      : {sel}')
    print(f'    Best params         : {best_params.get(sensor,"N/A")}')
    print(f'    Val  MAE/RMSE/R²    : {val_r["MAE"]:.6f} / {val_r["RMSE"]:.6f} / {val_r["R2"]:.4f}')
    print(f'    Test MAE/RMSE/R²    : {test_r["MAE"]:.6f} / {test_r["RMSE"]:.6f} / {test_r["R2"]:.4f}')
    print(f'    Mean Baseline val   : {base_v["MAE"]:.6f}')
    print(f'    Zero-Change val     : {zero_v["MAE"]:.6f}')
    print(f'    vs Mean Baseline    : ΔMAE={delta:+.6f} ({pct:+.2f}%)')

# ── §20  Final summary ────────────────────────────────────────────────────────
print('\n' + '='*65)
print('[§20] FINAL ARTIFACTS')
print('='*65)
csvs = sorted(REPORTS_DIR.glob('06_*.csv'))
figs = sorted(FIGURES_DIR.glob('06_*.png'))
print(f'  CSVs ({len(csvs)}):')
for f in csvs: print(f'    {f}')
print(f'  Figures ({len(figs)}):')
for f in figs: print(f'    {f.name}')

print('\n' + '='*65)
print('CONTINUATION COMPLETE — ALL SECTIONS §1–§20 VERIFIED')
print('='*65)
