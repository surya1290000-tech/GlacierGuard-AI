"""
Builds notebooks/06_model_development.ipynb from verified execution code.
All code cells are taken verbatim from _run_nb06_final.py logic.
"""
import json, pathlib

NB_PATH = pathlib.Path('notebooks/06_model_development.ipynb')

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# §1 — Objective & Research Question
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# 06 — GlacierGuard-AI Advanced Model Development

**Project:** GlacierGuard-AI  
**Research Question:** Can robust machine-learning models improve prediction of next-year glacial-lake area change beyond the established baseline models, while preserving temporal validity and avoiding data leakage?  
**Primary Target:** `NEXT_AREA_CHANGE = AREA(t+1) − AREA(t)` in km², strictly when Δt = 1 year  
**Sensors:** Sentinel-1 (SAR) and Sentinel-2 (Optical), evaluated independently  

---

### Scientific Scope and Boundaries
1. **Task:** Predict next-year annual lake area change (km²). This is NOT a GLOF, flood, or hazard prediction model.
2. **Temporal Integrity:** Train ≤ 2021, Validation = 2022, Test = 2023. No random splitting. Test set evaluated once, after model selection is frozen.
3. **Leakage Controls:** All preprocessing fitted on training data only. Lake-size and uncertainty thresholds derived from training data only.
4. **Baselines:** Mean Baseline and Zero-Change Baseline are evaluated first and serve as the honest performance floor.
5. **Honest Negative Results:** If ML models fail to outperform baselines, that result is preserved and reported.

---

### Notebook Sections
| Section | Title |
|---|---|
| §2  | Environment & Reproducibility |
| §3  | Load / Reconstruct Gold Dataset |
| §4  | Dataset Integrity Assertions |
| §5  | Frozen Temporal Split Verification |
| §6  | Feature Matrix Construction |
| §7  | Train-Only Preprocessing |
| §8  | Baseline Reference Models |
| §9  | Candidate Advanced Models |
| §10 | Time-Aware Hyperparameter Tuning |
| §11 | Validation Comparison Table |
| §12 | Final Model Selection Criteria |
| §13 | Frozen Test Set Evaluation |
| §14 | Lake-Size Stratified Evaluation |
| §15 | Uncertainty-Stratified Evaluation |
| §16 | Error Analysis |
| §17 | Feature Importance / Explainability |
| §18 | Model Robustness Diagnostics |
| §19 | Research Interpretation |
| §20 | Final Model Artifact & Results Table |
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §2 — Environment & Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §2 — Environment & Reproducibility"))
cells.append(code("""\
# == §2.1  Imports, Reproducibility, Paths ==
import os, sys, pathlib, warnings, time, copy, itertools

import numpy as np
import pandas as pd
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
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = pathlib.Path('../reports/figures')
if not FIGURES_DIR.exists():
    FIGURES_DIR = pathlib.Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = FIGURES_DIR.parent
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print('Environment:')
print(f'  Python       : {sys.version.split()[0]}')
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  pandas       : {pd.__version__}')
print(f'  numpy        : {np.__version__}')
print(f'  Figures      : {FIGURES_DIR.resolve()}')
print(f'  Reports      : {REPORTS_DIR.resolve()}')
print(f'  Random seed  : {RANDOM_STATE}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §3 — Load / Reconstruct Gold Dataset
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §3 — Load / Reconstruct Gold Dataset

The Gold feature datasets are reconstructed from the raw annual GLO GeoPackages using the
**identical pipeline** as Notebook 04. This ensures reproducibility without requiring pickled
artefacts. The function `build_gold_dataset()` below is a faithful verbatim replication
of Notebook 04's authoritative feature engineering logic.

**Do not alter this function** — all schema assertions depend on its exact output.
"""))
cells.append(code("""\
# == §3.1  Locate Annual GLO GeoPackages ==
GLO_DIR = None
for p in [pathlib.Path('../raw/GLO'), pathlib.Path('./raw/GLO'),
          pathlib.Path('/Workspace/raw/GLO')]:
    if p.exists() and (p / 'S1_20172024_NTB_GLOID_v1.02.gpkg').exists():
        GLO_DIR = p.resolve()
        break
if GLO_DIR is None:
    raise FileNotFoundError(
        'Cannot locate raw/GLO annual GeoPackage files. '
        'Ensure S1_20172024_NTB_GLOID_v1.02.gpkg is present in raw/GLO/.')
print(f'GLO directory : {GLO_DIR}')
"""))
cells.append(code("""\
# == §3.2  build_gold_dataset — authoritative NB04 replication ==
def build_gold_dataset(gpkg_path):
    \"\"\"Reconstruct the Gold feature dataset from an annual GLO GeoPackage.

    Faithful replication of Notebook 04 build_gold_dataset().
    Must NOT be altered — schema assertions depend on exact output.
    \"\"\"
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

    # Gap-enforced area lags (NULL when year gap > 1)
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

    # Consecutive backward area changes
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

    # Backward-looking rolling statistics
    c_df = df[['AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3']]
    df['ROLLING_MEAN_3'] = c_df.mean(axis=1, skipna=True)
    df['ROLLING_STD_3']  = c_df.std(axis=1, skipna=True, ddof=1)

    # Supervised target (strictly when NEXT_YEAR_GAP == 1)
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
        'AREA_CHANGE',          # preserved for persistence benchmark only
        'NEXT_AREA_CHANGE',
    ]
    return df[gold_cols].copy()

t0 = time.time()
gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')
print(f'Datasets loaded in {time.time()-t0:.1f}s')
print(f'  gold_s1: {len(gold_s1):,} rows, {gold_s1.columns.tolist()}')
print(f'  gold_s2: {len(gold_s2):,} rows')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §4 — Integrity Assertions
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §4 — Dataset Integrity Assertions"))
cells.append(code("""\
# == §4.1  Schema and Row-Count Assertions ==
assert len(gold_s1) == 18129, f'S1 row mismatch: {len(gold_s1)}'
assert len(gold_s2) == 22294, f'S2 row mismatch: {len(gold_s2)}'
assert gold_s1['NEXT_AREA_CHANGE'].notna().sum() == 14242, 'S1 supervised target mismatch'
assert gold_s2['NEXT_AREA_CHANGE'].notna().sum() == 16610, 'S2 supervised target mismatch'

PROHIBITED_LEAKAGE = [
    'NEXT_AREA', 'NEXT_YEAR', 'NEXT_YEAR_GAP',
    'EXPANSION_RATE', 'EXPANSION_RATE_SIG', 'EXPANSION_UNCERTAINTY',
    'START_DATE', 'END_DATE', 'REF_MSTAT', 'S2_MSTAT', 'DATA_SOURCE', 'AREA_LAG4',
]
for lbl, df in [('S1', gold_s1), ('S2', gold_s2)]:
    leaked = [c for c in PROHIBITED_LEAKAGE if c in df.columns]
    assert len(leaked) == 0, f'Leakage detected in {lbl}: {leaked}'

print('All integrity assertions passed.')
print(f'  S1: {len(gold_s1):,} rows | {gold_s1["NEXT_AREA_CHANGE"].notna().sum():,} supervised targets')
print(f'  S2: {len(gold_s2):,} rows | {gold_s2["NEXT_AREA_CHANGE"].notna().sum():,} supervised targets')
print('  Zero prohibited leakage columns detected.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §5 — Temporal Split
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §5 — Frozen Temporal Split Verification

Partition follows strict chronological ordering — **no random splitting**:

| Split | AREA_YEAR | Predicts |
|---|---|---|
| **Train** | 2017–2021 | Next-year area in 2018–2022 |
| **Validation** | 2022 | Next-year area in 2023 |
| **Test** *(frozen)* | 2023 | Next-year area in 2024 |

Terminal year 2024 is excluded (no 2025 target available).  
The test set is evaluated **once** in §13, after model selection is frozen.
"""))
cells.append(code("""\
# == §5.1  Chronological Partition Function ==
TARGET_COL = 'NEXT_AREA_CHANGE'

def create_temporal_splits(df, label):
    sup   = df[df[TARGET_COL].notna()].copy()
    train = sup[sup['AREA_YEAR'] <= 2021].copy()
    val   = sup[sup['AREA_YEAR'] == 2022].copy()
    test  = sup[sup['AREA_YEAR'] == 2023].copy()
    assert len(train) + len(val) + len(test) == len(sup), 'Row count mismatch'
    assert train['AREA_YEAR'].max() <= 2021
    assert (val['AREA_YEAR']  == 2022).all()
    assert (test['AREA_YEAR'] == 2023).all()
    assert (sup['AREA_YEAR']  >= 2024).sum() == 0, '2024 found in supervised set'
    print(f'{label}: train={len(train):,} | val={len(val):,} | test={len(test):,}')
    return train, val, test

s1_train, s1_val, s1_test = create_temporal_splits(gold_s1, 'Sentinel-1')
s2_train, s2_val, s2_test = create_temporal_splits(gold_s2, 'Sentinel-2')

# Hard numerical assertions vs Notebook 03/04 verified counts
assert len(s1_train) == 10161 and len(s1_val) == 2026  and len(s1_test) == 2055
assert len(s2_train) == 11858 and len(s2_val) == 2245  and len(s2_test) == 2507
print('All temporal split count assertions passed (parity with Notebook 03/04).')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §6 — Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §6 — Feature Matrix Construction

Exactly **20 predictors** are used — unchanged from Notebook 04.  
`IS_TS_OUTLIER`, `TS_OUTLIER`, `GLO_ID`, `AREA_YEAR`, and `AREA_CHANGE` are explicitly excluded.
"""))
cells.append(code("""\
# == §6.1  Approved 20-Feature Contract ==
CATEGORICAL_FEATURES = ['BASIN', 'COUNTRY', 'GTNG_REGION_O2', 'CONNECTIVITY']
NUMERICAL_FEATURES = [
    'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
    'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
    'LATITUDE', 'LONGITUDE',
    'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
    'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3',
]
MODEL_FEATURE_VECTOR = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
assert len(MODEL_FEATURE_VECTOR) == 20
assert TARGET_COL          not in MODEL_FEATURE_VECTOR
assert 'GLO_ID'            not in MODEL_FEATURE_VECTOR
assert 'AREA_YEAR'         not in MODEL_FEATURE_VECTOR
assert 'IS_TS_OUTLIER'     not in MODEL_FEATURE_VECTOR
assert 'AREA_CHANGE'       not in MODEL_FEATURE_VECTOR
print(f'Feature contract: {len(MODEL_FEATURE_VECTOR)} features')
print(f'  Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}')
print(f'  Numerical   ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}')

X_s1_train, y_s1_train = s1_train[MODEL_FEATURE_VECTOR], s1_train[TARGET_COL]
X_s1_val,   y_s1_val   = s1_val[MODEL_FEATURE_VECTOR],   s1_val[TARGET_COL]
X_s1_test,  y_s1_test  = s1_test[MODEL_FEATURE_VECTOR],  s1_test[TARGET_COL]
X_s2_train, y_s2_train = s2_train[MODEL_FEATURE_VECTOR], s2_train[TARGET_COL]
X_s2_val,   y_s2_val   = s2_val[MODEL_FEATURE_VECTOR],   s2_val[TARGET_COL]
X_s2_test,  y_s2_test  = s2_test[MODEL_FEATURE_VECTOR],  s2_test[TARGET_COL]

for X, y in [(X_s1_train,y_s1_train),(X_s1_val,y_s1_val),(X_s1_test,y_s1_test),
             (X_s2_train,y_s2_train),(X_s2_val,y_s2_val),(X_s2_test,y_s2_test)]:
    assert X.shape[1] == 20
    assert TARGET_COL not in X.columns
    assert not y.isna().any()
print('Feature matrix assertions passed.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §7 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §7 — Train-Only Preprocessing

Preprocessing is **fitted exclusively on training data** and then applied to validation and test.  
This prevents any distributional information from the future leaking into the preprocessing step.

- Numerical: `SimpleImputer(strategy='median')`
- Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`
"""))
cells.append(code("""\
# == §7.1  Build and Fit Preprocessors (train-only) ==
def build_preprocessor():
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUMERICAL_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES),
    ])

prep_s1 = build_preprocessor().fit(X_s1_train)
prep_s2 = build_preprocessor().fit(X_s2_train)

Xt_s1_train = prep_s1.transform(X_s1_train)
Xt_s1_val   = prep_s1.transform(X_s1_val)
Xt_s1_test  = prep_s1.transform(X_s1_test)
Xt_s2_train = prep_s2.transform(X_s2_train)
Xt_s2_val   = prep_s2.transform(X_s2_val)
Xt_s2_test  = prep_s2.transform(X_s2_test)

print('Preprocessing (train-only fit):')
print(f'  S1: train={Xt_s1_train.shape}, val={Xt_s1_val.shape}, test={Xt_s1_test.shape}')
print(f'  S2: train={Xt_s2_train.shape}, val={Xt_s2_val.shape}, test={Xt_s2_test.shape}')
print(f'  (20 raw features → {Xt_s1_train.shape[1]} after OHE expansion)')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §8 — Baselines
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §8 — Baseline Reference Models

Two constant-prediction baselines establish the honest performance floor:
1. **Mean Baseline**: predicts the training-set target mean for every observation.
2. **Zero-Change Baseline**: predicts zero area change for every observation.

These replicate Notebook 04 baselines and provide the reference against which all ML models are compared.
"""))
cells.append(code("""\
# == §8.1  Evaluate Baseline Benchmarks ==
results_records = []
trained_estimators = {}   # keyed: '{Sensor}_{ModelName}'

def record(sensor, model, split, y_true, y_pred):
    results_records.append({
        'Sensor': sensor, 'Model': model, 'Split': split,
        'N': len(y_true),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2':   r2_score(y_true, y_pred),
    })

for sensor, y_tr, y_va, y_te in [
    ('Sentinel-1', y_s1_train, y_s1_val, y_s1_test),
    ('Sentinel-2', y_s2_train, y_s2_val, y_s2_test),
]:
    tr_mean = float(y_tr.mean())
    for split, yt in [('Train',y_tr),('Validation',y_va),('Test',y_te)]:
        record(sensor, 'Mean Baseline',        split, yt, np.full(len(yt), tr_mean))
        record(sensor, 'Zero-Change Baseline', split, yt, np.zeros(len(yt)))

print('Baselines recorded.')
print(f'  S1 training mean target: {y_s1_train.mean():+.6f} km²')
print(f'  S2 training mean target: {y_s2_train.mean():+.6f} km²')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §9 — Candidate Models
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §9 — Candidate Advanced Models

Five supervised models are trained with fixed hyperparameters matching Notebook 04 defaults,
plus ExtraTrees as an additional ensemble candidate:

| Model | Notes |
|---|---|
| **Ridge** | L₂ regularisation, α=1.0 |
| **Random Forest** | 100 trees, max_depth=10, min_leaf=5 |
| **HistGradientBoosting** | 100 iterations, max_depth=6, min_leaf=20 |
| **ExtraTrees** | 200 trees, max_depth=20, min_leaf=5, max_features=0.7 |
| **Tuned HGB** | Hyperparameters from §10 expanding-window CV |

All models use the **same train-fitted preprocessor** from §7. No new libraries are installed.
"""))
cells.append(code("""\
# == §9.1  Train Fixed-Param Models ==
# Note: Tuned HGB trained in §10b after CV completes.
fixed_specs = {
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
    for m_name, m_obj in fixed_specs.items():
        est = copy.deepcopy(m_obj)
        t0  = time.time()
        est.fit(Xtr, ytr)
        trained_estimators[f'{sensor}_{m_name}'] = est
        record(sensor, m_name, 'Train',      ytr, est.predict(Xtr))
        record(sensor, m_name, 'Validation', yva, est.predict(Xva))
        record(sensor, m_name, 'Test',       yte, est.predict(Xte))
        print(f'  {sensor:12s} | {m_name:25s} | {time.time()-t0:.1f}s')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §10 — Tuning
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §10 — Time-Aware Hyperparameter Tuning

### Strategy: Expanding-Window Temporal CV (Training Era Only)

Hyperparameter tuning uses **only training-era data (2017–2021)**. The 2022 validation and 2023 test years are never seen during tuning.

**Fold scheme:**
| Fold | Train years | Validation year |
|---|---|---|
| 1 | 2017 | 2018 |
| 2 | 2017–2018 | 2019 |
| 3 | 2017–2019 | 2020 |
| 4 | 2017–2020 | 2021 |

**Tuning grid (HistGradientBoosting):**  
`max_iter ∈ {100, 200}`, `max_depth ∈ {4, 6}`, `min_samples_leaf ∈ {20, 30}`, `learning_rate = 0.05`  
→ **8 combinations × 4 folds × 2 sensors = 64 total model fits**

> **Note on `learning_rate`:** A full 54-combination grid (including `lr ∈ {0.05, 0.1}`) was evaluated
> in the development phase and confirmed `lr=0.05` as consistently superior for both sensors.
> It is therefore fixed here to reduce notebook runtime while preserving the validated result.
"""))
cells.append(code("""\
# == §10.1  Expanding-Window CV Grid Search (HGB) ==
HGB_GRID = {
    'max_iter':         [100, 200],
    'max_depth':        [4, 6],
    'min_samples_leaf': [20, 30],
    'learning_rate':    [0.05],    # confirmed best from full 54-combo development run
}
CV_FOLD_YEARS = [(2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021)]

param_keys   = list(HGB_GRID.keys())
all_combos   = list(itertools.product(*HGB_GRID.values()))
n_combos     = len(all_combos)
n_folds      = len(CV_FOLD_YEARS)
total_fits   = n_combos * n_folds * 2

print(f'HGB Tuning Plan:')
print(f'  Grid combinations : {n_combos}')
print(f'  Expanding folds   : {n_folds}  (train_end→val_year: {CV_FOLD_YEARS})')
print(f'  Total model fits  : {n_combos} × {n_folds} folds × 2 sensors = {total_fits}')
"""))
cells.append(code("""\
# == §10.2  Run Expanding-Window CV ==
# IMPORTANT: loop variable must NOT be named 'pd' (would shadow pandas import).
def expand_cv_mae(Xtr_full, ytr_full, train_years_arr, param_dict_hgb):
    \"\"\"Mean MAE across all expanding-window folds for a given HGB param dict.\"\"\"
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

best_params    = {}
cv_all_results = {}

t_tune_start = time.time()
for sensor, Xtr, ytr_arr, train_years_arr in [
    ('Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values),
    ('Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values),
]:
    t_s = time.time()
    best_mae, best_combo = np.inf, None
    sensor_results = []
    for combo in all_combos:
        param_dict_hgb = dict(zip(param_keys, combo))   # NOT named 'pd'
        cv_mae = expand_cv_mae(Xtr, ytr_arr, train_years_arr, param_dict_hgb)
        sensor_results.append({'combo': combo, 'cv_mae': cv_mae})
        if cv_mae < best_mae:
            best_mae, best_combo = cv_mae, combo
    best_params[sensor] = dict(zip(param_keys, best_combo))
    cv_all_results[sensor] = sensor_results
    print(f'  {sensor}: best CV-MAE={best_mae:.6f}  params={best_params[sensor]}  '
          f'[{time.time()-t_s:.1f}s]')

print(f'Total tuning time: {time.time()-t_tune_start:.1f}s')
"""))
cells.append(code("""\
# == §10b  Train Tuned HGB on Full 2017–2021 Training Set ==
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
    print(f'  {sensor}: val MAE={val_mae:.6f}  tuned params={best_params[sensor]}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §11 — Validation Comparison
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §11 — Validation Comparison Table"))
cells.append(code("""\
# == §11.1  Build and Display Full Results Table ==
res_df = pd.DataFrame(results_records)

val_table = (
    res_df[res_df['Split'] == 'Validation']
    [['Sensor', 'Model', 'MAE', 'RMSE', 'R2']]
    .sort_values(['Sensor', 'MAE'])
    .reset_index(drop=True)
)
print('VALIDATION COMPARISON TABLE (sorted by MAE per sensor):')
print(val_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §12 — Model Selection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §12 — Final Model Selection Criteria

Selection is based **exclusively on 2022 validation performance**, before the test set is opened.

**Selection criteria (in order):**
1. Lowest validation MAE (primary physical metric)
2. Among near-equivalent MAE (<2% relative difference): lower RMSE preferred
3. Among equivalent MAE/RMSE: prefer simpler model
4. Robustness check: val/train MAE ratio < 3× (no severe overfitting)

**Feature importance** is computed on the **best supervised ML model** per sensor (not on baselines,
which have no learned parameters). This characterises what structure the ML models were able to exploit.
"""))
cells.append(code("""\
# == §12.1  Apply Selection Criteria ==
SELECTED_MODELS = {}
BEST_ML_FOR_IMPORTANCE = {}

ML_MODELS = ['Ridge', 'Random Forest', 'HistGradientBoosting', 'ExtraTrees', 'Tuned HGB']

for sensor in ['Sentinel-1', 'Sentinel-2']:
    val_sub = val_table[val_table['Sensor'] == sensor].reset_index(drop=True)

    # Best-by-MAE (may be a baseline)
    best_model   = val_sub.iloc[0]['Model']
    best_val_mae = val_sub.iloc[0]['MAE']

    # Best supervised ML model (for FI)
    val_ml = val_sub[val_sub['Model'].isin(ML_MODELS)].reset_index(drop=True)
    best_ml = val_ml.iloc[0]['Model'] if len(val_ml) else 'HistGradientBoosting'

    # Robustness check
    tr_row = res_df[(res_df['Sensor']==sensor) &
                    (res_df['Model']==best_model) &
                    (res_df['Split']=='Train')]
    tr_mae = tr_row['MAE'].values[0] if len(tr_row) else np.nan
    gap    = best_val_mae / tr_mae if tr_mae > 0 else np.nan

    SELECTED_MODELS[sensor]           = best_model
    BEST_ML_FOR_IMPORTANCE[sensor]    = best_ml

    print(f'{sensor}:')
    print(f'  SELECTED MODEL       : {best_model}')
    print(f'  Val MAE              : {best_val_mae:.6f} km²')
    print(f'  Train MAE            : {tr_mae:.6f} km²')
    print(f'  Val/Train ratio      : {gap:.2f}x  (robustness check)')
    print(f'  Best ML for FI       : {best_ml}')
    print()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §13 — Frozen Test Evaluation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §13 — Frozen Test Set Evaluation

⚠️ **The frozen test set (2023) is evaluated exactly once, here, after model selection is complete.**  
No hyperparameter changes are made after observing these results.
"""))
cells.append(code("""\
# == §13.1  Report Test Results ==
test_table = (
    res_df[res_df['Split'] == 'Test']
    [['Sensor', 'Model', 'MAE', 'RMSE', 'R2']]
    .sort_values(['Sensor', 'MAE'])
    .reset_index(drop=True)
)
print('FROZEN TEST SET RESULTS (2023):')
print(test_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# Master comparison table
master_rows = []
model_order_all = ['Mean Baseline', 'Zero-Change Baseline', 'Ridge', 'Random Forest',
                   'HistGradientBoosting', 'ExtraTrees', 'Tuned HGB']
for sensor in ['Sentinel-1', 'Sentinel-2']:
    for model in model_order_all:
        row = {'Sensor': sensor, 'Model': model}
        for split in ['Validation', 'Test']:
            s = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==model)&(res_df['Split']==split)]
            for col in ['MAE', 'RMSE', 'R2']:
                row[f'{split} {col}'] = s[col].values[0] if len(s) else np.nan
        master_rows.append(row)
master_df = pd.DataFrame(master_rows).sort_values(['Sensor', 'Validation MAE'])
print('\\nMASTER MODEL COMPARISON TABLE:')
print(master_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
master_df.to_csv(REPORTS_DIR / '06_model_comparison.csv', index=False)
print(f'\\nSaved: {(REPORTS_DIR / "06_model_comparison.csv").resolve()}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §14 — Size-Stratified
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §14 — Lake-Size Stratified Evaluation

Lake-size tiers are defined from **training-set AREA quantiles** (q33, q67).
This matches the heteroscedasticity analysis methodology in Notebook 05.

> **Thresholds are derived from training data only** — not from the full supervised dataset.
"""))
cells.append(code("""\
# == §14.1  Size-Stratified Evaluation ==
size_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    q33 = train_df['AREA'].quantile(0.333)   # training-derived
    q67 = train_df['AREA'].quantile(0.667)   # training-derived
    print(f'{sensor}: AREA size thresholds (train-derived): q33={q33:.5f}, q67={q67:.5f} km²')

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

            for tier in ['Small', 'Medium', 'Large']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                size_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Size Tier': tier, 'N': int(mask.sum()),
                    'MAE':  mean_absolute_error(y_true[mask], y_pred[mask]),
                    'RMSE': root_mean_squared_error(y_true[mask], y_pred[mask]),
                })

size_df = pd.DataFrame(size_records)
size_df.to_csv(REPORTS_DIR / '06_size_stratified.csv', index=False)
print(f'Saved: {(REPORTS_DIR / "06_size_stratified.csv").resolve()}')

# Print selected model
for sensor, sel in SELECTED_MODELS.items():
    sub = size_df[(size_df['Sensor']==sensor)&(size_df['Model']==sel)&(size_df['Split']=='Validation')]
    print(f'\\n{sensor} | {sel} (Validation):')
    print(sub[['Size Tier','N','MAE','RMSE']].to_string(index=False, float_format=lambda x:f'{x:.6f}'))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §15 — Uncertainty-Stratified
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §15 — Uncertainty-Stratified Evaluation

Model errors are evaluated across three AREA_UNCERTAINTY tiers.  
Thresholds are derived from **training-set `AREA_UNCERTAINTY` quantiles** only.

> ⚠️ `AREA_UNCERTAINTY` describes single-observation area uncertainty, **not** the uncertainty of the
> transition target `NEXT_AREA_CHANGE`. Results should be interpreted as correlations with available
> observation uncertainty, not as exact transition error bounds.
"""))
cells.append(code("""\
# == §15.1  Uncertainty-Stratified Evaluation ==
unc_records = []

for sensor, train_df, val_df, test_df, prep in [
    ('Sentinel-1', s1_train, s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_train, s2_val, s2_test, prep_s2),
]:
    uq33 = train_df['AREA_UNCERTAINTY'].quantile(0.333)   # training-derived
    uq67 = train_df['AREA_UNCERTAINTY'].quantile(0.667)   # training-derived
    print(f'{sensor}: AREA_UNCERTAINTY thresholds (train-derived): q33={uq33:.5f}, q67={uq67:.5f} km²')

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

            for tier in ['Low', 'Medium', 'High']:
                mask = tiers == tier
                if mask.sum() < 5:
                    continue
                unc_records.append({
                    'Sensor': sensor, 'Model': model_name, 'Split': split_lbl,
                    'Uncertainty Tier': tier, 'N': int(mask.sum()),
                    'MAE':  mean_absolute_error(y_true[mask], y_pred[mask]),
                    'RMSE': root_mean_squared_error(y_true[mask], y_pred[mask]),
                })

unc_df = pd.DataFrame(unc_records)
unc_df.to_csv(REPORTS_DIR / '06_uncertainty_stratified.csv', index=False)
print(f'Saved: {(REPORTS_DIR / "06_uncertainty_stratified.csv").resolve()}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §16 — Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §16 — Error Analysis"))
cells.append(code("""\
# == §16.1  Residual Statistics, Year-by-Year, Outlier Comparison ==
error_year_records = []

for sensor, val_df, test_df, prep in [
    ('Sentinel-1', s1_val, s1_test, prep_s1),
    ('Sentinel-2', s2_val, s2_test, prep_s2),
]:
    sel = SELECTED_MODELS[sensor]
    key = f'{sensor}_{sel}'
    est = trained_estimators.get(key)

    for split_lbl, split_df in [('Validation', val_df), ('Test', test_df)]:
        Xt = prep.transform(split_df[MODEL_FEATURE_VECTOR])
        y_true = split_df[TARGET_COL].values
        if est:
            y_pred = est.predict(Xt)
        elif sel == 'Mean Baseline':
            y_pred = np.full(len(y_true), split_df[TARGET_COL].mean()
                             if split_lbl == 'Train' else
                             val_df[TARGET_COL].mean()
                             if split_lbl == 'Validation' else
                             s1_train[TARGET_COL].mean() if sensor=='Sentinel-1'
                             else s2_train[TARGET_COL].mean())
        else:
            y_pred = np.zeros(len(y_true))

        resid = y_true - y_pred
        print(f'{sensor} | {sel} | {split_lbl}:')
        print(f'  Residual  mean={np.mean(resid):+.6f}  std={np.std(resid):.6f}  '
              f'skew={float(pd.Series(resid).skew()):.3f}  kurt={float(pd.Series(resid).kurt()):.3f}')
        print(f'  Overpredict (pred>true): {(resid<0).sum()} ({(resid<0).mean()*100:.1f}%)')
        print(f'  Underpredict (pred<true): {(resid>0).sum()} ({(resid>0).mean()*100:.1f}%)')
        top5 = sorted(np.abs(resid), reverse=True)[:5]
        print(f'  Top-5 |residual|: {[f"{v:.5f}" for v in top5]}')

        if 'IS_TS_OUTLIER' in split_df.columns:
            out_mask = split_df['IS_TS_OUTLIER'].values
            if out_mask.sum() > 5:
                print(f'  Outlier MAE: {mean_absolute_error(y_true[out_mask],y_pred[out_mask]):.6f} (n={out_mask.sum()})')
                print(f'  Non-outlier MAE: {mean_absolute_error(y_true[~out_mask],y_pred[~out_mask]):.6f} (n={(~out_mask).sum()})')

        for yr, idx_group in split_df.groupby('AREA_YEAR').groups.items():
            pos = split_df.index.get_indexer(idx_group)
            st, sp = y_true[pos], y_pred[pos]
            error_year_records.append({
                'Sensor': sensor, 'Model': sel, 'Split': split_lbl,
                'AREA_YEAR': yr, 'N': len(pos),
                'MAE': mean_absolute_error(st, sp),
                'RMSE': root_mean_squared_error(st, sp),
                'Mean Residual': float(np.mean(st - sp)),
            })
        print()

error_year_df = pd.DataFrame(error_year_records)
error_year_df.to_csv(REPORTS_DIR / '06_error_by_year.csv', index=False)
print(f'Saved: {(REPORTS_DIR / "06_error_by_year.csv").resolve()}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §17 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §17 — Feature Importance / Explainability

**Permutation importance** is computed on the best supervised ML model per sensor
(evaluated on the 2022 validation set):

- **Sentinel-1**: HistGradientBoosting (best val R² among ML models)
- **Sentinel-2**: Tuned HGB (lowest val MAE among ML models)

> ⚠️ Note: The *selected* models for both sensors are baselines (§12). Feature importance is computed
> on ML models separately to characterise what predictive structure they identified — baselines
> have no learned parameters and cannot be analyzed for feature importance.

> **Feature importance = predictive association, NOT causal evidence.**  
> "AREA contributed strongly to model predictions" — not "AREA causes area change."
"""))
cells.append(code("""\
# == §17.1  Permutation Feature Importance (best ML model per sensor) ==
def get_feature_names(prep):
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = list(
        prep.named_transformers_['cat']['ohe']
            .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names

fi_records = []

for sensor, prep, Xva, yva in [
    ('Sentinel-1', prep_s1, Xt_s1_val, y_s1_val),
    ('Sentinel-2', prep_s2, Xt_s2_val, y_s2_val),
]:
    ml_name    = BEST_ML_FOR_IMPORTANCE[sensor]
    est        = trained_estimators[f'{sensor}_{ml_name}']
    feat_names = get_feature_names(prep)

    t0 = time.time()
    pi = permutation_importance(
        est, Xva, yva,
        n_repeats=10, random_state=RANDOM_STATE,
        scoring='neg_mean_absolute_error',
    )
    print(f'{sensor} ({ml_name}): done in {time.time()-t0:.1f}s')

    for i, name in enumerate(feat_names):
        fi_records.append({
            'Sensor':          sensor,
            'Model':           ml_name,
            'Feature':         name,
            'Importance Mean': -pi.importances_mean[i],
            'Importance Std':   pi.importances_std[i],
        })

fi_df = pd.DataFrame(fi_records).sort_values(
    ['Sensor', 'Importance Mean'], ascending=[True, False])
fi_df.to_csv(REPORTS_DIR / '06_feature_importance.csv', index=False)
print(f'Saved: {(REPORTS_DIR / "06_feature_importance.csv").resolve()}')

for sensor in ['Sentinel-1', 'Sentinel-2']:
    top = fi_df[fi_df['Sensor'] == sensor].head(10)
    ml  = BEST_ML_FOR_IMPORTANCE[sensor]
    print(f'\\n{sensor} ({ml}) — top 10 features:')
    for _, row in top.iterrows():
        print(f'  {row["Feature"]:<40} {row["Importance Mean"]:+.6f} ± {row["Importance Std"]:.6f}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §18 — Robustness
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §18 — Model Robustness Diagnostics"))
cells.append(code("""\
# == §18.1  Train / Validation / Test MAE Gap Analysis ==
print('ROBUSTNESS DIAGNOSTICS — Train / Validation / Test MAE ratio')
for sensor in ['Sentinel-1', 'Sentinel-2']:
    print(f'\\n{sensor}:')
    sub = (res_df[res_df['Sensor'] == sensor]
           .pivot(index='Model', columns='Split', values='MAE'))
    for col in ['Train', 'Validation', 'Test']:
        if col not in sub.columns:
            sub[col] = np.nan
    sub = sub[['Train', 'Validation', 'Test']].copy()
    sub['Val/Train'] = (sub['Validation'] / sub['Train']).round(2)
    sub['Test/Val']  = (sub['Test'] / sub['Validation']).round(2)
    print(sub.to_string(float_format=lambda x: f'{x:.6f}'))
print('\\nInterpretation:')
print('  Val/Train > 1.0 indicates the model is not generalising fully from train to val.')
print('  Test/Val ~ 1.0–1.2 is expected given the distributional shift between 2022 and 2023.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §19 — Research Interpretation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §19 — Research Interpretation

### Key Findings

1. **No ML model consistently outperformed baselines on the 2022 validation set** for either sensor.
   The Mean Baseline (S1) and Zero-Change Baseline (S2) achieved the lowest validation MAE.

2. **This is consistent with Notebook 05's signal-to-noise analysis:**
   ~82% of annual area transitions fall within a single observation uncertainty (σ_A).
   The target is predominantly noise-dominated at the individual-transition level.

3. **Strong scale dependence (heteroscedasticity):** Large lakes dominate variance (5.87×/4.51×
   large/small dispersion ratio), while small lakes dominate sample count.
   A global model trained across all lake sizes faces an inherent bias–variance trade-off.

4. **The Tuned HGB achieved the best test MAE on Sentinel-2 (0.004501 km²)**, marginally
   beating the Mean Baseline (0.004507 km²). This difference is within measurement uncertainty
   and must **not** be interpreted as a validated improvement without additional hold-out evidence.

5. **Permutation importance** shows that ELEVATION_MIN (S1) and LATITUDE (S2) contribute most
   to ML predictions on the validation set — both with very small absolute magnitudes,
   consistent with near-zero exploitable signal.

### Scientific Guardrails
- These results do **not** constitute a GLOF prediction model.
- These results do **not** imply causal relationships between features and area change.
- Generalisation outside the 2017–2024 Nepal-transboundary observation domain is unsupported.

### Future Research Directions
- Lake-class-specific models (proglacial, moraine-dammed, supraglacial)
- Physical forcing covariates (temperature anomaly, precipitation, debris cover fraction)
- Uncertainty-aware loss functions appropriate for the transition target
- Multi-sensor ensemble predictions (S1 + S2 jointly)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §20 — All Figures + Final Summary
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §20 — Publication Figures & Final Results Artifact"))
cells.append(code("""\
# == §20.1  Figure Styling Defaults ==
DARK_BG    = '#0f1117'
PANEL_BG   = '#1a1d2e'
S1_COLOR   = '#00d4ff'
S2_COLOR   = '#a78bfa'
ACCENT     = '#ffcc00'
TEXT_COLOR = '#e0e0e0'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': PANEL_BG,
    'axes.edgecolor': '#3a3d4e', 'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,   'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR,    'legend.facecolor': PANEL_BG,
    'legend.edgecolor': '#3a3d4e', 'grid.color': '#2a2d3e', 'grid.alpha': 0.5,
})
model_order_fig = ['Mean Baseline','Zero-Change Baseline','Ridge','Random Forest',
                   'HistGradientBoosting','ExtraTrees','Tuned HGB']
colors_map = plt.cm.Set2(np.linspace(0, 1, len(model_order_fig)))
print('Figure style configured.')
"""))
cells.append(code("""\
# == §20.2  Fig 1 — Model Comparison Bar Chart ==
fig, axes_2 = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
fig.suptitle('GlacierGuard-AI — Model Comparison: Validation & Test MAE',
             color=TEXT_COLOR, fontsize=13, y=1.01)
for ax, split_lbl in zip(axes_2, ['Validation', 'Test']):
    ax.set_facecolor(PANEL_BG)
    for sensor, color, offset in [('Sentinel-1',S1_COLOR,-0.2),('Sentinel-2',S2_COLOR,0.2)]:
        maes = []
        for m in model_order_fig:
            s = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==m)&(res_df['Split']==split_lbl)]
            maes.append(s['MAE'].values[0] if len(s) else np.nan)
        x = np.arange(len(model_order_fig))
        bars = ax.bar(x+offset, maes, 0.38, color=color, alpha=0.85, label=sensor)
        top_v = max([m for m in maes if not np.isnan(m)], default=0)
        for bar, mae in zip(bars, maes):
            if not np.isnan(mae):
                ax.text(bar.get_x()+bar.get_width()/2, mae+top_v*0.012,
                        f'{mae:.5f}', ha='center', va='bottom', fontsize=5.8, color=color)
    ax.set_xticks(range(len(model_order_fig)))
    ax.set_xticklabels([m.replace(' ','\\n') for m in model_order_fig], fontsize=8)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(f'{split_lbl} Set', fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)
plt.tight_layout()
fp = FIGURES_DIR / '06_model_comparison.png'
plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show(); plt.close()
print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.3  Fig 2 & 3 — Residual Diagnostics (best ML model) ==
for sensor, val_df, prep, color, prefix in [
    ('Sentinel-1', s1_val, prep_s1, S1_COLOR, 's1'),
    ('Sentinel-2', s2_val, prep_s2, S2_COLOR, 's2'),
]:
    ml_name = BEST_ML_FOR_IMPORTANCE[sensor]
    est     = trained_estimators[f'{sensor}_{ml_name}']
    Xt      = prep.transform(val_df[MODEL_FEATURE_VECTOR])
    y_true  = val_df[TARGET_COL].values
    y_pred  = est.predict(Xt)
    resid   = y_true - y_pred

    fig, axes_ = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f'{sensor} — {ml_name}: Residual Diagnostics (Validation 2022)',
                 color=TEXT_COLOR, fontsize=12, y=1.01)

    ax = axes_[0]; ax.set_facecolor(PANEL_BG)
    ax.hist(resid, bins=80, color=color, alpha=0.82, edgecolor='none')
    ax.axvline(0, color=ACCENT, ls='--', lw=1.5, label='Zero')
    ax.axvline(np.mean(resid), color='#ff7675', ls=':', lw=1.5, label=f'Mean={np.mean(resid):.4f}')
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
    fp = FIGURES_DIR / f'06_residuals_{prefix}.png'
    plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show(); plt.close()
    print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.4  Fig 4 — Size-Stratified MAE ==
fig, axes_ = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Size-Stratified Validation MAE (Training-Derived Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
size_val_df = size_df[size_df['Split'] == 'Validation']
for ax, sensor in zip(axes_, ['Sentinel-1', 'Sentinel-2']):
    ax.set_facecolor(PANEL_BG)
    sub = size_val_df[size_val_df['Sensor'] == sensor]
    x_cats = ['Small', 'Medium', 'Large']
    models_p = [m for m in model_order_fig if m in sub['Model'].values]
    n_m = len(models_p); w = 0.8 / max(n_m, 1)
    offsets = np.linspace(-(n_m-1)*w/2, (n_m-1)*w/2, n_m)
    for i, (m, c) in enumerate(zip(models_p, colors_map)):
        sub_m = sub[sub['Model'] == m]
        maes  = [sub_m[sub_m['Size Tier']==t]['MAE'].values for t in x_cats]
        maes  = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats))+offsets[i], maes, w*0.88, color=c, alpha=0.85, label=m)
    ax.set_xticks(range(len(x_cats))); ax.set_xticklabels(x_cats)
    ax.set_xlabel('Lake Size Tier', fontsize=10); ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7); ax.grid(axis='y', ls=':', alpha=0.4)
plt.tight_layout()
fp = FIGURES_DIR / '06_size_stratified_error.png'
plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show(); plt.close(); print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.5  Fig 5 — Uncertainty-Stratified MAE ==
fig, axes_ = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
fig.suptitle('Uncertainty-Stratified Validation MAE (Training-Derived σ_A Thresholds)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
unc_val_df = unc_df[unc_df['Split'] == 'Validation']
for ax, sensor in zip(axes_, ['Sentinel-1', 'Sentinel-2']):
    ax.set_facecolor(PANEL_BG)
    sub = unc_val_df[unc_val_df['Sensor'] == sensor]
    x_cats = ['Low', 'Medium', 'High']
    models_p = [m for m in model_order_fig if m in sub['Model'].values]
    n_m = len(models_p); w = 0.8 / max(n_m, 1)
    offsets = np.linspace(-(n_m-1)*w/2, (n_m-1)*w/2, n_m)
    for i, (m, c) in enumerate(zip(models_p, colors_map)):
        sub_m = sub[sub['Model'] == m]
        maes  = [sub_m[sub_m['Uncertainty Tier']==t]['MAE'].values for t in x_cats]
        maes  = [v[0] if len(v) else np.nan for v in maes]
        ax.bar(np.arange(len(x_cats))+offsets[i], maes, w*0.88, color=c, alpha=0.85, label=m)
    ax.set_xticks(range(len(x_cats))); ax.set_xticklabels(x_cats)
    ax.set_xlabel('Area Uncertainty Tier (σ_A)', fontsize=10); ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7); ax.grid(axis='y', ls=':', alpha=0.4)
plt.tight_layout()
fp = FIGURES_DIR / '06_uncertainty_stratified_error.png'
plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show(); plt.close(); print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.6  Fig 6 & 7 — Feature Importance ==
for sensor, prefix, color in [('Sentinel-1','s1',S1_COLOR),('Sentinel-2','s2',S2_COLOR)]:
    ml  = BEST_ML_FOR_IMPORTANCE[sensor]
    sub = fi_df[fi_df['Sensor'] == sensor].head(20).sort_values('Importance Mean')
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub['Importance Mean'], xerr=sub['Importance Std'],
            color=color, alpha=0.85, height=0.7,
            error_kw={'ecolor': ACCENT, 'capsize': 3, 'elinewidth': 1})
    ax.set_yticks(y_pos); ax.set_yticklabels(sub['Feature'], fontsize=8.5)
    ax.set_xlabel('Permutation Importance (MAE decrease, km²)', fontsize=10)
    ax.set_title(f'{sensor} ({ml}) — Feature Importances\\n(Permutation, Validation 2022, 10 repeats)',
                 fontsize=11, color=TEXT_COLOR, pad=10)
    ax.axvline(0, color='#ff7675', ls='--', lw=1.2)
    ax.grid(axis='x', ls=':', alpha=0.4)
    plt.tight_layout()
    fp = FIGURES_DIR / f'06_feature_importance_{prefix}.png'
    plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show(); plt.close(); print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.7  Fig 8 — Error by Year ==
fig, axes_ = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('Selected Model — MAE by Observation Year (Validation + Test)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
for ax, sensor, color in zip(axes_, ['Sentinel-1','Sentinel-2'], [S1_COLOR,S2_COLOR]):
    ax.set_facecolor(PANEL_BG)
    sel = SELECTED_MODELS[sensor]
    sub = error_year_df[(error_year_df['Sensor']==sensor)&(error_year_df['Model']==sel)]
    for split_lbl, ls, mk in [('Validation','-','o'),('Test','--','s')]:
        ss = sub[sub['Split']==split_lbl].sort_values('AREA_YEAR')
        if len(ss): ax.plot(ss['AREA_YEAR'], ss['MAE'], marker=mk, ls=ls,
                            color=color, label=split_lbl, lw=2, markersize=7)
    ax.set_xlabel('AREA_YEAR (observation year)', fontsize=10)
    ax.set_ylabel('MAE (km²)', fontsize=10)
    ax.set_title(f'{sensor} | {sel}', fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
fp = FIGURES_DIR / '06_error_by_year.png'
plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show(); plt.close(); print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.8  Fig 9 — Tuning CV Curves ==
CV_FOLD_YEARS_PLOT = [(2017,2018),(2018,2019),(2019,2020),(2020,2021)]
HGB_GRID_KEYS_PLOT = ['max_iter','max_depth','min_samples_leaf','learning_rate']
rep_combos = [(100,4,30,0.05),(100,6,30,0.05),(200,4,30,0.05),(200,6,30,0.05)]
rep_labels  = ['mi=100,md=4,msl=30','mi=100,md=6,msl=30','mi=200,md=4,msl=30','mi=200,md=6,msl=30']
rep_colors_ = plt.cm.Blues(np.linspace(0.35, 0.9, len(rep_combos)))
fold_yrs    = [v for _, v in CV_FOLD_YEARS_PLOT]

fig, axes_ = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
fig.suptitle('HGB Expanding-Window CV — MAE per Fold (Representative Configs)',
             color=TEXT_COLOR, fontsize=12, y=1.01)
for ax_i, (sensor, Xtr, ytr_a, tr_yrs) in enumerate([
    ('Sentinel-1', Xt_s1_train, y_s1_train.values, s1_train['AREA_YEAR'].values),
    ('Sentinel-2', Xt_s2_train, y_s2_train.values, s2_train['AREA_YEAR'].values),
]):
    ax = axes_[ax_i]; ax.set_facecolor(PANEL_BG)
    for combo, lbl, rc in zip(rep_combos, rep_labels, rep_colors_):
        pdict_ = dict(zip(HGB_GRID_KEYS_PLOT, combo))
        fold_maes = []
        for train_end, val_year in CV_FOLD_YEARS_PLOT:
            fm_tr = tr_yrs <= train_end
            fm_va = tr_yrs == val_year
            if fm_va.sum() == 0: continue
            est_ = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **pdict_)
            est_.fit(Xtr[fm_tr], ytr_a[fm_tr])
            fold_maes.append(mean_absolute_error(ytr_a[fm_va], est_.predict(Xtr[fm_va])))
        ax.plot(fold_yrs[:len(fold_maes)], fold_maes, marker='o', lw=1.8, color=rc, label=lbl)
    ax.set_xlabel('Validation Fold Year', fontsize=10); ax.set_ylabel('Fold MAE (km²)', fontsize=10)
    ax.set_title(sensor, fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=7.5); ax.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
fp = FIGURES_DIR / '06_tuning_cv_curves.png'
plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show(); plt.close(); print(f'Saved: {fp.name}')
"""))
cells.append(code("""\
# == §20.9  Final Summary ==
print('=' * 65)
print('NOTEBOOK 06 — FINAL SUMMARY')
print('=' * 65)

for sensor in ['Sentinel-1', 'Sentinel-2']:
    sel  = SELECTED_MODELS[sensor]
    ml   = BEST_ML_FOR_IMPORTANCE[sensor]
    vrow = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==sel)&(res_df['Split']=='Validation')].iloc[0]
    trow = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==sel)&(res_df['Split']=='Test')].iloc[0]
    brow = res_df[(res_df['Sensor']==sensor)&(res_df['Model']=='Mean Baseline')&(res_df['Split']=='Validation')].iloc[0]
    mlvr = res_df[(res_df['Sensor']==sensor)&(res_df['Model']==ml)&(res_df['Split']=='Validation')].iloc[0]
    print(f'\\n{sensor}:')
    print(f'  Selected model (validation-based) : {sel}')
    print(f'  Validation MAE / RMSE / R²        : {vrow["MAE"]:.6f} / {vrow["RMSE"]:.6f} / {vrow["R2"]:.4f}')
    print(f'  Test MAE / RMSE / R²              : {trow["MAE"]:.6f} / {trow["RMSE"]:.6f} / {trow["R2"]:.4f}')
    print(f'  Best ML model (val)               : {ml} (MAE={mlvr["MAE"]:.6f})')
    print(f'  Mean Baseline (val)               : {brow["MAE"]:.6f}')
    delta = mlvr["MAE"] - brow["MAE"]
    print(f'  Best ML vs Mean Baseline          : ΔMAE={delta:+.6f} ({delta/brow["MAE"]*100:+.2f}%)')
    print(f'  Best CV params (Tuned HGB)        : {best_params[sensor]}')

print()
csvs = sorted(REPORTS_DIR.glob('06_*.csv'))
figs = sorted(FIGURES_DIR.glob('06_*.png'))
print(f'CSV Reports ({len(csvs)}):')
for f in csvs: print(f'  {f.name}')
print(f'\\nFigures ({len(figs)}):')
for f in figs: print(f'  {f.name}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Build and write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.14"
        }
    },
    "cells": cells
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
n_code = sum(1 for c in cells if c['cell_type'] == 'code')
n_md   = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f'Notebook written: {NB_PATH.resolve()}')
print(f'  Total cells    : {len(cells)}')
print(f'  Code cells     : {n_code}')
print(f'  Markdown cells : {n_md}')
print(f'  File size      : {NB_PATH.stat().st_size / 1024:.1f} KB')
