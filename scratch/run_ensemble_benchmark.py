"""
GlacierGuard-AI — NB11 Ensemble Regression Benchmark
=====================================================
Compares: Persistence baseline, RandomForestRegressor, XGBRegressor,
          VotingRegressor (RF+XGB equal-weight), Weighted blend (val-tuned),
          and StackingRegressor (temporally-ordered OOF only if feasible).

Scientific constraints (non-negotiable):
- Target: NEXT_AREA_CHANGE = AREA(t+1) - AREA(t)  [km²]
- Temporal split: train ≤2021, val 2022, test 2023 (frozen)
- Same observations, features, preprocessing as NB08/NB09 pipeline
- Preprocessor fitted on train-only
- Ensemble weights and meta-learner selected on validation ONLY
- Test set consulted ONCE for confirmatory comparison
- Old model artifacts never overwritten
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import hashlib
import pathlib
import warnings
import numpy as np
import pandas as pd
import pickle
import platform
import importlib.metadata

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
GLO_DIR   = ROOT / 'raw' / 'GLO'
ERA5_DIR  = ROOT / 'raw' / 'ERA5'
CHIRPS_DIR = ROOT / 'raw' / 'CHIRPS'
RGI_DIR   = ROOT / 'raw' / 'RGI'
REPORTS   = ROOT / 'reports'
MODELS    = REPORTS / 'models'
FIGURES   = REPORTS / 'figures'
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ── Feature and target definitions (EXACT NB08 contract) ─────────────────────
TARGET_COL = 'NEXT_AREA_CHANGE'
CATEGORICAL_FEATURES = ['BASIN', 'COUNTRY', 'GTNG_REGION_O2', 'CONNECTIVITY']
NUMERICAL_FEATURES = [
    'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
    'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
    'LATITUDE', 'LONGITUDE',
    'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
    'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3',
]
# NB09 used 32 numerical features (S1+S2 fused via column prefix).
# Here we reproduce the same GLO-only 20-feature set for a fair per-sensor
# and fused experiment, matching what NB08/NB09 describes as NUMERICAL_FEATURES.
GLO_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
assert len(GLO_FEATURES) == 20, f"Expected 20, got {len(GLO_FEATURES)}"

# ── Gold dataset builder (exact replication of NB08 §4.1) ─────────────────────
def build_gold_dataset(gpkg_path: pathlib.Path) -> pd.DataFrame:
    """Recreate the NB08 gold dataset from a GLO GeoPackage."""
    import geopandas as gpd
    gdf = gpd.read_file(gpkg_path)

    # Drop geometry and rename columns for S2 cross-referencing compatibility
    df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))

    # Normalise column names: some gpkg files use LONGITUD vs LONGITUDE
    if 'LONGITUD' in df.columns and 'LONGITUDE' not in df.columns:
        df = df.rename(columns={'LONGITUD': 'LONGITUDE'})
    if 'GTNG_REGION' in df.columns and 'GTNG_REGION_O2' not in df.columns:
        df = df.rename(columns={'GTNG_REGION': 'GTNG_REGION_O2'})

    required = ['GLO_ID', 'AREA_YEAR', 'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
                'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
                'LATITUDE', 'LONGITUDE', 'BASIN', 'COUNTRY',
                'GTNG_REGION_O2', 'CONNECTIVITY']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {gpkg_path.name}: {missing}")

    df = df[required].copy()
    df = df.sort_values(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)

    # Temporal lag features (shift within each lake)
    grp = df.groupby('GLO_ID')
    df['PREV_YEAR']   = grp['AREA_YEAR'].shift(1)
    df['PREV_AREA']   = grp['AREA'].shift(1)
    df['NEXT_YEAR']   = grp['AREA_YEAR'].shift(-1)
    df['NEXT_AREA']   = grp['AREA'].shift(-1)
    df['NEXT_YEAR_GAP'] = df['NEXT_YEAR'] - df['AREA_YEAR']
    df['PREV_YEAR_GAP'] = df['AREA_YEAR'] - df['PREV_YEAR']

    df['AREA_LAG1'] = grp['AREA'].shift(1)
    df['AREA_LAG2'] = grp['AREA'].shift(2)
    df['AREA_LAG3'] = grp['AREA'].shift(3)

    df['AREA_CHANGE_LAG1'] = np.where(
        df['AREA_LAG1'].notna() & (df['PREV_YEAR_GAP'] == 1),
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
    df['ROLLING_STD_3'] = np.where(
        df['AREA_LAG3'].notna(),
        df[['AREA', 'AREA_LAG1', 'AREA_LAG2']].std(axis=1, ddof=0), np.nan)

    # Target: NEXT_AREA_CHANGE, only for consecutive year-pairs
    df['NEXT_AREA_CHANGE'] = np.where(
        df['NEXT_YEAR_GAP'] == 1,
        df['NEXT_AREA'] - df['AREA'], np.nan)

    return df


def create_temporal_splits(df: pd.DataFrame, label: str):
    """Chronological split matching NB08 §8.1 exactly."""
    sup   = df[df[TARGET_COL].notna()].copy()
    train = sup[sup['AREA_YEAR'] <= 2021].copy()
    val   = sup[sup['AREA_YEAR'] == 2022].copy()
    test  = sup[sup['AREA_YEAR'] == 2023].copy()

    assert len(train) + len(val) + len(test) == len(sup), 'Row count mismatch'
    assert train['AREA_YEAR'].max() <= 2021
    assert (val['AREA_YEAR'] == 2022).all()
    assert (test['AREA_YEAR'] == 2023).all()
    assert (sup['AREA_YEAR'] >= 2024).sum() == 0, '2024+ must not be in supervised set'
    assert TARGET_COL not in GLO_FEATURES

    print(f"{label}: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    return train, val, test


def build_preprocessor(num_cols, cat_cols):
    """Train-only fitted ColumnTransformer (impute + OHE), exact NB08 §12.1."""
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])


def compute_metrics(y_true, y_pred, prefix=''):
    """Return dict of MAE, RMSE, R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {f'{prefix}MAE': mae, f'{prefix}RMSE': rmse, f'{prefix}R2': r2}


# ── Temporal OOF stacking (expanding-window) ──────────────────────────────────
def temporal_oof_predictions(X_train_df: pd.DataFrame,
                              y_train: pd.Series,
                              year_col: pd.Series,
                              base_estimator,
                              num_cols: list,
                              cat_cols: list,
                              min_train_years: int = 3):
    """
    Generate out-of-fold predictions using expanding temporal windows.
    For each fold year t (starting from the (min_train_years+1)th available year):
      - Train preprocessor and model on years < t
      - Predict year t
    Returns predictions aligned to X_train_df index.
    Returns None if fewer than (min_train_years+1) distinct years exist.
    """
    years = sorted(year_col.unique())
    if len(years) < min_train_years + 1:
        print(f"[STACKING] Only {len(years)} distinct training years — "
              f"need at least {min_train_years+1}. Skipping stacking.")
        return None

    oof = pd.Series(np.nan, index=X_train_df.index)
    fold_years = years[min_train_years:]   # skip the first min_train_years as hold-out

    for fold_year in fold_years:
        tr_mask = year_col < fold_year
        ev_mask = year_col == fold_year
        if tr_mask.sum() == 0 or ev_mask.sum() == 0:
            continue

        prep = build_preprocessor(num_cols, cat_cols)
        Xtr = prep.fit_transform(
            X_train_df.loc[tr_mask, cat_cols + num_cols])
        Xev = prep.transform(
            X_train_df.loc[ev_mask, cat_cols + num_cols])
        ytr = y_train.loc[tr_mask]

        import copy
        est = copy.deepcopy(base_estimator)
        est.fit(Xtr, ytr)
        oof.loc[ev_mask] = est.predict(Xev)

    # Only rows that received a prediction
    valid_mask = oof.notna()
    n_valid = valid_mask.sum()
    n_total = len(oof)
    print(f"[STACKING] OOF: {n_valid}/{n_total} rows have predictions "
          f"({fold_years[0]}-{fold_years[-1]})")
    return oof, valid_mask


def paired_mae_comparison(y_true, preds_a, preds_b, label_a, label_b,
                           n_boot=2000, rng_seed=RANDOM_STATE):
    """Bootstrap paired MAE difference: delta = mean(|e_B| - |e_A|).

    Sign convention (candidate = preds_b, reference = preds_a):
      delta < 0  →  candidate has LOWER (better) MAE than reference
      delta > 0  →  candidate has HIGHER (worse)  MAE than reference
      delta == 0 →  candidate and reference are equivalent

    Module-level so it can be imported and unit-tested directly.
    """
    rng = np.random.default_rng(rng_seed)
    err_a = np.abs(np.asarray(y_true) - np.asarray(preds_a))
    err_b = np.abs(np.asarray(y_true) - np.asarray(preds_b))
    # delta = candidate_MAE - reference_MAE
    # Negative value means candidate is better (lower error).
    delta = np.mean(err_b - err_a)
    boot_deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_deltas.append(np.mean(err_b[idx] - err_a[idx]))
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    print(f"  Paired \u0394 MAE ({label_b} - {label_a}): {delta:+.6f} km\u00b2  "
          f"95% CI [{ci_lo:+.6f}, {ci_hi:+.6f}]"
          f"  ({'candidate better' if delta < 0 else 'candidate worse' if delta > 0 else 'equivalent'})")
    return delta, ci_lo, ci_hi


def run_ensemble_benchmark():
    print("=" * 70)
    print("GlacierGuard-AI — NB11 Ensemble Regression Benchmark")
    print("=" * 70)

    # ── Load gold datasets ────────────────────────────────────────────────────
    print("\n[1/7] Loading gold datasets from GLO GeoPackages...")
    t0 = time.time()
    gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
    gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')
    print(f"  S1: {len(gold_s1):,} rows | {gold_s1['GLO_ID'].nunique():,} lakes")
    print(f"  S2: {len(gold_s2):,} rows | {gold_s2['GLO_ID'].nunique():,} lakes")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # SHA-256 of source files for provenance
    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while chunk := f.read(1024*1024):
                h.update(chunk)
        return h.hexdigest()

    s1_gpkg_path = GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg'
    s2_gpkg_path = GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg'
    s1_sha = sha256_file(s1_gpkg_path)
    s2_sha = sha256_file(s2_gpkg_path)
    print(f"  S1 GeoPackage SHA-256: {s1_sha[:16]}...")
    print(f"  S2 GeoPackage SHA-256: {s2_sha[:16]}...")

    # ── Temporal splits ───────────────────────────────────────────────────────
    print("\n[2/7] Creating temporal splits...")
    s1_train, s1_val, s1_test = create_temporal_splits(gold_s1, 'S1')
    s2_train, s2_val, s2_test = create_temporal_splits(gold_s2, 'S2')

    # Hard assertion: match NB08 counts
    assert len(s1_train) == 10161, f"S1 train count mismatch: {len(s1_train)}"
    assert len(s1_val)   == 2026,  f"S1 val count mismatch: {len(s1_val)}"
    assert len(s1_test)  == 2055,  f"S1 test count mismatch: {len(s1_test)}"
    assert len(s2_train) == 11858, f"S2 train count mismatch: {len(s2_train)}"
    assert len(s2_val)   == 2245,  f"S2 val count mismatch: {len(s2_val)}"
    assert len(s2_test)  == 2507,  f"S2 test count mismatch: {len(s2_test)}"
    print("  All split-count assertions passed (parity with NB08).")

    # ── Verify frozen benchmark against existing artifact ─────────────────────
    print("\n[3/7] Verifying reported NB09 benchmark against frozen artifact...")
    frozen_path = MODELS / '09_final_fused_xgboost.pkl'
    if frozen_path.exists():
        frozen = pickle.load(open(frozen_path, 'rb'))
        reported_mae  = frozen.get('test_mae', None)
        reported_rmse = frozen.get('test_rmse', None)
        print(f"  Artifact test MAE  (reported): {reported_mae:.6f} km²")
        print(f"  Artifact test RMSE (reported): {reported_rmse:.6f} km²")
        print(f"  Training period: {frozen.get('training_period')}")
        print(f"  Test period:     {frozen.get('test_period')}")
        print("  [NOTE] NB09 uses a FUSED S1+S2 feature set (32 numerical features).")
        print("         NB11 uses the per-sensor GLO 20-feature set for fair comparison.")
        print("         Reported MAE 0.004792 is the NB09 fused benchmark; NB11 uses")
        print("         the same architecture on the same split to isolate ensemble gain.")
    else:
        print("  [WARNING] Frozen artifact not found — cannot cross-verify.")

    # ── Build preprocessors (train-only fit) ──────────────────────────────────
    print("\n[4/7] Fitting preprocessors on training data only...")

    def prep_split(train_df, val_df, test_df, label):
        """Fit on train, transform all three splits."""
        prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        X_feat_train = train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        X_feat_val   = val_df  [CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        X_feat_test  = test_df [CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        prep.fit(X_feat_train)
        Xtr = prep.transform(X_feat_train)
        Xva = prep.transform(X_feat_val)
        Xte = prep.transform(X_feat_test)
        y_tr = train_df[TARGET_COL].values
        y_va = val_df[TARGET_COL].values
        y_te = test_df[TARGET_COL].values
        print(f"  {label}: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")
        # Leakage guard
        assert TARGET_COL not in train_df.columns[train_df.columns.isin(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)]
        return prep, Xtr, Xva, Xte, y_tr, y_va, y_te

    prep_s1, Xtr_s1, Xva_s1, Xte_s1, ytr_s1, yva_s1, yte_s1 = prep_split(
        s1_train, s1_val, s1_test, 'S1')
    prep_s2, Xtr_s2, Xva_s2, Xte_s2, ytr_s2, yva_s2, yte_s2 = prep_split(
        s2_train, s2_val, s2_test, 'S2')

    # ── Define candidate models ───────────────────────────────────────────────
    print("\n[5/7] Fitting candidate models...")

    # XGB hyperparameters: taken from NB08 tuned config (EXP-09 best)
    xgb_params = {
        'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'min_child_weight': 5,
        'random_state': RANDOM_STATE, 'n_jobs': -1,
        'tree_method': 'hist', 'verbosity': 0,
    }
    rf_params = {
        'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 5,
        'max_features': 'sqrt', 'random_state': RANDOM_STATE, 'n_jobs': -1,
    }

    all_results = []

    def evaluate_sensor(sensor_label, Xtr, Xva, Xte, ytr, yva, yte,
                        train_df, year_col_train):
        results = []

        # --- 1. Persistence baseline ---
        pers_val_pred  = np.zeros(len(yva))
        pers_test_pred = np.zeros(len(yte))
        r_val  = compute_metrics(yva, pers_val_pred)
        r_test = compute_metrics(yte, pers_test_pred)
        results.append({'Sensor': sensor_label, 'Model': 'Persistence',
                         **{f'Val_{k}': v for k,v in r_val.items()},
                         **{f'Test_{k}': v for k,v in r_test.items()},
                         'val_preds': pers_val_pred, 'test_preds': pers_test_pred})
        print(f"  [{sensor_label}] Persistence  Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")

        # --- 2. RandomForest ---
        rf = RandomForestRegressor(**rf_params)
        rf.fit(Xtr, ytr)
        rf_val_pred  = rf.predict(Xva)
        rf_test_pred = rf.predict(Xte)
        r_val  = compute_metrics(yva, rf_val_pred)
        r_test = compute_metrics(yte, rf_test_pred)
        results.append({'Sensor': sensor_label, 'Model': 'RandomForest',
                         **{f'Val_{k}': v for k,v in r_val.items()},
                         **{f'Test_{k}': v for k,v in r_test.items()},
                         'val_preds': rf_val_pred, 'test_preds': rf_test_pred,
                         'fitted': rf})
        print(f"  [{sensor_label}] RandomForest  Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")

        # --- 3. XGBRegressor ---
        xgbm = xgb.XGBRegressor(**xgb_params)
        xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        xgb_val_pred  = xgbm.predict(Xva)
        xgb_test_pred = xgbm.predict(Xte)
        r_val  = compute_metrics(yva, xgb_val_pred)
        r_test = compute_metrics(yte, xgb_test_pred)
        results.append({'Sensor': sensor_label, 'Model': 'XGBoost',
                         **{f'Val_{k}': v for k,v in r_val.items()},
                         **{f'Test_{k}': v for k,v in r_test.items()},
                         'val_preds': xgb_val_pred, 'test_preds': xgb_test_pred,
                         'fitted': xgbm})
        print(f"  [{sensor_label}] XGBoost       Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")

        # --- 4. VotingRegressor (equal weights) ---
        # Note: VotingRegressor clones estimators; needs raw (unfitted) to avoid sklearn issue
        rf2  = RandomForestRegressor(**rf_params)
        xgb2 = xgb.XGBRegressor(**xgb_params)
        voter_eq = VotingRegressor([('rf', rf2), ('xgb', xgb2)], weights=[1, 1])
        voter_eq.fit(Xtr, ytr)
        vote_val_pred  = voter_eq.predict(Xva)
        vote_test_pred = voter_eq.predict(Xte)
        r_val  = compute_metrics(yva, vote_val_pred)
        r_test = compute_metrics(yte, vote_test_pred)
        results.append({'Sensor': sensor_label, 'Model': 'VotingEqualWeight',
                         **{f'Val_{k}': v for k,v in r_val.items()},
                         **{f'Test_{k}': v for k,v in r_test.items()},
                         'val_preds': vote_val_pred, 'test_preds': vote_test_pred})
        print(f"  [{sensor_label}] VotingEqWt    Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")

        # --- 5. Weighted blend (weights from val MAE minimization) ---
        # Search over alpha in [0,1]: final = alpha*RF + (1-alpha)*XGB
        best_alpha, best_val_mae = 0.5, np.inf
        for alpha in np.linspace(0.0, 1.0, 21):
            blend_val = alpha * rf_val_pred + (1 - alpha) * xgb_val_pred
            mae_blend = mean_absolute_error(yva, blend_val)
            if mae_blend < best_val_mae:
                best_val_mae  = mae_blend
                best_alpha    = alpha
        blend_val_pred  = best_alpha * rf_val_pred  + (1 - best_alpha) * xgb_val_pred
        blend_test_pred = best_alpha * rf_test_pred + (1 - best_alpha) * xgb_test_pred
        r_val  = compute_metrics(yva, blend_val_pred)
        r_test = compute_metrics(yte, blend_test_pred)
        results.append({'Sensor': sensor_label, 'Model': f'WeightedBlend(α={best_alpha:.2f})',
                         **{f'Val_{k}': v for k,v in r_val.items()},
                         **{f'Test_{k}': v for k,v in r_test.items()},
                         'val_preds': blend_val_pred, 'test_preds': blend_test_pred,
                         'blend_alpha': best_alpha})
        print(f"  [{sensor_label}] WeightedBlend α={best_alpha:.2f}  Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")

        # --- 6. Stacking (temporal OOF only if feasible) ---
        print(f"  [{sensor_label}] Attempting temporal OOF stacking...")
        oof_result = temporal_oof_predictions(
            train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES],
            train_df[TARGET_COL],
            year_col_train,
            xgb.XGBRegressor(**xgb_params),
            NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
            min_train_years=3,
        )
        if oof_result is not None:
            oof_preds, valid_mask = oof_result
            # Train meta-learner (Ridge) on OOF predictions from XGB + RF
            # Build RF OOF with same expanding windows
            rf_oof_result = temporal_oof_predictions(
                train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES],
                train_df[TARGET_COL],
                year_col_train,
                RandomForestRegressor(**rf_params),
                NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
                min_train_years=3,
            )
            rf_oof_preds = rf_oof_result[0] if rf_oof_result else None

            if rf_oof_preds is not None:
                combined_valid = valid_mask & rf_oof_result[1]
                meta_X_train = np.column_stack([
                    oof_preds[combined_valid],
                    rf_oof_preds[combined_valid],
                ])
                meta_y_train = train_df.loc[combined_valid, TARGET_COL].values
                ridge = Ridge(alpha=1.0)
                ridge.fit(meta_X_train, meta_y_train)

                # Re-train base models on ALL train data for val/test prediction
                xgb_full = xgb.XGBRegressor(**xgb_params)
                rf_full  = RandomForestRegressor(**rf_params)
                xgb_full.fit(Xtr, ytr)
                rf_full.fit(Xtr, ytr)

                meta_X_val  = np.column_stack([xgb_full.predict(Xva), rf_full.predict(Xva)])
                meta_X_test = np.column_stack([xgb_full.predict(Xte), rf_full.predict(Xte)])
                stack_val_pred  = ridge.predict(meta_X_val)
                stack_test_pred = ridge.predict(meta_X_test)
                r_val  = compute_metrics(yva, stack_val_pred)
                r_test = compute_metrics(yte, stack_test_pred)
                results.append({'Sensor': sensor_label, 'Model': 'StackingTemporalOOF',
                                 **{f'Val_{k}': v for k,v in r_val.items()},
                                 **{f'Test_{k}': v for k,v in r_test.items()},
                                 'val_preds': stack_val_pred, 'test_preds': stack_test_pred,
                                 'n_oof_valid': int(combined_valid.sum())})
                print(f"  [{sensor_label}] StackingOOF   Val MAE={r_val['MAE']:.6f}  Test MAE={r_test['MAE']:.6f}")
            else:
                print(f"  [{sensor_label}] StackingOOF: RF OOF failed — skipping stacking.")
        else:
            print(f"  [{sensor_label}] Stacking skipped (insufficient distinct training years).")

        return results

    print("\n  -- S1 Sentinel-1 models --")
    res_s1 = evaluate_sensor(
        'S1', Xtr_s1, Xva_s1, Xte_s1, ytr_s1, yva_s1, yte_s1,
        s1_train, s1_train['AREA_YEAR'])

    print("\n  -- S2 Sentinel-2 models --")
    res_s2 = evaluate_sensor(
        'S2', Xtr_s2, Xva_s2, Xte_s2, ytr_s2, yva_s2, yte_s2,
        s2_train, s2_train['AREA_YEAR'])

    all_results = res_s1 + res_s2

    # ── Model selection: use val MAE only ─────────────────────────────────────
    print("\n[6/7] Model selection (validation set only) ...")
    results_clean = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if k not in ('val_preds', 'test_preds', 'fitted', 'blend_alpha', 'n_oof_valid')}
        results_clean.append(row)

    df_results = pd.DataFrame(results_clean)
    print("\n  Validation MAE (model selection criterion):")
    print(df_results[['Sensor', 'Model', 'Val_MAE', 'Val_RMSE', 'Val_R2',
                        'Test_MAE', 'Test_RMSE', 'Test_R2']].to_string(index=False))

    # Best per sensor by val MAE
    best_by_sensor = {}
    for sensor, grp in df_results.groupby('Sensor'):
        best_row = grp.loc[grp['Val_MAE'].idxmin()]
        best_by_sensor[sensor] = best_row['Model']
        print(f"\n  Best model for {sensor} (by Val MAE): {best_row['Model']}")
        print(f"    Val  MAE={best_row['Val_MAE']:.6f}  RMSE={best_row['Val_RMSE']:.6f}  R²={best_row['Val_R2']:.4f}")
        print(f"    Test MAE={best_row['Test_MAE']:.6f}  RMSE={best_row['Test_RMSE']:.6f}  R²={best_row['Test_R2']:.4f}")

    # ── Residual diagnostics and paired comparison ────────────────────────────
    # paired_mae_comparison is defined at module level for direct importability
    # and unit-test coverage; call it here without redefining.

    print("\n  Paired comparisons (XGBoost as reference, vs Ensemble, on TEST):")
    paired_rows = []
    for sensor in ['S1', 'S2']:
        sensor_res = [r for r in all_results if r['Sensor'] == sensor]
        xgb_row  = next((r for r in sensor_res if r['Model'] == 'XGBoost'), None)
        pers_row = next((r for r in sensor_res if r['Model'] == 'Persistence'), None)

        if xgb_row is None:
            continue

        y_te = yte_s1 if sensor == 'S1' else yte_s2
        for cand_row in sensor_res:
            if cand_row['Model'] in ('XGBoost', 'Persistence'):
                continue
            delta, ci_lo, ci_hi = paired_mae_comparison(
                y_te,
                xgb_row['test_preds'],
                cand_row['test_preds'],
                'XGBoost',
                cand_row['Model'])
            paired_rows.append({
                'Sensor': sensor,
                'Reference': 'XGBoost',
                'Candidate': cand_row['Model'],
                'Delta_TestMAE': delta,
                'CI_95_Lower': ci_lo,
                'CI_95_Upper': ci_hi,
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[7/7] Saving results...")

    df_results.to_csv(REPORTS / '11_ensemble_regression_comparison.csv', index=False)
    print(f"  Saved: 11_ensemble_regression_comparison.csv ({len(df_results)} rows)")

    # Test predictions with residuals
    pred_rows = []
    for r in all_results:
        sensor = r['Sensor']
        y_te   = yte_s1 if sensor == 'S1' else yte_s2
        for pred, split in [(r.get('val_preds'), 'val'), (r.get('test_preds'), 'test')]:
            if pred is None:
                continue
            y_split = (yva_s1 if (sensor=='S1' and split=='val') else
                       yva_s2 if (sensor=='S2' and split=='val') else
                       yte_s1 if (sensor=='S1' and split=='test') else yte_s2)
            for i, (yt, yp) in enumerate(zip(y_split, pred)):
                pred_rows.append({
                    'Sensor': sensor, 'Split': split, 'Model': r['Model'],
                    'y_true': yt, 'y_pred': yp, 'residual': yp - yt,
                    'abs_error': abs(yt - yp),
                })

    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(REPORTS / '11_ensemble_test_predictions.csv', index=False)
    print(f"  Saved: 11_ensemble_test_predictions.csv ({len(df_preds)} rows)")

    if paired_rows:
        df_paired = pd.DataFrame(paired_rows)
        df_paired.to_csv(REPORTS / '11_ensemble_paired_comparison.csv', index=False)
        print(f"  Saved: 11_ensemble_paired_comparison.csv ({len(df_paired)} rows)")

    # Residual diagnostic plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        test_preds_only = df_preds[df_preds['Split'] == 'test'].copy()
        sensors = ['S1', 'S2']
        models  = ['Persistence', 'RandomForest', 'XGBoost', 'VotingEqualWeight']

        fig, axes = plt.subplots(len(sensors), len(models),
                                  figsize=(16, 8), facecolor='#1a1a2e')
        fig.suptitle('NB11 — Residual Diagnostics (Frozen 2023 Test)', fontsize=13,
                      color='white', fontweight='bold')

        for si, sensor in enumerate(sensors):
            for mi, model in enumerate(models):
                ax = axes[si][mi]
                ax.set_facecolor('#16213e')
                sub = test_preds_only[(test_preds_only['Sensor'] == sensor) &
                                       (test_preds_only['Model'] == model)]
                if len(sub) == 0:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                             color='white', transform=ax.transAxes)
                    ax.set_title(f'{sensor} {model}', color='white', fontsize=8)
                    continue

                ax.scatter(sub['y_true'], sub['residual'], s=3, alpha=0.35,
                            color='#00d4ff', linewidths=0)
                ax.axhline(0, color='#ff6b6b', lw=1, ls='--')
                ax.set_title(f'{sensor} — {model}', color='white', fontsize=8)
                ax.set_xlabel('True ΔAREA (km²)', color='#aaa', fontsize=7)
                ax.set_ylabel('Residual (km²)', color='#aaa', fontsize=7)
                ax.tick_params(colors='#aaa', labelsize=6)
                for sp in ax.spines.values():
                    sp.set_color('#333')

                mae_val = sub['abs_error'].mean()
                ax.text(0.97, 0.97, f'MAE={mae_val:.5f}', ha='right', va='top',
                         transform=ax.transAxes, color='#ffd700', fontsize=7)

        plt.tight_layout()
        fig_path = FIGURES / '11_ensemble_residual_diagnostics.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 11_ensemble_residual_diagnostics.png")
    except Exception as e:
        print(f"  [WARNING] Residual plot failed: {e}")

    # Experiment manifest
    manifest = {
        'description': 'NB11 RF+XGBoost ensemble regression benchmark',
        'target': 'NEXT_AREA_CHANGE = AREA(t+1) - AREA(t) [km²]',
        'data_sources': {
            's1_gpkg': str(s1_gpkg_path.relative_to(ROOT)),
            's1_sha256': s1_sha,
            's2_gpkg': str(s2_gpkg_path.relative_to(ROOT)),
            's2_sha256': s2_sha,
        },
        'splits': {
            'train': 'AREA_YEAR <= 2021',
            'val':   'AREA_YEAR == 2022',
            'test':  'AREA_YEAR == 2023 (frozen, consulted once)',
        },
        'sample_counts': {
            's1_train': 10161, 's1_val': 2026, 's1_test': 2055,
            's2_train': 11858, 's2_val': 2245, 's2_test': 2507,
        },
        'features': {
            'categorical': CATEGORICAL_FEATURES,
            'numerical': NUMERICAL_FEATURES,
            'total': len(GLO_FEATURES),
        },
        'models_evaluated': [r['Model'] for r in res_s1],
        'model_selection_criterion': 'Minimum Val_MAE (validation set, 2022)',
        'best_model_by_val_mae': best_by_sensor,
        'random_state': RANDOM_STATE,
        'hyperparameters': {'xgboost': xgb_params, 'randomforest': rf_params},
        'stacking': {
            'method': 'expanding-window temporal OOF',
            'min_train_years': 3,
            'meta_learner': 'Ridge(alpha=1.0)',
            'note': 'Implemented and run; see results table for feasibility outcome.',
        },
        'nb09_frozen_benchmark': {
            'model': 'FusedXGBoost(S1+S2, 32+4 features)',
            'reported_test_mae': 0.004792,
            'reported_test_rmse': 0.015079,
            'note': 'NB11 uses per-sensor 20-feature GLO set; not directly comparable to NB09 fused 36-feature artifact.',
        },
        'software': {
            'python': platform.python_version(),
            'xgboost': xgb.__version__,
            'sklearn': importlib.metadata.version('scikit-learn'),
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(REPORTS / '11_ensemble_experiment_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: 11_ensemble_experiment_manifest.json")

    # Save best S1 and S2 artifacts
    for sensor, res_list in [('S1', res_s1), ('S2', res_s2)]:
        best_model_name = best_by_sensor.get(sensor, 'XGBoost')
        best_row = next((r for r in res_list if r['Model'] == best_model_name), None)
        fitted   = best_row.get('fitted') if best_row else None
        if fitted is not None:
            tag = sensor.lower()
            artifact = {
                'model_class': type(fitted).__name__,
                'sensor': sensor,
                'model_name': best_model_name,
                'val_mae': best_row['Val_MAE'],
                'test_mae': best_row['Test_MAE'],
                'features_num': NUMERICAL_FEATURES,
                'features_cat': CATEGORICAL_FEATURES,
                'hyperparameters': xgb_params if 'XGB' in best_model_name else rf_params,
                'training_period': '2017-2021',
                'validation_period': '2022',
                'test_period': '2023',
                'random_seed': RANDOM_STATE,
                'model': fitted,
            }
            save_path = MODELS / f'11_best_{tag}.pkl'
            pickle.dump(artifact, open(save_path, 'wb'))
            print(f"  Saved model artifact: {save_path.name}")

    print("\n" + "="*70)
    print("NB11 BENCHMARK COMPLETE")
    print("="*70)
    print("\nFINAL SUMMARY TABLE (Test Set — Frozen 2023):")
    print(df_results[['Sensor', 'Model', 'Val_MAE', 'Test_MAE', 'Test_RMSE', 'Test_R2']
                     ].to_string(index=False, float_format='{:.6f}'.format))

    return df_results, manifest


if __name__ == '__main__':
    run_ensemble_benchmark()
