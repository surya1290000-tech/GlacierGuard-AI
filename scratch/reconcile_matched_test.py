import pathlib
import json
import joblib
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
GLO_DIR = ROOT / "raw" / "GLO"
REPORTS = ROOT / "reports"

# 1. Build gold datasets with exact features
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

    df['NEXT_AREA_CHANGE'] = np.where(
        df['NEXT_YEAR_GAP'] == 1,
        df['NEXT_AREA'] - df['AREA'], np.nan)
    return df

print("Loading gold datasets...")
gold_s1 = build_gold_dataset(GLO_DIR / "S1_20172024_NTB_GLOID_v1.02.gpkg")
gold_s2 = build_gold_dataset(GLO_DIR / "S2_20172024_NTB_GLOID_v1.02.gpkg")

s1_sup = gold_s1[gold_s1['NEXT_AREA_CHANGE'].notna()].copy()
s2_sup = gold_s2[gold_s2['NEXT_AREA_CHANGE'].notna()].copy()

s1_te = s1_sup[s1_sup['AREA_YEAR'] == 2023].copy().reset_index(drop=True)
s2_te = s2_sup[s2_sup['AREA_YEAR'] == 2023].copy().reset_index(drop=True)

print(f"S1 2023 Test Rows: {len(s1_te)}")
print(f"S2 2023 Test Rows: {len(s2_te)}")

# Find matched IDs
matched_idx = set(zip(s1_te['GLO_ID'], s1_te['AREA_YEAR'])) & set(zip(s2_te['GLO_ID'], s2_te['AREA_YEAR']))
print(f"Matched (GLO_ID, 2023) count: {len(matched_idx)}")

# Mask in s1_te and s2_te
m_s1_mask = s1_te.apply(lambda r: (r['GLO_ID'], r['AREA_YEAR']) in matched_idx, axis=1)
m_s2_mask = s2_te.apply(lambda r: (r['GLO_ID'], r['AREA_YEAR']) in matched_idx, axis=1)

m_s1_te = s1_te[m_s1_mask].copy().sort_values('GLO_ID').reset_index(drop=True)
m_s2_te = s2_te[m_s2_mask].copy().sort_values('GLO_ID').reset_index(drop=True)

assert (m_s1_te['GLO_ID'] == m_s2_te['GLO_ID']).all(), "GLO_ID alignment error in matched test!"
print("GLO_IDs aligned perfectly across matched S1 and S2.")

# Consensus target
y_s1_matched = m_s1_te['NEXT_AREA_CHANGE'].values
y_s2_matched = m_s2_te['NEXT_AREA_CHANGE'].values
y_fused_target = (y_s1_matched + y_s2_matched) / 2

# Check correlation between S1 and S2 targets on matched test
corr = np.corrcoef(y_s1_matched, y_s2_matched)[0, 1]
print(f"Pearson r between S1 target and S2 target on matched 2023 test (N=1742): {corr:.4f}")
