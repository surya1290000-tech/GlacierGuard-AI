"""
GlacierGuard-AI -- Imja Tsho Extent Calibration Pipeline
Executes the full calibration experiments across S1 and S2 scenes.
Generates all 7 CSV reports and 8 figures.
"""

import os, sys, json, io, math, pathlib, datetime, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.features
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')

ROOT = pathlib.Path('.').resolve()
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BBOX = [86.900, 27.885, 86.955, 27.930] # west, south, east, north
W, H = 512, 512
TRANSFORM = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], W, H)

# Projected pixel area in UTM Zone 45N
aoi_poly_4326 = gpd.GeoSeries.from_wkt([f'POLYGON(({BBOX[0]} {BBOX[1]}, {BBOX[2]} {BBOX[1]}, {BBOX[2]} {BBOX[3]}, {BBOX[0]} {BBOX[3]}, {BBOX[0]} {BBOX[1]}))'], crs='EPSG:4326')
aoi_poly_utm = aoi_poly_4326.to_crs('EPSG:32645').iloc[0]
PIXEL_AREA_KM2 = (aoi_poly_utm.area / 1e6) / (W * H)
print(f"AOI UTM Area: {aoi_poly_utm.area / 1e6:.4f} km2, Pixel area: {PIXEL_AREA_KM2:.8f} km2")

# GLO Polygon & Buffers
glo_dir = ROOT / 'raw' / 'GLO'
gdf_glo = gpd.read_file(glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg', where="GLO_ID = 'GLO_86.92845_27.89838'")
gdf_2023 = gdf_glo[gdf_glo['AREA_YEAR'] == 2023].to_crs('EPSG:32645')
poly_utm_0 = gdf_2023.geometry.iloc[0]
GLO_2023_AREA_KM2 = poly_utm_0.area / 1e6
print(f"GLO 2023 Reference Area (UTM): {GLO_2023_AREA_KM2:.4f} km2")

BUFFER_DISTANCES = [0, 50, 100, 200]
BUFFER_MASKS = {}
BUFFER_POLYS_4326 = {}
for b_m in BUFFER_DISTANCES:
    b_utm = poly_utm_0 if b_m == 0 else poly_utm_0.buffer(b_m)
    b_4326 = gpd.GeoSeries([b_utm], crs='EPSG:32645').to_crs('EPSG:4326').iloc[0]
    BUFFER_POLYS_4326[b_m] = b_4326
    m = rasterio.features.rasterize([(b_4326, 1)], out_shape=(H, W), transform=TRANSFORM, fill=0, dtype=np.uint8) == 1
    BUFFER_MASKS[b_m] = m
    print(f"Buffer {b_m:3d}m: {m.sum()} px ({m.sum()*PIXEL_AREA_KM2:.4f} km2)")

# Scene inventory
S1_METADATA = [
    {"date": "2026-09-28", "product_id": "S1D_IW_GRDH_1SDV_20260928T121313_20260928T121338_004778_008F55_1667_COG", "orbit": "Ascending (Rel 12)", "polarization": "VV+VH", "mode": "IW"},
    {"date": "2026-09-16", "product_id": "S1D_IW_GRDH_1SDV_20260916T121313_20260916T121338_004603_008949_F1C8_COG", "orbit": "Ascending (Rel 12)", "polarization": "VV+VH", "mode": "IW"},
    {"date": "2026-09-04", "product_id": "S1D_IW_GRDH_1SDV_20260904T121313_20260904T121338_004428_00833E_9092_COG", "orbit": "Ascending (Rel 12)", "polarization": "VV+VH", "mode": "IW"},
    {"date": "2026-08-23", "product_id": "S1D_IW_GRDH_1SDV_20260823T121313_20260823T121338_004253_007D19_461D_COG", "orbit": "Ascending (Rel 12)", "polarization": "VV+VH", "mode": "IW"},
    {"date": "2026-08-11", "product_id": "S1D_IW_GRDH_1SDV_20260811T121312_20260811T121337_004078_0076F9_676B_COG", "orbit": "Ascending (Rel 12)", "polarization": "VV+VH", "mode": "IW"},
]

S2_METADATA = [
    {"date": "2026-09-28", "product_id": "S2C_MSIL2A_20260928T044701_N0513_R076_T45RVL_20260928T075011", "tile_cloud": 6.13, "lake_cloud": 0.0, "lake_valid": 100.0},
    {"date": "2026-09-08", "product_id": "S2C_MSIL2A_20260908T044701_N0512_R076_T45RVL_20260908T094920", "tile_cloud": 65.73, "lake_cloud": 9.8, "lake_valid": 90.2},
    {"date": "2026-05-11", "product_id": "S2C_MSIL2A_20260511T044701_N0512_R076_T45RVL_20260511T080826", "tile_cloud": 35.42, "lake_cloud": 0.0, "lake_valid": 100.0},
    {"date": "2026-04-21", "product_id": "S2C_MSIL2A_20260421T044701_N0512_R076_T45RVL_20260421T080124", "tile_cloud": 3.79, "lake_cloud": 0.0, "lake_valid": 100.0},
    {"date": "2026-04-16", "product_id": "S2B_MSIL2A_20260416T044659_N0512_R076_T45RVL_20260416T083210", "tile_cloud": 25.72, "lake_cloud": 0.0, "lake_valid": 100.0},
]

# Helper functions
def lee_filter(img_linear, size=5, num_looks=4.4):
    img = np.maximum(img_linear, 1e-10)
    mean = uniform_filter(img, size=size)
    sqr_mean = uniform_filter(img**2, size=size)
    variance = np.maximum(sqr_mean - mean**2, 0)
    noise_var = (mean**2) / num_looks
    weight = variance / (variance + noise_var + 1e-10)
    weight = np.clip(weight, 0.0, 1.0)
    filtered = mean + weight * (img - mean)
    return np.maximum(filtered, 1e-10)

def remove_small_components(mask, min_size=10):
    labeled, n_feat = ndimage.label(mask)
    if n_feat == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n_feat + 1))
    clean = mask.copy()
    for i, s in enumerate(sizes, start=1):
        if s < min_size:
            clean[labeled == i] = 0
    return clean

def fill_holes(mask, max_hole_size=5):
    # Invert, label holes, remove small holes
    inv = ~mask
    labeled, n_feat = ndimage.label(inv)
    if n_feat <= 1:
        return mask
    # Background label is connected to borders
    border_labels = set(np.unique(np.concatenate([labeled[0,:], labeled[-1,:], labeled[:,0], labeled[:,-1]])))
    clean = mask.copy()
    sizes = ndimage.sum(inv, labeled, range(1, n_feat + 1))
    for i, s in enumerate(sizes, start=1):
        if i not in border_labels and s <= max_hole_size:
            clean[labeled == i] = 1
    return clean

print("\n--- Running S2 Calibrations ---")
# Cache load S2 scenes
s2_data = {}
for meta in S2_METADATA:
    fpath = ROOT / 'scratch' / 'cache_s2' / f"{meta['date']}.tif"
    with rasterio.open(fpath) as ds:
        arr = ds.read()
    green = arr[0]
    nir = arr[1]
    swir = arr[2]
    scl = np.round(arr[3]).astype(int)
    
    eps = 1e-10
    ndwi = (green - nir) / (green + nir + eps)
    mndwi = (green - swir) / (green + swir + eps)
    cloud_shadow = np.isin(scl, [0, 1, 3, 8, 9, 10])
    valid = ~cloud_shadow
    
    s2_data[meta['date']] = {
        'green': green, 'nir': nir, 'swir': swir, 'scl': scl,
        'ndwi': ndwi, 'mndwi': mndwi, 'valid': valid, 'cloud_shadow': cloud_shadow
    }

# Grid for S2: indices (MNDWI, NDWI), thresholds, buffers, cleanups
s2_rows = []
MNDWI_THRESHOLDS = [-0.2, -0.1, 0.0, 0.1, 0.2]
NDWI_THRESHOLDS = [-0.1, 0.0, 0.1, 0.2, 0.3]

for dt, d in s2_data.items():
    for buf_m, bmask in BUFFER_MASKS.items():
        vmask = d['valid'] & bmask
        cmask = d['cloud_shadow'] & bmask
        v_frac = vmask.sum() / max(bmask.sum(), 1)
        c_frac = cmask.sum() / max(bmask.sum(), 1)
        
        # Test MNDWI grid
        for th in MNDWI_THRESHOLDS:
            raw_w = (d['mndwi'] > th) & vmask
            clean_w = remove_small_components(raw_w.copy(), min_size=10)
            clean_w = fill_holes(clean_w, max_hole_size=5)
            
            labeled, n_comp = ndimage.label(clean_w)
            w_px = clean_w.sum()
            area_km2 = w_px * PIXEL_AREA_KM2
            w_frac = w_px / max(vmask.sum(), 1)
            
            s2_rows.append({
                'date': dt, 'sensor': 'S2', 'index': 'MNDWI', 'threshold': th,
                'buffer_m': buf_m, 'area_km2': area_km2, 'water_pixels': int(w_px),
                'valid_fraction': v_frac, 'cloud_fraction': c_frac,
                'water_fraction': w_frac, 'components': n_comp
            })
            
        # Test NDWI grid
        for th in NDWI_THRESHOLDS:
            raw_w = (d['ndwi'] > th) & vmask
            clean_w = remove_small_components(raw_w.copy(), min_size=10)
            clean_w = fill_holes(clean_w, max_hole_size=5)
            
            labeled, n_comp = ndimage.label(clean_w)
            w_px = clean_w.sum()
            area_km2 = w_px * PIXEL_AREA_KM2
            w_frac = w_px / max(vmask.sum(), 1)
            
            s2_rows.append({
                'date': dt, 'sensor': 'S2', 'index': 'NDWI', 'threshold': th,
                'buffer_m': buf_m, 'area_km2': area_km2, 'water_pixels': int(w_px),
                'valid_fraction': v_frac, 'cloud_fraction': c_frac,
                'water_fraction': w_frac, 'components': n_comp
            })

df_s2_cal = pd.DataFrame(s2_rows)
print(f"S2 calibration experiments: {len(df_s2_cal)} rows")

# Cache load S1 scenes
print("\n--- Running S1 Calibrations ---")
s1_data = {}
for meta in S1_METADATA:
    fpath = ROOT / 'scratch' / 'cache_s1' / f"{meta['date']}.tif"
    with rasterio.open(fpath) as ds:
        arr = ds.read()
    vv_lin = arr[0]
    vh_lin = arr[1] if arr.shape[0] > 1 else arr[0]
    
    vv_db = 10.0 * np.log10(np.maximum(vv_lin, 1e-10))
    vh_db = 10.0 * np.log10(np.maximum(vh_lin, 1e-10))
    valid = (vv_db > -35.0) & (vv_db < 5.0) & np.isfinite(vv_db)
    
    # Preprocessing filters
    lee_5 = 10.0 * np.log10(lee_filter(vv_lin, size=5, num_looks=4.4))
    lee_7 = 10.0 * np.log10(lee_filter(vv_lin, size=7, num_looks=4.4))
    
    s1_data[meta['date']] = {
        'vv_raw': vv_db, 'vh_raw': vh_db,
        'vv_lee5': lee_5, 'vv_lee7': lee_7,
        'valid': valid
    }

s1_rows = []
VV_THRESHOLDS = [-15.0, -14.0, -13.0, -12.0, -11.0]
PREPROC_METHODS = ['RAW', 'Lee 5x5', 'Lee 7x7']

for dt, d in s1_data.items():
    for buf_m, bmask in BUFFER_MASKS.items():
        vmask = d['valid'] & bmask
        v_frac = vmask.sum() / max(bmask.sum(), 1)
        
        for prep in PREPROC_METHODS:
            img = d['vv_raw'] if prep == 'RAW' else (d['vv_lee5'] if prep == 'Lee 5x5' else d['vv_lee7'])
            
            for th in VV_THRESHOLDS:
                raw_w = (img < th) & vmask
                clean_w = remove_small_components(raw_w.copy(), min_size=10)
                clean_w = fill_holes(clean_w, max_hole_size=5)
                
                labeled, n_comp = ndimage.label(clean_w)
                w_px = clean_w.sum()
                area_km2 = w_px * PIXEL_AREA_KM2
                w_frac = w_px / max(vmask.sum(), 1)
                
                s1_rows.append({
                    'date': dt, 'sensor': 'S1', 'preprocessing': prep, 'threshold': th,
                    'buffer_m': buf_m, 'area_km2': area_km2, 'water_pixels': int(w_px),
                    'valid_fraction': v_frac, 'water_fraction': w_frac, 'components': n_comp
                })

df_s1_cal = pd.DataFrame(s1_rows)
print(f"S1 calibration experiments: {len(df_s1_cal)} rows")

# Compute temporal stability summaries across scenes for each parameter configuration
print("\n--- Computing Temporal Stability & Parameter Optimization ---")
s2_stability = df_s2_cal.groupby(['index', 'threshold', 'buffer_m']).agg(
    mean_area=('area_km2', 'mean'),
    median_area=('area_km2', 'median'),
    std_area=('area_km2', 'std'),
    min_area=('area_km2', 'min'),
    max_area=('area_km2', 'max'),
    mean_comp=('components', 'mean'),
    mean_valid=('valid_fraction', 'mean'),
    n_scenes=('area_km2', 'count')
).reset_index()
s2_stability['cv_pct'] = (s2_stability['std_area'] / s2_stability['mean_area']) * 100
s2_stability['range_area'] = s2_stability['max_area'] - s2_stability['min_area']

s1_stability = df_s1_cal.groupby(['preprocessing', 'threshold', 'buffer_m']).agg(
    mean_area=('area_km2', 'mean'),
    median_area=('area_km2', 'median'),
    std_area=('area_km2', 'std'),
    min_area=('area_km2', 'min'),
    max_area=('area_km2', 'max'),
    mean_comp=('components', 'mean'),
    mean_valid=('valid_fraction', 'mean'),
    n_scenes=('area_km2', 'count')
).reset_index()
s1_stability['cv_pct'] = (s1_stability['std_area'] / s1_stability['mean_area']) * 100
s1_stability['range_area'] = s1_stability['max_area'] - s1_stability['min_area']

# Let's inspect the results
print("\nTop S2 configurations by lowest CV (temporal stability):")
print(s2_stability.sort_values('cv_pct').head(10)[['index', 'threshold', 'buffer_m', 'median_area', 'std_area', 'cv_pct', 'mean_comp']])

print("\nTop S1 configurations by lowest CV (temporal stability):")
print(s1_stability.sort_values('cv_pct').head(10)[['preprocessing', 'threshold', 'buffer_m', 'median_area', 'std_area', 'cv_pct', 'mean_comp']])
