"""
GlacierGuard-AI -- 5-Lake Operational Extent Monitoring Pipeline
Applies the calibrated procedure to all 5 focal lakes.
Produces reports/10_2_five_lake_monitoring_qa.csv and diagnostic figures.
"""

import os, sys, json, io, math, pathlib, datetime, warnings
import urllib.request, urllib.parse, urllib.error
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

warnings.filterwarnings('ignore')

ROOT = pathlib.Path('.').resolve()
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
CACHE_BASE = ROOT / 'scratch' / 'cache_5lakes'
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
CACHE_BASE.mkdir(exist_ok=True)

# Auth
cred_file = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
creds = json.loads(cred_file.read_text())
cid, csec = creds['client_id'], creds['client_secret']

token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
payload = urllib.parse.urlencode({'grant_type': 'client_credentials', 'client_id': cid, 'client_secret': csec}).encode()
req = urllib.request.Request(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req) as resp:
    token = json.loads(resp.read().decode())['access_token']

print("Token acquired successfully.")

glo_dir = ROOT / 'raw' / 'GLO'
gdf_all_glo = gpd.read_file(glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg')

LAKES_CONFIG = [
    {
        'lake_id': 'imja_tsho',
        'display_name': 'Imja Tsho',
        'glo_id': 'GLO_86.92845_27.89838',
        'old_hist_ref_km2': 1.32,
        'bbox': [86.900, 27.885, 86.955, 27.930],
        'utm_epsg': 'EPSG:32645'
    },
    {
        'lake_id': 'thulagi',
        'display_name': 'Thulagi Glacier Lake',
        'glo_id': 'GLO_84.485_28.48833',
        'old_hist_ref_km2': 0.73,
        'bbox': [84.460, 28.465, 84.515, 28.515],
        'utm_epsg': 'EPSG:32645'
    },
    {
        'lake_id': 'tsho_rolpa',
        'display_name': 'Tsho Rolpa',
        'glo_id': 'GLO_86.47904_27.85858',
        'old_hist_ref_km2': 1.65,
        'bbox': [86.455, 27.840, 86.515, 27.890],
        'utm_epsg': 'EPSG:32645'
    },
    {
        'lake_id': 'lower_barun',
        'display_name': 'Lower Barun',
        'glo_id': 'GLO_87.0815_27.84295',
        'old_hist_ref_km2': 0.48,
        'bbox': [87.060, 27.825, 87.110, 27.865],
        'utm_epsg': 'EPSG:32645'
    },
    {
        'lake_id': 'sabai_tsho',
        'display_name': 'Sabai Tsho',
        'glo_id': 'GLO_86.62025_27.79158',
        'old_hist_ref_km2': 0.22,
        'bbox': [86.600, 27.775, 86.645, 27.815],
        'utm_epsg': 'EPSG:32645'
    }
]

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
    inv = ~mask
    labeled, n_feat = ndimage.label(inv)
    if n_feat <= 1:
        return mask
    border_labels = set(np.unique(np.concatenate([labeled[0,:], labeled[-1,:], labeled[:,0], labeled[:,-1]])))
    clean = mask.copy()
    sizes = ndimage.sum(inv, labeled, range(1, n_feat + 1))
    for i, s in enumerate(sizes, start=1):
        if i not in border_labels and s <= max_hole_size:
            clean[labeled == i] = 1
    return clean

evalscript_s2 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B08", "B11", "SCL"]}],
    output: {bands: 4, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B08, sample.B11, sample.SCL];
}
"""

evalscript_s1 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""

def fetch_process_api(token, sensor_type, bbox, dt_str, evalscript, width=512, height=512):
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": sensor_type, "dataFilter": {"timeRange": {"from": f"{dt_str}T00:00:00Z", "to": f"{dt_str}T23:59:59Z"}}}]
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    req = urllib.request.Request(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'User-Agent': 'GlacierGuard-AI/1.0'}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()

# Dates for each lake:
# Imja Tsho, Tsho Rolpa, Lower Barun, Sabai Tsho share the eastern cluster:
# S2 dates: 2026-09-28, 2026-09-08, 2026-05-11, 2026-04-21, 2026-04-16
# S1 dates: 2026-09-28, 2026-09-16, 2026-09-04, 2026-08-23, 2026-08-11
EASTERN_S2_DATES = ["2026-09-28", "2026-09-08", "2026-05-11", "2026-04-21", "2026-04-16"]
EASTERN_S1_DATES = ["2026-09-28", "2026-09-16", "2026-09-04", "2026-08-23", "2026-08-11"]

# For Thulagi (central Nepal, Marsyangdi / Manaslu):
# S1: 2026-09-26, 2026-09-21, 2026-09-14, 2026-09-09, 2026-08-28
# S2: 2026-09-21, 2026-05-24, 2026-05-14, 2026-04-24, 2026-04-14
THULAGI_S1_DATES = ["2026-09-26", "2026-09-21", "2026-09-14", "2026-09-09", "2026-08-28"]
THULAGI_S2_DATES = ["2026-09-21", "2026-05-24", "2026-05-14", "2026-04-24", "2026-04-14"]

W, H = 512, 512
qa_records = []
lake_visuals = {}

for lk in LAKES_CONFIG:
    lid = lk['lake_id']
    name = lk['display_name']
    glo_id = lk['glo_id']
    old_hist = lk['old_hist_ref_km2']
    bbox = lk['bbox']
    utm_epsg = lk['utm_epsg']
    
    print(f"\n========================================================")
    print(f"PROCESSING LAKE: {name} ({lid})")
    print(f"========================================================")
    
    # 1. GLO Polygon & Projected Pixel Area
    gdf_lake_glo = gdf_all_glo[gdf_all_glo['GLO_ID'] == glo_id]
    gdf_2023 = gdf_lake_glo[gdf_lake_glo['AREA_YEAR'] == 2023].to_crs(utm_epsg)
    poly_utm_0 = gdf_2023.geometry.iloc[0]
    glo_2023_area = poly_utm_0.area / 1e6
    
    # 50m buffer search envelope
    poly_utm_50 = poly_utm_0.buffer(50)
    poly_4326_50 = gpd.GeoSeries([poly_utm_50], crs=utm_epsg).to_crs('EPSG:4326').iloc[0]
    poly_4326_0 = gpd.GeoSeries([poly_utm_0], crs=utm_epsg).to_crs('EPSG:4326').iloc[0]
    
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
    mask_buffer_50 = rasterio.features.rasterize([(poly_4326_50, 1)], out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8) == 1
    mask_glo_0 = rasterio.features.rasterize([(poly_4326_0, 1)], out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8) == 1
    
    aoi_poly_4326 = gpd.GeoSeries.from_wkt([f'POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))'], crs='EPSG:4326')
    aoi_poly_utm = aoi_poly_4326.to_crs(utm_epsg).iloc[0]
    pixel_area_km2 = (aoi_poly_utm.area / 1e6) / (W * H)
    
    print(f"GLO 2023 Vector Area : {glo_2023_area:.4f} km²")
    print(f"Old Historical Ref   : {old_hist:.2f} km² (Diff: {glo_2023_area - old_hist:+.4f} km², {(glo_2023_area - old_hist)/old_hist*100:+.1f}%)")
    print(f"50m Search Envelope  : {mask_buffer_50.sum()} px ({mask_buffer_50.sum()*pixel_area_km2:.4f} km²)")
    
    # Lake cache directory
    lake_cache = CACHE_BASE / lid
    lake_cache.mkdir(exist_ok=True)
    
    s1_dates = THULAGI_S1_DATES if lid == 'thulagi' else EASTERN_S1_DATES
    s2_dates = THULAGI_S2_DATES if lid == 'thulagi' else EASTERN_S2_DATES
    
    # 2. Fetch and Process S2 Observations
    print(f"\n--- Processing S2 ({len(s2_dates)} scenes) ---")
    s2_observations = []
    latest_s2_mask = None
    latest_s2_date = None
    latest_s2_valid_frac = 1.0
    latest_s2_cloud_frac = 0.0
    
    for dt_str in s2_dates:
        cache_file = lake_cache / f"s2_{dt_str}.tif"
        if not cache_file.exists():
            # If Imja, check scratch/cache_s2
            if lid == 'imja_tsho' and (ROOT / 'scratch' / 'cache_s2' / f"{dt_str}.tif").exists():
                cache_file.write_bytes((ROOT / 'scratch' / 'cache_s2' / f"{dt_str}.tif").read_bytes())
            else:
                try:
                    data_bytes = fetch_process_api(token, 'sentinel-2-l2a', bbox, dt_str, evalscript_s2)
                    cache_file.write_bytes(data_bytes)
                except Exception as ex:
                    print(f"  S2 {dt_str} download failed: {ex}")
                    continue
                    
        with rasterio.open(cache_file) as ds:
            arr = ds.read()
        green, nir, swir = arr[0], arr[1], arr[2]
        scl = np.round(arr[3]).astype(int)
        eps = 1e-10
        mndwi = (green - swir) / (green + swir + eps)
        
        # SCL masking
        cloud_shadow = np.isin(scl, [0, 1, 3, 8, 9, 10])
        valid = ~cloud_shadow
        
        lake_cloud_px = (cloud_shadow & mask_buffer_50).sum()
        lake_valid_px = (valid & mask_buffer_50).sum()
        cloud_frac = lake_cloud_px / max(mask_buffer_50.sum(), 1)
        valid_frac = lake_valid_px / max(mask_buffer_50.sum(), 1)
        
        # Water extraction: MNDWI > 0.0 inside 50m buffer
        raw_w = (mndwi > 0.0) & valid & mask_buffer_50
        clean_w = remove_small_components(raw_w.copy(), min_size=10)
        clean_w = fill_holes(clean_w, max_hole_size=5)
        
        area_km2 = clean_w.sum() * pixel_area_km2
        print(f"  S2 {dt_str}: Area = {area_km2:.4f} km², Valid = {valid_frac*100:.1f}%, Lake Cloud = {cloud_frac*100:.1f}%")
        
        if valid_frac >= 0.65:
            s2_observations.append({
                'date': dt_str, 'area_km2': area_km2,
                'valid_frac': valid_frac, 'cloud_frac': cloud_frac, 'mask': clean_w
            })
            if latest_s2_mask is None:
                latest_s2_mask = clean_w
                latest_s2_date = dt_str
                latest_s2_valid_frac = valid_frac
                latest_s2_cloud_frac = cloud_frac
                
    # 3. Fetch and Process S1 Observations
    print(f"\n--- Processing S1 ({len(s1_dates)} scenes) ---")
    s1_observations = []
    latest_s1_mask = None
    latest_s1_date = None
    
    for dt_str in s1_dates:
        cache_file = lake_cache / f"s1_{dt_str}.tif"
        if not cache_file.exists():
            if lid == 'imja_tsho' and (ROOT / 'scratch' / 'cache_s1' / f"{dt_str}.tif").exists():
                cache_file.write_bytes((ROOT / 'scratch' / 'cache_s1' / f"{dt_str}.tif").read_bytes())
            else:
                try:
                    data_bytes = fetch_process_api(token, 'sentinel-1-grd', bbox, dt_str, evalscript_s1)
                    cache_file.write_bytes(data_bytes)
                except Exception as ex:
                    print(f"  S1 {dt_str} download failed: {ex}")
                    continue
                    
        with rasterio.open(cache_file) as ds:
            arr = ds.read()
        vv_lin = arr[0]
        vv_filtered = lee_filter(vv_lin, size=5, num_looks=4.4)
        vv_db = 10.0 * np.log10(np.maximum(vv_filtered, 1e-10))
        valid_s1 = (vv_db > -35.0) & (vv_db < 5.0) & np.isfinite(vv_db)
        
        # Water extraction: VV < -12.0 dB inside 50m buffer
        raw_w = (vv_db < -12.0) & valid_s1 & mask_buffer_50
        clean_w = remove_small_components(raw_w.copy(), min_size=10)
        clean_w = fill_holes(clean_w, max_hole_size=5)
        
        area_km2 = clean_w.sum() * pixel_area_km2
        v_frac = (valid_s1 & mask_buffer_50).sum() / max(mask_buffer_50.sum(), 1)
        print(f"  S1 {dt_str}: Area = {area_km2:.4f} km², Valid = {v_frac*100:.1f}%")
        
        s1_observations.append({
            'date': dt_str, 'area_km2': area_km2, 'valid_frac': v_frac, 'mask': clean_w
        })
        if latest_s1_mask is None:
            latest_s1_mask = clean_w
            latest_s1_date = dt_str

    # 4. Multi-Temporal Statistics
    s2_areas = [obs['area_km2'] for obs in s2_observations]
    s1_areas = [obs['area_km2'] for obs in s1_observations]
    
    s2_med = float(np.median(s2_areas)) if s2_areas else 0.0
    s2_std = float(np.std(s2_areas)) if len(s2_areas) > 1 else 0.0
    s2_cv = (s2_std / s2_med * 100) if s2_med > 0 else 0.0
    
    s1_med = float(np.median(s1_areas)) if s1_areas else 0.0
    s1_std = float(np.std(s1_areas)) if len(s1_areas) > 1 else 0.0
    s1_cv = (s1_std / s1_med * 100) if s1_med > 0 else 0.0
    
    latest_s2_area = s2_observations[0]['area_km2'] if s2_observations else 0.0
    latest_s1_area = s1_observations[0]['area_km2'] if s1_observations else 0.0
    
    # 5. Cross-Sensor Disagreement on Nearest Temporal Observation Pair
    if latest_s1_area > 0 and latest_s2_area > 0:
        abs_diff = abs(latest_s2_area - latest_s1_area)
        rel_disagreement = abs_diff / ((latest_s2_area + latest_s1_area) / 2.0) * 100
    else:
        abs_diff = np.nan
        rel_disagreement = np.nan
        
    # 6. Quality Score & Monitoring Status
    if not s2_observations or latest_s2_area == 0.0:
        quality = "INSUFFICIENT_DATA"
        status = "INSUFFICIENT_DATA"
        diff_from_glo = np.nan
        pct_from_glo = np.nan
    else:
        # Quality logic based on observability and stability
        if latest_s2_valid_frac >= 0.90 and latest_s2_cloud_frac <= 0.10:
            quality = "HIGH"
        elif latest_s2_valid_frac >= 0.70 and latest_s2_cloud_frac <= 0.30:
            quality = "MEDIUM"
        else:
            quality = "LOW"
            
        diff_from_glo = latest_s2_area - glo_2023_area
        pct_from_glo = (diff_from_glo / glo_2023_area) * 100
        
        if abs(pct_from_glo) < 15.0 and rel_disagreement < 25.0:
            status = "NORMAL"
        elif abs(pct_from_glo) < 30.0 or rel_disagreement >= 25.0:
            status = "WATCH"
        else:
            status = "UNUSUAL_CHANGE"
        
    print(f"\n--- Summary for {name} ---")
    print(f"  Latest S2 Date: {latest_s2_date} | S2 Area: {latest_s2_area:.4f} km² | S2 CV: {s2_cv:.1f}%")
    print(f"  Latest S1 Date: {latest_s1_date} | S1 Area: {latest_s1_area:.4f} km² | S1 CV: {s1_cv:.1f}%")
    print(f"  Cross-Sensor Disagreement : {rel_disagreement:.2f}% (abs diff: {abs_diff:.4f} km²)")
    print(f"  GLO 2023 Vector Reference : {glo_2023_area:.4f} km² (Diff: {diff_from_glo:+.4f} km², {pct_from_glo:+.1f}%)")
    print(f"  Old Historical Literature : {old_hist:.2f} km²")
    print(f"  Quality: {quality} | Monitoring Status: {status}")
    
    qa_records.append({
        'lake_id': lid,
        'lake_name': name,
        'glo_id': glo_id,
        'latest_s1_date': latest_s1_date,
        'latest_s2_date': latest_s2_date,
        's1_area_km2': latest_s1_area,
        's2_area_km2': latest_s2_area,
        's1_median_area_km2': s1_med,
        's2_median_area_km2': s2_med,
        's1_cv_pct': s1_cv,
        's2_cv_pct': s2_cv,
        'cross_sensor_disagreement_pct': rel_disagreement,
        'valid_pixel_fraction': latest_s2_valid_frac,
        'cloud_fraction': latest_s2_cloud_frac,
        'glo_2023_vector_ref_km2': glo_2023_area,
        'diff_from_glo_2023_km2': diff_from_glo,
        'diff_from_glo_2023_pct': pct_from_glo,
        'old_hist_ref_km2': old_hist,
        'diff_glo_vs_old_hist_pct': (glo_2023_area - old_hist) / old_hist * 100,
        'quality': quality,
        'monitoring_status': status
    })
    
    lake_visuals[lid] = {
        'name': name, 's2_mask': latest_s2_mask, 's1_mask': latest_s1_mask,
        'glo_mask': mask_glo_0, 's2_area': latest_s2_area, 's1_area': latest_s1_area,
        'glo_area': glo_2023_area, 'pixel_area': pixel_area_km2
    }

df_qa = pd.DataFrame(qa_records)
df_qa.to_csv(REPORTS / '10_2_five_lake_monitoring_qa.csv', index=False)
print(f"\nSaved QA Table to {REPORTS / '10_2_five_lake_monitoring_qa.csv'}")

# Generate Multi-Lake QA Visual Figure
fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300)
cmap_s2 = plt.matplotlib.colors.ListedColormap(['#0F172A', '#0284C7'])
cmap_s1 = plt.matplotlib.colors.ListedColormap(['#0F172A', '#10B981'])

for col_idx, (lid, vis) in enumerate(lake_visuals.items()):
    # Row 0: S2 Optical Mask with GLO Boundary
    ax_s2 = axes[0, col_idx]
    if vis['s2_mask'] is not None:
        ax_s2.imshow(vis['s2_mask'], cmap=cmap_s2)
        ax_s2.contour(vis['glo_mask'], levels=[0.5], colors=['#F59E0B'], linewidths=[1.5])
    ax_s2.set_title(f"{vis['name']}\nS2 MNDWI: {vis['s2_area']:.3f} km²\nGLO 2023: {vis['glo_area']:.3f} km²", fontsize=10, fontweight='bold')
    ax_s2.set_xticks([])
    ax_s2.set_yticks([])
    
    # Row 1: S1 SAR Mask with GLO Boundary
    ax_s1 = axes[1, col_idx]
    if vis['s1_mask'] is not None:
        ax_s1.imshow(vis['s1_mask'], cmap=cmap_s1)
        ax_s1.contour(vis['glo_mask'], levels=[0.5], colors=['#F59E0B'], linewidths=[1.5])
    ax_s1.set_title(f"S1 SAR (VV < -12 dB)\nArea: {vis['s1_area']:.3f} km²", fontsize=10, fontweight='bold')
    ax_s1.set_xticks([])
    ax_s1.set_yticks([])

# Add row labels
axes[0, 0].set_ylabel('Sentinel-2 (Optical)\n[Blue=Water, Gold=GLO]', fontsize=11, fontweight='bold', labelpad=10)
axes[1, 0].set_ylabel('Sentinel-1 (SAR)\n[Green=Water, Gold=GLO]', fontsize=11, fontweight='bold', labelpad=10)

plt.suptitle('GlacierGuard-AI — Controlled 5-Lake Operational Extent Monitoring QA', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = FIGURES / '10_2_five_lake_extent_qa.png'
plt.savefig(fig_path, bbox_inches='tight')
plt.close()
print(f"Saved QA Figure to {fig_path}")

print("\n=== COMPLETE 5-LAKE OPERATIONAL QA TABLE ===")
display_cols = ['lake_name', 'latest_s2_date', 's2_area_km2', 's1_area_km2', 'glo_2023_vector_ref_km2', 'old_hist_ref_km2', 'cross_sensor_disagreement_pct', 'quality', 'monitoring_status']
print(df_qa[display_cols].to_string(index=False))
