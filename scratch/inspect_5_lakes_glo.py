import geopandas as gpd
import pathlib
import pyproj
from shapely.ops import transform
from shapely.geometry import box

glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'

# 5 lakes definitions from NB10
lakes_def = [
    {'lake_id': 'imja_tsho', 'display_name': 'Imja Tsho', 'bbox': [86.900, 27.885, 86.955, 27.930], 'old_hist_area': 1.32},
    {'lake_id': 'thulagi', 'display_name': 'Thulagi Glacier Lake', 'bbox': [84.472, 28.524, 84.530, 28.565], 'old_hist_area': 0.73},
    {'lake_id': 'tsho_rolpa', 'display_name': 'Tsho Rolpa', 'bbox': [86.467, 27.850, 86.520, 27.900], 'old_hist_area': 1.65},
    {'lake_id': 'lower_barun', 'display_name': 'Lower Barun', 'bbox': [87.077, 27.818, 87.125, 27.858], 'old_hist_area': 0.48},
    {'lake_id': 'sabai_tsho', 'display_name': 'Sabai Tsho', 'bbox': [86.571, 27.794, 86.620, 27.830], 'old_hist_area': 0.22},
]

gdf_sample = gpd.read_file(s2_annual, rows=1)
target_crs = gdf_sample.crs
wgs84_to_target = pyproj.Transformer.from_crs('EPSG:4326', target_crs, always_xy=True).transform

results = []
for lk in lakes_def:
    b = lk['bbox']
    w_box = box(b[0], b[1], b[2], b[3])
    t_box = transform(wgs84_to_target, w_box)
    gdf_sub = gpd.read_file(s2_annual, bbox=t_box.bounds)
    # Filter for 2023 or latest year
    gdf_2023 = gdf_sub[gdf_sub['AREA_YEAR'] == 2023]
    if len(gdf_2023) == 0:
        # fallback to latest year in subset
        latest_yr = gdf_sub['AREA_YEAR'].max()
        gdf_2023 = gdf_sub[gdf_sub['AREA_YEAR'] == latest_yr]
    
    # Pick the largest polygon in the bbox if multiple small ponds exist
    gdf_2023 = gdf_2023.sort_values('AREA', ascending=False)
    if len(gdf_2023) > 0:
        best_row = gdf_2023.iloc[0]
        # Reproject to UTM to get exact projected area in km2
        # Determine UTM zone from longitude center
        lon_c = (b[0] + b[2]) / 2.0
        utm_zone = int((lon_c + 180) / 6) + 1
        utm_crs = f'EPSG:326{utm_zone}'
        poly_utm = gpd.GeoSeries([best_row.geometry], crs=target_crs).to_crs(utm_crs).iloc[0]
        utm_area_km2 = poly_utm.area / 1e6
        
        results.append({
            'lake_id': lk['lake_id'],
            'display_name': lk['display_name'],
            'glo_id': best_row['GLO_ID'],
            'year': best_row['AREA_YEAR'],
            'glo_table_area_km2': best_row['AREA'],
            'glo_utm_area_km2': utm_area_km2,
            'old_hist_area_km2': lk['old_hist_area'],
            'diff_glo_vs_old': utm_area_km2 - lk['old_hist_area'],
            'diff_pct': (utm_area_km2 - lk['old_hist_area']) / lk['old_hist_area'] * 100,
            'lon': best_row['LONGITUDE'],
            'lat': best_row['LATITUDE'],
            'utm_crs': utm_crs,
            'bbox': b
        })
    else:
        print(f"Warning: No lake found in bbox for {lk['lake_id']}")

import pandas as pd
df_res = pd.DataFrame(results)
print("=== 5 LAKES GLO IDENTIFICATION ===")
print(df_res[['lake_id', 'display_name', 'glo_id', 'year', 'glo_utm_area_km2', 'old_hist_area_km2', 'diff_glo_vs_old', 'diff_pct']].to_string(index=False))
