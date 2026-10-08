import geopandas as gpd
import pathlib

glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'

lakes = [
    {'lake_id': 'imja_tsho', 'glo_id': 'GLO_86.92845_27.89838', 'display_name': 'Imja Tsho', 'old_hist': 1.32, 'bbox': [86.900, 27.885, 86.955, 27.930]},
    {'lake_id': 'thulagi', 'glo_id': 'GLO_84.485_28.48833', 'display_name': 'Thulagi Glacier Lake', 'old_hist': 0.73, 'bbox': [84.460, 28.465, 84.515, 28.515]},
    {'lake_id': 'tsho_rolpa', 'glo_id': 'GLO_86.47904_27.85858', 'display_name': 'Tsho Rolpa', 'old_hist': 1.65, 'bbox': [86.455, 27.840, 86.515, 27.890]},
    {'lake_id': 'lower_barun', 'glo_id': 'GLO_87.0815_27.84295', 'display_name': 'Lower Barun', 'old_hist': 0.48, 'bbox': [87.060, 27.825, 87.110, 27.865]},
    {'lake_id': 'sabai_tsho', 'glo_id': 'GLO_86.62025_27.79158', 'display_name': 'Sabai Tsho', 'old_hist': 0.22, 'bbox': [86.600, 27.775, 86.645, 27.815]},
]

for lk in lakes:
    gdf = gpd.read_file(s2_annual, where=f"GLO_ID = '{lk['glo_id']}'")
    gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023]
    poly_4326 = gdf_2023.to_crs('EPSG:4326').geometry.iloc[0]
    poly_utm = gdf_2023.to_crs('EPSG:32645').geometry.iloc[0]
    b = lk['bbox']
    minx, miny, maxx, maxy = poly_4326.bounds
    inside = (minx >= b[0] and miny >= b[1] and maxx <= b[2] and maxy <= b[3])
    print(f"Lake: {lk['display_name']:<20} | GLO 2023 Area: {poly_utm.area/1e6:.4f} km2 | Old Hist: {lk['old_hist']:.2f} km2 | Poly Bounds: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}) | Inside BBox: {inside}")
