import geopandas as gpd
import pathlib
import pyproj
from shapely.ops import transform
from shapely.geometry import box

glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'

gdf_sample = gpd.read_file(s2_annual, rows=1)
target_crs = gdf_sample.crs
wgs84_to_target = pyproj.Transformer.from_crs('EPSG:4326', target_crs, always_xy=True).transform

def search_lakes_near(lon, lat, delta=0.08):
    b = [lon - delta, lat - delta, lon + delta, lat + delta]
    w_box = box(b[0], b[1], b[2], b[3])
    t_box = transform(wgs84_to_target, w_box)
    gdf_sub = gpd.read_file(s2_annual, bbox=t_box.bounds)
    gdf_2023 = gdf_sub[gdf_sub['AREA_YEAR'] == 2023].sort_values('AREA', ascending=False)
    print(f"\n--- Lakes near Lon {lon}, Lat {lat} (count: {len(gdf_2023)}) ---")
    for _, r in gdf_2023.head(10).iterrows():
        print(f"GLO_ID: {r['GLO_ID']}, Area: {r['AREA']:.4f} km2, Lon: {r['LONGITUDE']:.5f}, Lat: {r['LATITUDE']:.5f}, Basin: {r.get('BASIN', '')}")

# Thulagi ~ 84.50, 28.50
search_lakes_near(84.50, 28.50)

# Sabai Tsho ~ 86.60, 27.80
search_lakes_near(86.60, 27.80)

# Lower Barun ~ 87.10, 27.83
search_lakes_near(87.10, 27.83)

# Tsho Rolpa ~ 86.49, 27.87
search_lakes_near(86.49, 27.87)
