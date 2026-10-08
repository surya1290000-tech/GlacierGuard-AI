import geopandas as gpd
import pathlib
import rasterio.features
from rasterio.transform import from_bounds
import numpy as np

glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'

# Load Imja polygon for 2023
gdf = gpd.read_file(s2_annual, where="GLO_ID = 'GLO_86.92845_27.89838'")
gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023].to_crs('EPSG:32645') # UTM Zone 45N
poly_utm = gdf_2023.geometry.iloc[0]
print(f'UTM Area (km2): {poly_utm.area / 1e6:.4f}')

# Test buffers in UTM (meters)
b0 = poly_utm
b50 = poly_utm.buffer(50)
b100 = poly_utm.buffer(100)
b200 = poly_utm.buffer(200)

print(f'Buffer 0m area: {b0.area / 1e6:.4f} km2')
print(f'Buffer 50m area: {b50.area / 1e6:.4f} km2')
print(f'Buffer 100m area: {b100.area / 1e6:.4f} km2')
print(f'Buffer 200m area: {b200.area / 1e6:.4f} km2')

gdf_buffers = gpd.GeoDataFrame({
    'buffer_m': [0, 50, 100, 200],
    'geometry': [b0, b50, b100, b200]
}, crs='EPSG:32645').to_crs('EPSG:4326')

# Rasterize onto 512x512 grid over BBOX [86.900, 27.885, 86.955, 27.930]
# west, south, east, north
bbox = [86.900, 27.885, 86.955, 27.930]
w, h = 512, 512
# from_bounds(west, south, east, north, width, height)
# note: rasterio transform standard is west, south, east, north
transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

# Compute pixel area in km2 using projected UTM coordinates
# In UTM, each pixel has a true ground size:
aoi_poly_4326 = gpd.GeoSeries.from_wkt([f'POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))'], crs='EPSG:4326')
aoi_poly_utm = aoi_poly_4326.to_crs('EPSG:32645').iloc[0]
aoi_utm_area_km2 = aoi_poly_utm.area / 1e6
pixel_area_km2 = aoi_utm_area_km2 / (w * h)
print(f'AOI UTM area: {aoi_utm_area_km2:.4f} km2, pixel area: {pixel_area_km2:.8f} km2')

for idx, row in gdf_buffers.iterrows():
    mask = rasterio.features.rasterize(
        [(row['geometry'], 1)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    px_count = mask.sum()
    calc_area = px_count * pixel_area_km2
    print(f'Buffer {row["buffer_m"]}m: {px_count} pixels, raster area: {calc_area:.4f} km2')
