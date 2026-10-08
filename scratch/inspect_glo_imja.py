import geopandas as gpd
import pathlib

glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'

# Load Imja polygon for all years
gdf = gpd.read_file(s2_annual, where="GLO_ID = 'GLO_86.92845_27.89838'")
print('Imja S2 rows count:', len(gdf))
for _, r in gdf.iterrows():
    print(f"Year: {r['AREA_YEAR']}, Area (km2 in table): {r['AREA']:.6f}, Uncertainty: {r.get('AREA_UNCERTAINTY', 'N/A')}")

gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023]
if len(gdf_2023) > 0:
    poly_proj = gdf_2023.geometry.iloc[0]
    print('Poly proj area (m2):', poly_proj.area, 'km2:', poly_proj.area / 1e6)
    gdf_4326 = gdf_2023.to_crs('EPSG:4326')
    poly_4326 = gdf_4326.geometry.iloc[0]
    print('Poly 4326 bounds:', poly_4326.bounds)
    print('Centroid 4326:', poly_4326.centroid.x, poly_4326.centroid.y)

s1_annual = glo_dir / 'S1_20172024_NTB_GLOID_v1.02.gpkg'
gdf_s1 = gpd.read_file(s1_annual, where="GLO_ID = 'GLO_86.92845_27.89838'")
print('\nImja S1 rows count:', len(gdf_s1))
for _, r in gdf_s1.iterrows():
    print(f"Year: {r['AREA_YEAR']}, Area (km2 in table): {r['AREA']:.6f}")
