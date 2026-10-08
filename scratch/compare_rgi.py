import pandas as pd
import geopandas as gpd

# RGI 7.0 (CSV)
r7 = pd.read_csv("raw/RGI/rgi7_region15_glaciers.csv")
print("=== RGI 7.0 Region 15 ===")
print(f"  Glaciers: {len(r7)}")
print(f"  Lat: {r7['cenlat'].min():.3f} to {r7['cenlat'].max():.3f}")
print(f"  Lon: {r7['cenlon'].min():.3f} to {r7['cenlon'].max():.3f}")
print(f"  Area: {r7['area_km2'].sum():.1f} km2 total")
print(f"  Columns (first 15): {list(r7.columns[:15])}")

# RGI 6.0 (Shapefile)
r6 = gpd.read_file("raw/RGI/15_rgi60_SouthAsiaEast.shp")
print(f"\n=== RGI 6.0 Region 15 ===")
print(f"  Glaciers: {len(r6)}")
print(f"  Columns (first 15): {list(r6.columns[:15])}")
cent = r6.geometry.centroid
print(f"  Lat: {cent.y.min():.3f} to {cent.y.max():.3f}")
print(f"  Lon: {cent.x.min():.3f} to {cent.x.max():.3f}")
if "Area" in r6.columns:
    print(f"  Area: {r6['Area'].sum():.1f} km2 total")

print(f"\n=== Difference ===")
print(f"  RGI 7.0: {len(r7)} glaciers")
print(f"  RGI 6.0: {len(r6)} glaciers")
print(f"  Delta: {len(r7) - len(r6):+d}")
