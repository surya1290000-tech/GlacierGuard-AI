"""
Compute lake-glacier distances using RGI 7.0 (18,587 glaciers) vs RGI 6.0 (13,119 glaciers).
Compare statistics for Sentinel-1 and Sentinel-2 unique lakes.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
import pathlib

# Load unique lakes
s1_lakes_path = "raw/GLO/S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg"
s2_lakes_path = "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_centroid_v1.02.gpkg"

s1_gdf = gpd.read_file(s1_lakes_path)
s2_gdf = gpd.read_file(s2_lakes_path)

print(f"Loaded S1 unique lakes: {len(s1_gdf)}")
print(f"Loaded S2 unique lakes: {len(s2_gdf)}")

# Ensure centroid lat/lon
for name, gdf in [("S1", s1_gdf), ("S2", s2_gdf)]:
    if "LONGITUDE" not in gdf.columns or "LATITUDE" not in gdf.columns:
        cent = gdf.geometry.centroid
        gdf["LONGITUDE"] = cent.x
        gdf["LATITUDE"] = cent.y
    print(f"  {name} GLO_ID count: {gdf['GLO_ID'].nunique()}")

# Combine unique lakes across both sensors
all_lakes = pd.concat([
    s1_gdf[["GLO_ID", "LONGITUDE", "LATITUDE"]],
    s2_gdf[["GLO_ID", "LONGITUDE", "LATITUDE"]]
]).drop_duplicates(subset=["GLO_ID"]).reset_index(drop=True)
print(f"Total unique lakes across S1+S2: {len(all_lakes)}")

# 1. RGI 6.0
r6 = gpd.read_file("raw/RGI/15_rgi60_SouthAsiaEast.shp")
cent6 = r6.geometry.centroid
r6_coords = np.column_stack([cent6.x, cent6.y])
tree6 = cKDTree(r6_coords)

# 2. RGI 7.0
r7 = pd.read_csv("raw/RGI/rgi7_region15_glaciers.csv")
r7_coords = np.column_stack([r7["cenlon"].values, r7["cenlat"].values])
tree7 = cKDTree(r7_coords)

# Convert degree distance to approximate km:
# 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(mean_lat) ~ 97 km (at 29 deg N)
lake_coords = np.column_stack([all_lakes["LONGITUDE"].values, all_lakes["LATITUDE"].values])
mean_lat_rad = np.radians(all_lakes["LATITUDE"].mean())
km_per_deg_lon = 111.320 * np.cos(mean_lat_rad)
km_per_deg_lat = 110.574

def calc_dist_km(tree, lake_pts):
    # Query nearest point
    d_deg, idx = tree.query(lake_pts)
    # Exact haversine-like or metric projection approximation
    # For nearest neighbor index, recompute ellipsoidal/metric distance:
    glac_pts = tree.data[idx]
    dlon = (glac_pts[:, 0] - lake_pts[:, 0]) * km_per_deg_lon
    dlat = (glac_pts[:, 1] - lake_pts[:, 1]) * km_per_deg_lat
    return np.sqrt(dlon**2 + dlat**2), d_deg, idx

dist6_km, _, idx6 = calc_dist_km(tree6, lake_coords)
dist7_km, _, idx7 = calc_dist_km(tree7, lake_coords)

all_lakes["DIST_RGI6_KM"] = dist6_km
all_lakes["DIST_RGI7_KM"] = dist7_km
all_lakes["DELTA_KM"] = dist7_km - dist6_km

print("\n=== LAKE-GLACIER DISTANCE COMPARISON ===")
print("RGI 6.0 (13,119 glaciers):")
print(f"  Mean distance:   {dist6_km.mean():.2f} km")
print(f"  Median distance: {np.median(dist6_km):.2f} km")
print(f"  Min / Max:       {dist6_km.min():.2f} / {dist6_km.max():.2f} km")
print(f"  Within 1 km:     {(dist6_km < 1.0).sum()} lakes ({(dist6_km < 1.0).mean()*100:.1f}%)")
print(f"  Within 5 km:     {(dist6_km < 5.0).sum()} lakes ({(dist6_km < 5.0).mean()*100:.1f}%)")

print("\nRGI 7.0 (18,587 glaciers):")
print(f"  Mean distance:   {dist7_km.mean():.2f} km")
print(f"  Median distance: {np.median(dist7_km):.2f} km")
print(f"  Min / Max:       {dist7_km.min():.2f} / {dist7_km.max():.2f} km")
print(f"  Within 1 km:     {(dist7_km < 1.0).sum()} lakes ({(dist7_km < 1.0).mean()*100:.1f}%)")
print(f"  Within 5 km:     {(dist7_km < 5.0).sum()} lakes ({(dist7_km < 5.0).mean()*100:.1f}%)")

closer_in_r7 = (dist7_km < dist6_km).sum()
print(f"\nLakes with closer glacier found in RGI 7.0: {closer_in_r7}/{len(all_lakes)} ({closer_in_r7/len(all_lakes)*100:.1f}%)")
print(f"Average distance reduction for those lakes: {(dist6_km - dist7_km)[dist7_km < dist6_km].mean():.2f} km")

# Save updated distances with both RGI versions
out_df = all_lakes[["GLO_ID", "LONGITUDE", "LATITUDE", "DIST_RGI6_KM", "DIST_RGI7_KM"]].copy()
out_df["DIST_NEAREST_GLACIER_KM"] = out_df["DIST_RGI7_KM"]  # Primary is RGI 7.0
out_df.to_csv("raw/RGI/lake_glacier_distances_rgi7.csv", index=False)
print("\nSaved raw/RGI/lake_glacier_distances_rgi7.csv")
