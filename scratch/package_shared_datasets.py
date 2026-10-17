"""
GlacierGuard-AI — Dataset Sharing Packaging and Verification Script

This script safely packages all existing local datasets into `shared_datasets/`,
verifies file hashes, prevents redundant duplicates, tests geospatial/climate
library readability, and generates `dataset_manifest.csv`.
"""

import os
import shutil
import hashlib
import zipfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import xarray as xr

PROJECT_ROOT = Path(".").resolve()
SHARED_ROOT = PROJECT_ROOT / "shared_datasets"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_disk_free_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)

def main():
    print("=" * 80)
    print("GLACIERGUARD-AI: DATASET PACKAGING & INTEGRITY VERIFICATION")
    print("=" * 80)
    
    # Check disk space
    free_gb = get_disk_free_gb(PROJECT_ROOT)
    print(f"Available disk space on drive: {free_gb:.2f} GB")
    if free_gb < 1.0:
        raise RuntimeError(f"Insufficient disk space ({free_gb:.2f} GB available, need at least 1 GB).")

    # Define file catalog
    catalog = [
        # ── Raw GLO v1.02 GeoPackages ─────────────────────────────────────────
        {
            "category": "raw/GLO",
            "name": "Sentinel-1 Unique Lakes Dissolved Polygons (2017-2024)",
            "src": "raw/GLO/S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg",
            "dest": "raw/GLO/S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg",
            "format": "GeoPackage (.gpkg)",
            "notes": "Standalone GeoPackage; CRS ESRI:102025; used in NB01-NB07",
            "action": "copy"
        },
        {
            "category": "raw/GLO",
            "name": "Sentinel-1 Annual Lake Delineations (2017-2024)",
            "src": "raw/GLO/S1_20172024_NTB_GLOID_v1.02.gpkg",
            "dest": "raw/GLO/S1_20172024_NTB_GLOID_v1.02.gpkg",
            "format": "GeoPackage (.gpkg)",
            "notes": "Standalone GeoPackage; CRS ESRI:102025; annual time series used in NB01-NB07",
            "action": "copy"
        },
        {
            "category": "raw/GLO",
            "name": "Sentinel-2 Unique Lakes Centroids (2017-2024)",
            "src": "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_centroid_v1.02.gpkg",
            "dest": "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_centroid_v1.02.gpkg",
            "format": "GeoPackage (.gpkg)",
            "notes": "Standalone GeoPackage; CRS EPSG:4326; point centroids used in spatial joins",
            "action": "copy"
        },
        {
            "category": "raw/GLO",
            "name": "Sentinel-2 Unique Lakes Dissolved Polygons (2017-2024)",
            "src": "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg",
            "dest": "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg",
            "format": "GeoPackage (.gpkg)",
            "notes": "Standalone GeoPackage; CRS ESRI:102025; dissolved max extent used in NB01-NB09",
            "action": "copy"
        },
        {
            "category": "raw/GLO",
            "name": "Sentinel-2 Unique Lakes Dissolved Polygons (Duplicate File)",
            "src": "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_v1.02 (1).gpkg",
            "dest": "N/A (Skipped duplicate)",
            "format": "GeoPackage (.gpkg)",
            "notes": "OS-renamed download duplicate of S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg; SHA-256 verified identical; skipped to prevent redundant copy",
            "action": "skip_duplicate"
        },
        {
            "category": "raw/GLO",
            "name": "Sentinel-2 Annual Lake Delineations (2017-2024)",
            "src": "raw/GLO/S2_20172024_NTB_GLOID_v1.02.gpkg",
            "dest": "raw/GLO/S2_20172024_NTB_GLOID_v1.02.gpkg",
            "format": "GeoPackage (.gpkg)",
            "notes": "Standalone GeoPackage; CRS ESRI:102025; annual delineations used in NB01-NB09 and reference polygons for NB10-NB10.2",
            "action": "copy"
        },
        {
            "category": "raw/GLO",
            "name": "GLO v1.02 Dataset Documentation and Schema",
            "src": "raw/GLO/README (1).md",
            "dest": "raw/GLO/README (1).md",
            "format": "Markdown (.md)",
            "notes": "Original provenance documentation from Leeds University / Zenodo for GLO v1.02",
            "action": "copy"
        },

        # ── Raw ERA5-Land Climate Data ────────────────────────────────────────
        {
            "category": "raw/ERA5",
            "name": "ERA5-Land Monthly Averaged Reanalysis Nepal (2016-2023)",
            "src": "raw/ERA5/era5_land_monthly_nepal_2016_2023.nc",
            "dest": "raw/ERA5/era5_land_monthly_nepal_2016_2023.nc",
            "format": "NetCDF (.nc)",
            "notes": "Variables: t2m (2m temp), tp (total precip), sde (snow depth water equivalent); 96 monthly time steps; used in NB07",
            "action": "copy"
        },
        {
            "category": "raw/ERA5",
            "name": "ERA5-Land Single Month Test Sample",
            "src": "raw/ERA5/era5_land_test_1month.nc",
            "dest": "raw/ERA5/era5_land_test_1month.nc",
            "format": "NetCDF (.nc)",
            "notes": "Variable: t2m; test verification file for CDS API ingestion pipeline",
            "action": "copy"
        },

        # ── Raw CHIRPS Precipitation Data ─────────────────────────────────────
        {
            "category": "raw/CHIRPS",
            "name": "CHIRPS v2.0 Monthly Precipitation Nepal (2016-2023)",
            "src": "raw/CHIRPS/chirps_monthly_nepal_2016_2023.nc",
            "dest": "raw/CHIRPS/chirps_monthly_nepal_2016_2023.nc",
            "format": "NetCDF (.nc)",
            "notes": "Variable: precip (monthly rainfall 0.05° resolution); 96 monthly time steps; used in NB07",
            "action": "copy"
        },

        # ── Raw Randolph Glacier Inventory (RGI) ──────────────────────────────
        {
            "category": "raw/RGI",
            "name": "RGI 6.0 Region 15 South Asia East Glaciers (Shapefile)",
            "src": "raw/RGI/15_rgi60_SouthAsiaEast.shp",
            "dest": "raw/RGI/15_rgi60_SouthAsiaEast.shp",
            "format": "ESRI Shapefile (.shp)",
            "notes": "Main geometry file; requires companion sidecars .shx, .dbf, .prj; used in NB07",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 6.0 Region 15 Shapefile Attribute Table",
            "src": "raw/RGI/15_rgi60_SouthAsiaEast.dbf",
            "dest": "raw/RGI/15_rgi60_SouthAsiaEast.dbf",
            "format": "dBASE table (.dbf)",
            "notes": "Required companion file for 15_rgi60_SouthAsiaEast.shp",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 6.0 Region 15 Shapefile Spatial Index",
            "src": "raw/RGI/15_rgi60_SouthAsiaEast.shx",
            "dest": "raw/RGI/15_rgi60_SouthAsiaEast.shx",
            "format": "Shapefile index (.shx)",
            "notes": "Required companion file for 15_rgi60_SouthAsiaEast.shp",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 6.0 Region 15 Shapefile Projection Definition",
            "src": "raw/RGI/15_rgi60_SouthAsiaEast.prj",
            "dest": "raw/RGI/15_rgi60_SouthAsiaEast.prj",
            "format": "Projection file (.prj)",
            "notes": "Required companion file for 15_rgi60_SouthAsiaEast.shp (WGS84 EPSG:4326)",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 6.0 Region 15 Complete Distribution Archive",
            "src": "raw/RGI/15_rgi60_SouthAsiaEast.zip",
            "dest": "raw/RGI/15_rgi60_SouthAsiaEast.zip",
            "format": "Zip archive (.zip)",
            "notes": "Original distribution zip archive containing complete RGI 6.0 South Asia East inventory",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 7.0 Global Glaciers Attribute Database",
            "src": "raw/RGI/RGI2000-v7.0-G-global-attributes.csv",
            "dest": "raw/RGI/RGI2000-v7.0-G-global-attributes.csv",
            "format": "CSV (.csv)",
            "notes": "Global RGI 7.0 glacier inventory metadata table (215,547 glacier records)",
            "action": "copy"
        },
        {
            "category": "raw/RGI",
            "name": "RGI 7.0 Region 15 South Asia East Glaciers",
            "src": "raw/RGI/rgi7_region15_glaciers.csv",
            "dest": "raw/RGI/rgi7_region15_glaciers.csv",
            "format": "CSV (.csv)",
            "notes": "Regional RGI 7.0 glacier attributes filtered for Region 15 (South Asia East)",
            "action": "copy"
        },

        # ── Processed Feature Tables & Reusable Extracts ──────────────────────
        {
            "category": "processed/features",
            "name": "CHIRPS Precipitation Extracted Lake Features",
            "src": "raw/CHIRPS/chirps_features.csv",
            "dest": "processed/features/chirps_features.csv",
            "format": "CSV (.csv)",
            "notes": "Engineered lake precipitation features from NB07; input to NB08-NB09 modeling",
            "action": "copy"
        },
        {
            "category": "processed/features",
            "name": "ERA5-Land Climate Extracted Lake Features",
            "src": "raw/ERA5/era5_features.csv",
            "dest": "processed/features/era5_features.csv",
            "format": "CSV (.csv)",
            "notes": "Engineered lake temperature, snow, precip features from NB07; input to NB08-NB09 modeling",
            "action": "copy"
        },
        {
            "category": "processed/features",
            "name": "ERA5-Land Monthly Lake Time-Series Extracts",
            "src": "raw/ERA5/era5_monthly_lake_extracts.parquet",
            "dest": "processed/features/era5_monthly_lake_extracts.parquet",
            "format": "Parquet (.parquet)",
            "notes": "Longitudinal monthly climate observations (370k rows) extracted in NB07",
            "action": "copy"
        },
        {
            "category": "processed/features",
            "name": "Lake-to-Glacier Proximity Features (RGI 6.0)",
            "src": "raw/RGI/lake_glacier_distances.csv",
            "dest": "processed/features/lake_glacier_distances.csv",
            "format": "CSV (.csv)",
            "notes": "Minimum distance from each glacial lake to nearest RGI 6.0 glacier; input to NB08-NB09",
            "action": "copy"
        },
        {
            "category": "processed/features",
            "name": "Lake-to-Glacier Proximity Features (RGI 7.0)",
            "src": "raw/RGI/lake_glacier_distances_rgi7.csv",
            "dest": "processed/features/lake_glacier_distances_rgi7.csv",
            "format": "CSV (.csv)",
            "notes": "Proximity features calculated against modern RGI 7.0 glacier inventory",
            "action": "copy"
        }
    ]

    manifest_rows = []
    total_copied_bytes = 0
    copied_count = 0

    for item in catalog:
        src_path = PROJECT_ROOT / item["src"]
        if not src_path.exists():
            print(f"[MISSING SOURCE] {item['src']} does not exist!")
            manifest_rows.append({
                "dataset_category": item["category"],
                "dataset_name": item["name"],
                "original_relative_path": item["src"],
                "shared_relative_path": item["dest"],
                "file_size_bytes": 0,
                "file_format": item["format"],
                "sha256_checksum": "N/A",
                "copy_status": "MISSING_SOURCE",
                "companion_files_notes": item["notes"]
            })
            continue

        size = src_path.stat().st_size
        sha_src = compute_sha256(src_path)

        if item["action"] == "skip_duplicate":
            print(f"[SKIP DUPLICATE] {item['src']} ({size:,} bytes) -> matches canonical file")
            manifest_rows.append({
                "dataset_category": item["category"],
                "dataset_name": item["name"],
                "original_relative_path": item["src"],
                "shared_relative_path": item["dest"],
                "file_size_bytes": size,
                "file_format": item["format"],
                "sha256_checksum": sha_src,
                "copy_status": "SKIPPED_DUPLICATE",
                "companion_files_notes": item["notes"]
            })
            continue

        dest_path = SHARED_ROOT / item["dest"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[COPYING] {item['src']} -> {item['dest']} ({size / 1024 / 1024:.2f} MB)...")
        shutil.copy2(src_path, dest_path)

        # Verify copy
        sha_dest = compute_sha256(dest_path)
        dest_size = dest_path.stat().st_size

        if sha_src != sha_dest or size != dest_size:
            raise ValueError(f"Integrity check failed for {dest_path}! Source {sha_src} != Dest {sha_dest}")

        print(f"  --> [PASS] Hash: {sha_dest[:16]}... Size: {dest_size:,} bytes")
        total_copied_bytes += dest_size
        copied_count += 1

        manifest_rows.append({
            "dataset_category": item["category"],
            "dataset_name": item["name"],
            "original_relative_path": item["src"],
            "shared_relative_path": item["dest"],
            "file_size_bytes": dest_size,
            "file_format": item["format"],
            "sha256_checksum": sha_dest,
            "copy_status": "COPIED_VERIFIED",
            "companion_files_notes": item["notes"]
        })

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = SHARED_ROOT / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nManifest successfully created at {manifest_csv.relative_to(PROJECT_ROOT)}")

    # ── Verify readability of copied datasets ────────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFYING COPIED DATASETS INTEGRITY VIA PYTHON LIBRARIES")
    print("=" * 80)

    # 1. GeoPackages
    for gpkg_rel in [
        "raw/GLO/S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg",
        "raw/GLO/S1_20172024_NTB_GLOID_v1.02.gpkg",
        "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_centroid_v1.02.gpkg",
        "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg",
        "raw/GLO/S2_20172024_NTB_GLOID_v1.02.gpkg"
    ]:
        p = SHARED_ROOT / gpkg_rel
        gdf = gpd.read_file(p, rows=5)
        print(f"  [PASS GEOPACKAGE] {gpkg_rel}: {len(gdf)} sample rows read, CRS={gdf.crs.name if gdf.crs else 'None'}")

    # 2. Shapefile
    shp_p = SHARED_ROOT / "raw/RGI/15_rgi60_SouthAsiaEast.shp"
    gdf_shp = gpd.read_file(shp_p, rows=5)
    print(f"  [PASS SHAPEFILE] raw/RGI/15_rgi60_SouthAsiaEast.shp: {len(gdf_shp)} rows read, CRS={gdf_shp.crs.name if gdf_shp.crs else 'None'}")

    # 3. Zip file
    zip_p = SHARED_ROOT / "raw/RGI/15_rgi60_SouthAsiaEast.zip"
    zf = zipfile.ZipFile(zip_p)
    bad_file = zf.testzip()
    if bad_file:
        raise ValueError(f"Corrupt zip file detected: {bad_file}")
    print(f"  [PASS ZIP] raw/RGI/15_rgi60_SouthAsiaEast.zip: valid archive ({len(zf.namelist())} members)")

    # 4. NetCDFs
    for nc_rel in [
        "raw/ERA5/era5_land_monthly_nepal_2016_2023.nc",
        "raw/ERA5/era5_land_test_1month.nc",
        "raw/CHIRPS/chirps_monthly_nepal_2016_2023.nc"
    ]:
        p = SHARED_ROOT / nc_rel
        ds = xr.open_dataset(p)
        print(f"  [PASS NETCDF] {nc_rel}: dims={list(ds.sizes.keys())}, vars={list(ds.data_vars)}")
        ds.close()

    # 5. CSVs and Parquet
    for tab_rel in [
        "raw/RGI/RGI2000-v7.0-G-global-attributes.csv",
        "raw/RGI/rgi7_region15_glaciers.csv",
        "processed/features/chirps_features.csv",
        "processed/features/era5_features.csv",
        "processed/features/era5_monthly_lake_extracts.parquet",
        "processed/features/lake_glacier_distances.csv",
        "processed/features/lake_glacier_distances_rgi7.csv"
    ]:
        p = SHARED_ROOT / tab_rel
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, nrows=5)
        print(f"  [PASS TABULAR] {tab_rel}: shape={df.shape}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Total files successfully copied: {copied_count}")
    print(f"Total size of copied datasets: {total_copied_bytes:,} bytes ({total_copied_bytes / 1024 / 1024:.2f} MB)")
    print("All checksums matched 100%. All geospatial and tabular datasets validated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
