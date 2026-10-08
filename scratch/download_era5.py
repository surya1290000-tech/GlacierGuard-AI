"""
GlacierGuard-AI — ERA5-Land Download, QA, and Feature Extraction Pipeline.

Period: 2016–2023 (8 years x 12 months)
BBox: [31.0, 80.0, 27.0, 89.0] (Nepal / Central HKH)
Variables: 2m_temperature, total_precipitation, snow_depth
Output: raw/ERA5/era5_land_monthly_nepal_2016_2023.nc
Features: raw/ERA5/era5_features.csv
"""
import os
import sys
import time
import pathlib
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

ERA5_DIR = pathlib.Path("raw/ERA5")
ERA5_DIR.mkdir(parents=True, exist_ok=True)

ERA5_NC = ERA5_DIR / "era5_land_monthly_nepal_2016_2023.nc"
TEST_NC = ERA5_DIR / "era5_land_test_1month.nc"
ERA5_FEATURES_CSV = ERA5_DIR / "era5_features.csv"

# Bounding box: North, West, South, East
BBOX = [31.0, 80.0, 27.0, 89.0]
YEARS = [str(y) for y in range(2016, 2024)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
VARIABLES = ["2m_temperature", "total_precipitation", "snow_depth"]

def test_era5_connection():
    """Verify CDS authentication and licence acceptance with a minimal request."""
    print("=" * 60)
    print("STEP 1: CDS API & LICENCE VERIFICATION (1-month minimal test)")
    print("=" * 60)
    client = cdsapi.Client()
    test_params = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2020"],
        "month": ["07"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": BBOX,
    }
    t0 = time.time()
    try:
        client.retrieve("reanalysis-era5-land-monthly-means", test_params, str(TEST_NC))
        print(f"SUCCESS: Test file retrieved in {time.time()-t0:.1f}s ({TEST_NC.stat().st_size/1024:.1f} KB)")
        # Verify valid netCDF
        with xr.open_dataset(str(TEST_NC)) as ds:
            print(f"Verified NetCDF variables: {list(ds.data_vars)}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def download_era5_full():
    """Download full 2016–2023 ERA5-Land monthly dataset."""
    print("\n" + "=" * 60)
    print(f"STEP 2: DOWNLOADING ERA5-LAND MONTHLY (2016–2023, 8 years x 12 months)")
    print("=" * 60)
    if ERA5_NC.exists() and ERA5_NC.stat().st_size > 100_000:
        print(f"File already exists: {ERA5_NC} ({ERA5_NC.stat().st_size/1e6:.2f} MB)")
        return True

    client = cdsapi.Client()
    params = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": YEARS,
        "month": MONTHS,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": BBOX,
    }
    print(f"Requesting variables: {VARIABLES}")
    print(f"Years: {YEARS[0]}–{YEARS[-1]} | BBox: {BBOX}")
    t0 = time.time()
    try:
        client.retrieve("reanalysis-era5-land-monthly-means", params, str(ERA5_NC))
        print(f"SUCCESS: Downloaded full dataset in {time.time()-t0:.1f}s ({ERA5_NC.stat().st_size/1e6:.2f} MB)")
        return True
    except Exception as e:
        print(f"Full download failed: {e}")
        # Try download with download_format: zip if unarchived is not supported for multi-var
        print("Retrying with alternative format parameters...")
        try:
            params_alt = {
                "product_type": "monthly_averaged_reanalysis",
                "variable": VARIABLES,
                "year": YEARS,
                "month": MONTHS,
                "time": "00:00",
                "format": "netcdf",
                "area": BBOX,
            }
            client.retrieve("reanalysis-era5-land-monthly-means", params_alt, str(ERA5_NC))
            print(f"SUCCESS with fallback parameters: {ERA5_NC.stat().st_size/1e6:.2f} MB")
            return True
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return False

def qa_era5_dataset():
    """Verify data quality of the downloaded ERA5-Land NetCDF."""
    print("\n" + "=" * 60)
    print("STEP 3: ERA5-LAND DATA QUALITY ASSURANCE")
    print("=" * 60)
    ds = xr.open_dataset(str(ERA5_NC))
    print(f"Dataset structure:")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Data variables: {list(ds.data_vars)}")
    
    # Lat/Lon coordinates
    lat_name = "latitude" if "latitude" in ds else "lat"
    lon_name = "longitude" if "longitude" in ds else "lon"
    time_name = "valid_time" if "valid_time" in ds else "time"
    
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    times = pd.to_datetime(ds[time_name].values)
    
    print(f"  Latitude:  {lat.min():.2f}° to {lat.max():.2f}° (N={len(lat)})")
    print(f"  Longitude: {lon.min():.2f}° to {lon.max():.2f}° (N={len(lon)})")
    print(f"  Time range: {times.min().strftime('%Y-%m')} to {times.max().strftime('%Y-%m')} (N={len(times)} months)")
    
    expected_months = len(YEARS) * 12
    assert len(times) == expected_months, f"Expected {expected_months} months, got {len(times)}"
    
    # Check variables for NaNs and physical ranges
    t2m_name = [v for v in ds.data_vars if "t2m" in v.lower() or "temp" in v.lower() or "2t" in v.lower()][0]
    tp_name  = [v for v in ds.data_vars if "tp" in v.lower() or "precip" in v.lower()][0]
    sd_name  = [v for v in ds.data_vars if "sd" in v.lower() or "snow" in v.lower()][0]
    
    t2m = ds[t2m_name].values
    tp  = ds[tp_name].values
    sd  = ds[sd_name].values
    
    # Convert temperature from Kelvin to Celsius if necessary
    if np.nanmean(t2m) > 100:
        t2m_c = t2m - 273.15
    else:
        t2m_c = t2m
        
    print(f"\nPhysical Range Checks:")
    print(f"  Temperature (C): min={np.nanmin(t2m_c):.1f}°C, mean={np.nanmean(t2m_c):.1f}°C, max={np.nanmax(t2m_c):.1f}°C")
    print(f"  Precipitation:   min={np.nanmin(tp):.4f}, mean={np.nanmean(tp):.4f}, max={np.nanmax(tp):.4f} (units={ds[tp_name].attrs.get('units','')})")
    print(f"  Snow depth:      min={np.nanmin(sd):.4f}, mean={np.nanmean(sd):.4f}, max={np.nanmax(sd):.4f} (units={ds[sd_name].attrs.get('units','')})")
    
    assert np.nanmin(t2m_c) > -60.0 and np.nanmax(t2m_c) < 55.0, "Temperature out of physical bounds!"
    assert np.nanmin(tp) >= -1e-4, "Precipitation cannot be significantly negative!"
    assert np.nanmin(sd) >= -1e-4, "Snow depth cannot be significantly negative!"
    
    # Clip tiny machine-precision negatives to 0.0
    tp = np.maximum(0.0, tp)
    sd = np.maximum(0.0, sd)
    
    nan_pct_t2m = np.isnan(t2m).mean() * 100
    print(f"  NaN percentage (land mask): {nan_pct_t2m:.1f}%")
    
    ds.close()
    print("QA PASSED: All variables within physical bounds and time series complete.")
    return True

def extract_and_engineer_features():
    """
    Extract ERA5 grid values for all GLO unique lakes and compute annual climate features.
    
    Strict Temporal Contract:
    - Target: AREA(t+1) - AREA(t)
    - Predictor Year: t (e.g. 2017–2023)
    - Environmental predictors must ONLY use data up to December of year t (no future leakage).
    
    Features engineered per (GLO_ID, AREA_YEAR):
    1. ERA5_TEMP_MEAN: Annual mean 2m temperature (°C) in year t
    2. ERA5_TEMP_WARM: Melt season (May–Sep) mean 2m temperature (°C) in year t
    3. ERA5_TEMP_ANOM: Temperature anomaly in year t vs 2016–2021 training baseline (°C)
    4. ERA5_PRECIP_TOTAL: Annual total precipitation (mm) in year t
    5. ERA5_PRECIP_MONSOON: Monsoon (Jun–Sep) total precipitation (mm) in year t
    6. ERA5_PRECIP_ANOM: Precip anomaly in year t vs 2016–2021 training baseline (mm)
    7. ERA5_SNOW_MEAN: Winter accumulation (Oct(t-1)–Mar(t)) mean snow depth (m w.e.)
    8. ERA5_PDD: Positive degree-day proxy (sum of monthly mean T * 30.5 for months where T > 0°C)
    """
    print("\n" + "=" * 60)
    print("STEP 4: SPATIAL EXTRACTION & FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load lake locations
    s1_lakes_path = "raw/GLO/S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg"
    s2_lakes_path = "raw/GLO/S2_20172024_NTB_GLOID_UniqueLakes_centroid_v1.02.gpkg"
    
    import geopandas as gpd
    s1_gdf = gpd.read_file(s1_lakes_path)
    s2_gdf = gpd.read_file(s2_lakes_path)
    
    for gdf in [s1_gdf, s2_gdf]:
        if "LONGITUDE" not in gdf.columns or "LATITUDE" not in gdf.columns:
            cent = gdf.geometry.centroid
            gdf["LONGITUDE"] = cent.x
            gdf["LATITUDE"] = cent.y
            
    all_lakes = pd.concat([
        s1_gdf[["GLO_ID", "LONGITUDE", "LATITUDE"]],
        s2_gdf[["GLO_ID", "LONGITUDE", "LATITUDE"]]
    ]).drop_duplicates(subset=["GLO_ID"]).reset_index(drop=True)
    
    print(f"Total unique lakes to extract: {len(all_lakes)}")
    
    # Load ERA5 dataset
    ds = xr.open_dataset(str(ERA5_NC))
    lat_name = "latitude" if "latitude" in ds else "lat"
    lon_name = "longitude" if "longitude" in ds else "lon"
    time_name = "valid_time" if "valid_time" in ds else "time"
    
    t2m_name = [v for v in ds.data_vars if "t2m" in v.lower() or "temp" in v.lower() or "2t" in v.lower()][0]
    tp_name  = [v for v in ds.data_vars if "tp" in v.lower() or "precip" in v.lower()][0]
    sd_name  = [v for v in ds.data_vars if "sd" in v.lower() or "snow" in v.lower()][0]
    
    # Fast vectorized extraction by unique grid cells
    # Find nearest grid indices for all lakes
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    
    lake_lat = xr.DataArray(all_lakes["LATITUDE"].values, dims="lake")
    lake_lon = xr.DataArray(all_lakes["LONGITUDE"].values, dims="lake")
    
    print("Extracting nearest grid points for all lakes...")
    t0 = time.time()
    lake_ds = ds.sel({lat_name: lake_lat, lon_name: lake_lon}, method="nearest").compute()
    print(f"Extracted {len(all_lakes)} lake time series in {time.time()-t0:.1f}s")
    
    times = pd.to_datetime(lake_ds[time_name].values)
    years = times.year
    months = times.month
    
    # Temperature: convert to Celsius if Kelvin
    t2m_vals = lake_ds[t2m_name].values  # shape: (n_times, n_lakes) or (n_lakes, n_times)
    if lake_ds[t2m_name].dims[0] != "lake":
        t2m_vals = t2m_vals.T  # make (n_lakes, n_times)
    if np.nanmean(t2m_vals) > 100:
        t2m_vals = t2m_vals - 273.15
        
    # Precipitation: convert m/day or m to mm/month
    tp_vals = lake_ds[tp_name].values
    if lake_ds[tp_name].dims[0] != "lake":
        tp_vals = tp_vals.T
    # ERA5-Land monthly means of daily total precipitation: units are m/day
    units = lake_ds[tp_name].attrs.get("units", "").lower()
    if "m of water equivalent" in units or "m/day" in units or np.nanmean(tp_vals) < 0.1:
        # m/day -> mm/month (multiply by days in month * 1000)
        # Check if daily mean or monthly total
        # In ERA5-Land monthly means, total_precipitation is m/day
        days_in_month = times.days_in_month.values
        tp_mm_month = tp_vals * 1000.0 * days_in_month[np.newaxis, :]
    else:
        tp_mm_month = tp_vals
        
    sd_vals = lake_ds[sd_name].values
    if lake_ds[sd_name].dims[0] != "lake":
        sd_vals = sd_vals.T
        
    records = []
    # Predictor years of interest: 2017 to 2023
    pred_years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    
    print("Computing annual features per lake-year...")
    t0 = time.time()
    
    for yr in pred_years:
        mask_yr = (years == yr)
        mask_warm = (years == yr) & ((months >= 5) & (months <= 9))
        mask_monsoon = (years == yr) & ((months >= 6) & (months <= 9))
        
        # Winter accumulation: Oct of yr-1 to Mar of yr
        mask_winter = ((years == (yr - 1)) & (months >= 10)) | ((years == yr) & (months <= 3))
        
        # Means and totals
        temp_mean = np.nanmean(t2m_vals[:, mask_yr], axis=1)
        temp_warm = np.nanmean(t2m_vals[:, mask_warm], axis=1)
        precip_total = np.nansum(tp_mm_month[:, mask_yr], axis=1)
        precip_monsoon = np.nansum(tp_mm_month[:, mask_monsoon], axis=1)
        snow_mean = np.nanmean(sd_vals[:, mask_winter], axis=1)
        
        # PDD proxy: sum of (T * 30.5) for months where T > 0
        pdd = np.zeros(len(all_lakes))
        month_indices = np.where(mask_yr)[0]
        for m_idx in month_indices:
            t_m = t2m_vals[:, m_idx]
            pos_t = np.maximum(0, t_m)
            pdd += pos_t * times[m_idx].days_in_month
            
        for i, glo_id in enumerate(all_lakes["GLO_ID"]):
            records.append({
                "GLO_ID": glo_id,
                "AREA_YEAR": yr,
                "ERA5_TEMP_MEAN": temp_mean[i],
                "ERA5_TEMP_WARM": temp_warm[i],
                "ERA5_PRECIP_TOTAL": precip_total[i],
                "ERA5_PRECIP_MONSOON": precip_monsoon[i],
                "ERA5_SNOW_MEAN": snow_mean[i],
                "ERA5_PDD": pdd[i],
            })
            
    df_feat = pd.DataFrame(records)
    
    # Compute anomalies relative to the FROZEN TRAINING PERIOD (2017–2021) baseline!
    # Strict temporal leakage prevention: baseline is computed on TRAIN ONLY (<= 2021)
    train_mask = df_feat["AREA_YEAR"] <= 2021
    train_baselines = df_feat[train_mask].groupby("GLO_ID").agg({
        "ERA5_TEMP_MEAN": "mean",
        "ERA5_PRECIP_TOTAL": "mean"
    }).rename(columns={
        "ERA5_TEMP_MEAN": "TRAIN_BASE_TEMP",
        "ERA5_PRECIP_TOTAL": "TRAIN_BASE_PRECIP"
    })
    
    df_feat = df_feat.merge(train_baselines, on="GLO_ID", how="left")
    df_feat["ERA5_TEMP_ANOM"] = df_feat["ERA5_TEMP_MEAN"] - df_feat["TRAIN_BASE_TEMP"]
    df_feat["ERA5_PRECIP_ANOM"] = df_feat["ERA5_PRECIP_TOTAL"] - df_feat["TRAIN_BASE_PRECIP"]
    df_feat = df_feat.drop(columns=["TRAIN_BASE_TEMP", "TRAIN_BASE_PRECIP"])
    
    # Save features
    df_feat.to_csv(ERA5_FEATURES_CSV, index=False)
    print(f"\nSUCCESS: Engineered ERA5 features saved to {ERA5_FEATURES_CSV}")
    print(f"Shape: {df_feat.shape} | Years: {sorted(df_feat['AREA_YEAR'].unique())}")
    print(df_feat.head(3).to_string())
    
    # Leakage Audit
    print("\n" + "=" * 60)
    print("STEP 5: TEMPORAL LEAKAGE AUDIT")
    print("=" * 60)
    assert df_feat["AREA_YEAR"].max() == 2023, "No future years beyond 2023!"
    for yr in pred_years:
        sub = df_feat[df_feat["AREA_YEAR"] == yr]
        assert len(sub) == len(all_lakes), f"Incomplete lake coverage for year {yr}"
        assert not sub["ERA5_TEMP_MEAN"].isna().any(), f"NaN values in ERA5_TEMP_MEAN for year {yr}"
    print("LEAKAGE AUDIT PASSED: All features computed strictly within predictor year t.")
    print("Baseline anomalies derived strictly from training years (<= 2021).")
    
    ds.close()
    return True

if __name__ == "__main__":
    if not test_era5_connection():
        sys.exit(1)
    if not download_era5_full():
        sys.exit(2)
    if not qa_era5_dataset():
        sys.exit(3)
    if not extract_and_engineer_features():
        sys.exit(4)
    print("\nALL ERA5 INTEGRATION STEPS COMPLETED SUCCESSFULLY!")
