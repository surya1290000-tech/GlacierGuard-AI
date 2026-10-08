"""
Small ERA5-Land test request: 1 month, 1 variable, Nepal bbox.
Verifies CDS authentication, dataset name, variable names, and download.
Uses the modern CDS API schema (Sept 2024+).
"""
import cdsapi
import pathlib
import time

out_dir = pathlib.Path("raw/ERA5")
out_dir.mkdir(parents=True, exist_ok=True)
test_file = out_dir / "era5_land_test_1month.nc"

if test_file.exists():
    test_file.unlink()

print("Connecting to CDS API (using ~/.cdsapirc)...")
client = cdsapi.Client()
print("Client created. Submitting small test request...")
print("  Dataset: reanalysis-era5-land-monthly-means")
print("  Variable: 2m_temperature")
print("  Year: 2020, Month: 07")
print("  Area: [31, 80, 27, 89] (Nepal/HKH)")

t0 = time.time()
try:
    # Modern CDS API request dictionary
    request_params = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2020"],
        "month": ["07"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [31.0, 80.0, 27.0, 89.0],  # North, West, South, East
    }
    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        request_params,
        str(test_file),
    )
    elapsed = time.time() - t0
    print(f"\nSUCCESS! Downloaded in {elapsed:.1f}s")
    print(f"  File: {test_file}")
    print(f"  Size: {test_file.stat().st_size / 1024:.1f} KB")

    # Validate contents with xarray
    import xarray as xr
    ds = xr.open_dataset(str(test_file))
    print(f"\n  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")
    if "latitude" in ds:
        print(f"  Lat range: {float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f}")
        print(f"  Lon range: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f}")
    elif "lat" in ds:
        print(f"  Lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
        print(f"  Lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")
    if "time" in ds:
        print(f"  Time: {str(ds.time.values[0])[:10]}")
    for v in ds.data_vars:
        da = ds[v]
        print(f"  {v}: min={float(da.min()):.2f}, max={float(da.max()):.2f}, units={da.attrs.get('units','?')}")
    ds.close()
    print("\nTEST PASSED — ERA5-Land access verified.")

except Exception as e:
    elapsed = time.time() - t0
    print(f"\nFAILED after {elapsed:.1f}s")
    print(f"Error: {e}")
    if "licence" in str(e).lower() or "forbidden" in str(e).lower():
        print("\n*** LICENCE ACCEPTANCE REQUIRED ***")
        print("Please visit:")
        print("  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download#manage-licences")
        print("Log in and accept the licence terms at the bottom of the page.")
    raise
