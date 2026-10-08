"""
Download and assemble CHIRPS v2.0 monthly precipitation for Nepal / HKH study area.
Saves to raw/CHIRPS/chirps_monthly_nepal_2016_2023.nc
"""
import urllib.request
import gzip
import time
import pathlib
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
import xarray as xr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

out_dir = pathlib.Path('raw/CHIRPS')
out_dir.mkdir(parents=True, exist_ok=True)
out_nc = out_dir / 'chirps_monthly_nepal_2016_2023.nc'

# Bounding box for Nepal & surrounding HKH:
# Lon: 79.5 to 89.5, Lat: 26.5 to 31.5
min_lon, min_lat, max_lon, max_lat = 79.5, 26.5, 89.5, 31.5

def fetch_month(ym):
    year, month = ym
    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz"
    req = urllib.request.Request(url, headers={'User-Agent': 'GlacierGuard-AI/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                compressed = resp.read()
            decomp = gzip.decompress(compressed)
            with MemoryFile(decomp) as mem:
                with mem.open() as src:
                    win = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                    arr = src.read(1, window=win)
                    trans = rasterio.windows.transform(win, src.transform)
                    h, w = arr.shape
                    lons = trans.c + (np.arange(w) + 0.5) * trans.a
                    lats = trans.f + (np.arange(h) + 0.5) * trans.e
                    arr = np.where(arr < 0, 0.0, arr).astype(np.float32)
                    return (year, month, arr, lats, lons)
        except Exception as e:
            if attempt == 2:
                print(f"FAILED {year}-{month:02d}: {e}")
                return None
            time.sleep(2)

def main():
    if out_nc.exists() and out_nc.stat().st_size > 100000:
        print(f"File already exists: {out_nc} ({out_nc.stat().st_size/1e6:.2f} MB)")
        ds = xr.open_dataset(out_nc)
        print("Dataset dims:", ds.dims)
        print("Time range:", str(ds.time.values[0])[:10], "to", str(ds.time.values[-1])[:10])
        return

    tasks = [(y, m) for y in range(2016, 2024) for m in range(1, 13)]
    print(f"Fetching {len(tasks)} months (2016-2023) using ThreadPoolExecutor...")
    t0 = time.time()
    
    results = {}
    ref_lats, ref_lons = None, None
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_month, ym): ym for ym in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                y, m, arr, lats, lons = res
                results[(y, m)] = arr
                if ref_lats is None:
                    ref_lats = lats
                    ref_lons = lons
                print(f"  Downloaded {y}-{m:02d} ({len(results)}/{len(tasks)})", end='\r')

    print(f"\nFetched {len(results)}/{len(tasks)} months in {time.time()-t0:.1f}s")
    if len(results) < len(tasks):
        print("WARNING: Some months were missing!")

    # Sort in chronological order
    times = []
    arr_list = []
    for y in range(2016, 2024):
        for m in range(1, 13):
            if (y, m) in results:
                times.append(pd.Timestamp(year=y, month=m, day=1))
                arr_list.append(results[(y, m)])

    da_precip = xr.DataArray(
        np.stack(arr_list, axis=0),
        coords={'time': times, 'latitude': ref_lats, 'longitude': ref_lons},
        dims=['time', 'latitude', 'longitude'],
        name='precip',
        attrs={'units': 'mm/month', 'long_name': 'CHIRPS v2.0 Monthly Precipitation'}
    )
    
    ds = xr.Dataset({'precip': da_precip}, attrs={
        'title': 'CHIRPS v2.0 Monthly Precipitation — Nepal & Central/Eastern Himalaya',
        'spatial_resolution': '0.05 degree (~5.5 km)',
        'source': 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/',
        'bounding_box': f'lon=[{min_lon}, {max_lon}], lat=[{min_lat}, {max_lat}]'
    })
    
    ds.to_netcdf(out_nc)
    print(f"Saved to {out_nc} ({out_nc.stat().st_size/1e6:.2f} MB)")
    print("Dimensions:", dict(ds.dims))
    print("Summary mean:", float(ds['precip'].mean()))

if __name__ == '__main__':
    main()
