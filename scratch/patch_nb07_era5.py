"""
Patches scratch/build_nb07.py to fully activate ERA5 and RGI 7.0.
"""
import pathlib
import re

p = pathlib.Path("scratch/build_nb07.py")
text = p.read_text(encoding="utf-8")

# 1. Update ERA5_NC detection in §6.3
old_63_nc = "ERA5_NC = ERA5_DIR / 'era5_land_monthly_nepal_20102023.nc'"
new_63_nc = """# Detect existing ERA5 NetCDF
era5_candidates = list(ERA5_DIR.glob('era5_land_monthly_nepal_*.nc'))
if era5_candidates:
    ERA5_NC = era5_candidates[0]
else:
    ERA5_NC = ERA5_DIR / 'era5_land_monthly_nepal_2016_2023.nc'"""

if old_63_nc in text:
    text = text.replace(old_63_nc, new_63_nc, 1)
    print("Patched ERA5_NC path in §6.3")

# 2. Update §7.1 and §7.2 to support 'valid_time'
old_71 = """    print(f'  Time range : {str(ERA5_DS.time.values[0])[:7]} to {str(ERA5_DS.time.values[-1])[:7]}')
    print(f'  N timesteps: {len(ERA5_DS.time)}')"""

new_71 = """    t_dim = 'valid_time' if 'valid_time' in ERA5_DS.dims else 'time'
    print(f'  Time range : {str(ERA5_DS[t_dim].values[0])[:7]} to {str(ERA5_DS[t_dim].values[-1])[:7]}')
    print(f'  N timesteps: {len(ERA5_DS[t_dim])}')"""

if old_71 in text:
    text = text.replace(old_71, new_71, 1)
    print("Patched time dimension in §7.1")

old_72 = """    times_pd = _pd.to_datetime(ERA5_DS.time.values)"""
new_72 = """    t_dim = 'valid_time' if 'valid_time' in ERA5_DS.dims else 'time'
    times_pd = _pd.to_datetime(ERA5_DS[t_dim].values)"""

if old_72 in text:
    text = text.replace(old_72, new_72, 1)
    print("Patched time dimension in §7.2")

# 3. Update §13.2 to load pre-computed era5_features.csv if available
old_132_start = """# == §13.2  Run ERA5 Extraction (or load from cache) ==
era5_features = None

if ERA5_AVAILABLE:
    import xarray as xr
    ERA5_DS = xr.open_dataset(str(ERA5_NC))"""

new_132_start = """# == §13.2  Run ERA5 Extraction (or load from cache) ==
era5_features = None

if ERA5_AVAILABLE:
    era5_csv_path = ERA5_DIR / 'era5_features.csv'
    if era5_csv_path.exists():
        print(f'Loading pre-computed ERA5 features from {era5_csv_path}...')
        era5_features = pd.read_csv(era5_csv_path)
        print(f'  Loaded {len(era5_features)} ERA5 feature records.')
        print(f'  Columns: {list(era5_features.columns)}')
    else:
        import xarray as xr
        ERA5_DS = xr.open_dataset(str(ERA5_NC))"""

if old_132_start in text:
    text = text.replace(old_132_start, new_132_start, 1)
    print("Patched fast loading in §13.2")

# 4. In §19.1: make sure all branches run properly
# Check B_GLO+ERA5 in §19.1
old_b_branch = """    # B: GLO + ERA5
    if era5_cols:
        r = run_ablation_branch(train, val, test, sensor, 'B_GLO+ERA5', era5_cols)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'B_GLO+ERA5_UNAVAIL', [])
        r['Branch'] = 'B_GLO+ERA5'
        r['_era5_unavailable'] = True
    ablation_results.append(r)"""

# Check if old_b_branch exists in text
if old_b_branch in text:
    print("Found §19.1 B_GLO+ERA5 branch definition — it will run with era5_cols!")
else:
    print("Checking §19.1 branch definition...")

p.write_text(text, encoding="utf-8")
print("Saved patched build_nb07.py")
