"""
Applies targeted modifications to scratch/build_nb07.py using line range replacements.
"""
import pathlib

p = pathlib.Path('scratch/build_nb07.py')
lines = p.read_text(encoding='utf-8').splitlines(keepends=True)

def replace_lines(start_pat, end_pat, new_block, find_start_idx=0):
    start = None
    for i in range(find_start_idx, len(lines)):
        if start_pat in lines[i]:
            start = i
            break
    if start is None:
        raise ValueError(f"Pattern '{start_pat}' not found after line {find_start_idx}")
    
    end = None
    for i in range(start, len(lines)):
        if end_pat in lines[i]:
            end = i
            break
    if end is None:
        raise ValueError(f"End pattern '{end_pat}' not found after line {start}")
    
    lines[start:end] = [new_block]
    print(f"Replaced lines {start+1} to {end} for '{start_pat}'")

# 1. §6.5 CHIRPS Availability
block_65 = """# == §6.5  CHIRPS Availability Check ==
CHIRPS_NC = CHIRPS_DIR / 'chirps_monthly_nepal_2016_2023.nc'
CHIRPS_AVAILABLE = False
CHIRPS_REASON = ''

if CHIRPS_NC.exists():
    CHIRPS_AVAILABLE = True
    CHIRPS_REASON = f'Local NetCDF ready: {CHIRPS_NC.name} ({CHIRPS_NC.stat().st_size/1e6:.2f} MB)'
    print(f'CHIRPS dataset detected: {CHIRPS_NC}')
    print('  Coverage: 2016-2023 (96 monthly rasters, Nepal/HKH domain)')
elif HAS_RASTERIO:
    print('rasterio is installed; CHIRPS NetCDF not yet compiled locally.')
    CHIRPS_REASON = 'rasterio installed, NetCDF pending compilation'
else:
    CHIRPS_REASON = 'rasterio not installed'
    print('rasterio not available — CHIRPS GeoTIFF extraction disabled.')

print()
"""
replace_lines('# == §6.5  CHIRPS Availability Check ==', 'print(\'Data availability summary:\')', block_65)

# 2. §10 and §11
block_10_11 = """cells.append(md("## §10 — CHIRPS Data Inspection"))
cells.append(code(\"\"\"\\
# == §10.1  CHIRPS Availability & Inspection ==
print('CHIRPS v2.0 Monthly Global Precipitation')
print('=' * 50)
print(f'Status: {"AVAILABLE" if CHIRPS_AVAILABLE else "UNAVAILABLE"}')
print(f'Details: {CHIRPS_REASON}')
print()

chirps_ds = None
if CHIRPS_AVAILABLE and CHIRPS_NC.exists():
    import xarray as xr
    chirps_ds = xr.open_dataset(str(CHIRPS_NC))
    print('CHIRPS Nepal/HKH dataset:')
    print(f'  Dimensions : {dict(chirps_ds.sizes)}')
    print(f'  Variables  : {list(chirps_ds.data_vars)}')
    print(f'  Lat range  : {float(chirps_ds.latitude.min()):.2f} to {float(chirps_ds.latitude.max()):.2f}°N')
    print(f'  Lon range  : {float(chirps_ds.longitude.min()):.2f} to {float(chirps_ds.longitude.max()):.2f}°E')
    print(f'  Time range : {str(chirps_ds.time.values[0])[:10]} to {str(chirps_ds.time.values[-1])[:10]}')
    print(f'  Timesteps  : {len(chirps_ds.time)} months (2016-2023 complete)')
    p = chirps_ds['precip']
    print(f'  Precip stats: min={float(p.min()):.2f}, max={float(p.max()):.2f}, mean={float(p.mean()):.2f} mm/month')
    print(f'  Null count : {int(p.isnull().sum())} (0% missing)')
else:
    print('CHIRPS dataset not available for inspection.')
\"\"\"))

# ─────────────────────────────────────────────────────────────────────────────
# §11 — CHIRPS Alignment
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §11 — CHIRPS Spatial / Temporal Alignment"))
cells.append(code(\"\"\"\\
# == §11.1  CHIRPS Spatial & Temporal Alignment ==
chirps_features = None
CHIRPS_FEATURE_COLS = []

if CHIRPS_AVAILABLE and CHIRPS_NC.exists():
    import xarray as xr
    chirps_ds = xr.open_dataset(str(CHIRPS_NC))
    precip_arr = chirps_ds['precip'].values  # (96, n_lat, n_lon)
    c_lats = chirps_ds['latitude'].values
    c_lons = chirps_ds['longitude'].values
    c_times = pd.to_datetime(chirps_ds['time'].values)
    time_df = pd.DataFrame({'year': c_times.year, 'month': c_times.month})

    # Unique lakes across S1 and S2
    all_lakes = pd.concat([
        gold_s1[['GLO_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates('GLO_ID'),
        gold_s2[['GLO_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates('GLO_ID')
    ]).drop_duplicates('GLO_ID').reset_index(drop=True)

    print(f'Extracting CHIRPS time series for {len(all_lakes)} unique lakes...')
    # Nearest neighbor grid cell indices
    lat_idx = np.abs(c_lats[:, None] - all_lakes['LATITUDE'].values[None, :]).argmin(axis=0)
    lon_idx = np.abs(c_lons[:, None] - all_lakes['LONGITUDE'].values[None, :]).argmin(axis=0)

    # Fast numpy indexing: shape (96, N_lakes)
    lake_precip = precip_arr[:, lat_idx, lon_idx]

    # Pre-training baseline: 2016 total precipitation per lake (zero future leakage)
    mask_2016 = (time_df['year'] == 2016).values
    baseline_total = lake_precip[mask_2016].sum(axis=0)

    records = []
    for yr in range(2017, 2024):
        mask_yr = (time_df['year'] == yr).values
        mask_jjas = ((time_df['year'] == yr) & (time_df['month'].isin([6, 7, 8, 9]))).values
        tp_total = lake_precip[mask_yr].sum(axis=0)
        tp_jjas  = lake_precip[mask_jjas].sum(axis=0)
        tp_anom  = tp_total - baseline_total

        for i, glo_id in enumerate(all_lakes['GLO_ID']):
            records.append({
                'GLO_ID': glo_id,
                'AREA_YEAR': yr,
                'CHIRPS_TP_TOTAL': float(tp_total[i]),
                'CHIRPS_TP_JJAS':  float(tp_jjas[i]),
                'CHIRPS_TP_ANOM':  float(tp_anom[i]),
            })

    chirps_features = pd.DataFrame(records)
    CHIRPS_FEATURE_COLS = ['CHIRPS_TP_TOTAL', 'CHIRPS_TP_JJAS', 'CHIRPS_TP_ANOM']
    print(f'CHIRPS features computed for {len(chirps_features)} lake-year rows.')
    print(f'  Features: {CHIRPS_FEATURE_COLS}')
    print(f'  TP_TOTAL mean: {chirps_features["CHIRPS_TP_TOTAL"].mean():.1f} mm/yr')
    print(f'  TP_JJAS  mean: {chirps_features["CHIRPS_TP_JJAS"].mean():.1f} mm/monsoon')
    print(f'  TP_ANOM  mean: {chirps_features["CHIRPS_TP_ANOM"].mean():.1f} mm')
    chirps_features.to_csv(CHIRPS_DIR / 'chirps_features.csv', index=False)
    print(f'Saved: {CHIRPS_DIR}/chirps_features.csv')
else:
    print('CHIRPS extraction skipped (data unavailable).')
    CHIRPS_FEATURE_COLS = []
\"\"\"))

# ─────────────────────────────────────────────────────────────────────────────
"""
replace_lines('cells.append(md("## §10 — CHIRPS Data Inspection"))', '# §12 — RGI Glacier Context Extraction', block_10_11)

# 3. §14 CHIRPS QA
block_14 = """# CHIRPS QA
if chirps_features is not None:
    print('CHIRPS QA:')
    print(f'  Rows in chirps_features: {len(chirps_features)}')
    for col in CHIRPS_FEATURE_COLS:
        n_nan = chirps_features[col].isna().sum()
        print(f'    {col:<20}: {n_nan} missing ({n_nan/len(chirps_features)*100:.1f}%)')
    assert (chirps_features['CHIRPS_TP_TOTAL'] >= 0).all(), 'Negative total precipitation in CHIRPS!'
    assert (chirps_features['CHIRPS_TP_JJAS'] >= 0).all(), 'Negative monsoon precipitation in CHIRPS!'
    print('  [PASS] All CHIRPS values physically valid.')
else:
    print('CHIRPS QA: SKIPPED (data unavailable)')
\"\"\"))

# ─────────────────────────────────────────────────────────────────────────────
"""
replace_lines('# CHIRPS QA', '# §15 — Temporal Alignment / Leakage Audit', block_14)

# 4. §16 make_augmented_gold
block_16 = """# == §16.1  Build Augmented Datasets ==
def make_augmented_gold(gold_df, era5_feat, chirps_feat, rgi_dist, sensor_name):
    \"\"\"Merge GLO gold with environmental features.

    Args:
        gold_df  : gold dataset from build_gold_dataset()
        era5_feat: ERA5 features DataFrame or None
        chirps_feat: CHIRPS features DataFrame or None
        rgi_dist : RGI distance DataFrame or None
        sensor_name: 'S1' or 'S2'

    Returns:
        augmented DataFrame with all GLO + environmental columns
    \"\"\"
    df = gold_df.copy()

    # Merge ERA5
    if era5_feat is not None:
        df = df.merge(era5_feat[['GLO_ID', 'AREA_YEAR'] + ERA5_FEATURE_COLS],
                      on=['GLO_ID', 'AREA_YEAR'], how='left')
        n_matched = df[ERA5_FEATURE_COLS[0]].notna().sum()
        n_total = len(df)
        print(f'  {sensor_name} ERA5 match: {n_matched}/{n_total} ({n_matched/n_total*100:.1f}%)')
    else:
        for col in ERA5_FEATURE_COLS:
            df[col] = np.nan
        print(f'  {sensor_name} ERA5: all NaN (unavailable)')

    # Merge CHIRPS
    if chirps_feat is not None:
        df = df.merge(chirps_feat[['GLO_ID', 'AREA_YEAR'] + CHIRPS_FEATURE_COLS],
                      on=['GLO_ID', 'AREA_YEAR'], how='left')
        n_matched = df['CHIRPS_TP_TOTAL'].notna().sum()
        print(f'  {sensor_name} CHIRPS match: {n_matched}/{len(df)} ({n_matched/len(df)*100:.1f}%)')
    else:
        for col in CHIRPS_FEATURE_COLS:
            df[col] = np.nan
        print(f'  {sensor_name} CHIRPS: all NaN (unavailable)')

    # Merge RGI
    if rgi_dist is not None:
        df = df.merge(rgi_dist[['GLO_ID', 'DIST_NEAREST_GLACIER_KM']],
                      on='GLO_ID', how='left')
        n_matched = df['DIST_NEAREST_GLACIER_KM'].notna().sum()
        print(f'  {sensor_name} RGI match: {n_matched}/{len(df)} ({n_matched/len(df)*100:.1f}%)')
    else:
        df['DIST_NEAREST_GLACIER_KM'] = np.nan
        print(f'  {sensor_name} RGI: all NaN (unavailable)')

    return df

print('Building augmented datasets...')
aug_s1 = make_augmented_gold(gold_s1, era5_features, chirps_features, rgi_distances, 'S1')
aug_s2 = make_augmented_gold(gold_s2, era5_features, chirps_features, rgi_distances, 'S2')
print()
print(f'S1 augmented: {len(aug_s1)} rows, {len(aug_s1.columns)} columns')
print(f'S2 augmented: {len(aug_s2)} rows, {len(aug_s2.columns)} columns')
\"\"\"))

cells.append(code(\"\"\"\\
"""
replace_lines('# == §16.1  Build Augmented Datasets ==', '# == §16.2  Temporal Splits on Augmented Data ==', block_16)

# 5. §19 ablation branches
block_19 = """    # C: GLO + CHIRPS
    if chirps_cols:
        r = run_ablation_branch(train, val, test, sensor, 'C_GLO+CHIRPS', chirps_cols)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'C_GLO+CHIRPS_UNAVAIL', [])
        r['Branch'] = 'C_GLO+CHIRPS'
        r['_chirps_unavailable'] = True
    ablation_results.append(r)
"""
replace_lines('    # C: GLO + CHIRPS', '    # D: GLO + RGI', block_19)

# 6. §20 env_status in ablation summary
block_20 = """    # Determine if branch ran with actual environmental data
    era5_flag    = r.get('_era5_unavailable', False)
    chirps_flag  = r.get('_chirps_unavailable', False)
    env_status   = []
    if 'ERA5' in branch and branch != 'E_GLO+ALL':
        env_status.append('ERA5:UNAVAIL' if era5_flag else 'ERA5:OK')
    if 'CHIRPS' in branch and branch != 'E_GLO+ALL':
        env_status.append('CHIRPS:UNAVAIL' if chirps_flag else 'CHIRPS:OK')
    if 'RGI' in branch and branch != 'E_GLO+ALL':
        env_status.append('RGI:OK')
    if branch == 'E_GLO+ALL':
        sub_flags = []
        if era5_cols: sub_flags.append('ERA5:OK')
        else: sub_flags.append('ERA5:UNAVAIL')
        if chirps_cols: sub_flags.append('CHIRPS:OK')
        else: sub_flags.append('CHIRPS:UNAVAIL')
        if rgi_cols: sub_flags.append('RGI:OK')
        else: sub_flags.append('RGI:UNAVAIL')
        env_status.append('; '.join(sub_flags))
"""
replace_lines('    # Determine if branch ran with actual environmental data', '    for model in [\'Mean Baseline\', \'Zero-Change\', \'Tuned HGB\']:', block_20)

# 7. §23 Permutation importance candidate branch order
for i, l in enumerate(lines):
    if "for branch_prefix in ['E_GLO+ALL', 'B_GLO+ERA5', 'D_GLO+RGI', 'A_GLO_only']:" in l:
        lines[i] = "    for branch_prefix in ['E_GLO+ALL', 'C_GLO+CHIRPS', 'D_GLO+RGI', 'B_GLO+ERA5', 'A_GLO_only']:\n"
        print(f"Updated permutation importance candidate branches at line {i+1}")
        break

# 8. §25 QA Summary CSV
for i, l in enumerate(lines):
    if "('CHIRPS', CHIRPS_AVAILABLE," in l:
        lines[i] = "    ('CHIRPS v2.0', CHIRPS_AVAILABLE, '96 monthly NetCDF rasters (2016-2023) loaded via rasterio/xarray' if CHIRPS_AVAILABLE else CHIRPS_REASON),\n"
        print(f"Updated QA summary dataset name at line {i+1}")
        break

p.write_text(''.join(lines), encoding='utf-8')
print("Successfully patched scratch/build_nb07.py!")
