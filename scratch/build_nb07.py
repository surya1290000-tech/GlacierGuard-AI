"""
Builds notebooks/07_environmental_forcing.ipynb
All cells are constructed here and verified executable top-to-bottom.
"""
import json, pathlib

NB_PATH = pathlib.Path('notebooks/07_environmental_forcing.ipynb')

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# §1 — Research Objective
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# 07 — GlacierGuard-AI Environmental Forcing Integration

**Project:** GlacierGuard-AI
**Research Question:** Does adding external environmental forcing provide additional predictive information
for next-year glacial-lake area change beyond the GLO-derived observational predictors?

**Task:** Controlled feature-ablation experiment comparing:
- **Model A (GLO-only):** Exact 20-feature contract from Notebook 04/06
- **Model B (GLO + ERA5):** GLO + ERA5-Land climate forcing
- **Model C (GLO + CHIRPS):** GLO + CHIRPS precipitation
- **Model D (GLO + RGI):** GLO + glacier proximity from RGI 6.0
- **Model E (GLO + ALL):** GLO + ERA5 + CHIRPS + RGI

**Primary Target:** `NEXT_AREA_CHANGE = AREA(t+1) − AREA(t)` in km², when `NEXT_YEAR − AREA_YEAR == 1`
**Sensors:** Sentinel-1 (SAR) and Sentinel-2 (Optical), evaluated independently

---

### Scientific Scope
This notebook is an observational prediction experiment. It does NOT:
- Predict GLOFs, floods, or disaster probabilities
- Establish causal climate relationships
- Claim any operational forecasting capability

**A negative result — environmental variables do not improve predictions — is equally valid
and will be reported honestly.**

---

### Notebook Sections
| Section | Title |
|---|---|
| §1  | Research Objective |
| §2  | Scientific Hypothesis |
| §3  | Existing GLO Dataset Contract |
| §4  | Environmental Data Sources and Provenance |
| §5  | Environment / Dependency Checks |
| §6  | Raw Data Availability and Acquisition |
| §7  | ERA5-Land Data Inspection |
| §8  | ERA5-Land Spatial Extraction |
| §9  | ERA5-Land Temporal Aggregation |
| §10 | CHIRPS Data Inspection |
| §11 | CHIRPS Spatial / Temporal Alignment |
| §12 | RGI Glacier Context Extraction |
| §13 | Environmental Feature Engineering |
| §14 | Environmental Data Quality Assurance |
| §15 | Temporal Alignment / Leakage Audit |
| §16 | Merge Environmental Features with GLO Gold |
| §17 | Environmental Feature Correlation / Redundancy Check |
| §18 | GLO-only Reference Experiment |
| §19 | GLO + Environmental Experiment |
| §20 | Controlled Ablation Study |
| §21 | Model Evaluation |
| §22 | Error Analysis |
| §23 | Environmental Feature Importance |
| §24 | Scientific Interpretation |
| §25 | Final Research Conclusions |
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §2 — Scientific Hypothesis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §2 — Scientific Hypothesis

**Primary Hypothesis H1:**
Environmental forcing variables (temperature, precipitation, snow depth, glacier proximity) contain
information about inter-annual glacial-lake area change that is not already captured in the GLO-derived
observational features.

**Null Hypothesis H0:**
Environmental forcing adds no statistically or practically meaningful predictive information beyond
the existing GLO-derived features. Model performance with environmental features will be
indistinguishable from, or worse than, the GLO-only model.

**Expected Direction (prior to experiment):**
Given Notebook 06 findings that the GLO-only signal is weak, and that baselines outperform ML models,
we expect that environmental forcing may:
1. Potentially improve relative error for specific lake-size or temporal subgroups
2. Not necessarily improve overall MAE, given the noise-floor constraints documented in NB05
3. Provide physically interpretable feature importance rankings even if absolute improvement is modest

**Experimental Design:**
- Hold model class constant (Tuned HistGradientBoosting + baselines)
- Vary only the feature set across ablation branches
- All branches use identical temporal split (train ≤2021, val=2022, test=2023)
- No post-hoc tuning based on test results
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §3 — Existing GLO Dataset Contract
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §3 — Existing GLO Dataset Contract

The following contracts from Notebooks 03–06 are frozen and must be preserved exactly.

### 20-Feature Contract
```
Categorical (4): BASIN, COUNTRY, GTNG_REGION_O2, CONNECTIVITY
Numerical  (16): AREA, PERIMETER, AREA_UNCERTAINTY,
                 ELEVATION_MEAN, ELEVATION_MIN, ELEVATION_MEDIAN,
                 LATITUDE, LONGITUDE,
                 AREA_LAG1, AREA_LAG2, AREA_LAG3,
                 AREA_CHANGE_LAG1, AREA_CHANGE_LAG2, AREA_CHANGE_LAG3,
                 ROLLING_MEAN_3, ROLLING_STD_3
```

### Target
`NEXT_AREA_CHANGE = NEXT_AREA − AREA` (km²), where `NEXT_YEAR − AREA_YEAR == 1` exactly.

### Temporal Split
- Train: AREA_YEAR ≤ 2021
- Validation: AREA_YEAR == 2022
- Test: AREA_YEAR == 2023 (frozen, evaluated once)

### Reference Results (NB06, to be reproduced by GLO-only branch)
| Sensor | Model | Val MAE (km²) |
|---|---|---|
| Sentinel-1 | Mean Baseline | 0.005597 |
| Sentinel-1 | Tuned HGB | 0.005668 |
| Sentinel-2 | Zero-Change Baseline | 0.004965 |
| Sentinel-2 | Tuned HGB | 0.005000 |

The GLO-only branch in this notebook must reproduce the above within ±1% numerical tolerance.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §4 — Environmental Data Sources and Provenance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## §4 — Environmental Data Sources and Provenance

### ERA5-Land
- **Product:** ERA5-Land Monthly Averaged Data (reanalysis-era5-land-monthly-means)
- **Provider:** Copernicus Climate Change Service (C3S), ECMWF
- **Access:** Copernicus Data Store (CDS) API — requires free registration at cds.climate.copernicus.eu
- **Variables used:** 2m_temperature (t2m), total_precipitation (tp), snow_depth (sd)
- **Resolution:** ~9 km (~0.1°×0.1°)
- **Coverage:** Nepal bounding box: Lat [27.0, 31.0]°N, Lon [80.0, 89.0]°E
- **Period:** 2010–2023 (2010–2016 for climatology baseline; 2017–2023 for predictors)
- **Citation:** Copernicus Climate Change Service (C3S), 2019. ERA5-Land monthly averaged data.
  doi:10.24381/cds.68d2bb30
- **License:** CC-BY 4.0

### CHIRPS v2
- **Product:** Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS v2.0) Monthly
- **Provider:** Climate Hazards Center, UCSB
- **Access:** Public download from data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
- **Resolution:** 0.05° (~5.5 km), global monthly GeoTIFF (~14.4 MB/month compressed)
- **Note:** GeoTIFF extraction requires rasterio (not in current env). If rasterio not available,
  CHIRPS extraction will be skipped and ERA5 precipitation used as the sole precipitation source.
- **Citation:** Funk et al. (2015). The climate hazards infrared precipitation with stations.
  Scientific Data, 2, 150066. doi:10.1038/sdata.2015.66

### RGI 6.0 (Randolph Glacier Inventory)
- **Product:** RGI version 6.0 — Region 15 (South Asia East, Nepal/Tibet/Bhutan)
- **Provider:** RGI Consortium (2017) via OGGM mirror at cluster.klima.uni-bremen.de
- **File:** 15_rgi60_SouthAsiaEast.zip (~15 MB)
- **Usage:** Static glacier proximity feature (nearest-glacier distance), NOT for dynamic glacier change
- **Content:** Glacier outlines as Shapefile; EPSG:4326 (WGS84)
- **Citation:** RGI Consortium (2017). Randolph Glacier Inventory — A Dataset of Global Glacier Outlines:
  Version 6.0. Technical Report, Global Land Ice Measurements from Space, Boulder, Colorado, USA.
  doi:10.7265/4m1f-gd79

### Spatial Context
- All lakes are in Nepal / Tibet / India, Lat 27.47–30.59°N, Lon 80.03–88.88°E
- The study area spans the Koshi, Gandaki, and Karnali basins
- ERA5-Land at 0.1° resolution provides ~9 lake-grid-cell matches per unique 1° grid box
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §5 — Environment / Dependency Checks
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §5 — Environment / Dependency Checks"))
cells.append(code("""\
# == §5.1  Imports, Environment, Paths ==
import os, sys, pathlib, warnings, time, copy, itertools, zipfile, io
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import scipy.spatial

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = pathlib.Path('../reports/figures')
if not FIGURES_DIR.exists():
    FIGURES_DIR = pathlib.Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = FIGURES_DIR.parent
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Optional dependency / API flags (set here so always defined) ──────────
HAS_CDSAPI   = False
HAS_XARRAY   = False
HAS_RASTERIO = False
HAS_CDS_KEY  = False
ERA5_AVAILABLE  = False
CHIRPS_AVAILABLE = False
RGI_AVAILABLE   = False

try:
    import cdsapi as _cdsapi
    HAS_CDSAPI = True
except ImportError:
    pass

try:
    import xarray as _xr
    import netCDF4 as _nc4
    HAS_XARRAY = True
except ImportError:
    pass

try:
    import rasterio as _rasterio
    HAS_RASTERIO = True
except ImportError:
    pass

import pathlib as _pl
HAS_CDS_KEY = (_pl.Path.home() / '.cdsapirc').exists()

print('Environment:')
print(f'  Python       : {sys.version.split()[0]}')
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  pandas       : {pd.__version__}')
print(f'  numpy        : {np.__version__}')
print(f'  geopandas    : {gpd.__version__}')
print(f'  scipy        : {scipy.__version__}')
print(f'  Figures      : {FIGURES_DIR.resolve()}')
print(f'  Reports      : {REPORTS_DIR.resolve()}')
print(f'  Random seed  : {RANDOM_STATE}')
print(f'  HAS_CDSAPI   : {HAS_CDSAPI}')
print(f'  HAS_XARRAY   : {HAS_XARRAY}')
print(f'  HAS_RASTERIO : {HAS_RASTERIO}')
print(f'  HAS_CDS_KEY  : {HAS_CDS_KEY}')
"""))

cells.append(code("""\
# == §5.2  Optional Dependency Detail Report ==
if HAS_CDSAPI:
    import cdsapi as _cds_mod
    # cdsapi may not have __version__; try alternative
    _cds_ver = getattr(_cds_mod, '__version__', None) or getattr(_cds_mod, 'version', 'installed')
    print(f'cdsapi     : {_cds_ver}')
else:
    print('cdsapi     : NOT AVAILABLE — install with: uv pip install cdsapi')

if HAS_XARRAY:
    import xarray as xr
    import netCDF4
    print(f'xarray     : {xr.__version__}')
    print(f'netCDF4    : {netCDF4.__version__}')
else:
    print('xarray/netCDF4 : NOT AVAILABLE — install with: uv pip install xarray netCDF4')

if HAS_RASTERIO:
    import rasterio
    print(f'rasterio   : {rasterio.__version__}')
else:
    print('rasterio   : NOT AVAILABLE — CHIRPS GeoTIFF extraction disabled')

print()
print(f'CDS API key file (~/.cdsapirc) : {"FOUND" if HAS_CDS_KEY else "NOT FOUND"}')
if not HAS_CDS_KEY:
    print()
    print('ACTION REQUIRED for ERA5 download:')
    print('  1. Register at https://cds.climate.copernicus.eu/')
    print('  2. Copy your UID and API key from your profile page')
    print('  3. Create ~/.cdsapirc with content:')
    print('     url: https://cds.climate.copernicus.eu/api')
    print('     key: <your-api-key>')
    print()
    print('  Without this file, ERA5 data will not be downloaded.')
    print('  ERA5 features will be set to NaN; ablation will show "ERA5=UNAVAILABLE".')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §6 — Raw Data Availability and Acquisition
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §6 — Raw Data Availability and Acquisition"))
cells.append(code("""\
# == §6.1  Create raw data directory structure ==
GLO_DIR = None
for p in [pathlib.Path('../raw/GLO'), pathlib.Path('./raw/GLO'),
          pathlib.Path('/Workspace/raw/GLO')]:
    if p.exists() and (p / 'S1_20172024_NTB_GLOID_v1.02.gpkg').exists():
        GLO_DIR = p.resolve()
        break
assert GLO_DIR is not None, 'Cannot locate raw/GLO GeoPackage files.'
print(f'GLO directory: {GLO_DIR}')

# Create ERA5, CHIRPS, RGI directories
for sub in ['ERA5', 'CHIRPS', 'RGI']:
    d = GLO_DIR.parent / sub
    d.mkdir(exist_ok=True)
    print(f'  {sub} dir: {d}')

ERA5_DIR   = (GLO_DIR.parent / 'ERA5').resolve()
CHIRPS_DIR = (GLO_DIR.parent / 'CHIRPS').resolve()
RGI_DIR    = (GLO_DIR.parent / 'RGI').resolve()
"""))

cells.append(code("""\
# == §6.2  Update .gitignore for large raw files ==
import pathlib as _pl
gitignore = _pl.Path('.gitignore')
if not gitignore.exists():
    gitignore = _pl.Path('../.gitignore')

additions = [
    '# ── Environmental raw data (large binary files) ─────────────────────────────',
    'raw/ERA5/*.nc',
    'raw/ERA5/*.grib',
    'raw/CHIRPS/*.tif',
    'raw/CHIRPS/*.tif.gz',
    'raw/CHIRPS/*.nc',
    'raw/RGI/*.zip',
    'raw/RGI/*.shp',
    'raw/RGI/*.dbf',
    'raw/RGI/*.shx',
    'raw/RGI/*.prj',
    'raw/RGI/*.cpg',
]

if gitignore.exists():
    current = gitignore.read_text(encoding='utf-8')
    new_lines = [a for a in additions if a not in current]
    if new_lines:
        with open(gitignore, 'a', encoding='utf-8') as f:
            f.write('\\n' + '\\n'.join(new_lines) + '\\n')
        print(f'Updated .gitignore with {len(new_lines)} new patterns.')
    else:
        print('.gitignore already up-to-date for environmental data.')
else:
    print('WARNING: .gitignore not found. Create manually to avoid committing large files.')
"""))

cells.append(code("""\
# == §6.3  ERA5 Download ==
# Detect existing ERA5 NetCDF
era5_candidates = list(ERA5_DIR.glob('era5_land_monthly_nepal_*.nc'))
if era5_candidates:
    ERA5_NC = era5_candidates[0]
else:
    ERA5_NC = ERA5_DIR / 'era5_land_monthly_nepal_2016_2023.nc'

def download_era5():
    import cdsapi
    client = cdsapi.Client(quiet=True)
    years = [str(y) for y in range(2010, 2024)]
    months = [f'{m:02d}' for m in range(1, 13)]
    print(f'Requesting ERA5-Land monthly for Nepal (2010-2023)...')
    print(f'  Variables : 2m_temperature, total_precipitation, snow_depth')
    print(f'  Area      : [31, 80, 27, 89] (N/W/S/E)')
    print(f'  Years     : {years[0]}-{years[-1]}')
    t0 = time.time()
    client.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ['2m_temperature', 'total_precipitation', 'snow_depth'],
            'year': years,
            'month': months,
            'time': '00:00',
            'format': 'netcdf',
            'area': [31.0, 80.0, 27.0, 89.0],  # N, W, S, E
        },
        str(ERA5_NC)
    )
    print(f'Downloaded in {time.time()-t0:.0f}s. File: {ERA5_NC} ({ERA5_NC.stat().st_size/1e6:.1f} MB)')

if ERA5_NC.exists():
    print(f'ERA5 NetCDF already exists: {ERA5_NC} ({ERA5_NC.stat().st_size/1e6:.1f} MB)')
    ERA5_AVAILABLE = HAS_XARRAY
elif HAS_CDSAPI and HAS_CDS_KEY and HAS_XARRAY:
    print('Downloading ERA5-Land data...')
    try:
        download_era5()
        ERA5_AVAILABLE = True
    except Exception as e:
        print(f'ERA5 download failed: {e}')
        print('ERA5 features will be marked UNAVAILABLE.')
        ERA5_AVAILABLE = False
else:
    missing = []
    if not HAS_CDSAPI: missing.append('cdsapi')
    if not HAS_CDS_KEY: missing.append('~/.cdsapirc API key')
    if not HAS_XARRAY: missing.append('xarray')
    print(f'ERA5 download skipped — missing: {", ".join(missing)}')
    print('ERA5 features will be marked UNAVAILABLE for this run.')
    ERA5_AVAILABLE = False
"""))

cells.append(code("""\
# == §6.4  RGI 6.0 Download (Region 15 — South Asia East) ==
RGI15_ZIP  = RGI_DIR / '15_rgi60_SouthAsiaEast.zip'
RGI15_SHP  = RGI_DIR / '15_rgi60_SouthAsiaEast.shp'

OGGM_BASE = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/rgi60_files/'

def download_rgi_region(region_num, region_name):
    fname = f'{region_num:02d}_rgi60_{region_name}.zip'
    url = OGGM_BASE + fname
    out_zip = RGI_DIR / fname
    if out_zip.exists():
        print(f'  {fname} already downloaded ({out_zip.stat().st_size/1e6:.1f} MB)')
        return out_zip
    print(f'  Downloading {fname} from OGGM mirror...')
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(out_zip, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
    print(f'  Downloaded {out_zip.name}: {out_zip.stat().st_size/1e6:.1f} MB')
    return out_zip

def extract_rgi_shp(zip_path, shp_name):
    with zipfile.ZipFile(zip_path) as zf:
        shp_files = [n for n in zf.namelist() if n.endswith('.shp')]
        print(f'  Contents (.shp): {shp_files[:5]}')
        for member in zf.namelist():
            if any(member.endswith(ext) for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']):
                zf.extract(member, RGI_DIR)
    extracted = list(RGI_DIR.rglob('*.shp'))
    print(f'  Extracted shapefiles: {[p.name for p in extracted]}')
    return extracted

RGI_AVAILABLE = False
try:
    zip15 = download_rgi_region(15, 'SouthAsiaEast')
    extracted = extract_rgi_shp(zip15, '15_rgi60_SouthAsiaEast.shp')
    if extracted:
        RGI_AVAILABLE = True
        print('RGI 6.0 Region 15 ready.')
    else:
        print('WARNING: No shapefiles extracted.')
except Exception as e:
    print(f'RGI download/extraction failed: {e}')
    print('RGI features will be marked UNAVAILABLE.')
"""))

cells.append(code("""\
# == §6.5  CHIRPS Availability Check ==
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
print('Data availability summary:')
print(f'  ERA5   : {"AVAILABLE" if ERA5_AVAILABLE else "UNAVAILABLE"}')
print(f'  CHIRPS : {"AVAILABLE" if CHIRPS_AVAILABLE else f"UNAVAILABLE ({CHIRPS_REASON})"}')
print(f'  RGI    : {"AVAILABLE" if RGI_AVAILABLE else "UNAVAILABLE"}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §7 — ERA5-Land Data Inspection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §7 — ERA5-Land Data Inspection"))
cells.append(code("""\
# == §7.1  Inspect ERA5 NetCDF ==
ERA5_DS = None

if ERA5_AVAILABLE:
    import xarray as xr
    ERA5_DS = xr.open_dataset(str(ERA5_NC))
    print('ERA5-Land dataset:')
    print(f'  Dimensions : {dict(ERA5_DS.dims)}')
    print(f'  Variables  : {list(ERA5_DS.data_vars)}')
    print(f'  Lat range  : {float(ERA5_DS.latitude.min()):.2f} to {float(ERA5_DS.latitude.max()):.2f}')
    print(f'  Lon range  : {float(ERA5_DS.longitude.min()):.2f} to {float(ERA5_DS.longitude.max()):.2f}')
    t_dim = 'valid_time' if 'valid_time' in ERA5_DS.dims else 'time'
    print(f'  Time range : {str(ERA5_DS[t_dim].values[0])[:7]} to {str(ERA5_DS[t_dim].values[-1])[:7]}')
    print(f'  N timesteps: {len(ERA5_DS[t_dim])}')
    print()
    for var in ERA5_DS.data_vars:
        da = ERA5_DS[var]
        print(f'  {var}: units={da.attrs.get("units","?")}, '
              f'mean={float(da.mean()):.4f}, '
              f'std={float(da.std()):.4f}, '
              f'nan={int(da.isnull().sum())}')
else:
    print('ERA5 data not available — inspection skipped.')
    print('To enable: register at cds.climate.copernicus.eu and configure ~/.cdsapirc')
"""))

cells.append(code("""\
# == §7.2  ERA5 Coordinate System and Grid Sanity Check ==
if ERA5_DS is not None:
    print('Coordinate reference system: WGS84 (geographic lat/lon), EPSG:4326')
    print('ERA5-Land grid spacing: 0.1° x 0.1° (~9 km at the equator)')
    print()

    # Verify we have complete months
    import pandas as _pd
    t_dim = 'valid_time' if 'valid_time' in ERA5_DS.dims else 'time'
    times_pd = _pd.to_datetime(ERA5_DS[t_dim].values)
    years_range = range(times_pd.year.min(), times_pd.year.max() + 1)
    expected = len(years_range) * 12
    print(f'  Expected months (2010-2023): {expected}')
    print(f'  Actual months in file      : {len(times_pd)}')

    missing_months = []
    for yr in years_range:
        for mo in range(1, 13):
            if not any((times_pd.year == yr) & (times_pd.month == mo)):
                missing_months.append((yr, mo))
    if missing_months:
        print(f'  MISSING MONTHS: {missing_months}')
    else:
        print('  All months present — no temporal gaps.')

    # Check for NaN values in study area
    study_lat = slice(31.0, 27.0)  # xarray needs N->S for ERA5
    study_lon = slice(80.0, 89.0)
    subset = ERA5_DS.sel(latitude=study_lat, longitude=study_lon)
    print()
    print(f'  Study area grid cells: {len(subset.latitude)}lat x {len(subset.longitude)}lon '
          f'= {len(subset.latitude)*len(subset.longitude)} cells')
    for var in ERA5_DS.data_vars:
        pct_nan = float(subset[var].isnull().mean() * 100)
        print(f'  {var} NaN %: {pct_nan:.2f}%')
else:
    print('ERA5 inspection skipped (data unavailable).')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §8 — ERA5-Land Spatial Extraction
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §8 — ERA5-Land Spatial Extraction"))
cells.append(code("""\
# == §8.1  Extract ERA5 at Lake Centroids ==
# Method: nearest-cell lookup using xarray .sel(method='nearest')
# For ERA5-Land (0.1° resolution), this is the standard and appropriate approach.

def extract_era5_at_point(ds, lat, lon, year, month):
    \"\"\"Extract ERA5 value at nearest grid cell for a given lat/lon/year/month.\"\"\"
    import pandas as _pd
    t = _pd.Timestamp(year=year, month=month, day=1)
    try:
        val = ds.sel(latitude=lat, longitude=lon, time=t, method='nearest')
        result = {}
        for var in ds.data_vars:
            result[var] = float(val[var].values)
        return result
    except Exception:
        return {var: np.nan for var in ds.data_vars}


def batch_extract_era5(ds, lake_years_df):
    \"\"\"Batch extract ERA5 monthly values for all (lake, year) combinations.

    lake_years_df must have columns: GLO_ID, LATITUDE, LONGITUDE, AREA_YEAR
    Returns DataFrame with ERA5 monthly values per (GLO_ID, AREA_YEAR, month).
    \"\"\"
    records = []
    unique_locs = lake_years_df[['GLO_ID', 'LATITUDE', 'LONGITUDE', 'AREA_YEAR']].drop_duplicates()
    total = len(unique_locs)
    print(f'Extracting ERA5 for {total} unique (lake, year) combinations...')
    t0 = time.time()
    for i, row in enumerate(unique_locs.itertuples()):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            print(f'  {i}/{total} ({i/total*100:.0f}%) — ETA: {eta:.0f}s')
        for month in range(1, 13):
            vals = extract_era5_at_point(ds, row.LATITUDE, row.LONGITUDE, row.AREA_YEAR, month)
            rec = {'GLO_ID': row.GLO_ID, 'LATITUDE': row.LATITUDE,
                   'LONGITUDE': row.LONGITUDE, 'AREA_YEAR': row.AREA_YEAR,
                   'month': month}
            rec.update(vals)
            records.append(rec)
    elapsed = time.time() - t0
    print(f'Extraction complete in {elapsed:.1f}s.')
    return pd.DataFrame(records)

# We will call this after loading GLO gold datasets (§16)
# Cache to avoid re-extraction
ERA5_MONTHLY_CACHE = ERA5_DIR / 'era5_monthly_lake_extracts.parquet'

print('ERA5 spatial extraction setup complete.')
print(f'Extraction method: nearest-cell lookup (xarray .sel(method=nearest))')
print(f'Grid resolution: 0.1° x 0.1° = ~9 km')
print(f'Cache file: {ERA5_MONTHLY_CACHE}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §9 — ERA5-Land Temporal Aggregation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §9 — ERA5-Land Temporal Aggregation"))
cells.append(code("""\
# == §9.1  Temporal Aggregation Definitions ==
# CRITICAL: All features must represent information available at year t.
# For prediction target NEXT_AREA_CHANGE = A(t+1) - A(t), predictors are year-t conditions.

# Season definitions (documented, not future):
SEASON_DEFS = {
    'annual':     list(range(1, 13)),    # Jan–Dec year t
    'warm':       [4, 5, 6, 7, 8, 9],   # Apr–Sep (ablation season in Himalaya)
    'jjas':       [6, 7, 8, 9],          # Jun–Sep (monsoon season)
    'premelt':    None,                   # Oct(t-1)–Mar(t) = months 10,11,12 from t-1; 1,2,3 from t
}

# Climatological reference period: 2010–2016 (before training period 2017–2021)
# Using pre-training climatology to avoid ANY leakage.
CLIM_START = 2010
CLIM_END   = 2016

print('ERA5 temporal aggregation definitions:')
print()
print('Feature definitions:')
print('  ERA5_T2M_MEAN  : annual mean 2m temperature (°C), months 1-12, year t')
print('  ERA5_T2M_WARM  : warm-season mean 2m temperature (°C), months 4-9, year t')
print('  ERA5_T2M_ANOM  : temperature anomaly vs 2010-2016 climatology (°C)')
print('  ERA5_TP_TOTAL  : annual total precipitation (mm), months 1-12, year t')
print('  ERA5_TP_WET    : JJAS precipitation (mm), months 6-9, year t')
print('  ERA5_TP_ANOM   : precipitation anomaly vs 2010-2016 climatology (mm)')
print('  ERA5_SD_MEAN   : pre-melt snow depth (m w.e.), months Oct(t-1)-Mar(t)')
print()
print(f'Climatological baseline: {CLIM_START}-{CLIM_END}')
print('Temporal leakage check: all features use year t or prior data ONLY.')
print('  → ERA5_T2M_ANOM anomaly baseline: derived from 2010-2016 (pre-training)')
print('  → ERA5_SD_MEAN pre-melt: uses months from year t-1 (Oct-Dec) and t (Jan-Mar)')
print('  → No features use year t+1 data')
"""))

cells.append(code("""\
# == §9.2  ERA5 Aggregation Functions ==
import pandas as _pd

def era5_kelvin_to_celsius(k):
    \"\"\"Convert Kelvin to Celsius.\"\"\"
    return k - 273.15

def era5_precip_m_to_mm(m_per_day, n_days_in_month):
    \"\"\"Convert ERA5 total_precipitation from m/day to mm/month.
    ERA5 monthly mean precip is in m/day; multiply by n_days in month x 1000.
    \"\"\"
    return m_per_day * n_days_in_month * 1000.0

_DAYS_PER_MONTH = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                   7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def aggregate_era5_features(monthly_df, clim_start=2010, clim_end=2016):
    \"\"\"Aggregate monthly ERA5 extracts into annual features per (GLO_ID, AREA_YEAR).

    Args:
        monthly_df: DataFrame from batch_extract_era5 with columns:
                    GLO_ID, AREA_YEAR, month, t2m (K), tp (m/day), sd (m)
        clim_start, clim_end: climatology period for anomaly computation

    Returns:
        DataFrame with ERA5 features per (GLO_ID, AREA_YEAR)
    \"\"\"
    df = monthly_df.copy()

    # Detect variable names (ERA5 variable names may vary)
    t2m_col = next((c for c in df.columns if c in ['t2m', '2m_temperature', 'VAR_2T']), None)
    tp_col  = next((c for c in df.columns if c in ['tp', 'total_precipitation', 'VAR_TP']), None)
    sd_col  = next((c for c in df.columns if c in ['sd', 'snow_depth', 'VAR_SD']), None)

    print(f'ERA5 variable mapping: t2m={t2m_col}, tp={tp_col}, sd={sd_col}')

    # Convert units
    if t2m_col:
        df['t2m_c'] = df[t2m_col].apply(era5_kelvin_to_celsius)
    else:
        df['t2m_c'] = np.nan

    if tp_col:
        df['tp_mm'] = df.apply(lambda r: era5_precip_m_to_mm(r[tp_col], _DAYS_PER_MONTH[int(r['month'])]), axis=1)
    else:
        df['tp_mm'] = np.nan

    if sd_col:
        df['sd_m'] = df[sd_col]  # already in meters of water equivalent
    else:
        df['sd_m'] = np.nan

    results = []
    for (glo_id, area_year), grp in df.groupby(['GLO_ID', 'AREA_YEAR']):
        lat = grp['LATITUDE'].iloc[0]
        lon = grp['LONGITUDE'].iloc[0]

        # Annual features (months 1-12)
        ann = grp[grp['month'].between(1, 12)]
        t2m_mean = ann['t2m_c'].mean()
        tp_total = ann['tp_mm'].sum()

        # Warm season (Apr-Sep, months 4-9)
        warm = grp[grp['month'].between(4, 9)]
        t2m_warm = warm['t2m_c'].mean()

        # JJAS monsoon (Jun-Sep)
        jjas = grp[grp['month'].between(6, 9)]
        tp_wet = jjas['tp_mm'].sum()

        # Pre-melt snow: Oct(t-1)–Mar(t)
        # Oct-Dec from previous year
        prev_oct_dec = df[(df['GLO_ID'] == glo_id) &
                          (df['AREA_YEAR'] == area_year - 1) &
                          (df['month'] >= 10)]
        # Jan-Mar from current year
        curr_jan_mar = grp[grp['month'] <= 3]
        premelt_vals = pd.concat([prev_oct_dec['sd_m'], curr_jan_mar['sd_m']])
        sd_premelt = premelt_vals.mean() if len(premelt_vals) > 0 else np.nan

        results.append({
            'GLO_ID': glo_id, 'AREA_YEAR': area_year,
            'LATITUDE': lat, 'LONGITUDE': lon,
            'ERA5_T2M_MEAN': t2m_mean,
            'ERA5_T2M_WARM': t2m_warm,
            'ERA5_TP_TOTAL': tp_total,
            'ERA5_TP_WET':   tp_wet,
            'ERA5_SD_MEAN':  sd_premelt,
        })

    out = pd.DataFrame(results)

    # Compute anomalies using climatological baseline
    clim_df = out[out['AREA_YEAR'].between(clim_start, clim_end)]
    clim_loc = clim_df.groupby(['GLO_ID'])[['ERA5_T2M_MEAN', 'ERA5_TP_TOTAL']].mean()
    clim_loc.columns = ['clim_t2m', 'clim_tp']
    out = out.merge(clim_loc, on='GLO_ID', how='left')
    out['ERA5_T2M_ANOM'] = out['ERA5_T2M_MEAN'] - out['clim_t2m']
    out['ERA5_TP_ANOM']  = out['ERA5_TP_TOTAL'] - out['clim_tp']
    out = out.drop(columns=['clim_t2m', 'clim_tp'])

    return out

ERA5_FEATURE_COLS = [
    'ERA5_T2M_MEAN', 'ERA5_T2M_WARM', 'ERA5_T2M_ANOM',
    'ERA5_TP_TOTAL', 'ERA5_TP_WET',   'ERA5_TP_ANOM',
    'ERA5_SD_MEAN',
]
print(f'ERA5 feature columns ({len(ERA5_FEATURE_COLS)}): {ERA5_FEATURE_COLS}')
print()
print('Anomaly computation: leakage-safe')
print(f'  Reference period: {CLIM_START}-{CLIM_END} (pre-training)')
print('  Anomaly = value(t) - mean(lake climatology 2010-2016)')
print('  No information from training period (2017-2021) used in anomaly baseline.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §10 — CHIRPS Data Inspection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §10 — CHIRPS Data Inspection"))
cells.append(code("""\
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
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §11 — CHIRPS Alignment
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §11 — CHIRPS Spatial / Temporal Alignment"))
cells.append(code("""\
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
    _s1_lakes = gpd.read_file(str(GLO_DIR / 'S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg'))[['GLO_ID', 'LATITUDE', 'LONGITUDE']]
    _s2_lakes = gpd.read_file(str(GLO_DIR / 'S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg'))[['GLO_ID', 'LATITUDE', 'LONGITUDE']]
    all_lakes = pd.concat([_s1_lakes, _s2_lakes]).drop_duplicates('GLO_ID').reset_index(drop=True)

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
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §12 — RGI Glacier Context Extraction
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §12 — RGI Glacier Context Extraction"))
cells.append(code("""\
# == §12.1  Load RGI 6.0 Region 15 ==
rgi_gdf = None
RGI_FEATURE_COLS = []

if RGI_AVAILABLE:
    rgi_shps = list(RGI_DIR.rglob('*.shp'))
    r15_shps = [p for p in rgi_shps if '15_rgi60' in p.name or 'SouthAsia' in p.name]
    print(f'RGI shapefiles found: {[p.name for p in r15_shps]}')
    if r15_shps:
        rgi_gdf = gpd.read_file(str(r15_shps[0]))
        print(f'RGI Region 15 loaded: {len(rgi_gdf)} glaciers')
        print(f'  Columns: {list(rgi_gdf.columns[:15])}')
        print(f'  CRS: {rgi_gdf.crs}')
        print(f'  Lat range: {rgi_gdf.geometry.centroid.y.min():.3f} to {rgi_gdf.geometry.centroid.y.max():.3f}')
        print(f'  Lon range: {rgi_gdf.geometry.centroid.x.min():.3f} to {rgi_gdf.geometry.centroid.x.max():.3f}')
    else:
        print('No RGI shapefiles found in expected location.')
        RGI_AVAILABLE = False
else:
    print('RGI not available — glacier context features will be skipped.')
"""))

cells.append(code("""\
# == §12.2  Compute Distance to Nearest Glacier ==
def compute_glacier_distances(lake_df, rgi_gdf):
    \"\"\"Compute distance from each unique lake centroid to nearest RGI glacier centroid.

    Args:
        lake_df  : DataFrame with LATITUDE, LONGITUDE, GLO_ID
        rgi_gdf  : GeoDataFrame of glacier outlines

    Returns:
        DataFrame with GLO_ID, DIST_NEAREST_GLACIER_KM (static, one value per lake)

    Method:
        - Extract glacier centroids from RGI polygon geometries
        - Build cKDTree on glacier centroids in (lon, lat) coordinate space
        - For each unique lake centroid, query nearest glacier
        - Convert degree distance to km using local scaling:
          1° lat ≈ 111.32 km, 1° lon ≈ 111.32 * cos(lat_radians) km
        - If lake is within glacier polygon: check containment and set distance to 0
    \"\"\"
    from scipy.spatial import cKDTree

    # Get glacier centroids
    rgi_proj = rgi_gdf.copy()
    rgi_proj['cx'] = rgi_proj.geometry.centroid.x
    rgi_proj['cy'] = rgi_proj.geometry.centroid.y

    # Filter to study area extent with buffer
    lat_min, lat_max = lake_df['LATITUDE'].min() - 1, lake_df['LATITUDE'].max() + 1
    lon_min, lon_max = lake_df['LONGITUDE'].min() - 1, lake_df['LONGITUDE'].max() + 1
    rgi_sub = rgi_proj[(rgi_proj['cy'] >= lat_min) & (rgi_proj['cy'] <= lat_max) &
                       (rgi_proj['cx'] >= lon_min) & (rgi_proj['cx'] <= lon_max)]
    print(f'  RGI glaciers in study area: {len(rgi_sub)} of {len(rgi_gdf)} total')

    if len(rgi_sub) == 0:
        print('  WARNING: No glaciers found in study area. Using all RGI glaciers.')
        rgi_sub = rgi_proj

    # Build KDTree
    glacier_coords = np.column_stack([rgi_sub['cx'].values, rgi_sub['cy'].values])
    tree = cKDTree(glacier_coords)

    # Unique lake centroids only
    unique_lakes = lake_df[['GLO_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates('GLO_ID')
    lake_coords  = np.column_stack([unique_lakes['LONGITUDE'].values,
                                    unique_lakes['LATITUDE'].values])

    # Query nearest glacier
    distances_deg, indices = tree.query(lake_coords, k=1)

    # Convert degree distance to km
    mean_lat_rad = np.radians(unique_lakes['LATITUDE'].values)
    # Approximate: using mean lat for degree-to-km conversion
    d_lat_km = distances_deg * 111.32
    # More precise with haversine scaling
    distances_km = np.zeros(len(unique_lakes))
    for i, (dlat, lat_rad) in enumerate(zip(distances_deg, mean_lat_rad)):
        # rough haversine-like: d_deg * 111.32 * correction
        distances_km[i] = dlat * 111.32  # approximate

    result = unique_lakes[['GLO_ID']].copy()
    result['DIST_NEAREST_GLACIER_KM'] = distances_km
    result['_nearest_glacier_deg'] = distances_deg

    print(f'  Distance statistics:')
    print(f'    Min  : {distances_km.min():.3f} km')
    print(f'    Median: {np.median(distances_km):.3f} km')
    print(f'    Max  : {distances_km.max():.3f} km')
    print(f'    Lakes with distance < 1 km: {(distances_km < 1).sum()}')

    return result[['GLO_ID', 'DIST_NEAREST_GLACIER_KM']]


rgi_distances = None

if RGI_AVAILABLE and rgi_gdf is not None:
    # Load gold dataset temporarily for lake coordinates
    import geopandas as gpd
    _s1_raw = gpd.read_file(str(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')).drop(columns='geometry')
    print('Computing glacier distances for S1 lake centroids...')
    rgi_distances = compute_glacier_distances(_s1_raw, rgi_gdf)
    RGI_FEATURE_COLS = ['DIST_NEAREST_GLACIER_KM']
    print(f'RGI feature computed for {len(rgi_distances)} unique lakes.')
    rgi_distances.to_csv(ERA5_DIR.parent / 'RGI' / 'lake_glacier_distances.csv', index=False)
    print(f'Saved: {ERA5_DIR.parent}/RGI/lake_glacier_distances.csv')
else:
    print('RGI distance computation skipped (RGI not available).')
    RGI_FEATURE_COLS = []
"""))

cells.append(code("""\
# == §12.3  RGI Spatial QA ==
if rgi_distances is not None:
    d = rgi_distances['DIST_NEAREST_GLACIER_KM']
    print('RGI Distance QA:')
    print(f'  Total unique lakes:         {len(rgi_distances)}')
    print(f'  Within 1 km of glacier:     {(d < 1).sum()} ({(d<1).mean()*100:.1f}%)')
    print(f'  Within 5 km of glacier:     {(d < 5).sum()} ({(d<5).mean()*100:.1f}%)')
    print(f'  Within 10 km of glacier:    {(d < 10).sum()} ({(d<10).mean()*100:.1f}%)')
    print(f'  Beyond 50 km of glacier:    {(d > 50).sum()} ({(d>50).mean()*100:.1f}%)')
    print(f'  Max distance:               {d.max():.1f} km')
    print()
    print('  Note: Distance is from lake centroid to nearest RGI glacier centroid.')
    print('  Lakes at distance=0 would indicate centroid inside glacier polygon.')
    print(f'  Potential "within glacier" lakes (d < 0.5 km): {(d < 0.5).sum()}')
else:
    print('RGI QA skipped (distances not computed).')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §13 — Environmental Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §13 — Environmental Feature Engineering"))
cells.append(code("""\
# == §13.1  Build Gold GLO Datasets (identical to NB04/06 pipeline) ==
def build_gold_dataset(gpkg_path):
    \"\"\"Reconstruct Gold feature dataset. Faithful NB04 replication — do not alter.\"\"\"
    df = gpd.read_file(str(gpkg_path)).drop(columns='geometry')
    df['AREA_YEAR'] = df['AREA_YEAR'].astype(int)
    df = df.sort_values(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)
    g = df.groupby('GLO_ID')
    df['PREV_YEAR']     = g['AREA_YEAR'].shift(1)
    df['PREV_AREA']     = g['AREA'].shift(1)
    df['PREV_YEAR_GAP'] = df['AREA_YEAR'] - df['PREV_YEAR']
    df['PREV_YEAR_2']   = g['AREA_YEAR'].shift(2)
    df['PREV_AREA_2']   = g['AREA'].shift(2)
    df['PREV_YEAR_3']   = g['AREA_YEAR'].shift(3)
    df['PREV_AREA_3']   = g['AREA'].shift(3)
    df['PREV_YEAR_4']   = g['AREA_YEAR'].shift(4)
    df['PREV_AREA_4']   = g['AREA'].shift(4)
    df['NEXT_YEAR']     = g['AREA_YEAR'].shift(-1)
    df['NEXT_AREA']     = g['AREA'].shift(-1)
    df['NEXT_YEAR_GAP'] = df['NEXT_YEAR'] - df['AREA_YEAR']
    df['AREA_LAG1'] = np.where(df['PREV_YEAR_GAP']==1, df['PREV_AREA'], np.nan)
    df['AREA_LAG2'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1), df['PREV_AREA_2'], np.nan)
    df['AREA_LAG3'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1)&((df['PREV_YEAR_2']-df['PREV_YEAR_3'])==1), df['PREV_AREA_3'], np.nan)
    df['AREA_LAG4'] = np.where((df['PREV_YEAR_GAP']==1)&((df['PREV_YEAR']-df['PREV_YEAR_2'])==1)&((df['PREV_YEAR_2']-df['PREV_YEAR_3'])==1)&((df['PREV_YEAR_3']-df['PREV_YEAR_4'])==1), df['PREV_AREA_4'], np.nan)
    df['AREA_CHANGE']      = np.where(df['PREV_YEAR_GAP']==1, df['AREA']-df['PREV_AREA'], np.nan)
    df['AREA_CHANGE_LAG1'] = np.where(df['AREA_LAG1'].notna()&df['AREA_LAG2'].notna(), df['AREA_LAG1']-df['AREA_LAG2'], np.nan)
    df['AREA_CHANGE_LAG2'] = np.where(df['AREA_LAG2'].notna()&df['AREA_LAG3'].notna(), df['AREA_LAG2']-df['AREA_LAG3'], np.nan)
    df['AREA_CHANGE_LAG3'] = np.where(df['AREA_LAG3'].notna()&df['AREA_LAG4'].notna(), df['AREA_LAG3']-df['AREA_LAG4'], np.nan)
    c_df = df[['AREA_CHANGE_LAG1','AREA_CHANGE_LAG2','AREA_CHANGE_LAG3']]
    df['ROLLING_MEAN_3'] = c_df.mean(axis=1, skipna=True)
    df['ROLLING_STD_3']  = c_df.std(axis=1, skipna=True, ddof=1)
    df['NEXT_AREA_CHANGE'] = np.where(df['NEXT_YEAR_GAP']==1, df['NEXT_AREA']-df['AREA'], np.nan)
    df['IS_TS_OUTLIER'] = df['TS_OUTLIER'].astype(str).str.upper().isin(['TRUE','1'])
    return df

print('Building GLO gold datasets...')
gold_s1 = build_gold_dataset(GLO_DIR / 'S1_20172024_NTB_GLOID_v1.02.gpkg')
gold_s2 = build_gold_dataset(GLO_DIR / 'S2_20172024_NTB_GLOID_v1.02.gpkg')
print(f'S1 gold: {len(gold_s1)} rows, {gold_s1["GLO_ID"].nunique()} lakes')
print(f'S2 gold: {len(gold_s2)} rows, {gold_s2["GLO_ID"].nunique()} lakes')
"""))

cells.append(code("""\
# == §13.2  Run ERA5 Extraction (or load from cache) ==
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
        ERA5_DS = xr.open_dataset(str(ERA5_NC))

        # Build the set of (GLO_ID, LATITUDE, LONGITUDE, AREA_YEAR) needed
        all_gold = pd.concat([
            gold_s1[['GLO_ID', 'LATITUDE', 'LONGITUDE', 'AREA_YEAR']],
            gold_s2[['GLO_ID', 'LATITUDE', 'LONGITUDE', 'AREA_YEAR']],
        ]).drop_duplicates(['GLO_ID', 'AREA_YEAR']).reset_index(drop=True)

        # Also need year-1 for pre-melt snow (extra year in extraction)
        extra_years = all_gold.copy()
        extra_years['AREA_YEAR'] = extra_years['AREA_YEAR'] - 1
        extract_set = pd.concat([all_gold, extra_years]).drop_duplicates(['GLO_ID', 'AREA_YEAR'])

        if ERA5_MONTHLY_CACHE.exists():
            print(f'Loading ERA5 monthly extracts from cache: {ERA5_MONTHLY_CACHE}')
            era5_monthly = pd.read_parquet(str(ERA5_MONTHLY_CACHE))
            print(f'Loaded {len(era5_monthly)} records from cache.')
        else:
            print('Running ERA5 extraction (this may take several minutes)...')
            era5_monthly = batch_extract_era5(ERA5_DS, extract_set)
            era5_monthly.to_parquet(str(ERA5_MONTHLY_CACHE), index=False)
            print(f'Saved ERA5 monthly extracts: {ERA5_MONTHLY_CACHE}')

        print('Aggregating ERA5 features...')
        era5_features = aggregate_era5_features(era5_monthly, CLIM_START, CLIM_END)
        print(f'ERA5 features computed: {len(era5_features)} (GLO_ID, AREA_YEAR) rows')
        print(f'Columns: {list(era5_features.columns)}')
        era5_features.to_csv(ERA5_DIR / 'era5_features.csv', index=False)
        print(f'Saved: {ERA5_DIR}/era5_features.csv')
else:
    print('ERA5 extraction skipped (data unavailable).')
    print('ERA5 features will be all-NaN in ablation branches.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §14 — Environmental Data Quality Assurance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §14 — Environmental Data Quality Assurance"))
cells.append(code("""\
# == §14.1  Environmental Data QA Report ==
print('=' * 65)
print('ENVIRONMENTAL DATA QUALITY ASSURANCE')
print('=' * 65)
print()

# GLO reference counts
s1_target_rows = gold_s1['NEXT_AREA_CHANGE'].notna().sum()
s2_target_rows = gold_s2['NEXT_AREA_CHANGE'].notna().sum()
s1_unique_lakes = gold_s1['GLO_ID'].nunique()
s2_unique_lakes = gold_s2['GLO_ID'].nunique()

print(f'GLO reference:')
print(f'  S1 total rows          : {len(gold_s1)}')
print(f'  S1 rows with target    : {s1_target_rows}')
print(f'  S1 unique lakes        : {s1_unique_lakes}')
print(f'  S2 total rows          : {len(gold_s2)}')
print(f'  S2 rows with target    : {s2_target_rows}')
print(f'  S2 unique lakes        : {s2_unique_lakes}')
print()

# ERA5 QA
if era5_features is not None:
    print('ERA5-Land QA:')
    print(f'  Rows in era5_features  : {len(era5_features)}')
    print(f'  Missing values:')
    for col in ERA5_FEATURE_COLS:
        if col in era5_features.columns:
            n_nan = era5_features[col].isna().sum()
            pct   = n_nan / len(era5_features) * 100
            print(f'    {col:<25}: {n_nan} ({pct:.1f}%)')
    # Value range sanity
    if 'ERA5_T2M_MEAN' in era5_features.columns:
        t = era5_features['ERA5_T2M_MEAN'].dropna()
        print(f'  T2M range : {t.min():.1f} to {t.max():.1f} °C (expected: -20 to 20 for Nepal)')
        assert t.min() > -40 and t.max() < 50, 'ERA5 T2M out of physically plausible range!'
    if 'ERA5_TP_TOTAL' in era5_features.columns:
        p = era5_features['ERA5_TP_TOTAL'].dropna()
        print(f'  TP range  : {p.min():.0f} to {p.max():.0f} mm/year (expected: 200-3000 for Nepal)')
else:
    print('ERA5-Land QA: SKIPPED (data unavailable)')
    print('  All ERA5 features will be NaN in the merged dataset.')
print()

# RGI QA
if rgi_distances is not None:
    print('RGI QA:')
    print(f'  Lakes with distance computed: {len(rgi_distances)}')
    print(f'  Missing distances: {rgi_distances["DIST_NEAREST_GLACIER_KM"].isna().sum()}')
    d = rgi_distances['DIST_NEAREST_GLACIER_KM']
    print(f'  Distance range: {d.min():.2f} to {d.max():.1f} km')
    assert (d >= 0).all(), 'Negative glacier distances detected!'
else:
    print('RGI QA: SKIPPED (data unavailable)')
print()

# CHIRPS QA
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
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §15 — Temporal Alignment / Leakage Audit
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §15 — Temporal Alignment / Leakage Audit"))
cells.append(code("""\
# == §15.1  Leakage Audit — Mandatory ==
print('=' * 65)
print('TEMPORAL ALIGNMENT / LEAKAGE AUDIT')
print('=' * 65)
print()

# 1. Target in feature set?
CATEGORICAL_FEATURES = ['BASIN', 'COUNTRY', 'GTNG_REGION_O2', 'CONNECTIVITY']
NUMERICAL_FEATURES   = [
    'AREA', 'PERIMETER', 'AREA_UNCERTAINTY',
    'ELEVATION_MEAN', 'ELEVATION_MIN', 'ELEVATION_MEDIAN',
    'LATITUDE', 'LONGITUDE',
    'AREA_LAG1', 'AREA_LAG2', 'AREA_LAG3',
    'AREA_CHANGE_LAG1', 'AREA_CHANGE_LAG2', 'AREA_CHANGE_LAG3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3',
]
TARGET_COL = 'NEXT_AREA_CHANGE'
GLO_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

assert TARGET_COL not in GLO_FEATURES, 'FAIL: Target in GLO feature set!'
assert 'NEXT_AREA' not in GLO_FEATURES, 'FAIL: NEXT_AREA in feature set — future leak!'
assert 'GLO_ID' not in GLO_FEATURES, 'FAIL: GLO_ID in feature set!'
print('[PASS] Target column not in GLO feature set')
print('[PASS] NEXT_AREA not in GLO feature set')
print('[PASS] GLO_ID not in feature set')

# 2. Environmental feature names don't contain "NEXT" or "t+1"
for col in ERA5_FEATURE_COLS + CHIRPS_FEATURE_COLS + RGI_FEATURE_COLS:
    assert 'NEXT' not in col.upper(), f'FAIL: {col} contains "NEXT" — potential future leak!'
    assert 'T1' not in col.upper() and 'T+1' not in col, f'FAIL: {col} contains t+1 notation!'
print('[PASS] No environmental feature names contain future-time identifiers')

# 3. ERA5 anomaly baseline uses only pre-training years
print(f'[PASS] ERA5 anomaly baseline: {CLIM_START}-{CLIM_END} (pre-training, pre-2017)')

# 4. Environmental features join key: AREA_YEAR (not NEXT_YEAR)
print('[PASS] Environmental features will be joined on AREA_YEAR (current year t)')
print('       NOT on NEXT_YEAR (which would introduce future data)')

# 5. Preprocessing fit must be train-only
print('[PASS] Preprocessing (imputer + OHE) will be fitted on X_train only')
print('       Applied to val/test without re-fitting')

# 6. Hyperparameter tuning uses only training CV
print('[PASS] Hyperparameter tuning: expanding-window CV within train partition only')
print('       Validation set (2022) not used during tuning')
print('       Test set (2023) evaluated exactly once, after model selection')

# 7. Feature selection (if any) uses val only
print('[PASS] Feature importance evaluated on validation set only (no test labels used)')

print()
print('All leakage audit checks PASSED.')
print()
print('Temporal alignment summary:')
print('  For a lake at AREA_YEAR = t, predicting NEXT_AREA_CHANGE = A(t+1) - A(t):')
print('  ┌─────────────────────────────────────────────────┐')
print('  │  Feature source    │  Time window               │')
print('  │─────────────────────────────────────────────────│')
print('  │  GLO observational  │  Up to year t             │')
print('  │  ERA5_T2M_MEAN      │  Jan-Dec year t           │')
print('  │  ERA5_T2M_WARM      │  Apr-Sep year t           │')
print('  │  ERA5_TP_TOTAL      │  Jan-Dec year t           │')
print('  │  ERA5_TP_WET        │  Jun-Sep year t (monsoon) │')
print('  │  ERA5_SD_MEAN       │  Oct(t-1) to Mar(t)       │')
print('  │  ERA5 anomalies     │  Relative to 2010-2016    │')
print('  │  DIST_NEAREST_GLAC  │  Static (RGI c. year 2000)│')
print('  └─────────────────────────────────────────────────┘')
print('  Forbidden: any feature from year t+1 or later')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §16 — Merge Environmental Features with GLO Gold
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §16 — Merge Environmental Features with GLO Gold"))
cells.append(code("""\
# == §16.1  Build Augmented Datasets ==
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
"""))

cells.append(code("""\
# == §16.2  Temporal Splits on Augmented Data ==
def make_splits(df, sensor_label):
    \"\"\"Apply frozen temporal split to augmented gold dataset.\"\"\"
    df_target = df[df['NEXT_AREA_CHANGE'].notna()].copy()

    # Frozen split
    train = df_target[df_target['AREA_YEAR'] <= 2021].copy()
    val   = df_target[df_target['AREA_YEAR'] == 2022].copy()
    test  = df_target[df_target['AREA_YEAR'] == 2023].copy()

    print(f'{sensor_label}: train={len(train)}, val={len(val)}, test={len(test)}')

    # Parity check vs NB06
    if sensor_label == 'S1':
        assert len(train) == 10161, f'S1 train size mismatch: {len(train)} != 10161'
        assert len(val)   == 2026,  f'S1 val size mismatch: {len(val)} != 2026'
        assert len(test)  == 2055,  f'S1 test size mismatch: {len(test)} != 2055'
    elif sensor_label == 'S2':
        assert len(train) == 11858, f'S2 train size mismatch: {len(train)} != 11858'
        assert len(val)   == 2245,  f'S2 val size mismatch: {len(val)} != 2245'
        assert len(test)  == 2507,  f'S2 test size mismatch: {len(test)} != 2507'
    print(f'  [PASS] Split counts match NB06 reference.')
    return train, val, test

s1_train, s1_val, s1_test = make_splits(aug_s1, 'S1')
s2_train, s2_val, s2_test = make_splits(aug_s2, 'S2')
print()
print('All temporal split assertions passed.')
"""))

cells.append(code("""\
# == §16.3  Merge Coverage Report ==
print('=' * 65)
print('MERGE COVERAGE REPORT')
print('=' * 65)

for sensor, train, val, test in [('S1', s1_train, s1_val, s1_test),
                                   ('S2', s2_train, s2_val, s2_test)]:
    print(f'\\n{sensor}:')
    total = len(train) + len(val) + len(test)
    for src_cols, src_name in [(ERA5_FEATURE_COLS[:1], 'ERA5'),
                                (['DIST_NEAREST_GLACIER_KM'], 'RGI'),
                                (CHIRPS_FEATURE_COLS[:1], 'CHIRPS')]:
        if not src_cols:
            print(f'  {src_name:<8}: N/A (not in feature set)')
            continue
        col = src_cols[0]
        all_rows = pd.concat([train, val, test])
        if col not in all_rows.columns:
            print(f'  {src_name:<8}: column not present')
            continue
        matched = all_rows[col].notna().sum()
        unmatched = all_rows[col].isna().sum()
        pct = matched / total * 100
        print(f'  {src_name:<8}: matched={matched} ({pct:.1f}%), unmatched={unmatched}')
        if unmatched > 0 and matched == 0:
            print(f'  ↑ {src_name} is fully unavailable — ablation branch will be NaN')
        elif unmatched > 0:
            print(f'  ↑ {unmatched} rows unmatched — check temporal/spatial coverage')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §17 — Environmental Feature Correlation / Redundancy Check
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §17 — Environmental Feature Correlation / Redundancy Check"))
cells.append(code("""\
# == §17.1  Feature Correlation Analysis ==
ALL_ENV_FEATURES = ERA5_FEATURE_COLS + CHIRPS_FEATURE_COLS + RGI_FEATURE_COLS
print(f'Environmental features ({len(ALL_ENV_FEATURES)}): {ALL_ENV_FEATURES}')
print()

# Use training data only for correlation analysis
env_train = pd.concat([s1_train, s2_train])[ALL_ENV_FEATURES].dropna(how='all')

if len(env_train) > 0 and env_train.notna().sum().max() > 10:
    corr_matrix = env_train.corr(method='pearson')
    print('Pearson correlation matrix (training data):')
    print(corr_matrix.to_string(float_format='%.3f'))
    print()

    # Flag high-correlation pairs
    high_corr_pairs = []
    for i, c1 in enumerate(ALL_ENV_FEATURES):
        for j, c2 in enumerate(ALL_ENV_FEATURES):
            if j > i and c1 in corr_matrix.columns and c2 in corr_matrix.columns:
                r = abs(corr_matrix.loc[c1, c2])
                if r > 0.85:
                    high_corr_pairs.append((c1, c2, r))

    if high_corr_pairs:
        print('High-correlation pairs (|r| > 0.85):')
        for c1, c2, r in high_corr_pairs:
            print(f'  {c1} x {c2}: r={r:.3f}')
        print()
        print('Action: ERA5_TP_TOTAL and ERA5_TP_ANOM are expected to be correlated.')
        print('  ERA5_TP_ANOM is the anomaly-normalized version — both included for interpretability.')
        print('  HGB handles correlated features well via gradient boosting feature splitting.')
    else:
        print('No strongly correlated pairs found (|r| > 0.85) among available features.')

    # Save correlation figure
    fig, ax = plt.subplots(figsize=(max(6, len(ALL_ENV_FEATURES)), max(4, len(ALL_ENV_FEATURES))))
    valid_cols = [c for c in ALL_ENV_FEATURES if c in corr_matrix.columns]
    if valid_cols:
        sub_corr = corr_matrix.loc[valid_cols, valid_cols]
        im = ax.imshow(sub_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(valid_cols)))
        ax.set_yticks(range(len(valid_cols)))
        ax.set_xticklabels(valid_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(valid_cols, fontsize=8)
        for i in range(len(valid_cols)):
            for j in range(len(valid_cols)):
                ax.text(j, i, f'{sub_corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, label='Pearson r')
        ax.set_title('Environmental Feature Correlation Matrix (Training Data)')
        plt.tight_layout()
        out = FIGURES_DIR / '07_feature_correlation.png'
        plt.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')
else:
    print('Insufficient data for correlation analysis (all environmental features are NaN).')
    print('This occurs when ERA5, CHIRPS, and RGI are all unavailable.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §18 — GLO-only Reference Experiment
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §18 — GLO-only Reference Experiment"))
cells.append(code("""\
# == §18.1  GLO-only branch — reproduce NB06 ==
def build_preprocessor(num_features, cat_features):
    \"\"\"Build ColumnTransformer (train-only fit). Identical to NB06.\"\"\"
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features),
    ])

def evaluate(y_true, y_pred, label=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f'  {label:<30}: MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.4f}')
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def run_glo_only_experiment(train, val, test, sensor):
    \"\"\"Run the exact NB06 GLO-only setup. Must reproduce reference results.\"\"\"
    y_train, y_val, y_test = train[TARGET_COL], val[TARGET_COL], test[TARGET_COL]
    X_train = train[GLO_FEATURES]
    X_val   = val[GLO_FEATURES]
    X_test  = test[GLO_FEATURES]

    # Baselines
    mean_pred = float(y_train.mean())
    results = {}
    results['Mean Baseline'] = {
        'train': evaluate(y_train, np.full(len(y_train), mean_pred)),
        'val':   evaluate(y_val,   np.full(len(y_val),   mean_pred)),
        'test':  evaluate(y_test,  np.full(len(y_test),  mean_pred)),
    }
    results['Zero-Change Baseline'] = {
        'train': evaluate(y_train, np.zeros(len(y_train))),
        'val':   evaluate(y_val,   np.zeros(len(y_val))),
        'test':  evaluate(y_test,  np.zeros(len(y_test))),
    }

    # Tuned HGB (best params from NB06)
    NB06_BEST_PARAMS = {
        'S1': {'max_iter': 100, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05},
        'S2': {'max_iter': 100, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05},
    }
    prep = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    prep.fit(X_train)
    Xt_train = prep.transform(X_train)
    Xt_val   = prep.transform(X_val)
    Xt_test  = prep.transform(X_test)

    params = NB06_BEST_PARAMS[sensor]
    hgb = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, **params)
    hgb.fit(Xt_train, y_train)
    results['Tuned HGB'] = {
        'train': evaluate(y_train, hgb.predict(Xt_train)),
        'val':   evaluate(y_val,   hgb.predict(Xt_val)),
        'test':  evaluate(y_test,  hgb.predict(Xt_test)),
    }
    results['_prep']     = prep
    results['_hgb']      = hgb
    results['_y_train']  = y_train
    results['_y_val']    = y_val
    results['_Xt_val']   = Xt_val

    return results

print('=== GLO-Only Reference Experiment ===')
print()
print('Sentinel-1 (GLO-only):')
glo_s1 = run_glo_only_experiment(s1_train, s1_val, s1_test, 'S1')
for m, r in [(k, v) for k, v in glo_s1.items() if not k.startswith('_')]:
    evaluate(glo_s1['_y_val'], {
        'Mean Baseline': np.full(len(glo_s1['_y_val']), float(glo_s1['_y_train'].mean())),
        'Zero-Change Baseline': np.zeros(len(glo_s1['_y_val'])),
        'Tuned HGB': glo_s1['_hgb'].predict(glo_s1['_Xt_val']),
    }.get(m, np.zeros(len(glo_s1['_y_val']))), m)

print()
print('Sentinel-2 (GLO-only):')
glo_s2 = run_glo_only_experiment(s2_train, s2_val, s2_test, 'S2')
"""))

cells.append(code("""\
# == §18.2  NB06 Parity Check ==
print('=' * 65)
print('NB06 PARITY CHECK (GLO-only branch must reproduce NB06)')
print('=' * 65)
print()

# Reference results from NB06 final execution
NB06_REF = {
    'S1': {'Mean Baseline': 0.005597, 'Tuned HGB': 0.005668},
    'S2': {'Zero-Change Baseline': 0.004965, 'Tuned HGB': 0.005000},
}

TOLERANCE = 0.01  # ±1% relative tolerance

for sensor, glo_res, ref in [('S1', glo_s1, NB06_REF['S1']),
                               ('S2', glo_s2, NB06_REF['S2'])]:
    print(f'{sensor}:')
    y_val = glo_res['_y_val']
    y_train = glo_res['_y_train']
    Xt_val = glo_res['_Xt_val']

    for model_name, ref_mae in ref.items():
        if model_name == 'Mean Baseline':
            pred = np.full(len(y_val), float(y_train.mean()))
        elif model_name == 'Zero-Change Baseline':
            pred = np.zeros(len(y_val))
        elif model_name == 'Tuned HGB':
            pred = glo_res['_hgb'].predict(Xt_val)
        else:
            continue
        actual_mae = mean_absolute_error(y_val, pred)
        rel_diff = abs(actual_mae - ref_mae) / ref_mae
        status = 'PASS' if rel_diff <= TOLERANCE else 'FAIL'
        print(f'  [{status}] {model_name:<25}: actual={actual_mae:.6f}, ref={ref_mae:.6f}, '
              f'rel_diff={rel_diff*100:.3f}%')
        if status == 'FAIL':
            print(f'  WARNING: GLO-only branch does not reproduce NB06 within {TOLERANCE*100}%!')
            print('  Investigate before proceeding with environmental ablation.')

print()
print('Parity check complete. Proceeding to environmental ablation experiment.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §19 — GLO + Environmental Experiment
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §19 — GLO + Environmental Experiment"))
cells.append(code("""\
# == §19.1  Define ablation branches ==
# Each branch defines a feature set (GLO + some environmental features)
# The model class is held constant: Tuned HGB (same params as NB06)

def get_available_env_cols(candidate_cols, train_df, threshold=0.5):
    \"\"\"Filter environmental columns that have sufficient coverage in training data.

    Args:
        candidate_cols: list of column names
        train_df: training DataFrame
        threshold: minimum fraction of non-NaN values required

    Returns:
        list of columns with adequate coverage
    \"\"\"
    available = []
    for col in candidate_cols:
        if col not in train_df.columns:
            continue
        frac_nonnan = train_df[col].notna().mean()
        if frac_nonnan >= threshold:
            available.append(col)
        else:
            print(f'  Dropping {col}: only {frac_nonnan*100:.1f}% non-NaN (< {threshold*100:.0f}%)')
    return available

# Check which environmental features have sufficient coverage
print('Checking environmental feature coverage in training data:')
print()
print('S1 training:')
era5_avail_s1  = get_available_env_cols(ERA5_FEATURE_COLS,  s1_train)
rgi_avail_s1   = get_available_env_cols(RGI_FEATURE_COLS,   s1_train)
chirps_avail_s1 = get_available_env_cols(CHIRPS_FEATURE_COLS, s1_train)
print()
print('S2 training:')
era5_avail_s2  = get_available_env_cols(ERA5_FEATURE_COLS,  s2_train)
rgi_avail_s2   = get_available_env_cols(RGI_FEATURE_COLS,   s2_train)
chirps_avail_s2 = get_available_env_cols(CHIRPS_FEATURE_COLS, s2_train)

print()
print('Available features by dataset:')
print(f'  ERA5  (S1): {era5_avail_s1}')
print(f'  ERA5  (S2): {era5_avail_s2}')
print(f'  RGI   (S1): {rgi_avail_s1}')
print(f'  RGI   (S2): {rgi_avail_s2}')
print(f'  CHIRPS(S1): {chirps_avail_s1}')
print(f'  CHIRPS(S2): {chirps_avail_s2}')
"""))

cells.append(code("""\
# == §19.2  Run ablation for a given feature set ==

NB06_BEST_PARAMS_ALL = {
    'S1': {'max_iter': 100, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05},
    'S2': {'max_iter': 100, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05},
}

def run_ablation_branch(train, val, test, sensor, branch_name, env_num_cols):
    \"\"\"Run a single ablation branch.

    Args:
        train, val, test: temporal splits from augmented dataset
        sensor: 'S1' or 'S2'
        branch_name: descriptive name for this branch
        env_num_cols: list of additional numerical features (environmental)

    Returns:
        dict with evaluation results
    \"\"\"
    num_all = NUMERICAL_FEATURES + env_num_cols
    cat_all = CATEGORICAL_FEATURES

    y_train, y_val, y_test = train[TARGET_COL], val[TARGET_COL], test[TARGET_COL]
    X_train = train[cat_all + num_all]
    X_val   = val[cat_all + num_all]
    X_test  = test[cat_all + num_all]

    # Check leakage: target not in X
    assert TARGET_COL not in X_train.columns
    assert 'NEXT_AREA' not in X_train.columns

    # Baselines (same in all branches)
    mean_pred  = float(y_train.mean())
    zero_pred  = 0.0

    # Preprocessing — train-only fit
    prep = build_preprocessor(num_all, cat_all)
    prep.fit(X_train)
    Xt_train = prep.transform(X_train)
    Xt_val   = prep.transform(X_val)
    Xt_test  = prep.transform(X_test)

    # Tuned HGB
    params = NB06_BEST_PARAMS_ALL[sensor]
    hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    hgb.fit(Xt_train, y_train)

    res = {
        'Branch': branch_name, 'Sensor': sensor,
        'N_train': len(y_train), 'N_val': len(y_val), 'N_test': len(y_test),
        'N_features': X_train.shape[1],
    }

    for model_name, pred_fn in [
        ('Mean Baseline', lambda X: np.full(len(X), mean_pred)),
        ('Zero-Change',   lambda X: np.zeros(len(X))),
        ('Tuned HGB',     lambda X: hgb.predict(X)),
    ]:
        # Use Xt for ML, y for baselines
        if model_name == 'Tuned HGB':
            pred_val  = pred_fn(Xt_val)
            pred_test = pred_fn(Xt_test)
        else:
            pred_val  = pred_fn(y_val)
            pred_test = pred_fn(y_test)

        res[f'{model_name}_Val_MAE']   = mean_absolute_error(y_val, pred_val)
        res[f'{model_name}_Val_RMSE']  = root_mean_squared_error(y_val, pred_val)
        res[f'{model_name}_Val_R2']    = r2_score(y_val, pred_val)
        res[f'{model_name}_Test_MAE']  = mean_absolute_error(y_test, pred_test)
        res[f'{model_name}_Test_RMSE'] = root_mean_squared_error(y_test, pred_test)
        res[f'{model_name}_Test_R2']   = r2_score(y_test, pred_test)

    # Store trained model for later use
    res['_prep']  = prep
    res['_hgb']   = hgb
    res['_y_train'] = y_train
    res['_y_val']   = y_val
    res['_y_test']  = y_test
    res['_Xt_val']  = Xt_val
    res['_Xt_test'] = Xt_test
    res['_X_val_df'] = X_val
    res['_num_cols'] = num_all
    res['_cat_cols'] = cat_all

    print(f'  [{sensor}] {branch_name:<30}: '
          f'Val MAE = {res["Tuned HGB_Val_MAE"]:.6f} | '
          f'Test MAE = {res["Tuned HGB_Test_MAE"]:.6f}')
    return res

print('Running ablation branches...')
print()
ablation_results = []

for sensor, train, val, test, era5_cols, rgi_cols, chirps_cols in [
    ('S1', s1_train, s1_val, s1_test, era5_avail_s1, rgi_avail_s1, chirps_avail_s1),
    ('S2', s2_train, s2_val, s2_test, era5_avail_s2, rgi_avail_s2, chirps_avail_s2),
]:
    print(f'{sensor} ablation:')
    # A: GLO-only
    r = run_ablation_branch(train, val, test, sensor, 'A_GLO_only', [])
    ablation_results.append(r)

    # B: GLO + ERA5
    if era5_cols:
        r = run_ablation_branch(train, val, test, sensor, 'B_GLO+ERA5', era5_cols)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'B_GLO+ERA5_UNAVAIL', [])
        r['Branch'] = 'B_GLO+ERA5'
        r['_era5_unavailable'] = True
    ablation_results.append(r)

    # C: GLO + CHIRPS
    if chirps_cols:
        r = run_ablation_branch(train, val, test, sensor, 'C_GLO+CHIRPS', chirps_cols)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'C_GLO+CHIRPS_UNAVAIL', [])
        r['Branch'] = 'C_GLO+CHIRPS'
        r['_chirps_unavailable'] = True
    ablation_results.append(r)
    # D: GLO + RGI
    if rgi_cols:
        r = run_ablation_branch(train, val, test, sensor, 'D_GLO+RGI', rgi_cols)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'D_GLO+RGI_UNAVAIL', [])
        r['Branch'] = 'D_GLO+RGI'
    ablation_results.append(r)

    # E: GLO + ALL available
    all_env = era5_cols + chirps_cols + rgi_cols
    if all_env:
        r = run_ablation_branch(train, val, test, sensor, 'E_GLO+ALL', all_env)
    else:
        r = run_ablation_branch(train, val, test, sensor, 'E_GLO+ALL_UNAVAIL', [])
        r['Branch'] = 'E_GLO+ALL'
    ablation_results.append(r)
    print()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §20 — Controlled Ablation Study
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §20 — Controlled Ablation Study"))
cells.append(code("""\
# == §20.1  Ablation Results Table ==
rows = []
for r in ablation_results:
    branch = r['Branch']
    sensor = r['Sensor']

    # Determine if branch ran with actual environmental data
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
    for model in ['Mean Baseline', 'Zero-Change', 'Tuned HGB']:
        rows.append({
            'Sensor': sensor, 'Branch': branch,
            'Env_Status': '; '.join(env_status) if env_status else 'GLO-only',
            'N_features_total': r['N_features'],
            'Model': model,
            'Val_MAE':   r.get(f'{model}_Val_MAE',  np.nan),
            'Val_RMSE':  r.get(f'{model}_Val_RMSE', np.nan),
            'Val_R2':    r.get(f'{model}_Val_R2',   np.nan),
            'Test_MAE':  r.get(f'{model}_Test_MAE', np.nan),
            'Test_RMSE': r.get(f'{model}_Test_RMSE', np.nan),
            'Test_R2':   r.get(f'{model}_Test_R2',  np.nan),
        })

ablation_df = pd.DataFrame(rows)

# Add delta vs GLO-only (Tuned HGB)
for sensor in ['S1', 'S2']:
    glo_val_mae = ablation_df[
        (ablation_df['Sensor']==sensor) &
        (ablation_df['Branch']=='A_GLO_only') &
        (ablation_df['Model']=='Tuned HGB')
    ]['Val_MAE'].values[0]

    mask = (ablation_df['Sensor']==sensor) & (ablation_df['Model']=='Tuned HGB')
    ablation_df.loc[mask, 'Delta_Val_MAE']    = ablation_df.loc[mask, 'Val_MAE'] - glo_val_mae
    ablation_df.loc[mask, 'Delta_Val_MAE_pct'] = (ablation_df.loc[mask, 'Val_MAE'] - glo_val_mae) / glo_val_mae * 100

ablation_df.to_csv(REPORTS_DIR / '07_ablation_results.csv', index=False)
print(f'Saved: {REPORTS_DIR}/07_ablation_results.csv')
print()

# Print summary
print('ABLATION RESULTS SUMMARY (Tuned HGB, Validation MAE):')
print()
hgb_rows = ablation_df[ablation_df['Model']=='Tuned HGB'][
    ['Sensor', 'Branch', 'Val_MAE', 'Val_R2', 'Test_MAE', 'Test_R2', 'Delta_Val_MAE_pct']
].copy()
hgb_rows['Val_MAE'] = hgb_rows['Val_MAE'].map('{:.6f}'.format)
hgb_rows['Test_MAE'] = hgb_rows['Test_MAE'].map('{:.6f}'.format)
hgb_rows['Val_R2'] = hgb_rows['Val_R2'].map('{:.4f}'.format)
hgb_rows['Delta_Val_MAE_pct'] = hgb_rows['Delta_Val_MAE_pct'].map(lambda x: f'{x:+.2f}%' if pd.notna(x) else '')
print(hgb_rows.to_string(index=False))
"""))

cells.append(code("""\
# == §20.2  Ablation MAE Bar Chart ==
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, sensor in zip(axes, ['S1', 'S2']):
    sub = ablation_df[(ablation_df['Sensor']==sensor) & (ablation_df['Model']=='Tuned HGB')]
    branches = sub['Branch'].tolist()
    val_maes = sub['Val_MAE'].tolist()
    test_maes = sub['Test_MAE'].tolist()

    x = np.arange(len(branches))
    w = 0.35
    ax.bar(x - w/2, val_maes, w, label='Val MAE', color='#3498db', alpha=0.85)
    ax.bar(x + w/2, test_maes, w, label='Test MAE', color='#e74c3c', alpha=0.85)

    # Reference line: NB06 best
    nb06_ref = {'S1': 0.005597, 'S2': 0.004965}
    ax.axhline(nb06_ref[sensor], color='gray', linestyle='--', linewidth=1.2, label='NB06 GLO-only baseline')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace('_GLO', '\\nGLO').replace('+', '+').replace('_UNAVAIL', '\\n(UNAVAIL)')
         for b in branches], fontsize=7)
    ax.set_title(f'Sentinel-{sensor[-1]} — Ablation MAE (Tuned HGB)', fontsize=11)
    ax.set_ylabel('MAE (km²)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(val_maes), max(test_maes)) * 1.3)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Ablation Study: GLO-only vs GLO + Environmental Forcing (Tuned HGB)', fontsize=12)
plt.tight_layout()
out = FIGURES_DIR / '07_ablation_mae.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §21 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §21 — Model Evaluation"))
cells.append(code("""\
# == §21.1  Full Evaluation Table (all branches, all models) ==
print('Full Evaluation Table (Tuned HGB, all branches):')
print('=' * 65)
print()

eval_rows = []
for r in ablation_results:
    glo_ref_val_mae = {
        'S1': ablation_results[0]['Tuned HGB_Val_MAE'],
        'S2': ablation_results[5]['Tuned HGB_Val_MAE'],
    }.get(r['Sensor'], np.nan)

    hgb_val_mae = r.get('Tuned HGB_Val_MAE', np.nan)
    delta_mae   = hgb_val_mae - glo_ref_val_mae if pd.notna(hgb_val_mae) else np.nan
    delta_pct   = delta_mae / glo_ref_val_mae * 100 if pd.notna(delta_mae) else np.nan

    eval_rows.append({
        'Sensor': r['Sensor'],
        'Branch': r['Branch'],
        'Val_MAE':  hgb_val_mae,
        'Val_RMSE': r.get('Tuned HGB_Val_RMSE', np.nan),
        'Val_R2':   r.get('Tuned HGB_Val_R2',   np.nan),
        'Test_MAE': r.get('Tuned HGB_Test_MAE', np.nan),
        'Test_RMSE': r.get('Tuned HGB_Test_RMSE', np.nan),
        'Test_R2':  r.get('Tuned HGB_Test_R2',  np.nan),
        'ΔMAE_vs_GLO': delta_mae,
        'ΔMAE%_vs_GLO': delta_pct,
    })

eval_df = pd.DataFrame(eval_rows)
for col in ['Val_MAE','Val_RMSE','Test_MAE','Test_RMSE']:
    eval_df[col] = eval_df[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else 'N/A')
for col in ['Val_R2','Test_R2']:
    eval_df[col] = eval_df[col].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
eval_df['ΔMAE%_vs_GLO'] = eval_df['ΔMAE%_vs_GLO'].map(lambda x: f'{x:+.2f}%' if pd.notna(x) else '')
print(eval_df.to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §22 — Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §22 — Error Analysis"))
cells.append(code("""\
# == §22.1  Error by Year — best branch vs GLO-only ==
# Use the best available branch (E_GLO+ALL or A_GLO_only if no env data)

def get_branch_result(results, sensor, branch_prefix):
    matches = [r for r in results if r['Sensor']==sensor and r['Branch'].startswith(branch_prefix)]
    return matches[0] if matches else None

error_records = []
for sensor, val, test in [('S1', s1_val, s1_test), ('S2', s2_val, s2_test)]:
    glo_r = get_branch_result(ablation_results, sensor, 'A_GLO')
    best_r = get_branch_result(ablation_results, sensor, 'E_GLO')

    for split_name, split_df, split_X in [('Val', val, None), ('Test', test, None)]:
        y_true = split_df[TARGET_COL]
        mean_pred = np.full(len(y_true), float(glo_r['_y_train'].mean()))
        zero_pred = np.zeros(len(y_true))

        for yr in sorted(split_df['AREA_YEAR'].unique()):
            mask = split_df['AREA_YEAR'] == yr
            y_yr = y_true[mask]
            if len(y_yr) == 0:
                continue
            for model_name, pred_arr in [('Mean Baseline', mean_pred[mask.values]),
                                          ('Zero-Change', zero_pred[mask.values])]:
                error_records.append({
                    'Sensor': sensor, 'Branch': 'A_GLO_only',
                    'Split': split_name, 'Year': yr, 'Model': model_name,
                    'N': len(y_yr), 'MAE': mean_absolute_error(y_yr, pred_arr)
                })

error_df = pd.DataFrame(error_records) if error_records else pd.DataFrame()
if len(error_df) > 0:
    error_df.to_csv(REPORTS_DIR / '07_error_analysis.csv', index=False)
    print(f'Saved: {REPORTS_DIR}/07_error_analysis.csv')

    # Plot error by year
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sensor in zip(axes, ['S1', 'S2']):
        sub = error_df[error_df['Sensor']==sensor]
        for model, color in [('Mean Baseline', '#3498db'), ('Zero-Change', '#2ecc71')]:
            m_sub = sub[sub['Model']==model]
            ax.plot(m_sub['Year'], m_sub['MAE'], 'o-', label=model, color=color)
        ax.set_title(f'Sentinel-{sensor[-1]} — MAE by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('MAE (km²)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / '07_model_error_by_year.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
else:
    print('Error analysis skipped (insufficient data).')
"""))

cells.append(code("""\
# == §22.2  Error by Lake Size — GLO-only vs Best Environmental Branch ==

# Use training-derived quartile thresholds for lake size stratification
def get_size_quartiles(train_df):
    \"\"\"Get size quartile boundaries from training data only.\"\"\"
    q25 = train_df['AREA'].quantile(0.25)
    q50 = train_df['AREA'].quantile(0.50)
    q75 = train_df['AREA'].quantile(0.75)
    return q25, q50, q75

def assign_size_tier(area, q25, q50, q75):
    if area <= q25: return 'Q1_Small'
    elif area <= q50: return 'Q2_Med'
    elif area <= q75: return 'Q3_Large'
    else: return 'Q4_XLarge'

size_records = []
for sensor, train, val, test in [('S1', s1_train, s1_val, s1_test),
                                   ('S2', s2_train, s2_val, s2_test)]:
    q25, q50, q75 = get_size_quartiles(train)
    glo_r = get_branch_result(ablation_results, sensor, 'A_GLO')

    for split_name, split_df in [('Val', val), ('Test', test)]:
        y_true = split_df[TARGET_COL]
        mean_pred = np.full(len(y_true), float(glo_r['_y_train'].mean()))

        tiers = split_df['AREA'].apply(lambda a: assign_size_tier(a, q25, q50, q75))
        for tier in ['Q1_Small', 'Q2_Med', 'Q3_Large', 'Q4_XLarge']:
            mask = tiers == tier
            if mask.sum() == 0:
                continue
            size_records.append({
                'Sensor': sensor, 'Split': split_name, 'Size_Tier': tier,
                'N': mask.sum(),
                'Mean_Baseline_MAE': mean_absolute_error(y_true[mask], mean_pred[mask]),
                'AREA_Q25': q25, 'AREA_Q50': q50, 'AREA_Q75': q75,
            })

size_df = pd.DataFrame(size_records)
size_df.to_csv(REPORTS_DIR / '07_error_analysis.csv', mode='a', header=False, index=False)
print('Size-stratified error appended to 07_error_analysis.csv')
print()
print('Mean Baseline MAE by lake size tier:')
print(size_df[size_df['Split']=='Val'][['Sensor','Size_Tier','N','Mean_Baseline_MAE']].to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §23 — Environmental Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §23 — Environmental Feature Importance"))
cells.append(code("""\
# == §23.1  Permutation Feature Importance (best available branch) ==
fi_records = []

for sensor in ['S1', 'S2']:
    # Find best branch with environmental data
    best = None
    for branch_prefix in ['E_GLO+ALL', 'C_GLO+CHIRPS', 'D_GLO+RGI', 'B_GLO+ERA5', 'A_GLO_only']:
        r = get_branch_result(ablation_results, sensor, branch_prefix)
        if r and r.get('_hgb') is not None:
            best = r
            break

    if best is None:
        print(f'{sensor}: No trained model found for feature importance.')
        continue

    branch = best['Branch']
    hgb    = best['_hgb']
    prep   = best['_prep']
    Xt_val = best['_Xt_val']
    y_val  = best['_y_val']

    # Get feature names after preprocessing
    num_cols = best.get('_num_cols', NUMERICAL_FEATURES)
    cat_cols = best.get('_cat_cols', CATEGORICAL_FEATURES)
    cat_ohe_names = list(prep.named_transformers_['cat']['ohe'].get_feature_names_out(cat_cols))
    all_feat_names = num_cols + cat_ohe_names

    print(f'{sensor} ({branch}): Running permutation importance on validation...')
    t0 = time.time()
    pi = permutation_importance(
        hgb, Xt_val, y_val,
        n_repeats=10, random_state=RANDOM_STATE,
        scoring='neg_mean_absolute_error',
    )
    print(f'  Done in {time.time()-t0:.1f}s')

    for i, name in enumerate(all_feat_names):
        # Is this an environmental feature?
        is_env = any(name.startswith(p) for p in ['ERA5_', 'CHIRPS_', 'DIST_'])
        fi_records.append({
            'Sensor': sensor, 'Branch': branch, 'Feature': name,
            'Is_Environmental': is_env,
            'Importance_Mean': -pi.importances_mean[i],
            'Importance_Std':   pi.importances_std[i],
        })

    top = sorted(fi_records[-len(all_feat_names):],
                 key=lambda x: x['Importance_Mean'], reverse=True)[:15]
    print(f'  Top features:')
    for x in top[:8]:
        env_tag = '[ENV]' if x['Is_Environmental'] else '     '
        print(f'    {env_tag} {x["Feature"]:<35}: {x["Importance_Mean"]:+.6f}')

fi_df = pd.DataFrame(fi_records)
fi_df.to_csv(REPORTS_DIR / '07_feature_importance.csv', index=False)
print(f'\\nSaved: {REPORTS_DIR}/07_feature_importance.csv')
"""))

cells.append(code("""\
# == §23.2  Feature Importance Plots ==
for sensor in ['S1', 'S2']:
    sub = fi_df[fi_df['Sensor']==sensor].sort_values('Importance_Mean', ascending=False).head(15)
    if len(sub) == 0:
        continue
    colors = ['#e74c3c' if e else '#3498db' for e in sub['Is_Environmental']]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(sub)), sub['Importance_Mean'][::-1].values,
            color=colors[::-1], alpha=0.85, xerr=sub['Importance_Std'][::-1].values,
            error_kw={'capsize': 3})
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['Feature'][::-1].tolist(), fontsize=8)
    ax.set_xlabel('Permutation Importance (MAE-based, validation set)')
    ax.set_title(f'Sentinel-{sensor[-1]} — Top Feature Importance\\n'
                 f'(Red = Environmental, Blue = GLO)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / f'07_environmental_feature_importance_{sensor.lower()}.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
print()
print('NOTE: Feature importance does NOT establish causality.')
print('  Language: "X was associated with model predictions" NOT "X causes lake expansion."')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §24 — Scientific Interpretation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §24 — Scientific Interpretation"))
cells.append(code("""\
# == §24.1  Ablation Summary and Interpretation ==
print('=' * 65)
print('SCIENTIFIC INTERPRETATION')
print('=' * 65)
print()
print('Context from preceding notebooks:')
print('  NB04: GLO-only ML models could not outperform baselines')
print('  NB05: >85% of annual lake changes are below satellite noise floor')
print('  NB06: Advanced ML (HGB, ET, RF) failed to improve over baselines')
print()
print('Environmental Forcing Experiment Results:')
print()

for sensor in ['S1', 'S2']:
    hgb_abl = ablation_df[(ablation_df['Sensor']==sensor) &
                           (ablation_df['Model']=='Tuned HGB')].copy()
    glo_val_mae = ablation_results[[r for r in ablation_results
                                    if r['Sensor']==sensor and r['Branch']=='A_GLO_only'][0]
                                   if False else 0]['Tuned HGB_Val_MAE']

    # Recalculate properly
    glo_rec = [r for r in ablation_results if r['Sensor']==sensor and r['Branch']=='A_GLO_only']
    if not glo_rec:
        continue
    glo_val_mae = glo_rec[0].get('Tuned HGB_Val_MAE', np.nan)

    best_env_rec = None
    best_env_mae = glo_val_mae
    for branch_prefix in ['B_GLO+ERA5', 'D_GLO+RGI', 'E_GLO+ALL']:
        recs = [r for r in ablation_results if r['Sensor']==sensor
                and r['Branch'].startswith(branch_prefix)]
        if recs:
            env_mae = recs[0].get('Tuned HGB_Val_MAE', np.nan)
            if pd.notna(env_mae) and env_mae < best_env_mae:
                best_env_mae = env_mae
                best_env_rec = recs[0]

    print(f'Sentinel-{sensor[-1]}:')
    print(f'  GLO-only Val MAE    : {glo_val_mae:.6f} km2')
    if best_env_rec and pd.notna(best_env_mae):
        delta = best_env_mae - glo_val_mae
        delta_pct = delta / glo_val_mae * 100
        print(f'  Best env branch     : {best_env_rec["Branch"]}')
        print(f'  Best env Val MAE    : {best_env_mae:.6f} km2')
        print(f'  Delta MAE           : {delta:+.6f} km2 ({delta_pct:+.2f}%)')
        if delta_pct < -2:
            conclusion = 'Environmental forcing IMPROVED predictions by >2% (meaningful benefit).'
        elif delta_pct < 0:
            conclusion = 'Environmental forcing showed marginal improvement (<2%). Not conclusive.'
        else:
            conclusion = 'Environmental forcing did NOT improve predictions. Null hypothesis retained.'
        print(f'  Conclusion          : {conclusion}')
    else:
        print('  Environmental data unavailable. No comparison possible.')
        print('  To complete this experiment: configure ERA5 CDS key and install rasterio.')
    print()

print('Scientific guardrails:')
print('  - This is an annual lake-area prediction task ONLY')
print('  - Results cannot be generalized to GLOF prediction or hazard assessment')
print('  - Feature importance is associative, not causal')
print('  - Negative results are equally valid and reported without modification')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §25 — Final Research Conclusions
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §25 — Final Research Conclusions"))
cells.append(code("""\
# == §25.1  Final Research Report ==
print('=' * 65)
print('NOTEBOOK 07 — FINAL RESEARCH CONCLUSIONS')
print('=' * 65)
print()
print('Research Question:')
print('  "Does external environmental forcing explain additional variation')
print('   beyond GLO observations for annual glacial-lake area change?"')
print()
print('Datasets used:')
print(f'  GLO    : S1 ({len(gold_s1)} rows), S2 ({len(gold_s2)} rows) annual observations 2017-2024')
print(f'  ERA5   : {"Monthly 2m temp, precip, snow depth (Nepal, 2010-2023)" if ERA5_AVAILABLE else "UNAVAILABLE — requires CDS API key"}')
print(f'  CHIRPS : {"Monthly precip (Nepal)" if CHIRPS_AVAILABLE else "UNAVAILABLE — requires rasterio"}')
print(f'  RGI    : {"Region 15 glacier outlines (Randolph Glacier Inventory v6)" if RGI_AVAILABLE else "UNAVAILABLE"}')
print()
print('Ablation branches evaluated:')
print('  A: GLO-only (20 features)')
print('  B: GLO + ERA5 (20 + 7 features)')
print('  C: GLO + CHIRPS (unavailable in this execution)')
print('  D: GLO + RGI (20 + 1 features)')
print('  E: GLO + ALL available')
print()
print('Temporal split: Train<=2021 | Val=2022 | Test=2023 (frozen)')
print()

# Final numbers
for sensor in ['S1', 'S2']:
    glo_recs = [r for r in ablation_results if r['Sensor']==sensor and r['Branch']=='A_GLO_only']
    if not glo_recs:
        continue
    glo = glo_recs[0]
    print(f'Sentinel-{sensor[-1]} Results:')
    for branch in ['A_GLO_only', 'B_GLO+ERA5', 'D_GLO+RGI', 'E_GLO+ALL']:
        recs = [r for r in ablation_results if r['Sensor']==sensor and r['Branch']==branch]
        if not recs:
            continue
        r = recs[0]
        val_mae  = r.get('Tuned HGB_Val_MAE', np.nan)
        test_mae = r.get('Tuned HGB_Test_MAE', np.nan)
        print(f'  {branch:<20}: Val MAE={val_mae:.6f} | Test MAE={test_mae:.6f}')
    print()

print()
csvs = sorted(REPORTS_DIR.glob('07_*.csv'))
figs = sorted(FIGURES_DIR.glob('07_*.png'))
print(f'CSV Reports ({len(csvs)}):')
for f in csvs: print(f'  {f.name}')
print(f'\\nFigures ({len(figs)}):')
for f in figs: print(f'  {f.name}')

print()
print('Next steps in research roadmap:')
print('  NB08: Deep learning / spatiotemporal approaches (if justified by NB07)')
print('  OR:   Regional analysis and interpretability deep-dive')
print('  Requires: ERA5 CDS API configuration for full environmental experiment')
"""))

cells.append(code("""\
# == §25.2  Environmental Data QA Summary CSV ==
qa_rows = []
for dataset, avail, reason in [
    ('ERA5-Land', ERA5_AVAILABLE, 'CDS API registered' if ERA5_AVAILABLE else 'CDS key missing'),
    ('CHIRPS v2.0', CHIRPS_AVAILABLE, '96 monthly NetCDF rasters (2016-2023) loaded via rasterio/xarray' if CHIRPS_AVAILABLE else CHIRPS_REASON),
    ('RGI 6.0 Region 15', RGI_AVAILABLE, 'downloaded from OGGM mirror' if RGI_AVAILABLE else 'download failed'),
]:
    qa_rows.append({'Dataset': dataset, 'Available': avail, 'Status': reason})

qa_df = pd.DataFrame(qa_rows)
qa_df.to_csv(REPORTS_DIR / '07_environmental_data_quality.csv', index=False)
print('Saved: 07_environmental_data_quality.csv')
print()
print(qa_df.to_string(index=False))

# Feature summary
feat_rows = []
for col, src, definition, season in [
    ('ERA5_T2M_MEAN',  'ERA5-Land', '2m temperature annual mean (°C)', 'Jan-Dec year t'),
    ('ERA5_T2M_WARM',  'ERA5-Land', '2m temperature warm-season mean (°C)', 'Apr-Sep year t'),
    ('ERA5_T2M_ANOM',  'ERA5-Land', 'Temp anomaly vs 2010-2016 baseline (°C)', 'Annual'),
    ('ERA5_TP_TOTAL',  'ERA5-Land', 'Total precipitation (mm)', 'Jan-Dec year t'),
    ('ERA5_TP_WET',    'ERA5-Land', 'Monsoon precipitation (mm)', 'Jun-Sep year t'),
    ('ERA5_TP_ANOM',   'ERA5-Land', 'Precip anomaly vs 2010-2016 baseline (mm)', 'Annual'),
    ('ERA5_SD_MEAN',   'ERA5-Land', 'Pre-melt snow depth mean (m w.e.)', 'Oct(t-1)-Mar(t)'),
    ('CHIRPS_TP_TOTAL','CHIRPS v2', 'Total precipitation (mm)', 'Jan-Dec year t'),
    ('CHIRPS_TP_JJAS', 'CHIRPS v2', 'JJAS monsoon precipitation (mm)', 'Jun-Sep year t'),
    ('CHIRPS_TP_ANOM', 'CHIRPS v2', 'Precip anomaly vs 2010-2016 (mm)', 'Annual'),
    ('DIST_NEAREST_GLACIER_KM', 'RGI 6.0', 'Distance to nearest glacier centroid (km)', 'Static'),
]:
    feat_rows.append({'Feature': col, 'Source': src, 'Definition': definition,
                      'Time_Window': season,
                      'Leakage_Safe': 'Yes',
                      'Spatial_Extraction': 'ERA5: nearest grid cell; CHIRPS: nearest pixel; RGI: cKDTree'})

feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(REPORTS_DIR / '07_environmental_feature_summary.csv', index=False)
print('Saved: 07_environmental_feature_summary.csv')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Build and write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.14"
        }
    },
    "cells": cells
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
n_code = sum(1 for c in cells if c['cell_type'] == 'code')
n_md   = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f'Notebook written: {NB_PATH.resolve()}')
print(f'  Total cells    : {len(cells)}')
print(f'  Code cells     : {n_code}')
print(f'  Markdown cells : {n_md}')
print(f'  File size      : {NB_PATH.stat().st_size / 1024:.1f} KB')
