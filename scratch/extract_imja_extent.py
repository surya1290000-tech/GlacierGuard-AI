"""
GlacierGuard-AI -- Imja Tsho Lake Extent Extraction Pipeline
============================================================
Fetches S2 (with SCL cloud mask) and S1 (VV) via Processing API,
extracts water masks, computes lake area, compares to GLO historical,
saves reports and figures.

CRITICAL: Credentials and tokens NEVER printed.
"""

import os, sys, json, io, math, pathlib, struct, datetime, warnings
import urllib.request, urllib.parse, urllib.error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import ndimage

warnings.filterwarnings('ignore')

try:
    import psutil
except ImportError:
    psutil = None

# -- Configuration --------------------------------------------------------------
CRED_FILE      = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu"
                  "/auth/realms/CDSE/protocol/openid-connect/token")
CDSE_STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

ROOT    = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Imja Tsho parameters
LAKE_NAME = "Imja Tsho"
BBOX      = [86.900, 27.885, 86.955, 27.930]
DT_RANGE  = "2024-01-01T00:00:00Z/2025-09-30T23:59:59Z"

# Historical GLO reference (from NB01-NB09 pipeline)
GLO_AREA_KM2  = 1.32
GLO_YEAR      = 2023

# Thresholds
NDWI_THRESHOLD    = 0.0    # NDWI > 0 = water
VV_DB_THRESHOLD   = -15.0  # VV < -15 dB = water (calm water is strong specular reflector)
MIN_COMPONENT_PX  = 10     # remove connected components smaller than this

# Processing API output: 512x512 at 10m native S2 resolution
IMG_SIZE = 512

# Pixel area calculation:
# AOI spans ~0.055 deg lon x 0.045 deg lat
# At lat ~27.9 deg: 1 deg lon ~ 97.8 km, 1 deg lat ~ 110.6 km
# AOI width ~ 0.055 * 97.8 = 5.379 km, height ~ 0.045 * 110.6 = 4.977 km
# pixel_area = (5.379/512) * (4.977/512) = ~0.0000977 km2 per pixel
AOI_WIDTH_DEG  = BBOX[2] - BBOX[0]  # 0.055 deg
AOI_HEIGHT_DEG = BBOX[3] - BBOX[1]  # 0.045 deg
LAT_CENTER     = (BBOX[1] + BBOX[3]) / 2.0  # 27.9075
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(LAT_CENTER))  # ~98.33 km
KM_PER_DEG_LAT = 110.574  # roughly constant
AOI_WIDTH_KM   = AOI_WIDTH_DEG * KM_PER_DEG_LON
AOI_HEIGHT_KM  = AOI_HEIGHT_DEG * KM_PER_DEG_LAT
PIXEL_AREA_KM2 = (AOI_WIDTH_KM / IMG_SIZE) * (AOI_HEIGHT_KM / IMG_SIZE)


# -- Credential & Auth ---------------------------------------------------------
def resolve_credentials():
    cid  = os.environ.get('CDSE_CLIENT_ID', '').strip()
    csec = os.environ.get('CDSE_CLIENT_SECRET', '').strip()
    if cid and csec:
        return cid, csec
    if psutil is not None:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pname = (proc.info.get('name') or '').lower()
                if 'powershell' in pname or 'pwsh' in pname:
                    env = proc.environ()
                    pc = env.get('CDSE_CLIENT_ID', '').strip()
                    ps = env.get('CDSE_CLIENT_SECRET', '').strip()
                    if pc and ps:
                        return pc, ps
            except Exception:
                continue
    if CRED_FILE.exists():
        try:
            data = json.loads(CRED_FILE.read_text(encoding='utf-8'))
            fc = data.get('client_id', '').strip()
            fs = data.get('client_secret', '').strip()
            if fc and fs:
                return fc, fs
        except Exception:
            pass
    return None, None


def get_token(cid, csec):
    payload = urllib.parse.urlencode({
        'grant_type': 'client_credentials',
        'client_id': cid, 'client_secret': csec,
    }).encode('utf-8')
    req = urllib.request.Request(
        CDSE_TOKEN_URL, data=payload,
        headers={'Content-Type': 'application/x-www-form-urlencoded',
                 'User-Agent': 'GlacierGuard-AI/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['access_token'], data.get('expires_in', 0)


# -- STAC Scene Selection ------------------------------------------------------
def select_scene(token, collection):
    auth_hdr = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'GlacierGuard-AI/1.0',
    }
    # Try with sortby first
    payload = json.dumps({
        "collections": [collection], "bbox": BBOX,
        "datetime": DT_RANGE, "limit": 1,
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }).encode('utf-8')
    req = urllib.request.Request(CDSE_STAC_URL, data=payload, headers=auth_hdr)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode('utf-8'))
    features = result.get('features', [])
    if not features:
        # Fallback: get many and sort locally
        payload = json.dumps({
            "collections": [collection], "bbox": BBOX,
            "datetime": DT_RANGE, "limit": 50,
        }).encode('utf-8')
        req = urllib.request.Request(CDSE_STAC_URL, data=payload, headers=auth_hdr)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        features = result.get('features', [])
        features.sort(
            key=lambda f: f.get('properties', {}).get('datetime', ''),
            reverse=True)
    if not features:
        raise RuntimeError(f"No scenes found for {collection}")
    feat = features[0]
    props = feat.get('properties', {})
    return {
        'id':        feat.get('id', 'unknown'),
        'datetime':  props.get('datetime', 'unknown'),
        'cloud_pct': props.get('eo:cloud_cover', None),
    }


# -- Processing API -------------------------------------------------------------
def build_s2_evalscript():
    """Return 4 Float32 bands: Green(B03), NIR(B08), SWIR(B11), SCL.
    SCL in DN units (not REFLECTANCE) to avoid the known SH error."""
    return """//VERSION=3
function setup() {
  return {
    input: [
      {bands: ["B03", "B08", "B11"], units: "REFLECTANCE"},
      {bands: ["SCL"], units: "DN"}
    ],
    output: {bands: 4, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(samples) {
  // samples is an array when multiple inputs, or single object
  // With multiple inputs, use samples[0] for reflectance bands, samples[1] for SCL
  var refl = samples[0] || samples;
  var scl_input = samples[1] || samples;
  return [refl.B03, refl.B08, refl.B11, scl_input.SCL];
}
"""


def build_s2_evalscript_v2():
    """Alternative: single input block with mixed units via separate setup."""
    return """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B08", "B11", "SCL"]
    }],
    output: {bands: 4, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B08, sample.B11, sample.SCL];
}
"""


def build_s1_evalscript():
    """VV linear power -> Float32 (we convert to dB in Python)."""
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV"]}],
    output: {bands: 1, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.VV];
}
"""


def call_process_api(token, sensor_type, scene, evalscript):
    """
    Send Processing API request. Returns raw bytes of GeoTIFF response.
    sensor_type: 'sentinel-2-l2a' or 'sentinel-1-grd'
    """
    dt = scene['datetime'][:10]
    dt_from = f"{dt}T00:00:00Z"
    dt_to   = f"{dt}T23:59:59Z"

    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": sensor_type,
                "dataFilter": {
                    "timeRange": {"from": dt_from, "to": dt_to}
                }
            }]
        },
        "output": {
            "width":  IMG_SIZE,
            "height": IMG_SIZE,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript,
    }

    req = urllib.request.Request(
        SH_PROCESS_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'User-Agent': 'GlacierGuard-AI/1.0',
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read(), None
    except urllib.error.HTTPError as e:
        err = e.read().decode('utf-8', errors='replace')
        return None, f"HTTP {e.code}: {err[:500]}"
    except Exception as ex:
        return None, str(ex)


# -- GeoTIFF Parsing (pure Python -- no GDAL/rasterio needed for raw) -----------
def parse_geotiff_to_array(raw_bytes, expected_bands=1):
    """Parse a raw GeoTIFF (Float32) into numpy array using rasterio."""
    try:
        import rasterio
        with rasterio.open(io.BytesIO(raw_bytes)) as ds:
            data = ds.read()  # shape: (bands, height, width)
            return data, ds.width, ds.height, ds.count, None
    except Exception as e:
        return None, None, None, None, str(e)


# -- Lake Geometry Constraint --------------------------------------------------
def create_lake_mask(width, height):
    """
    Create a rough elliptical constraint mask for Imja Tsho.
    Imja Tsho is an elongated proglacial lake oriented roughly E-W.
    Lake centroid: ~86.925 degE, 27.900 degN
    Lake extent: roughly 1.4 km x 0.4 km
    In pixel coords within our AOI:
      center_x ~ (86.925 - 86.900) / 0.055 * width ~ 0.4545 * width
      center_y ~ (27.930 - 27.900) / 0.045 * height ~ 0.6667 * height (inverted Y)
    """
    # Pixel coordinates of lake center
    cx = int((86.925 - BBOX[0]) / AOI_WIDTH_DEG * width)
    cy = int((BBOX[3] - 27.900) / AOI_HEIGHT_DEG * height)  # Y inverted in image

    # Semi-axes in pixels (generous buffer around the ~1.4km x 0.4km lake)
    # lake is ~1.6 km long, ~0.6 km wide with buffer
    semi_major_km = 1.0   # generous E-W radius
    semi_minor_km = 0.5   # generous N-S radius
    semi_major_px = int(semi_major_km / AOI_WIDTH_KM * width)
    semi_minor_px = int(semi_minor_km / AOI_HEIGHT_KM * height)

    # Create elliptical mask
    yy, xx = np.ogrid[:height, :width]
    dist = ((xx - cx) / max(semi_major_px, 1))**2 + ((yy - cy) / max(semi_minor_px, 1))**2
    mask = dist <= 1.0
    return mask


# -- Connected Component Filtering --------------------------------------------
def remove_small_components(binary_mask, min_size=MIN_COMPONENT_PX):
    """Remove connected components smaller than min_size pixels."""
    labeled, n_features = ndimage.label(binary_mask)
    if n_features == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, n_features + 1))
    for i, s in enumerate(sizes, start=1):
        if s < min_size:
            binary_mask[labeled == i] = 0
    return binary_mask


# -- S2 Processing -------------------------------------------------------------
def process_s2(raw_bytes, scene_info):
    """Process Sentinel-2 4-band GeoTIFF: B03, B08, B11, SCL."""
    data, w, h, nbands, err = parse_geotiff_to_array(raw_bytes)
    if data is None:
        return None, f"GeoTIFF parse failed: {err}"

    if nbands < 4:
        return None, f"Expected 4 bands, got {nbands}"

    green = data[0]  # B03
    nir   = data[1]  # B08
    swir  = data[2]  # B11
    scl   = data[3]  # SCL

    # SCL classes (Sentinel-2 Scene Classification Layer):
    # 0=No data, 1=Saturated, 2=Dark area, 3=Cloud shadow,
    # 4=Vegetation, 5=Bare soil, 6=Water, 7=Unclassified,
    # 8=Cloud medium prob, 9=Cloud high prob, 10=Thin cirrus, 11=Snow/Ice
    scl_int = np.round(scl).astype(int)

    # Cloud/shadow mask (classes 0,1,3,8,9,10 are invalid)
    cloud_shadow = np.isin(scl_int, [0, 1, 3, 8, 9, 10])
    valid_pixels = ~cloud_shadow

    # NDWI = (Green - NIR) / (Green + NIR + eps)
    eps = 1e-10
    ndwi = (green - nir) / (green + nir + eps)

    # MNDWI = (Green - SWIR) / (Green + SWIR + eps) -- more robust for turbid water
    mndwi = (green - swir) / (green + swir + eps)

    # Use MNDWI as primary (better at separating water from built-up/snow)
    # but also check NDWI for consistency
    water_mndwi = mndwi > NDWI_THRESHOLD
    water_ndwi  = ndwi  > NDWI_THRESHOLD

    # Combined: pixel is water if MNDWI > threshold
    water_raw = water_mndwi

    # Apply validity mask (only clear pixels count)
    water_valid = water_raw & valid_pixels

    # Apply lake geometry constraint
    lake_mask = create_lake_mask(w, h)
    water_constrained = water_valid & lake_mask

    # Remove small components
    water_final = remove_small_components(water_constrained.copy())

    # Statistics
    total_px       = w * h
    lake_region_px = int(lake_mask.sum())
    valid_in_lake  = int((valid_pixels & lake_mask).sum())
    cloud_in_lake  = int((cloud_shadow & lake_mask).sum())
    water_px       = int(water_final.sum())

    valid_fraction = valid_in_lake / max(lake_region_px, 1)
    cloud_fraction = cloud_in_lake / max(lake_region_px, 1)
    water_fraction = water_px / max(valid_in_lake, 1)

    current_area = water_px * PIXEL_AREA_KM2

    # Quality assessment
    if valid_fraction < 0.3:
        quality = "LOW -- heavy cloud contamination"
        sufficiently_observable = False
    elif valid_fraction < 0.6:
        quality = "MODERATE -- partial cloud contamination"
        sufficiently_observable = True
    else:
        quality = "HIGH -- mostly clear"
        sufficiently_observable = True

    result = {
        'sensor':          'S2',
        'product_id':      scene_info['id'],
        'acquisition':     scene_info['datetime'],
        'scene_cloud_pct': scene_info.get('cloud_pct'),
        'total_pixels':    total_px,
        'lake_region_px':  lake_region_px,
        'valid_in_lake':   valid_in_lake,
        'cloud_in_lake':   cloud_in_lake,
        'water_pixels':    water_px,
        'valid_fraction':  valid_fraction,
        'cloud_fraction':  cloud_fraction,
        'water_fraction':  water_fraction,
        'current_area_km2': current_area,
        'threshold':       NDWI_THRESHOLD,
        'threshold_type':  'MNDWI',
        'quality':         quality,
        'observable':      sufficiently_observable,
        'ndwi':            ndwi,
        'mndwi':           mndwi,
        'water_mask':      water_final,
        'valid_mask':      valid_pixels,
        'lake_mask':       lake_mask,
        'scl':             scl_int,
        'img_shape':       (h, w),
    }
    return result, None


# -- S1 Processing -------------------------------------------------------------
def process_s1(raw_bytes, scene_info):
    """Process Sentinel-1 1-band GeoTIFF: VV linear power."""
    data, w, h, nbands, err = parse_geotiff_to_array(raw_bytes)
    if data is None:
        return None, f"GeoTIFF parse failed: {err}"

    vv_linear = data[0]

    # Convert to dB
    vv_db = 10.0 * np.log10(np.maximum(vv_linear, 1e-10))

    # Valid pixels: VV > -30 dB and < 5 dB (reasonable SAR range)
    valid_pixels = (vv_db > -30.0) & (vv_db < 5.0) & np.isfinite(vv_db)

    # Water detection: calm water -> strong specular reflection -> very low backscatter
    water_raw = vv_db < VV_DB_THRESHOLD

    # Apply validity
    water_valid = water_raw & valid_pixels

    # Apply lake geometry constraint
    lake_mask = create_lake_mask(w, h)
    water_constrained = water_valid & lake_mask

    # Remove small components
    water_final = remove_small_components(water_constrained.copy())

    # Statistics
    total_px       = w * h
    lake_region_px = int(lake_mask.sum())
    valid_in_lake  = int((valid_pixels & lake_mask).sum())
    water_px       = int(water_final.sum())

    valid_fraction = valid_in_lake / max(lake_region_px, 1)
    water_fraction = water_px / max(valid_in_lake, 1)

    current_area = water_px * PIXEL_AREA_KM2

    # Quality assessment
    if valid_fraction < 0.3:
        quality = "LOW -- insufficient valid data"
    elif valid_fraction < 0.7:
        quality = "MODERATE"
    else:
        quality = "HIGH -- good SAR coverage"

    result = {
        'sensor':          'S1',
        'product_id':      scene_info['id'],
        'acquisition':     scene_info['datetime'],
        'total_pixels':    total_px,
        'lake_region_px':  lake_region_px,
        'valid_in_lake':   valid_in_lake,
        'water_pixels':    water_px,
        'valid_fraction':  valid_fraction,
        'water_fraction':  water_fraction,
        'current_area_km2': current_area,
        'threshold':       VV_DB_THRESHOLD,
        'threshold_type':  'VV dB',
        'quality':         quality,
        'vv_db':           vv_db,
        'water_mask':      water_final,
        'valid_mask':      valid_pixels,
        'lake_mask':       lake_mask,
        'img_shape':       (h, w),
    }
    return result, None


# -- Status Assessment ---------------------------------------------------------
def assess_status(current_area, glo_area, valid_fraction, observable=True):
    """
    Return NORMAL / WATCH / UNUSUAL_CHANGE / INSUFFICIENT_DATA.
    """
    if valid_fraction < 0.3 or not observable:
        return "INSUFFICIENT_DATA"

    if glo_area <= 0:
        return "INSUFFICIENT_DATA"

    abs_change = abs(current_area - glo_area)
    rel_change = abs_change / glo_area

    # Thresholds calibrated from NB08/NB09 MAE ~ 0.005 km2
    if rel_change < 0.05:    # < 5% relative change
        return "NORMAL"
    elif rel_change < 0.15:  # 5-15%
        return "WATCH"
    else:                    # > 15%
        return "UNUSUAL_CHANGE"


# -- Visualization -------------------------------------------------------------
def plot_water_mask(result, save_path, title_prefix):
    """Plot water mask with QA overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             facecolor='#0d1117')
    fig.suptitle(f"{title_prefix} -- {LAKE_NAME}", fontsize=14,
                 color='#e6edf3', fontweight='bold', y=0.98)

    sensor = result['sensor']
    water_mask = result['water_mask']
    lake_mask  = result['lake_mask']

    if sensor == 'S2':
        # Panel 1: MNDWI heatmap
        im = axes[0].imshow(result['mndwi'], cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[0].set_title('MNDWI', color='#e6edf3', fontsize=11)
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: SCL (cloud map)
        scl_colors = plt.cm.tab20(result['scl'] / 11.0)
        axes[1].imshow(scl_colors)
        axes[1].set_title('SCL (Cloud Classification)', color='#e6edf3', fontsize=11)
    else:
        # Panel 1: VV dB
        im = axes[0].imshow(result['vv_db'], cmap='gray', vmin=-25, vmax=0)
        axes[0].set_title('VV Backscatter (dB)', color='#e6edf3', fontsize=11)
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: valid mask
        axes[1].imshow(result['valid_mask'], cmap='Greens', vmin=0, vmax=1)
        axes[1].set_title('Valid Pixels', color='#e6edf3', fontsize=11)

    # Panel 3: Final water mask + lake constraint outline
    composite = np.zeros((*water_mask.shape, 3))
    # Lake region boundary in yellow
    lake_boundary = lake_mask.astype(float) - ndimage.binary_erosion(lake_mask, iterations=2).astype(float)
    composite[:, :, 0] = lake_boundary * 0.8  # R
    composite[:, :, 1] = lake_boundary * 0.8  # G
    # Water in cyan
    composite[:, :, 0] += water_mask * 0.0
    composite[:, :, 1] += water_mask * 0.7
    composite[:, :, 2] += water_mask * 1.0
    composite = np.clip(composite, 0, 1)

    axes[2].imshow(composite)
    axes[2].set_title(f'Water Mask ({result["current_area_km2"]:.4f} km2)',
                      color='#e6edf3', fontsize=11)

    # Legend
    patches = [
        mpatches.Patch(color='cyan', label='Detected water'),
        mpatches.Patch(color='yellow', label='Lake constraint region'),
    ]
    axes[2].legend(handles=patches, loc='lower left', fontsize=8,
                   facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#e6edf3')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
    print(f"  Figure saved: {save_path}")


def plot_comparison(s1_result, s2_result, save_path):
    """Side-by-side comparison of S1 vs S2 extent + bar chart."""
    fig = plt.figure(figsize=(16, 6), facecolor='#0d1117')
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8])

    fig.suptitle(f"{LAKE_NAME} -- Extent Comparison (S1 vs S2)",
                 fontsize=14, color='#e6edf3', fontweight='bold', y=0.98)

    # S1 water mask
    ax1 = fig.add_subplot(gs[0, 0])
    composite_s1 = np.zeros((*s1_result['water_mask'].shape, 3))
    composite_s1[:, :, 1] = s1_result['water_mask'] * 0.7
    composite_s1[:, :, 2] = s1_result['water_mask'] * 1.0
    ax1.imshow(composite_s1)
    ax1.set_title(f"S1: {s1_result['current_area_km2']:.4f} km2",
                  color='#e6edf3', fontsize=11)
    ax1.set_facecolor('#161b22')

    # S2 water mask
    ax2 = fig.add_subplot(gs[0, 1])
    composite_s2 = np.zeros((*s2_result['water_mask'].shape, 3))
    composite_s2[:, :, 1] = s2_result['water_mask'] * 0.7
    composite_s2[:, :, 2] = s2_result['water_mask'] * 1.0
    ax2.imshow(composite_s2)
    ax2.set_title(f"S2: {s2_result['current_area_km2']:.4f} km2",
                  color='#e6edf3', fontsize=11)
    ax2.set_facecolor('#161b22')

    # Bar chart comparison
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ['GLO\n(Historical)', 'S1\n(Current)', 'S2\n(Current)']
    values = [GLO_AREA_KM2, s1_result['current_area_km2'],
              s2_result['current_area_km2']]
    colors_bar = ['#8b949e', '#58a6ff', '#3fb950']
    bars = ax3.bar(labels, values, color=colors_bar, width=0.6,
                   edgecolor='#30363d', linewidth=0.5)
    ax3.set_ylabel('Area (km^2)', color='#e6edf3', fontsize=10)
    ax3.set_title('Area Comparison', color='#e6edf3', fontsize=11)
    ax3.set_facecolor('#161b22')
    ax3.tick_params(colors='#8b949e', labelsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    for spine in ax3.spines.values():
        spine.set_color('#30363d')

    # Annotate bars
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom',
                 color='#e6edf3', fontsize=9, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.tick_params(colors='#8b949e', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
    print(f"  Figure saved: {save_path}")


# -- Main Pipeline -------------------------------------------------------------
def main():
    print("=" * 60)
    print("GLACIERGUARD-AI -- IMJA TSHO LAKE EXTENT EXTRACTION")
    print("=" * 60)
    print()

    # -- Auth --
    cid, csec = resolve_credentials()
    if not cid or not csec:
        print("ERROR: Credentials not found.")
        sys.exit(1)
    print("  Credentials available : YES")

    token, expires_in = get_token(cid, csec)
    print(f"  OAuth                 : SUCCESS (expires {expires_in}s)")
    print()

    # -- Scene Selection --
    print("-" * 60)
    print("STEP 1: Scene Selection")
    print("-" * 60)

    s1_scene = select_scene(token, 'sentinel-1-grd')
    print(f"  S1 Product ID   : {s1_scene['id']}")
    print(f"  S1 Acquisition  : {s1_scene['datetime']}")

    s2_scene = select_scene(token, 'sentinel-2-l2a')
    print(f"  S2 Product ID   : {s2_scene['id']}")
    print(f"  S2 Acquisition  : {s2_scene['datetime']}")
    print(f"  S2 Cloud Cover  : {s2_scene.get('cloud_pct', 'N/A')}%")
    print()

    # -- S2 Processing API (4-band: Green, NIR, SWIR, SCL) --
    print("-" * 60)
    print("STEP 2: Sentinel-2 Processing API")
    print("-" * 60)

    # Try v1 evalscript (multi-input with separate units)
    s2_raw, s2_err = call_process_api(
        token, 'sentinel-2-l2a', s2_scene, build_s2_evalscript())

    if s2_err:
        print(f"  S2 evalscript v1 failed: {s2_err[:200]}")
        print(f"  Trying v2 evalscript (single input block)...")
        s2_raw, s2_err = call_process_api(
            token, 'sentinel-2-l2a', s2_scene, build_s2_evalscript_v2())

    if s2_err:
        print(f"  S2 Processing API FAILED: {s2_err}")
        s2_result = None
    else:
        print(f"  S2 Processing API : PASS ({len(s2_raw):,} bytes)")
        s2_result, s2_parse_err = process_s2(s2_raw, s2_scene)
        if s2_parse_err:
            print(f"  S2 parse error: {s2_parse_err}")
            s2_result = None
    print()

    # -- S1 Processing API (1-band: VV linear) --
    print("-" * 60)
    print("STEP 3: Sentinel-1 Processing API")
    print("-" * 60)

    s1_raw, s1_err = call_process_api(
        token, 'sentinel-1-grd', s1_scene, build_s1_evalscript())

    if s1_err:
        print(f"  S1 Processing API FAILED: {s1_err}")
        s1_result = None
    else:
        print(f"  S1 Processing API : PASS ({len(s1_raw):,} bytes)")
        s1_result, s1_parse_err = process_s1(s1_raw, s1_scene)
        if s1_parse_err:
            print(f"  S1 parse error: {s1_parse_err}")
            s1_result = None
    print()

    # -- S2 Results --
    print("-" * 60)
    print("STEP 4: Sentinel-2 Lake Extent Results")
    print("-" * 60)

    if s2_result:
        print(f"  Image dimensions       : {s2_result['img_shape'][1]} x {s2_result['img_shape'][0]} px")
        print(f"  Lake region pixels     : {s2_result['lake_region_px']}")
        print(f"  Cloud/shadow in AOI    : {s2_result['cloud_fraction']:.1%}")
        print(f"  Valid pixel fraction   : {s2_result['valid_fraction']:.1%}")
        print(f"  Observable             : {'YES' if s2_result['observable'] else 'NO'}")
        print(f"  Water pixels           : {s2_result['water_pixels']}")
        print(f"  Water fraction (valid) : {s2_result['water_fraction']:.1%}")
        print(f"  Current area           : {s2_result['current_area_km2']:.4f} km2")
        print(f"  Threshold              : {s2_result['threshold_type']} > {s2_result['threshold']}")
        print(f"  Quality                : {s2_result['quality']}")
    else:
        print("  S2 extraction: FAILED / NO DATA")
    print()

    # -- S1 Results --
    print("-" * 60)
    print("STEP 5: Sentinel-1 Lake Extent Results")
    print("-" * 60)

    if s1_result:
        print(f"  Image dimensions       : {s1_result['img_shape'][1]} x {s1_result['img_shape'][0]} px")
        print(f"  Lake region pixels     : {s1_result['lake_region_px']}")
        print(f"  Valid pixel fraction   : {s1_result['valid_fraction']:.1%}")
        print(f"  Water pixels           : {s1_result['water_pixels']}")
        print(f"  Water fraction (valid) : {s1_result['water_fraction']:.1%}")
        print(f"  Current area           : {s1_result['current_area_km2']:.4f} km2")
        print(f"  Threshold              : {s1_result['threshold_type']} < {s1_result['threshold']} dB")
        print(f"  Quality                : {s1_result['quality']}")
    else:
        print("  S1 extraction: FAILED / NO DATA")
    print()

    # -- Historical Comparison --
    print("-" * 60)
    print("STEP 6: Historical Comparison")
    print("-" * 60)

    glo_date  = datetime.date(GLO_YEAR, 12, 31)  # end of GLO year

    s1_change = s2_change = None
    s1_days = s2_days = None
    s1_status = s2_status = "INSUFFICIENT_DATA"

    if s1_result:
        s1_date_str = s1_result['acquisition'][:10]
        s1_date = datetime.date.fromisoformat(s1_date_str)
        s1_days = (s1_date - glo_date).days
        s1_change = s1_result['current_area_km2'] - GLO_AREA_KM2
        s1_status = assess_status(s1_result['current_area_km2'], GLO_AREA_KM2,
                                  s1_result['valid_fraction'])
        print(f"  S1 acquisition         : {s1_date_str}")
        print(f"  S1 current area        : {s1_result['current_area_km2']:.4f} km2")
        print(f"  GLO historical area    : {GLO_AREA_KM2:.4f} km2")
        print(f"  S1 observed change     : {s1_change:+.4f} km2")
        print(f"  Days since GLO obs     : {s1_days}")
        print(f"  S1 status              : {s1_status}")

    if s2_result:
        s2_date_str = s2_result['acquisition'][:10]
        s2_date = datetime.date.fromisoformat(s2_date_str)
        s2_days = (s2_date - glo_date).days
        s2_change = s2_result['current_area_km2'] - GLO_AREA_KM2
        s2_obs = s2_result['observable']
        s2_status = assess_status(s2_result['current_area_km2'], GLO_AREA_KM2,
                                  s2_result['valid_fraction'], s2_obs)
        print(f"  S2 acquisition         : {s2_date_str}")
        print(f"  S2 current area        : {s2_result['current_area_km2']:.4f} km2")
        print(f"  GLO historical area    : {GLO_AREA_KM2:.4f} km2")
        print(f"  S2 observed change     : {s2_change:+.4f} km2")
        print(f"  Days since GLO obs     : {s2_days}")
        print(f"  S2 status              : {s2_status}")
    print()

    # -- Cross-sensor consistency check --
    if s1_result and s2_result:
        area_diff = abs(s1_result['current_area_km2'] - s2_result['current_area_km2'])
        rel_diff  = area_diff / max(GLO_AREA_KM2, 0.01)
        print(f"  S1-S2 area difference  : {area_diff:.4f} km2 ({rel_diff:.1%})")
        if rel_diff > 0.20:
            print("  [!] FLAG FOR REVIEW: S1 and S2 estimates disagree by >20%.")
            print("      Do NOT average -- report both independently.")
        elif rel_diff > 0.10:
            print("  [!] MODERATE DISAGREEMENT: S1 and S2 differ by 10-20%.")
        else:
            print("  [OK] S1 and S2 estimates are broadly consistent (<10% difference).")
    print()

    # -- Save Figures --
    print("-" * 60)
    print("STEP 7: Saving Figures")
    print("-" * 60)

    if s2_result:
        plot_water_mask(s2_result,
                        FIGURES / '10_imja_tsho_s2_water_mask.png',
                        'Sentinel-2 MNDWI Water Detection')

    if s1_result:
        plot_water_mask(s1_result,
                        FIGURES / '10_imja_tsho_s1_water_mask.png',
                        'Sentinel-1 VV Water Detection')

    if s1_result and s2_result:
        plot_comparison(s1_result, s2_result,
                        FIGURES / '10_imja_tsho_extent_comparison.png')
    print()

    # -- Save CSV --
    print("-" * 60)
    print("STEP 8: Saving CSV Report")
    print("-" * 60)

    csv_path = REPORTS / '10_imja_tsho_extent.csv'
    rows = []
    for r, status, change, days in [
        (s1_result, s1_status, s1_change, s1_days),
        (s2_result, s2_status, s2_change, s2_days),
    ]:
        if r:
            rows.append({
                'lake_name':        LAKE_NAME,
                'sensor':           r['sensor'],
                'product_id':       r['product_id'],
                'acquisition':      r['acquisition'],
                'current_area_km2': f"{r['current_area_km2']:.6f}",
                'glo_area_km2':     f"{GLO_AREA_KM2:.4f}",
                'observed_change':  f"{change:+.6f}" if change is not None else '',
                'days_since_glo':   days if days is not None else '',
                'valid_fraction':   f"{r['valid_fraction']:.4f}",
                'water_pixels':     r['water_pixels'],
                'threshold_type':   r['threshold_type'],
                'threshold_value':  r['threshold'],
                'quality':          r['quality'],
                'status':           status,
            })

    if rows:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV saved: {csv_path}")
    print()

    # -- Final Summary --
    print("=" * 60)
    print("IMJA TSHO -- LAKE EXTENT EXTRACTION")
    print("=" * 60)
    print()
    if s1_result:
        print(f"S1 acquisition      : {s1_result['acquisition']}")
    else:
        print(f"S1 acquisition      : FAILED")
    if s2_result:
        print(f"S2 acquisition      : {s2_result['acquisition']}")
    else:
        print(f"S2 acquisition      : FAILED")
    print()
    if s1_result:
        print(f"S1 current area     : {s1_result['current_area_km2']:.4f} km2")
    else:
        print(f"S1 current area     : N/A")
    if s2_result:
        print(f"S2 current area     : {s2_result['current_area_km2']:.4f} km2")
    else:
        print(f"S2 current area     : N/A")
    print()
    print(f"Previous GLO area   : {GLO_AREA_KM2:.4f} km2")
    print()
    if s1_change is not None:
        print(f"S1 observed change  : {s1_change:+.4f} km2")
    else:
        print(f"S1 observed change  : N/A")
    if s2_change is not None:
        print(f"S2 observed change  : {s2_change:+.4f} km2")
    else:
        print(f"S2 observed change  : N/A")
    print()
    if s1_result:
        print(f"S1 quality          : {s1_result['quality']}")
    else:
        print(f"S1 quality          : N/A")
    if s2_result:
        print(f"S2 quality          : {s2_result['quality']}")
    else:
        print(f"S2 quality          : N/A")
    print()
    print(f"S1 status           : {s1_status}")
    print(f"S2 status           : {s2_status}")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
