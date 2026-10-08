import os, sys, json, io, math, pathlib, datetime
import urllib.request, urllib.parse, urllib.error
import numpy as np
import geopandas as gpd
import rasterio, rasterio.features
from rasterio.transform import from_bounds
from scipy import ndimage

# Credential
cred_file = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
creds = json.loads(cred_file.read_text())
cid, csec = creds['client_id'], creds['client_secret']

token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
payload = urllib.parse.urlencode({'grant_type': 'client_credentials', 'client_id': cid, 'client_secret': csec}).encode()
req = urllib.request.Request(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req) as resp:
    token = json.loads(resp.read().decode())['access_token']

bbox = [86.900, 27.885, 86.955, 27.930]
w, h = 512, 512
transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

# True pixel area via UTM projection
aoi_poly_4326 = gpd.GeoSeries.from_wkt([f'POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))'], crs='EPSG:4326')
aoi_poly_utm = aoi_poly_4326.to_crs('EPSG:32645').iloc[0]
pixel_area_km2 = (aoi_poly_utm.area / 1e6) / (w * h)

# Load GLO polygon & prepare buffer masks
glo_dir = pathlib.Path('raw/GLO')
s2_annual = glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg'
gdf = gpd.read_file(s2_annual, where="GLO_ID = 'GLO_86.92845_27.89838'")
gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023].to_crs('EPSG:32645')
poly_utm = gdf_2023.geometry.iloc[0]

buffer_masks = {}
for buf_m in [0, 50, 100, 200]:
    b_poly = poly_utm if buf_m == 0 else poly_utm.buffer(buf_m)
    b_4326 = gpd.GeoSeries([b_poly], crs='EPSG:32645').to_crs('EPSG:4326').iloc[0]
    m = rasterio.features.rasterize([(b_4326, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
    buffer_masks[buf_m] = (m == 1)

print("Buffer masks prepared successfully.")

# Process S2 on 2026-09-28
evalscript_s2 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B08", "B11", "SCL"]}],
    output: {bands: 4, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B08, sample.B11, sample.SCL];
}
"""

dt_s2 = "2026-09-28"
payload_s2 = {
    "input": {
        "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{dt_s2}T00:00:00Z", "to": f"{dt_s2}T23:59:59Z"}}}]
    },
    "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
    "evalscript": evalscript_s2,
}

req_s2 = urllib.request.Request(
    "https://sh.dataspace.copernicus.eu/api/v1/process",
    data=json.dumps(payload_s2).encode('utf-8'),
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'User-Agent': 'GlacierGuard-AI/1.0'}
)

print("Requesting S2 Processing API...")
try:
    with urllib.request.urlopen(req_s2, timeout=60) as resp:
        s2_bytes = resp.read()
except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code, e.read().decode('utf-8'))
    sys.exit(1)

with rasterio.open(io.BytesIO(s2_bytes)) as ds:
    s2_data = ds.read()

green = s2_data[0] # B03
nir = s2_data[1]   # B08
swir = s2_data[2]  # B11
scl = np.round(s2_data[3]).astype(int)

# SCL cloud/shadow mask: 0=no data, 1=saturated, 3=cloud shadow, 8=cloud med, 9=cloud high, 10=cirrus
cloud_shadow = np.isin(scl, [0, 1, 3, 8, 9, 10])
valid_s2 = ~cloud_shadow

eps = 1e-10
ndwi = (green - nir) / (green + nir + eps)
mndwi = (green - swir) / (green + swir + eps)

print("\n--- S2 EVALUATION (2026-09-28) ---")
for buf_m, bmask in buffer_masks.items():
    valid_in_buf = valid_s2 & bmask
    cloud_in_buf = cloud_shadow & bmask
    cloud_frac = cloud_in_buf.sum() / max(bmask.sum(), 1)
    valid_frac = valid_in_buf.sum() / max(bmask.sum(), 1)
    
    # Test threshold MNDWI > 0.0
    water_raw = (mndwi > 0.0) & valid_in_buf
    # remove small components (< 10 px)
    labeled, n_feat = ndimage.label(water_raw)
    sizes = ndimage.sum(water_raw, labeled, range(1, n_feat + 1))
    water_clean = water_raw.copy()
    for i, s in enumerate(sizes, start=1):
        if s < 10:
            water_clean[labeled == i] = 0
            
    area_km2 = water_clean.sum() * pixel_area_km2
    print(f"Buffer {buf_m:3d}m: Valid {valid_frac:.3f}, Cloud {cloud_frac:.3f}, Water px {water_clean.sum()}, Area: {area_km2:.4f} km2")

# Now S1
evalscript_s1 = """//VERSION=3
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

dt_s1 = "2026-09-28"
payload_s1 = {
    "input": {
        "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": f"{dt_s1}T00:00:00Z", "to": f"{dt_s1}T23:59:59Z"}}}]
    },
    "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
    "evalscript": evalscript_s1,
}

req_s1 = urllib.request.Request(
    "https://sh.dataspace.copernicus.eu/api/v1/process",
    data=json.dumps(payload_s1).encode('utf-8'),
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'User-Agent': 'GlacierGuard-AI/1.0'}
)

print("\nRequesting S1 Processing API...")
with urllib.request.urlopen(req_s1, timeout=60) as resp:
    s1_bytes = resp.read()

with rasterio.open(io.BytesIO(s1_bytes)) as ds:
    s1_data = ds.read()

vv_linear = s1_data[0]
vv_db = 10.0 * np.log10(np.maximum(vv_linear, 1e-10))
valid_s1 = (vv_db > -35.0) & (vv_db < 5.0) & np.isfinite(vv_db)

print("\n--- S1 RAW EVALUATION (2026-09-28) ---")
for buf_m, bmask in buffer_masks.items():
    valid_in_buf = valid_s1 & bmask
    # Test threshold VV < -15 dB
    water_raw = (vv_db < -15.0) & valid_in_buf
    labeled, n_feat = ndimage.label(water_raw)
    sizes = ndimage.sum(water_raw, labeled, range(1, n_feat + 1))
    water_clean = water_raw.copy()
    for i, s in enumerate(sizes, start=1):
        if s < 10:
            water_clean[labeled == i] = 0
            
    area_km2 = water_clean.sum() * pixel_area_km2
    print(f"Buffer {buf_m:3d}m (VV < -15 dB): Water px {water_clean.sum()}, Area: {area_km2:.4f} km2")
