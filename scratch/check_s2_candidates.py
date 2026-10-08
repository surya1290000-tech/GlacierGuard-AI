import json, urllib.request, urllib.parse, io, pathlib
import rasterio
import numpy as np

# Credentials
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

candidates = [
    ("2026-09-28", "S2C_MSIL2A_20260928T044701_N0513_R076_T45RVL_20260928T075011", 6.13),
    ("2026-09-20", "S2A_MSIL2A_20260920T045231_N0512_R076_T45RVL_20260920T095509", 44.28),
    ("2026-09-08", "S2C_MSIL2A_20260908T044701_N0512_R076_T45RVL_20260908T094920", 65.73),
    ("2026-08-11", "S2A_MSIL2A_20260811T045241_N0512_R076_T45RVL_20260811T100012", 34.73),
    ("2026-05-26", "S2B_MSIL2A_20260526T044659_N0512_R076_T45RVL_20260526T083435", 15.00),
    ("2026-04-21", "S2C_MSIL2A_20260421T044701_N0512_R076_T45RVL_20260421T080124", 3.79),
    ("2026-04-16", "S2B_MSIL2A_20260416T044659_N0512_R076_T45RVL_20260416T083210", 25.72),
]

# Load GLO polygon buffer 50m
import geopandas as gpd, rasterio.features
from rasterio.transform import from_bounds
glo_dir = pathlib.Path('raw/GLO')
gdf = gpd.read_file(glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg', where="GLO_ID = 'GLO_86.92845_27.89838'")
gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023].to_crs('EPSG:32645')
poly_utm_50 = gdf_2023.geometry.iloc[0].buffer(50)
poly_4326_50 = gpd.GeoSeries([poly_utm_50], crs='EPSG:32645').to_crs('EPSG:4326').iloc[0]

transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
lake_mask_50 = rasterio.features.rasterize([(poly_4326_50, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8) == 1

results = []
for dt_str, pid, tile_cloud in candidates:
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{dt_str}T00:00:00Z", "to": f"{dt_str}T23:59:59Z"}}}]
        },
        "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript_s2,
    }
    req = urllib.request.Request(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'User-Agent': 'GlacierGuard-AI/1.0'}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data_bytes = resp.read()
        with rasterio.open(io.BytesIO(data_bytes)) as ds:
            arr = ds.read()
        scl = np.round(arr[3]).astype(int)
        cloud_shadow = np.isin(scl, [0, 1, 3, 8, 9, 10])
        
        aoi_cloud_frac = cloud_shadow.sum() / (w * h)
        lake_cloud_frac = (cloud_shadow & lake_mask_50).sum() / lake_mask_50.sum()
        lake_valid_frac = 1.0 - lake_cloud_frac
        
        # Save raw GeoTIFF for caching so we don't need to re-download
        cache_dir = pathlib.Path('scratch/cache_s2')
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / f"{dt_str}.tif").write_bytes(data_bytes)
        
        status = "USABLE" if lake_cloud_frac <= 0.30 else "CONTAMINATED"
        print(f"Date: {dt_str} | Tile Cloud: {tile_cloud:5.1f}% | Lake Cloud: {lake_cloud_frac*100:5.1f}% | Lake Valid: {lake_valid_frac*100:5.1f}% -> {status}")
        results.append({
            'date': dt_str, 'product_id': pid, 'tile_cloud': tile_cloud,
            'lake_cloud_pct': lake_cloud_frac * 100, 'lake_valid_pct': lake_valid_frac * 100,
            'status': status
        })
    except Exception as ex:
        print(f"Date: {dt_str} FAILED: {ex}")

print(f"\nTotal candidate scenes evaluated: {len(results)}")
print(f"Usable scenes (<=30% lake cloud): {sum(1 for r in results if r['status'] == 'USABLE')}")
