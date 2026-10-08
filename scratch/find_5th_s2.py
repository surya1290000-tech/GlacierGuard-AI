import json, urllib.request, urllib.parse, io, pathlib
import rasterio
import numpy as np
import geopandas as gpd, rasterio.features
from rasterio.transform import from_bounds

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

glo_dir = pathlib.Path('raw/GLO')
gdf = gpd.read_file(glo_dir / 'S2_20172024_NTB_GLOID_v1.02.gpkg', where="GLO_ID = 'GLO_86.92845_27.89838'")
gdf_2023 = gdf[gdf['AREA_YEAR'] == 2023].to_crs('EPSG:32645')
poly_utm_50 = gdf_2023.geometry.iloc[0].buffer(50)
poly_4326_50 = gpd.GeoSeries([poly_utm_50], crs='EPSG:32645').to_crs('EPSG:4326').iloc[0]
lake_mask_50 = rasterio.features.rasterize([(poly_4326_50, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8) == 1

more_candidates = [
    ("2026-05-11", "S2C_MSIL2A_20260511T044701_N0512_R076_T45RVL_20260511T080826", 35.42),
    ("2026-04-11", "S2C_MSIL2A_20260411T044701_N0512_R076_T45RVL_20260411T080112", 36.92),
    ("2026-05-31", "S2C_MSIL2A_20260531T044701_N0512_R076_T45RVL_20260531T094908", 40.66),
    ("2026-04-06", "S2B_MSIL2A_20260406T044659_N0512_R076_T45RVL_20260406T083422", 48.48),
]

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

cache_dir = pathlib.Path('scratch/cache_s2')
for dt_str, pid, tile_cloud in more_candidates:
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
        lake_cloud_frac = (cloud_shadow & lake_mask_50).sum() / lake_mask_50.sum()
        status = "USABLE" if lake_cloud_frac <= 0.30 else "CONTAMINATED"
        print(f"Date: {dt_str} | Tile Cloud: {tile_cloud:5.1f}% | Lake Cloud: {lake_cloud_frac*100:5.1f}% -> {status}")
        if lake_cloud_frac <= 0.30:
            (cache_dir / f"{dt_str}.tif").write_bytes(data_bytes)
            print(f"  --> Cached as usable 5th S2 scene: {dt_str}")
            break
    except Exception as ex:
        print(f"Date: {dt_str} FAILED: {ex}")
