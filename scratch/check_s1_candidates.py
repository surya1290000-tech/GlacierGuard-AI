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

evalscript_s1 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""

s1_scenes = [
    ("2026-09-28", "S1D_IW_GRDH_1SDV_20260928T121313_20260928T121338_004778_008F55_1667_COG"),
    ("2026-09-16", "S1D_IW_GRDH_1SDV_20260916T121313_20260916T121338_004603_008949_F1C8_COG"),
    ("2026-09-04", "S1D_IW_GRDH_1SDV_20260904T121313_20260904T121338_004428_00833E_9092_COG"),
    ("2026-08-23", "S1D_IW_GRDH_1SDV_20260823T121313_20260823T121338_004253_007D19_461D_COG"),
    ("2026-08-11", "S1D_IW_GRDH_1SDV_20260811T121312_20260811T121337_004078_0076F9_676B_COG"),
]

cache_dir = pathlib.Path('scratch/cache_s1')
cache_dir.mkdir(exist_ok=True)

results = []
for dt_str, pid in s1_scenes:
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": f"{dt_str}T00:00:00Z", "to": f"{dt_str}T23:59:59Z"}}}]
        },
        "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript_s1,
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
            nbands = ds.count
        (cache_dir / f"{dt_str}.tif").write_bytes(data_bytes)
        vv = arr[0]
        vv_db = 10.0 * np.log10(np.maximum(vv, 1e-10))
        valid_px = int(((vv_db > -35.0) & (vv_db < 5.0) & np.isfinite(vv_db)).sum())
        valid_frac = valid_px / (w * h)
        print(f"Date: {dt_str} | Bands: {nbands} | Valid: {valid_frac*100:5.1f}% | Cached: {cache_dir / f'{dt_str}.tif'}")
        results.append({'date': dt_str, 'product_id': pid, 'valid_frac': valid_frac, 'bands': nbands})
    except Exception as ex:
        print(f"Date: {dt_str} FAILED: {ex}")

print(f"\nTotal S1 scenes fetched & cached: {len(results)}")
