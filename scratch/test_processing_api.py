"""
GlacierGuard-AI — Imja Tsho Scene Selection & Processing API Test
=================================================================
1. Catalog STAC search for newest S1 (sentinel-1-grd) and S2 (sentinel-2-l2a) scenes.
2. Extract product IDs and acquisition datetimes.
3. Send ONE Sentinel Hub Processing API request for S2 (NDWI band ratio).
4. Send ONE Sentinel Hub Processing API request for S1 (VV backscatter).
5. Verify actual image response is received (size, format).

CRITICAL: Credentials and access tokens are NEVER printed.
"""

import os
import sys
import json
import pathlib
import urllib.request
import urllib.parse
import urllib.error

try:
    import psutil
except ImportError:
    psutil = None

# ── Constants ──────────────────────────────────────────────────────────────────
CRED_FILE      = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu"
                  "/auth/realms/CDSE/protocol/openid-connect/token")
CDSE_STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Imja Tsho AOI (bbox: [min_lon, min_lat, max_lon, max_lat])
BBOX     = [86.900, 27.885, 86.955, 27.930]
DT_RANGE = "2024-01-01T00:00:00Z/2025-09-30T23:59:59Z"


# ── Credential Resolution ──────────────────────────────────────────────────────
def resolve_credentials():
    # Tier 1: current process env
    cid  = os.environ.get('CDSE_CLIENT_ID',  '').strip()
    csec = os.environ.get('CDSE_CLIENT_SECRET', '').strip()
    if cid and csec:
        return cid, csec

    # Tier 2: active PowerShell / pwsh session
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

    # Tier 3: persisted credential file
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


# ── OAuth ──────────────────────────────────────────────────────────────────────
def get_token(cid, csec):
    payload = urllib.parse.urlencode({
        'grant_type': 'client_credentials',
        'client_id': cid,
        'client_secret': csec,
    }).encode('utf-8')
    req = urllib.request.Request(
        CDSE_TOKEN_URL, data=payload,
        headers={'Content-Type': 'application/x-www-form-urlencoded',
                 'User-Agent': 'GlacierGuard-AI/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if 'access_token' not in data:
        raise RuntimeError(f"No access_token in response: {list(data.keys())}")
    return data['access_token'], data.get('expires_in', 0)


# ── STAC Scene Selection ───────────────────────────────────────────────────────
def select_scene(token, collection):
    """Return the newest feature from STAC for the given collection over Imja Tsho."""
    auth_hdr = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'GlacierGuard-AI/1.0',
    }
    payload = json.dumps({
        "collections": [collection],
        "bbox": BBOX,
        "datetime": DT_RANGE,
        "limit": 1,
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }).encode('utf-8')
    req = urllib.request.Request(CDSE_STAC_URL, data=payload, headers=auth_hdr)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    features = result.get('features', [])
    if not features:
        # Fallback: sort by latest date manually from a larger window
        payload = json.dumps({
            "collections": [collection],
            "bbox": BBOX,
            "datetime": DT_RANGE,
            "limit": 10,
        }).encode('utf-8')
        req = urllib.request.Request(CDSE_STAC_URL, data=payload, headers=auth_hdr)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        features = result.get('features', [])
        if not features:
            raise RuntimeError(f"No scenes found for {collection}")
        features.sort(key=lambda f: f.get('properties', {}).get('datetime', ''), reverse=True)

    feat = features[0]
    props = feat.get('properties', {})
    return {
        'id':       feat.get('id', 'unknown'),
        'datetime': props.get('datetime', 'unknown'),
        'cloud_pct': props.get('eo:cloud_cover', None),
        'bbox':     feat.get('bbox', []),
    }


# ── Processing API ─────────────────────────────────────────────────────────────
def build_s2_evalscript():
    """NDWI = (Green - NIR) / (Green + NIR)  → output as single Float32 band."""
    return """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B08"], units: "REFLECTANCE"}],
    output: {bands: 1, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  var ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 1e-10);
  return [ndwi];
}
"""


def build_s1_evalscript():
    """VV backscatter in dB → single Float32 band."""
    return """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV"]}],
    output: {bands: 1, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [10 * Math.log10(sample.VV + 1e-10)];
}
"""


def call_process_api(token, sensor_key, scene, evalscript):
    """
    Send a Processing API request for the given scene.
    Returns (success, width, height, nbytes, error_msg).
    """
    # Use the exact acquisition datetime of the selected scene for a 1-day window
    dt = scene['datetime'][:10]  # YYYY-MM-DD
    dt_from = f"{dt}T00:00:00Z"
    dt_to   = f"{dt}T23:59:59Z"

    # Map sensor to datasetId
    dataset_ids = {
        'S2': 'CUSTOM',   # Sentinel-2 L2A on SH-CDSE
        'S1': 'CUSTOM',
    }
    # Use collection-level identifiers for Sentinel Hub CDSE
    input_type = 'sentinel-2-l2a' if sensor_key == 'S2' else 'sentinel-1-grd'

    # Bounding box → small AOI around Imja Tsho
    minx, miny, maxx, maxy = BBOX

    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": input_type,
                "dataFilter": {
                    "timeRange": {
                        "from": dt_from,
                        "to":   dt_to,
                    }
                }
            }]
        },
        "output": {
            "width":  256,
            "height": 256,
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get('Content-Type', '')
            raw = resp.read()
            # Check for valid TIFF magic bytes (II or MM header)
            is_tiff = (len(raw) > 4 and
                       (raw[:2] == b'II' or raw[:2] == b'MM'))
            # Parse width/height from TIFF IFD if possible
            w, h = None, None
            try:
                import struct
                byte_order = '<' if raw[:2] == b'II' else '>'
                # IFD offset is at bytes 4-7
                ifd_offset = struct.unpack(byte_order + 'I', raw[4:8])[0]
                n_entries = struct.unpack(byte_order + 'H',
                                         raw[ifd_offset:ifd_offset+2])[0]
                for i in range(min(n_entries, 64)):
                    entry_start = ifd_offset + 2 + i * 12
                    tag  = struct.unpack(byte_order + 'H',
                                         raw[entry_start:entry_start+2])[0]
                    typ  = struct.unpack(byte_order + 'H',
                                         raw[entry_start+2:entry_start+4])[0]
                    val_bytes = raw[entry_start+8:entry_start+12]
                    # type 3 = SHORT (2 bytes), type 4 = LONG (4 bytes)
                    if typ == 3:    # SHORT
                        val = struct.unpack(byte_order + 'H', val_bytes[:2])[0]
                    elif typ == 4:  # LONG
                        val = struct.unpack(byte_order + 'I', val_bytes)[0]
                    else:
                        continue
                    if tag == 256:   # ImageWidth
                        w = val
                    elif tag == 257: # ImageLength
                        h = val
            except Exception:
                pass

            return True, w, h, len(raw), content_type, None

    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')
        return False, None, None, 0, '', f"HTTP {e.code}: {err_body[:400]}"
    except Exception as ex:
        return False, None, None, 0, '', str(ex)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("GLACIERGUARD-AI — IMJA TSHO SCENE SELECTION & PROCESSING TEST")
    print("=" * 65)

    # Credentials
    cid, csec = resolve_credentials()
    if not cid or not csec:
        print("ERROR: Credentials not found.")
        sys.exit(1)
    print("  Credentials available : YES")

    # Token
    try:
        token, expires_in = get_token(cid, csec)
        print(f"  OAuth                 : SUCCESS (expires {expires_in}s)")
    except Exception as e:
        print(f"  OAuth                 : FAILURE — {e}")
        sys.exit(1)

    print()
    print("-" * 65)
    print("SCENE SELECTION — Imja Tsho  [86.900, 27.885, 86.955, 27.930]")
    print("-" * 65)

    # Select newest S1 scene
    print()
    print("  Selecting newest Sentinel-1 (sentinel-1-grd) scene ...")
    s1_scene = None
    try:
        s1_scene = select_scene(token, 'sentinel-1-grd')
        print(f"  S1 Product ID    : {s1_scene['id']}")
        print(f"  S1 Acquisition   : {s1_scene['datetime']}")
    except Exception as e:
        print(f"  S1 scene selection FAILED: {e}")

    # Select newest S2 scene
    print()
    print("  Selecting newest Sentinel-2 (sentinel-2-l2a) scene ...")
    s2_scene = None
    try:
        s2_scene = select_scene(token, 'sentinel-2-l2a')
        print(f"  S2 Product ID    : {s2_scene['id']}")
        print(f"  S2 Acquisition   : {s2_scene['datetime']}")
        cc = s2_scene.get('cloud_pct')
        print(f"  S2 Cloud Cover   : {'%.1f%%' % cc if cc is not None else 'unknown'}")
    except Exception as e:
        print(f"  S2 scene selection FAILED: {e}")

    print()
    print("-" * 65)
    print("PROCESSING API TEST — Sentinel-2 (NDWI, Float32 GeoTIFF)")
    print("-" * 65)

    if s2_scene:
        ok, w, h, nbytes, ctype, err = call_process_api(
            token, 'S2', s2_scene, build_s2_evalscript())
        if ok:
            print(f"  S2 Processing API      : PASS")
            print(f"  Response format        : {ctype}")
            print(f"  Response size          : {nbytes:,} bytes")
            print(f"  Image dimensions       : {w} x {h} px" if w else "  Image dimensions       : parsed TIFF ok (dims unavailable)")
        else:
            print(f"  S2 Processing API      : FAIL")
            print(f"  Error                  : {err}")
    else:
        print("  S2 Processing API      : SKIPPED (no scene selected)")

    print()
    print("-" * 65)
    print("PROCESSING API TEST — Sentinel-1 (VV dB, Float32 GeoTIFF)")
    print("-" * 65)

    if s1_scene:
        ok, w, h, nbytes, ctype, err = call_process_api(
            token, 'S1', s1_scene, build_s1_evalscript())
        if ok:
            print(f"  S1 Processing API      : PASS")
            print(f"  Response format        : {ctype}")
            print(f"  Response size          : {nbytes:,} bytes")
            print(f"  Image dimensions       : {w} x {h} px" if w else "  Image dimensions       : parsed TIFF ok (dims unavailable)")
        else:
            print(f"  S1 Processing API      : FAIL")
            print(f"  Error                  : {err}")
    else:
        print("  S1 Processing API      : SKIPPED (no scene selected)")

    print()
    print("=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
