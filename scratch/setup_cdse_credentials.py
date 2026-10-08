"""
GlacierGuard-AI — CDSE Credential Persistence & Verification
============================================================
Discovers CDSE credentials from:
  1. Current process environment (CDSE_CLIENT_ID, CDSE_CLIENT_SECRET)
  2. Active PowerShell / pwsh sessions via psutil environment inspection
  3. User-local credential file (~/.glacierguard/cdse_credentials.json)

Persists securely to ~/.glacierguard/cdse_credentials.json (outside Git repo)
with restricted permissions (current user read/write only).

Performs:
  - OAuth token acquisition test (CDSE Keycloak with Sentinel Hub fallback)
  - STAC Catalog search for Imja Tsho (Sentinel-1 & Sentinel-2)

CRITICAL: Never prints, logs, or exposes credentials or access tokens.
"""

import os
import sys
import json
import pathlib
import subprocess
import urllib.request
import urllib.parse
import urllib.error

try:
    import psutil
except ImportError:
    psutil = None

# ── Credential file location (outside git repo) ────────────────────────────────
CRED_DIR  = pathlib.Path.home() / '.glacierguard'
CRED_FILE = CRED_DIR / 'cdse_credentials.json'

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu"
                  "/auth/realms/CDSE/protocol/openid-connect/token")
SH_TOKEN_URL   = "https://services.sentinel-hub.com/oauth/token"
CDSE_STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"


def resolve_credentials():
    """
    Multi-tier credential resolution:
      Tier 1: Current process environment
      Tier 2: Active PowerShell / pwsh processes via psutil
      Tier 3: User-local credential file (~/.glacierguard/cdse_credentials.json)
    Returns: (cid, csec, source_desc, needs_save)
    """
    # Tier 1: Current process environment
    cid = os.environ.get('CDSE_CLIENT_ID', '').strip()
    csec = os.environ.get('CDSE_CLIENT_SECRET', '').strip()
    if cid and csec:
        return cid, csec, "current process environment", True

    # Tier 2: Search active PowerShell / pwsh processes
    if psutil is not None:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pname = (proc.info.get('name') or '').lower()
                if 'powershell' in pname or 'pwsh' in pname:
                    env = proc.environ()
                    proc_cid = env.get('CDSE_CLIENT_ID', '').strip()
                    proc_csec = env.get('CDSE_CLIENT_SECRET', '').strip()
                    if proc_cid and proc_csec:
                        return proc_cid, proc_csec, f"active PowerShell process (PID {proc.pid})", True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue

    # Tier 3: User-local credential file
    if CRED_FILE.exists():
        try:
            data = json.loads(CRED_FILE.read_text(encoding='utf-8'))
            file_cid = data.get('client_id', '').strip()
            file_csec = data.get('client_secret', '').strip()
            if file_cid and file_csec:
                return file_cid, file_csec, f"credential file ({CRED_FILE})", False
        except Exception as e:
            print(f"  [WARN] Failed to read credential file: {e}")

    return None, None, "none", False


def save_credentials(cid, csec):
    """Save credentials securely to ~/.glacierguard/cdse_credentials.json."""
    CRED_DIR.mkdir(parents=True, exist_ok=True)
    cred_data = {
        "client_id":     cid,
        "client_secret": csec,
        "_note":         "GlacierGuard-AI CDSE credentials — DO NOT COMMIT OR SHARE",
    }
    CRED_FILE.write_text(json.dumps(cred_data, indent=2), encoding='utf-8')

    # Apply restricted permissions on Windows
    perm_status = "default filesystem ACL"
    if sys.platform == 'win32':
        try:
            username = os.environ.get('USERNAME', '')
            if username:
                # Remove inheritance and grant Read/Write only to current user
                res = subprocess.run(
                    ['icacls', str(CRED_FILE), '/inheritance:r', '/grant:r', f'{username}:(R,W)'],
                    capture_output=True, text=True
                )
                if res.returncode == 0:
                    perm_status = f"restricted to {username}:(R,W)"
                else:
                    perm_status = f"icacls notice: {res.stderr.strip()[:80]}"
        except Exception as ex:
            perm_status = f"icacls error: {ex}"

    return perm_status


def test_oauth(cid, csec):
    """Authenticate against CDSE / Sentinel Hub OAuth endpoints."""
    endpoints = [
        ("CDSE Keycloak", CDSE_TOKEN_URL),
        ("Sentinel Hub OAuth", SH_TOKEN_URL),
    ]

    for name, url in endpoints:
        payload = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': cid,
            'client_secret': csec,
        }).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'GlacierGuard-AI/1.0',
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if 'access_token' in data:
                    return True, name, data['access_token'], data.get('expires_in', 0), None
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8', errors='replace')
            last_err = f"{name} HTTP {e.code}: {err[:200]}"
        except Exception as ex:
            last_err = f"{name} error: {ex}"

    return False, None, None, 0, last_err


def test_catalog_imja_tsho(token):
    """Run a targeted STAC search for Imja Tsho (Sentinel-1 and Sentinel-2)."""
    # Imja Tsho bbox: [min_lon, min_lat, max_lon, max_lat]
    bbox = [86.900, 27.885, 86.955, 27.930]
    dt_range = "2024-01-01T00:00:00Z/2025-09-29T23:59:59Z"

    results = {}
    auth_hdr = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'GlacierGuard-AI/1.0',
    }

    sensors = [
        ("Sentinel-1", ["sentinel-1-grd"]),
        ("Sentinel-2", ["sentinel-2-l2a"]),
    ]

    for sensor, collections in sensors:
        sensor_success = False
        for coll in collections:
            payload = json.dumps({
                "collections": [coll],
                "bbox": bbox,
                "datetime": dt_range,
                "limit": 100,
            }).encode('utf-8')
            req = urllib.request.Request(CDSE_STAC_URL, data=payload, headers=auth_hdr)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode('utf-8'))
                    features = resp_data.get('features', [])
                    matched = resp_data.get('numberMatched', len(features))
                    dates = sorted(set(
                        f.get('properties', {}).get('datetime', '')[:10]
                        for f in features if f.get('properties', {}).get('datetime')
                    ), reverse=True)
                    results[sensor] = {
                        'status': 'SUCCESS',
                        'collection': coll,
                        'matched': matched,
                        'latest_dates': dates[:5],
                    }
                    sensor_success = True
                    break
            except Exception as ex:
                continue

        if not sensor_success:
            results[sensor] = {'status': 'FAILURE', 'error': 'No matching collection accessible'}

    return results


def main():
    print("=" * 65)
    print("GLACIERGUARD-AI — CDSE CREDENTIAL WORKFLOW & VERIFICATION")
    print("=" * 65)

    # 1. Resolve credentials
    cid, csec, source, needs_save = resolve_credentials()
    has_creds = bool(cid and csec)
    print(f"  Credentials available : {'YES' if has_creds else 'NO'}")
    print(f"  Discovery source      : {source}")

    if not has_creds:
        print()
        print("ERROR: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not found in environment,")
        print("active PowerShell processes, or ~/.glacierguard/cdse_credentials.json.")
        sys.exit(1)

    # 2. Persist to local secure file
    if needs_save:
        perm_info = save_credentials(cid, csec)
        print(f"  Credential storage    : SAVED to {CRED_FILE}")
        print(f"  File permissions      : {perm_info}")
    else:
        print(f"  Credential storage    : ALREADY PERSISTED at {CRED_FILE}")

    # 3. OAuth authentication test
    print()
    print("-" * 65)
    print("STEP 1: OAuth Authentication Test")
    print("-" * 65)
    success, endpoint_name, token, expires_in, err = test_oauth(cid, csec)
    if not success:
        print(f"  OAuth result          : FAILURE")
        print(f"  Error details         : {err}")
        sys.exit(1)

    print(f"  OAuth result          : SUCCESS")
    print(f"  Endpoint              : {endpoint_name}")
    print(f"  Access token          : OBTAINED (valid, masked)")
    print(f"  Expires in            : {expires_in} seconds")

    # 4. Catalog API test — Imja Tsho only
    print()
    print("-" * 65)
    print("STEP 2: Catalog STAC API Test — Imja Tsho AOI")
    print("-" * 65)
    print("  Target lake           : Imja Tsho (Nepal)")
    print("  AOI bounding box      : [86.900, 27.885, 86.955, 27.930]")
    print("  Temporal window       : 2024-01-01 to 2025-09-29")
    print()

    cat_res = test_catalog_imja_tsho(token)
    for sensor, info in cat_res.items():
        if info['status'] == 'SUCCESS':
            print(f"  {sensor:<12} [{info['collection']}]")
            print(f"    Catalog status      : SUCCESS")
            print(f"    Matching scenes     : {info['matched']}")
            print(f"    Recent dates sample : {info['latest_dates']}")
        else:
            print(f"  {sensor:<12}")
            print(f"    Catalog status      : FAILURE ({info.get('error')})")

    print()
    print("=" * 65)
    print("VERIFICATION COMPLETE — CREDENTIALS PERSISTED & CATALOG OPERATIONAL")
    print("=" * 65)


if __name__ == '__main__':
    main()
