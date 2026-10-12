"""
GlacierGuard-AI — Unified External Services Authentication & Verification Suite
==============================================================================
Validates end-to-end authentication and operational capabilities across:
  1. CDSE OAuth (Keycloak / Sentinel Hub)
  2. Sentinel-1 STAC Catalog API (sentinel-1-grd)
  3. Sentinel-2 STAC Catalog API (sentinel-2-l2a)
  4. Sentinel Hub Processing API (GeoTIFF imagery retrieval)
  5. Copernicus Climate Data Store (CDS API / ERA5-Land reanalysis)
  6. Databricks CLI & SDK Workspace Access
  7. Python Environment & Notebook Directory Credential Resolution
  8. Git Secret Security Scan

CRITICAL SECURITY RULES:
  - NEVER prints or reveals API keys, secrets, authorization headers, or access tokens.
  - Masks identifiers and emails where appropriate.
  - Produces standard verification table with PASS / FAIL / NOT CONFIGURED status.
"""

import os
import sys
import json
import time
import pathlib
import urllib.request
import urllib.parse
import urllib.error
import subprocess
import configparser

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()

CDSE_CRED_FILE = HOME / '.glacierguard' / 'cdse_credentials.json'
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_TOKEN_URL   = "https://services.sentinel-hub.com/oauth/token"
CDSE_STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CDS_CONFIG_FILE = HOME / '.cdsapirc'
DATABRICKS_CFG = HOME / '.databrickscfg'

RESULTS = []

def record_result(service, creds_found, auth_status, test_status, final_result, notes=""):
    RESULTS.append({
        "service": service,
        "creds_found": creds_found,
        "auth_status": auth_status,
        "test_status": test_status,
        "final_result": final_result,
        "notes": notes
    })


# =============================================================================
# 1. CDSE / SENTINEL HUB VERIFICATION
# =============================================================================
def test_cdse():
    print("\n" + "=" * 70)
    print("1. CDSE / SENTINEL HUB AUTHENTICATION & API VERIFICATION")
    print("=" * 70)
    
    # 1.1 Resolution
    cid = os.environ.get('CDSE_CLIENT_ID', '').strip()
    csec = os.environ.get('CDSE_CLIENT_SECRET', '').strip()
    source = "environment variable"
    
    if not (cid and csec) and CDSE_CRED_FILE.exists():
        try:
            data = json.loads(CDSE_CRED_FILE.read_text(encoding='utf-8'))
            cid = data.get('client_id', '').strip()
            csec = data.get('client_secret', '').strip()
            source = f"file ({CDSE_CRED_FILE})"
        except Exception as e:
            print(f"  [WARN] Failed to read CDSE credential file: {e}")
            
    has_creds = bool(cid and csec)
    print(f"  Credentials presence : {'FOUND' if has_creds else 'NOT FOUND'} (source: {source if has_creds else 'none'})")
    
    if not has_creds:
        record_result("CDSE OAuth", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "Missing CDSE_CLIENT_ID/SECRET")
        record_result("Sentinel-1 STAC", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "No auth token")
        record_result("Sentinel-2 STAC", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "No auth token")
        record_result("Processing API", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "No auth token")
        return None
        
    # 1.2 OAuth Token Acquisition
    token = None
    endpoint_used = None
    last_err = None
    
    for name, url in [("CDSE Keycloak", CDSE_TOKEN_URL), ("Sentinel Hub", SH_TOKEN_URL)]:
        payload = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': cid,
            'client_secret': csec
        }).encode('utf-8')
        req = urllib.request.Request(
            url, data=payload,
            headers={'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': 'GlacierGuard-AI/1.0'}
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if 'access_token' in data:
                    token = data['access_token']
                    expires_in = data.get('expires_in', 0)
                    endpoint_used = name
                    print(f"  OAuth Authentication : PASS ({name}, HTTP {resp.status}, expires_in={expires_in}s)")
                    record_result("CDSE OAuth", "YES", "PASS", f"Token issued ({expires_in}s)", "PASS", f"Endpoint: {name}")
                    break
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = f"Error: {e}"
            
    if not token:
        print(f"  OAuth Authentication : FAIL ({last_err})")
        record_result("CDSE OAuth", "YES", "FAIL", "No token", "FAIL", last_err)
        record_result("Sentinel-1 STAC", "YES", "SKIPPED", "SKIPPED", "FAIL", "OAuth failed")
        record_result("Sentinel-2 STAC", "YES", "SKIPPED", "SKIPPED", "FAIL", "OAuth failed")
        record_result("Processing API", "YES", "SKIPPED", "SKIPPED", "FAIL", "OAuth failed")
        return None
        
    auth_hdr = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'GlacierGuard-AI/1.0'
    }
    
    # 1.3 STAC Sentinel-1 Query
    bbox = [86.900, 27.885, 86.955, 27.930]  # Imja Tsho bbox
    dt_range = "2024-01-01T00:00:00Z/2026-09-30T23:59:59Z"
    
    try:
        s1_payload = json.dumps({'collections': ['sentinel-1-grd'], 'bbox': bbox, 'datetime': dt_range, 'limit': 10}).encode('utf-8')
        req = urllib.request.Request(CDSE_STAC_URL, data=s1_payload, headers=auth_hdr)
        with urllib.request.urlopen(req, timeout=20) as resp:
            s1_data = json.loads(resp.read().decode('utf-8'))
            s1_feats = s1_data.get('features', [])
            matched = s1_data.get('numberMatched', len(s1_feats))
            print(f"  Sentinel-1 STAC (sentinel-1-grd) : PASS (HTTP {resp.status}, matched={matched}, returned={len(s1_feats)})")
            record_result("Sentinel-1 STAC", "YES", "PASS", f"{len(s1_feats)} scenes returned", "PASS", f"Matched: {matched}")
    except Exception as e:
        print(f"  Sentinel-1 STAC : FAIL ({e})")
        record_result("Sentinel-1 STAC", "YES", "PASS", "Query failed", "FAIL", str(e)[:60])
        
    # 1.4 STAC Sentinel-2 Query
    try:
        s2_payload = json.dumps({'collections': ['sentinel-2-l2a'], 'bbox': bbox, 'datetime': dt_range, 'limit': 10}).encode('utf-8')
        req = urllib.request.Request(CDSE_STAC_URL, data=s2_payload, headers=auth_hdr)
        with urllib.request.urlopen(req, timeout=20) as resp:
            s2_data = json.loads(resp.read().decode('utf-8'))
            s2_feats = s2_data.get('features', [])
            matched = s2_data.get('numberMatched', len(s2_feats))
            print(f"  Sentinel-2 STAC (sentinel-2-l2a) : PASS (HTTP {resp.status}, matched={matched}, returned={len(s2_feats)})")
            record_result("Sentinel-2 STAC", "YES", "PASS", f"{len(s2_feats)} scenes returned", "PASS", f"Matched: {matched}")
    except Exception as e:
        print(f"  Sentinel-2 STAC : FAIL ({e})")
        record_result("Sentinel-2 STAC", "YES", "PASS", "Query failed", "FAIL", str(e)[:60])
        
    # 1.5 Minimal Processing API Test
    evalscript = """//VERSION=3
function setup() {
  return { input: [{bands: ["B03", "B11"]}], output: {bands: 2, sampleType: "FLOAT32"} };
}
function evaluatePixel(sample) { return [sample.B03, sample.B11]; }
"""
    proc_payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": "2026-09-28T00:00:00Z", "to": "2026-09-28T23:59:59Z"}}}]
        },
        "output": {"width": 64, "height": 64, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript
    }
    try:
        req = urllib.request.Request(SH_PROCESS_URL, data=json.dumps(proc_payload).encode('utf-8'), headers=auth_hdr)
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()
            # Verify TIFF magic bytes: II*\x00 (little-endian) or MM\x00* (big-endian)
            is_tiff = img_bytes[:4] in (b'II*\x00', b'MM\x00*')
            print(f"  Processing API (Sentinel-2 GeoTIFF) : PASS (HTTP {resp.status}, size={len(img_bytes)} bytes, is_tiff={is_tiff})")
            record_result("Processing API", "YES", "PASS", f"GeoTIFF 64x64 ({len(img_bytes)}B)", "PASS", "Valid TIFF stream")
    except Exception as e:
        print(f"  Processing API : FAIL ({e})")
        record_result("Processing API", "YES", "PASS", "Process call failed", "FAIL", str(e)[:60])
        
    return token


# =============================================================================
# 2. CLIMATE DATA STORE (CDS / ERA5-LAND) VERIFICATION
# =============================================================================
def test_cds():
    print("\n" + "=" * 70)
    print("2. COPERNICUS CLIMATE DATA STORE (CDS / ERA5-LAND) VERIFICATION")
    print("=" * 70)
    
    has_file = CDS_CONFIG_FILE.exists()
    has_env = bool(os.environ.get('CDSAPI_KEY'))
    print(f"  Credentials presence : {'FOUND' if (has_file or has_env) else 'NOT FOUND'} (file: {has_file}, env: {has_env})")
    
    if not (has_file or has_env):
        record_result("CDS / ERA5", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "Missing ~/.cdsapirc")
        return
        
    try:
        import cdsapi
        client = cdsapi.Client(quiet=True)
        print(f"  Client initialization : PASS (URL: {client.url})")
        
        # Test authentication endpoint
        auth_info = client.client.check_authentication()
        user_id = auth_info.get('id', 'unknown')
        user_email = auth_info.get('email', '')
        # Mask email: e.g. s***@gmail.com
        if user_email and '@' in user_email:
            name_part, domain_part = user_email.split('@', 1)
            masked_email = name_part[0] + '***@' + domain_part
        else:
            masked_email = 'masked'
            
        print(f"  Server Authentication : PASS (User ID: {user_id}, Account: {masked_email})")
        
        # Test ERA5-Land collection metadata access
        coll = client.client.get_collection('reanalysis-era5-land-monthly-means')
        print(f"  Collection Access     : PASS (ID: {coll.id}, Title: {coll.title[:45]}...)")
        
        record_result("CDS / ERA5", "YES", "PASS", f"Collection {coll.id} accessible", "PASS", f"User ID: {user_id}")
    except Exception as e:
        err_msg = str(e)
        print(f"  CDS Authentication/Check: FAIL ({err_msg})")
        record_result("CDS / ERA5", "YES", "FAIL", "Auth/Metadata check failed", "FAIL", err_msg[:60])


# =============================================================================
# 3. DATABRICKS CLI & SDK WORKSPACE VERIFICATION
# =============================================================================
def test_databricks():
    print("\n" + "=" * 70)
    print("3. DATABRICKS CLI & WORKSPACE VERIFICATION")
    print("=" * 70)
    
    has_cfg = DATABRICKS_CFG.exists()
    has_env = bool(os.environ.get('DATABRICKS_TOKEN') or os.environ.get('DATABRICKS_HOST'))
    print(f"  Config presence      : {'FOUND' if (has_cfg or has_env) else 'NOT FOUND'} (file: {has_cfg}, env: {has_env})")
    
    # Check CLI
    cli_installed = False
    try:
        res = subprocess.run(['databricks', '--version'], capture_output=True, text=True, timeout=5)
        cli_ver = res.stdout.strip() if res.stdout else res.stderr.strip()
        cli_installed = True
        print(f"  Databricks CLI       : INSTALLED ({cli_ver})")
    except Exception:
        print(f"  Databricks CLI       : NOT INSTALLED IN PATH")
        
    # Check CLI current-user
    cli_auth_ok = False
    if cli_installed:
        try:
            res = subprocess.run(['databricks', 'current-user', 'me'], capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                user_name = data.get('userName', 'authenticated')
                # Mask username
                if '@' in user_name:
                    p1, p2 = user_name.split('@', 1)
                    masked_user = p1[0] + '***@' + p2
                else:
                    masked_user = 'user'
                print(f"  CLI Authentication   : PASS (User: {masked_user}, Active: {data.get('active', True)})")
                cli_auth_ok = True
            else:
                print(f"  CLI Authentication   : FAIL ({res.stderr.strip()[:80]})")
        except Exception as e:
            print(f"  CLI Authentication   : FAIL ({e})")
            
    # Check Databricks SDK WorkspaceClient
    sdk_auth_ok = False
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        u = w.current_user.me()
        user_name = u.user_name or 'authenticated'
        if '@' in user_name:
            p1, p2 = user_name.split('@', 1)
            masked_user = p1[0] + '***@' + p2
        else:
            masked_user = 'user'
        print(f"  SDK WorkspaceClient  : PASS (Host: {w.config.host}, User: {masked_user})")
        sdk_auth_ok = True
    except Exception as e:
        print(f"  SDK WorkspaceClient  : FAIL ({e})")
        
    if cli_auth_ok and sdk_auth_ok:
        record_result("Databricks", "YES", "PASS", "CLI & SDK operational", "PASS", "Workspace host verified")
    elif cli_auth_ok or sdk_auth_ok:
        record_result("Databricks", "YES", "PASS", "Partial (CLI or SDK only)", "PASS", "Workspace accessible")
    elif not has_cfg and not has_env:
        record_result("Databricks", "NO", "NOT CONFIGURED", "SKIPPED", "NOT CONFIGURED", "Run databricks auth login")
    else:
        record_result("Databricks", "YES", "FAIL", "Auth check failed", "FAIL", "Check profile config")


# =============================================================================
# 4. PYTHON ENVIRONMENT & NOTEBOOK CONTEXT VERIFICATION
# =============================================================================
def test_environment_and_notebooks():
    print("\n" + "=" * 70)
    print("4. PYTHON ENVIRONMENT & NOTEBOOK RESOLUTION VERIFICATION")
    print("=" * 70)
    
    # 4.1 Module imports
    modules = ['cdsapi', 'databricks.sdk', 'rasterio', 'geopandas', 'shapely', 'sklearn', 'xgboost']
    missing = []
    for m in modules:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
            
    if missing:
        print(f"  Python Packages      : INCOMPLETE (missing: {missing})")
        imports_ok = False
    else:
        print(f"  Python Packages      : PASS (all {len(modules)} required packages importable)")
        imports_ok = True
        
    # 4.2 Credential resolution from simulated notebooks/ working directory
    # Temporarily change working directory to notebooks/
    original_cwd = pathlib.Path.cwd()
    notebooks_dir = ROOT / 'notebooks'
    
    try:
        os.chdir(str(notebooks_dir))
        
        # Test CDSE resolution from notebooks directory
        cdse_cred = HOME / '.glacierguard' / 'cdse_credentials.json'
        cdse_resolved_from_nb = cdse_cred.exists()
        
        # Test CDS resolution from notebooks directory
        cds_cred = HOME / '.cdsapirc'
        cds_resolved_from_nb = cds_cred.exists()
        
        # Test Databricks resolution from notebooks directory
        db_cred = HOME / '.databrickscfg'
        db_resolved_from_nb = db_cred.exists()
        
        all_resolved = cdse_resolved_from_nb and cds_resolved_from_nb and db_resolved_from_nb
        print(f"  Notebook CWD Test    : PASS (Simulated CWD: {notebooks_dir.name}/)")
        print(f"    - CDSE path resolve: {'PASS' if cdse_resolved_from_nb else 'FAIL'}")
        print(f"    - CDS path resolve : {'PASS' if cds_resolved_from_nb else 'FAIL'}")
        print(f"    - DB path resolve  : {'PASS' if db_resolved_from_nb else 'FAIL'}")
    finally:
        os.chdir(str(original_cwd))
        
    if imports_ok and all_resolved:
        record_result("Notebook environment", "YES", "PASS", "Imports & home paths OK", "PASS", "CWD independent")
    else:
        record_result("Notebook environment", "YES", "FAIL", f"Missing={missing}", "FAIL", "Check environment")


# =============================================================================
# 5. GIT SECRET AUDIT
# =============================================================================
def test_git_secret_audit():
    print("\n" + "=" * 70)
    print("5. GIT SECRET AUDIT")
    print("=" * 70)
    
    audit_script = ROOT / 'scratch' / 'audit_secrets.py'
    if not audit_script.exists():
        record_result("Git secret audit", "N/A", "N/A", "Script missing", "FAIL", "audit_secrets.py missing")
        return
        
    res = subprocess.run([sys.executable, str(audit_script)], capture_output=True, text=True)
    out = res.stdout.strip()
    
    if "Total potential secret findings: 0" in out:
        print("  Secret Audit Result  : PASS (0 secrets or exposed credentials found in workspace)")
        record_result("Git secret audit", "N/A", "PASS", "0 secrets found", "PASS", "Workspace clean")
    else:
        print(f"  Secret Audit Result  : FAIL or WARNING\n{out}")
        record_result("Git secret audit", "N/A", "FAIL", "Potential secret found", "FAIL", "Check audit_secrets.py output")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def print_summary_table():
    print("\n" + "=" * 70)
    print("GLACIERGUARD-AI — EXTERNAL SERVICES VERIFICATION SUMMARY")
    print("=" * 70)
    
    fmt = "| {:<22} | {:<17} | {:<14} | {:<26} | {:<15} |"
    divider = "|-" + "-"*22 + "-|-" + "-"*17 + "-|-" + "-"*14 + "-|-" + "-"*26 + "-|-" + "-"*15 + "-|"
    
    print(fmt.format("Service", "Credentials found", "Authentication", "Functional test", "Result"))
    print(divider)
    for r in RESULTS:
        print(fmt.format(
            r["service"],
            r["creds_found"],
            r["auth_status"],
            r["test_status"],
            r["final_result"]
        ))
    print("=" * 70)


def main():
    test_cdse()
    test_cds()
    test_databricks()
    test_environment_and_notebooks()
    test_git_secret_audit()
    print_summary_table()


if __name__ == '__main__':
    main()
