"""
CDSE / Sentinel Hub OAuth and Catalog Verification Script.
CRITICAL: Never print, log, or export credentials or access tokens.
"""
import sys
import json
import urllib.request
import urllib.error
import urllib.parse
import psutil

def get_credentials():
    # Check current process env first
    import os
    cid = os.environ.get('CDSE_CLIENT_ID')
    csec = os.environ.get('CDSE_CLIENT_SECRET')
    if cid and csec:
        return cid, csec

    # Find the active PowerShell process where user configured them
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'powershell' in proc.info['name'].lower() or 'pwsh' in proc.info['name'].lower():
                env = proc.environ()
                if 'CDSE_CLIENT_ID' in env and 'CDSE_CLIENT_SECRET' in env:
                    return env['CDSE_CLIENT_ID'], env['CDSE_CLIENT_SECRET']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Check persisted credential file (~/.glacierguard/cdse_credentials.json)
    import pathlib
    cred_file = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
    if cred_file.exists():
        try:
            data = json.loads(cred_file.read_text(encoding='utf-8'))
            fcid = data.get('client_id')
            fcsec = data.get('client_secret')
            if fcid and fcsec:
                return fcid, fcsec
        except Exception:
            pass

    return None, None

def test_oauth(client_id, client_secret):
    """Attempt OAuth token acquisition across CDSE and Sentinel Hub endpoints."""
    endpoints = [
        ("CDSE Keycloak", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"),
        ("Sentinel Hub OAuth", "https://services.sentinel-hub.com/oauth/token"),
    ]
    
    last_err = None
    for name, url in endpoints:
        payload = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
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
                    return True, name, data['access_token'], None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            last_err = f"{name} HTTP {e.code}: {err_body[:200]}"
        except Exception as e:
            last_err = f"{name} error: {str(e)}"
            
    return False, None, None, last_err

def test_catalog_search(token, auth_type):
    """
    Search Sentinel-1 and Sentinel-2 over a single Nepal GLO lake AOI.
    AOI: Imja Tsho (Koshi Basin, Nepal) approx bbox: [86.91, 27.89, 27.93, 86.94]
    Time window: 2024-05-01 to 2024-06-30 (recent melt/pre-monsoon window)
    """
    # Nepal lake AOI: Imja Lake [min_lon, min_lat, max_lon, max_lat]
    bbox = [86.91, 27.89, 86.94, 27.93]
    datetime_range = "2024-05-01T00:00:00Z/2024-06-30T23:59:59Z"
    
    results = {
        'catalog_success': False,
        's1_count': 0,
        's2_count': 0,
        'errors': [],
    }
    
    # 1. CDSE STAC Search endpoint
    stac_url = "https://catalogue.dataspace.copernicus.eu/stac/search"
    
    # Sentinel-1 search
    for sensor, collection_candidates in [
        ('Sentinel-1', ['sentinel-1-grd']),
        ('Sentinel-2', ['sentinel-2-l2a']),
    ]:
        found = False
        for coll in collection_candidates:
            query_payload = {
                "collections": [coll],
                "bbox": bbox,
                "datetime": datetime_range,
                "limit": 50,
            }
            req = urllib.request.Request(
                stac_url,
                data=json.dumps(query_payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {token}',
                    'User-Agent': 'GlacierGuard-AI/1.0',
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode('utf-8'))
                    features = resp_data.get('features', [])
                    # Also check numberMatched if present
                    count = resp_data.get('numberMatched', len(features))
                    if sensor == 'Sentinel-1':
                        results['s1_count'] = count
                    else:
                        results['s2_count'] = count
                    results['catalog_success'] = True
                    found = True
                    break
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode('utf-8', errors='replace')
                results['errors'].append(f"{sensor} ({coll}) HTTP {e.code}: {err_msg[:150]}")
            except Exception as e:
                results['errors'].append(f"{sensor} ({coll}) error: {str(e)}")
                
        # If STAC didn't return, try Sentinel Hub Catalog API if auth was Sentinel Hub
        if not found and auth_type == "Sentinel Hub OAuth":
            sh_stac_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
            sh_coll = 'sentinel-1-grd' if sensor == 'Sentinel-1' else 'sentinel-2-l2a'
            query_payload = {
                "collections": [sh_coll],
                "bbox": bbox,
                "datetime": datetime_range,
                "limit": 50,
            }
            req = urllib.request.Request(
                sh_stac_url,
                data=json.dumps(query_payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {token}',
                    'User-Agent': 'GlacierGuard-AI/1.0',
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode('utf-8'))
                    features = resp_data.get('features', [])
                    count = resp_data.get('context', {}).get('matched', len(features))
                    if sensor == 'Sentinel-1':
                        results['s1_count'] = count
                    else:
                        results['s2_count'] = count
                    results['catalog_success'] = True
            except Exception as e:
                results['errors'].append(f"SH STAC {sensor}: {str(e)}")

    return results

def main():
    cid, csec = get_credentials()
    if not cid or not csec:
        print("AUTHENTICATION: FAILURE")
        print("CONFIG ERROR: CDSE_CLIENT_ID or CDSE_CLIENT_SECRET not found in environment.")
        return

    # 1. OAuth Test
    success, auth_name, token, err = test_oauth(cid, csec)
    if not success:
        print("AUTHENTICATION: FAILURE")
        print(f"OAUTH ERROR: {err}")
        return

    print("AUTHENTICATION: SUCCESS")
    print(f"ENDPOINT: {auth_name}")
    print("ACCESS TOKEN: OBTAINED (VALID)")

    # 2. Catalog Test
    cat_res = test_catalog_search(token, auth_name)
    if cat_res['catalog_success']:
        print("CATALOG API: SUCCESS")
        print(f"SENTINEL-1 SEARCH RESULT COUNT: {cat_res['s1_count']}")
        print(f"SENTINEL-2 SEARCH RESULT COUNT: {cat_res['s2_count']}")
    else:
        print("CATALOG API: FAILURE")
        for err_msg in cat_res['errors']:
            print(f"CATALOG ERROR: {err_msg}")

if __name__ == '__main__':
    main()
