"""
GlacierGuard-AI — Interactive Secure Local Credential Configuration Helper
==========================================================================
Guides developers through configuring local external services outside Git:
  1. CDSE OAuth:  ~/.glacierguard/cdse_credentials.json
  2. CDS API:     ~/.cdsapirc
  3. Databricks:  Native Databricks CLI OAuth profile (databricks auth login)

SECURITY PRINCIPLES:
  - Masked hidden input via getpass for all secrets and API keys.
  - Never prints, logs, or echoes credentials.
  - Never stores secrets in tracked repository files or project root .env.
  - Applies restrictive OS permissions (icacls on Windows).
  - Never overwrites existing valid configurations without explicit confirmation.
"""

import os
import sys
import json
import getpass
import pathlib
import subprocess

HOME = pathlib.Path.home()
CDSE_DIR = HOME / '.glacierguard'
CDSE_FILE = CDSE_DIR / 'cdse_credentials.json'
CDS_FILE = HOME / '.cdsapirc'
DATABRICKS_CFG = HOME / '.databrickscfg'


def set_restricted_permissions_windows(filepath):
    """Grant Read/Write access only to current user; strip inheritance."""
    if sys.platform != 'win32':
        try:
            os.chmod(filepath, 0o600)
            return "chmod 600"
        except Exception:
            return "default"
    username = os.environ.get('USERNAME', '')
    if not username:
        return "default ACL"
    try:
        res = subprocess.run(
            ['icacls', str(filepath), '/inheritance:r', '/grant:r', f'{username}:(R,W)'],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            return f"restricted to {username}:(R,W)"
        return f"icacls notice: {res.stderr.strip()[:60]}"
    except Exception as e:
        return f"icacls error: {e}"


def configure_cdse(check_only=False):
    print("\n" + "-" * 60)
    print("1. COPERNICUS DATA SPACE ECOSYSTEM (CDSE) OAUTH SETUP")
    print("-" * 60)
    
    # Check existing
    existing_ok = False
    if CDSE_FILE.exists():
        try:
            data = json.loads(CDSE_FILE.read_text(encoding='utf-8'))
            if data.get('client_id') and data.get('client_secret'):
                existing_ok = True
                print(f"  [OK] Credentials already present at: {CDSE_FILE}")
                print(f"       client_id length    : {len(data['client_id'])} chars")
                print(f"       client_secret length: {len(data['client_secret'])} chars")
        except Exception:
            print(f"  [WARN] Existing file at {CDSE_FILE} is malformed.")
            
    if existing_ok and check_only:
        return True
        
    if existing_ok:
        ans = input("  Overwrite existing CDSE credentials? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  Skipping CDSE setup (existing configuration preserved).")
            return True
            
    if check_only:
        print("  [ACTION NEEDED] CDSE credentials are missing.")
        return False
        
    print("\n  Enter your Copernicus Data Space OAuth Client Credentials")
    print("  (Obtain from: https://dataspace.copernicus.eu/ -> Dashboard -> OAuth)")
    cid = input("  Enter CDSE Client ID: ").strip()
    csec = getpass.getpass("  Enter CDSE Client Secret (hidden): ").strip()
    
    if not (cid and csec):
        print("  [ERROR] Client ID and Secret cannot be empty. Setup aborted.")
        return False
        
    CDSE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "client_id": cid,
        "client_secret": csec,
        "_note": "GlacierGuard-AI CDSE credentials — DO NOT COMMIT OR SHARE"
    }
    CDSE_FILE.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    perm = set_restricted_permissions_windows(CDSE_FILE)
    print(f"  [SUCCESS] Saved to {CDSE_FILE} ({perm})")
    return True


def configure_cds(check_only=False):
    print("\n" + "-" * 60)
    print("2. COPERNICUS CLIMATE DATA STORE (CDS / ERA5-LAND) SETUP")
    print("-" * 60)
    
    existing_ok = False
    if CDS_FILE.exists():
        try:
            content = CDS_FILE.read_text(encoding='utf-8')
            has_url = any(l.strip().lower().startswith('url:') for l in content.splitlines())
            has_key = any(l.strip().lower().startswith('key:') for l in content.splitlines())
            if has_url and has_key:
                existing_ok = True
                print(f"  [OK] Configuration already present at: {CDS_FILE}")
        except Exception:
            print(f"  [WARN] Existing file at {CDS_FILE} is malformed.")
            
    if existing_ok and check_only:
        return True
        
    if existing_ok:
        ans = input("  Overwrite existing CDS API credentials? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  Skipping CDS setup (existing configuration preserved).")
            return True
            
    if check_only:
        print("  [ACTION NEEDED] CDS API configuration is missing.")
        return False
        
    print("\n  Enter your Copernicus Climate Data Store (CDS) API key")
    print("  (Obtain from: https://cds.climate.copernicus.eu/ -> User Profile)")
    key = getpass.getpass("  Enter CDS API Key / Personal Access Token (hidden): ").strip()
    
    if not key:
        print("  [ERROR] API key cannot be empty. Setup aborted.")
        return False
        
    cds_text = f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n"
    CDS_FILE.write_text(cds_text, encoding='utf-8')
    perm = set_restricted_permissions_windows(CDS_FILE)
    print(f"  [SUCCESS] Saved to {CDS_FILE} ({perm})")
    return True


def configure_databricks(check_only=False):
    print("\n" + "-" * 60)
    print("3. DATABRICKS WORKSPACE SETUP")
    print("-" * 60)
    
    workspace_host = "https://dbc-0be231d6-04fb.cloud.databricks.com"
    
    # Check CLI
    try:
        res = subprocess.run(['databricks', '--version'], capture_output=True, text=True, timeout=5)
        cli_ok = True
    except Exception:
        cli_ok = False
        
    if not cli_ok:
        print("  [NOTE] Databricks CLI not found in PATH.")
        print("         To install, visit: https://docs.databricks.com/dev-tools/cli/install.html")
        return False
        
    # Test current-user
    try:
        res = subprocess.run(['databricks', 'current-user', 'me'], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            print(f"  [OK] Databricks CLI is authenticated and active.")
            return True
    except Exception:
        pass
        
    print(f"  [ACTION NEEDED] Databricks CLI authentication required for {workspace_host}")
    print(f"  Run the following command in PowerShell:")
    print(f"    databricks auth login --host {workspace_host}")
    return False


def main():
    check_only = '--check-only' in sys.argv
    print("=" * 60)
    print("GLACIERGUARD-AI — LOCAL CREDENTIAL CONFIGURATION HELPER")
    print("=" * 60)
    
    cdse_ok = configure_cdse(check_only=check_only)
    cds_ok = configure_cds(check_only=check_only)
    db_ok = configure_databricks(check_only=check_only)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  CDSE OAuth  : {'CONFIGURED' if cdse_ok else 'ACTION NEEDED'}")
    print(f"  CDS API     : {'CONFIGURED' if cds_ok else 'ACTION NEEDED'}")
    print(f"  Databricks  : {'CONFIGURED' if db_ok else 'ACTION NEEDED'}")
    print("=" * 60)
    print("\nRun verification with:")
    print("  python scratch/verify_api_credentials.py\n")


if __name__ == '__main__':
    main()
