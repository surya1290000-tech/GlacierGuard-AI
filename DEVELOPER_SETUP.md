# GlacierGuard-AI — Developer Setup & Credentials Guide

This guide details how to configure and verify external services, virtual environments, and API credentials for **GlacierGuard-AI** on Windows (PowerShell) or Unix systems.

---

## 1. Architecture: Why Credentials Live Outside the Repository

GlacierGuard-AI uses a **security-first, user-local credential strategy**:

- **No Secrets in Project Root:** The project does **not** use a root `.env` file for real credentials. This prevents accidental commits, leaked Git history, and cross-developer credential collisions.
- **Persistent Across Working Directories:** Notebooks (`notebooks/`) and scripts (`scratch/`) execute from different working directories. Using user-local paths (`%USERPROFILE%` / `~`) guarantees that credentials resolve identically regardless of which folder is active.
- **Independent of Terminal Environment:** Jupyter kernels started from VS Code, Databricks, or JupyterLab do not always inherit volatile PowerShell terminal variables. User-local files ensure kernels authenticate reliably.

### Precedence of Configuration Sources

For each external service, the project resolves configuration in the following order:

1. **CDSE / Sentinel Hub (NRT Satellite Imagery):**
   - **Tier 1:** Environment variables (`CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`) if explicitly set in the process.
   - **Tier 2 (Standard):** User-local credential file at `~/.glacierguard/cdse_credentials.json`.
   - **Tier 3 (Fallback):** Scans active parent PowerShell sessions (for interactive dev).

2. **Copernicus Climate Data Store (CDS / ERA5-Land Reanalysis):**
   - **Tier 1:** Environment variables (`CDSAPI_URL`, `CDSAPI_KEY`) if set.
   - **Tier 2 (Standard):** User-local configuration file at `~/.cdsapirc`.

3. **Databricks Workspace:**
   - **Tier 1:** Active CLI profile `[data]` or default profile in `~/.databrickscfg` (authenticated via OAuth keyring).
   - **Tier 2:** Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`) if set.
   - **Tier 3:** Declarative bundle settings in `databricks.yml`.

---

## 2. Quickstart on a Fresh Clone (Windows PowerShell)

```powershell
# 1. Clone the existing repository
git clone https://github.com/surya1290000-tech/GlacierGuard-AI.git
cd GlacierGuard-AI

# 2. Synchronize Python 3.12 virtual environment with uv
uv sync

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Check credential status or configure interactively
python scratch/configure_credentials.py

# 5. Run the complete automated verification suite
python scratch/verify_api_credentials.py
```

---

## 3. Step-by-Step Service Configuration

### A. Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub

Used for near-real-time Sentinel-1 SAR and Sentinel-2 optical imagery retrieval.

1. Register an account at [Copernicus Data Space](https://dataspace.copernicus.eu/).
2. Navigate to your user dashboard -> **OAuth Clients** (or Sentinel Hub API Keys).
3. Create an OAuth client and copy the `Client ID` and `Client Secret`.
4. Run the interactive setup helper:
   ```powershell
   python scratch/configure_credentials.py
   ```
   *Alternatively, create the file manually:*
   Create `C:\Users\<username>\.glacierguard\cdse_credentials.json`:
   ```json
   {
     "client_id": "your-client-id",
     "client_secret": "your-client-secret"
   }
   ```
   Apply restrictive file permissions on Windows:
   ```powershell
   icacls "$HOME\.glacierguard\cdse_credentials.json" /inheritance:r /grant:r "$($env:USERNAME):(R,W)"
   ```

### B. Copernicus Climate Data Store (CDS / ERA5-Land)

Used for downloading ERA5-Land monthly reanalysis (temperature, runoff, snowmelt).

1. Register an account at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Log in, view your user profile, and copy your **Personal Access Token** (API key).
3. Accept the **Terms of Use** and **ERA5-Land licence** in your CDS dashboard.
4. Run the setup helper or create `C:\Users\<username>\.cdsapirc`:
   ```ini
   url: https://cds.climate.copernicus.eu/api
   key: your-personal-access-token
   ```
5. Apply restrictive file permissions:
   ```powershell
   icacls "$HOME\.cdsapirc" /inheritance:r /grant:r "$($env:USERNAME):(R,W)"
   ```

### C. Databricks Workspace & CLI

Used for distributed PySpark processing in `notebooks/02_glo_databricks_eda.ipynb`.

1. Ensure the Databricks CLI is installed:
   ```powershell
   databricks --version
   ```
2. Log into the project's Databricks workspace using OAuth:
   ```powershell
   databricks auth login --host https://dbc-0be231d6-04fb.cloud.databricks.com
   ```
3. When prompted, choose profile name `data` (or press Enter for default).
4. Verify authentication:
   ```powershell
   databricks current-user me
   ```

---

## 4. Verification Suite

Run the unified verification suite to test real server authentication and functional API calls across all services:

```powershell
python scratch/verify_api_credentials.py
```

### Expected Output Summary
```
======================================================================
GLACIERGUARD-AI — EXTERNAL SERVICES VERIFICATION SUMMARY
======================================================================
| Service                | Credentials found | Authentication | Functional test            | Result          |
|------------------------|-------------------|----------------|----------------------------|-----------------|
| CDSE OAuth             | YES               | PASS           | Token issued (1800s)       | PASS            |
| Sentinel-1 STAC        | YES               | PASS           | 10 scenes returned         | PASS            |
| Sentinel-2 STAC        | YES               | PASS           | 10 scenes returned         | PASS            |
| Processing API         | YES               | PASS           | GeoTIFF 64x64 (25226B)     | PASS            |
| CDS / ERA5             | YES               | PASS           | Collection accessible      | PASS            |
| Databricks             | YES               | PASS           | CLI & SDK operational      | PASS            |
| Notebook environment   | YES               | PASS           | Imports & home paths OK    | PASS            |
| Git secret audit       | N/A               | PASS           | 0 secrets found            | PASS            |
======================================================================
```

---

## 5. Security & Pre-Commit Audit

Before pushing any commits or submitting pull requests, run the static secret scanner:

```powershell
python scratch/audit_secrets.py
```

- Verifies that zero hardcoded API keys, tokens, or private secrets exist in tracked files.
- Checks that `.env` files, `.cdsapirc`, and `.glacierguard/` directories are properly ignored by Git.
- Never commit credentials to Git.
