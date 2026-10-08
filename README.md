# GlacierGuard-AI

> **Satellite-based monitoring and prediction of annual glacial lake area change, with operational near-real-time Sentinel-1 and Sentinel-2 multi-sensor monitoring.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://docs.astral.sh/uv/)
[![Target: Annual Area Change](https://img.shields.io/badge/target-NEXT__AREA__CHANGE-green.svg)](#6-target-definition)
[![Validated Model: Fused XGBoost](https://img.shields.io/badge/model-S1%2BS2%20Fused%20XGBoost-orange.svg)](#8-current-best-ml-model)

---

## 1. Project Overview

**GlacierGuard-AI** is a dual-tier remote sensing and machine learning platform engineered to study and monitor glacial lake expansion across High Mountain Asia (focusing on Nepal and transboundary Himalayan catchments). 

### The Scientific Problem
Glacial lakes in the Himalayas are expanding rapidly due to accelerated glacier retreat under climatic warming. Unstable moraine-dammed and ice-dammed lakes pose acute hazards of Glacial Lake Outburst Floods (GLOFs), which threaten downstream communities, hydropower installations, and infrastructure. Understanding which lakes expand most rapidly and establishing reliable continuous surveillance are paramount scientific challenges.

### Prediction vs. Near-Real-Time (NRT) Monitoring
GlacierGuard-AI explicitly distinguishes between two complementary operational regimes:

1. **Annual Area Change Prediction (Machine Learning Tier):**  
   Models multi-year glacial lake surface area evolution using historical satellite inventories (GLO v1.02), morphological indicators (RGI 6.0/7.0), and environmental drivers (ERA5-Land reanalysis and CHIRPS precipitation). It predicts the *next-year area change* ($\text{AREA}_{t+1} - \text{AREA}_t$) to identify lakes undergoing anomalous expansion rates.
2. **Near-Real-Time Multi-Sensor Surveillance (Satellite Monitoring Tier):**  
   Dynamically queries the Copernicus Data Space Ecosystem (CDSE) for fresh Sentinel-1 (SAR) and Sentinel-2 (Multispectral Optical) imagery over priority lake areas of interest (AOIs), extracts current water masks, computes calibrated lake surface areas, and flags statistical deviations relative to historical baselines.

> [!IMPORTANT]
> **Scientific Scope and Limitation:**  
> The current system predicts **annual lake area evolution** and provides **near-real-time satellite surveillance**. It does **NOT** claim 24/7 continuous live sensing, nor does it claim instantaneous or confirmed GLOF detection. A change in surface area indicates physical evolution or dam deformation, which serves as a decision-support indicator, not an automated flood alarm.

---

## 2. Current Architecture

```
═════════════════════════════════════════════════════════════════════════════════
TIER 1: HISTORICAL RESEARCH & ANNUAL ML PREDICTION PIPELINE
═════════════════════════════════════════════════════════════════════════════════

  GLO v1.02 Dataset (S1 SAR & S2 Optical Inventories: 2017–2024)
                           │
                           ▼
  Databricks EDA & Quality Screening (Geometry, Nulls, Missing-Year Filters)
                           │
                           ▼
  Feature Engineering (Lags, Ratios, Glacier Distance [RGI], Climate [ERA5/CHIRPS])
                           │
                           ▼
  Temporal Split (Train <= 2021 | Validation = 2022 | Frozen Test = 2023)
                           │
                           ▼
  Cross-Sensor Model Fusion (S1 SAR Features + S2 Optical Features)
                           │
                           ▼
  Validated Model: Multi-Sensor XGBoost Regression
  [ Matched Test: MAE = 0.004792 km² | RMSE = 0.015079 km² | R² = 0.3366 ]

═════════════════════════════════════════════════════════════════════════════════
TIER 2: OPERATIONAL NEAR-REAL-TIME (NRT) SATELLITE MONITORING PIPELINE
═════════════════════════════════════════════════════════════════════════════════

  Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub
  OAuth Authentication (Keycloak / Sentinel Hub fallback)
                           │
                           ▼
  STAC Catalog Search (sentinel-1-grd & sentinel-2-l2a)
                           │
                           ▼
  Processing API / GeoTIFF Retrieval (Recent AOI Scenes)
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Sentinel-2 Optical (L2A)    Sentinel-1 SAR (GRD)
    - SCL Cloud/Shadow Mask     - Border & Thermal Noise Cut
    - Calibrated MNDWI (>0.0)   - Lee 5x5 Speckle Filter
    - Core Lake Extraction      - Calibrated Threshold (-12 dB)
             └─────────────┬─────────────┘
                           ▼
  Standardized Spatial Envelopes (GLO Geometry Prior + Search Buffers)
                           │
                           ▼
  Lake Water Extent & Area Calculation (km²)
                           │
                           ▼
  Multi-Sensor Cross-Agreement & Operational QA Status Flagging
  (NORMAL_EXPANSION | WATCH | SENSOR_DISAGREEMENT | UNUSUAL_CHANGE)
```

---

## 3. Repository Structure

```
GlacierGuard-AI/
├── .gitignore                      # Git exclusion rules (protects raw data and secrets)
├── .env.example                    # Template environment variables (no secrets)
├── .vscode/
│   └── settings.json               # VS Code / Databricks interactive window settings
├── databricks.yml                  # Databricks asset bundle definition
├── pyproject.toml                  # Python 3.12 dependencies and Databricks Connect
├── uv.lock                         # Deterministic package dependency lockfile
├── README.md                       # Comprehensive project documentation
│
├── notebooks/                      # Sequential research & operational notebooks
│   ├── 01_glo_dataset_inspection.ipynb
│   ├── 02_glo_databricks_eda.ipynb
│   ├── 03_glo_feature_engineering.ipynb
│   ├── 04_glo_baseline_ml.ipynb
│   ├── 05_glo_target_uncertainty_analysis.ipynb
│   ├── 06_model_development.ipynb
│   ├── 07_environmental_forcing.ipynb
│   ├── 08_multisource_advanced_modeling.ipynb
│   ├── 09_final_model_validation.ipynb
│   ├── 10_nrt_lake_monitoring.ipynb
│   └── 10_1_imja_extent_calibration.ipynb
│
├── reports/                        # Curated validation, ablation & benchmark CSVs
│   ├── 06_*.csv                    # Single-sensor baseline model comparisons
│   ├── 07_*.csv                    # ERA5 & CHIRPS climate ablation reports
│   ├── 08_*.csv                    # Multi-source sensor fusion results
│   ├── 09_*.csv                    # Confirmatory validation & bootstrap CIs
│   ├── 10_*.csv                    # NRT monitoring results & scene inventories
│   ├── 10_1_*.csv                  # Water extraction calibration tables
│   ├── 10_2_five_lake_monitoring_qa.csv # Multi-lake operational assessment
│   │
│   ├── figures/                    # High-resolution publication plots (PNG)
│   │   ├── 01_*.png                # Lake inventory distributions & maps
│   │   ├── 06_*.png to 09_*.png    # Model residual diagnostics & ablation curves
│   │   ├── 10_*.png                # NRT monitoring dashboards
│   │   ├── 10_1_*.png              # SAR/Optical threshold sensitivity & masks
│   │   └── 10_2_five_lake_extent_qa.png # Five-lake multi-sensor comparison
│   │
│   └── models/                     # Serialized production model artifacts
│       ├── 08_best_model_S1.pkl    # Standalone S1 XGBoost model
│       ├── 08_best_model_S2.pkl    # Standalone S2 XGBoost model
│       └── 09_final_fused_xgboost.pkl # Validated cross-sensor fused XGBoost
│
├── scratch/                        # Execution runners, audit scripts & test suites
│   ├── audit_secrets.py            # Static analysis secret scanner
│   ├── audit_large_files.py        # Workspace file size auditor
│   ├── setup_cdse_credentials.py   # Multi-tier CDSE credential resolver & tester
│   ├── test_cdse_oauth.py          # OAuth and STAC catalog verification
│   ├── test_processing_api.py      # Sentinel Hub Processing API verification
│   ├── execute_nb*.py              # Headless notebook execution harnesses
│   └── generate_all_reports_and_figures.py # Full report generation suite
│
└── raw/                            # Local raw datasets (GIT-IGNORED)
    ├── GLO/                        # GLO v1.02 GeoPackages (S1/S2 annual & unique)
    ├── ERA5/                       # ERA5-Land monthly reanalysis netCDF/CSV
    ├── CHIRPS/                     # CHIRPS monthly precipitation netCDF/CSV
    └── RGI/                        # Randolph Glacier Inventory 6.0/7.0 shapefiles/CSVs
```

---

## 4. Research Notebook Roadmap

| Notebook | Title | Core Objective | Status | Key Artifacts |
|---|---|---|---|---|
| **NB01** | Dataset Inspection | Schema inspection, CRS verification, and spatial bounds of GLO v1.02 | Completed | Baseline geometry verification |
| **NB02** | Databricks EDA | Distributed PySpark profiling, null distributions, lake size histograms | Completed | Multi-basin area distributions |
| **NB03** | Feature Engineering | Constructing temporal lags ($t-1$), ratios, expansion rates, missing-year flags | Completed | Feature matrix specification |
| **NB04** | Baseline ML | Linear regression, Ridge, Random Forest baselines on temporal splits | Completed | Baseline MAE benchmarks |
| **NB05** | Target Uncertainty | Noise floor characterization, area-dependent measurement errors | Completed | Residual variance vs. lake size |
| **NB06** | Model Development | Hyperparameter tuning of LightGBM, CatBoost, XGBoost per sensor | Completed | `06_model_comparison.csv` |
| **NB07** | Environmental Forcing | Integration of ERA5-Land (temperature, runoff) and CHIRPS precipitation | Completed | Climate ablation analysis |
| **NB08** | Multisource ML | Fusing Sentinel-1 SAR and Sentinel-2 Optical features into joint models | Completed | Sensor fusion discovery |
| **NB09** | Final Model Validation | Confirmatory validation, frozen 2023 test set, bootstrap CIs, robustness | **Validated Baseline** | `09_final_fused_xgboost.pkl` |
| **NB10** | NRT Satellite Monitoring | CDSE OAuth, STAC catalog queries, Sentinel-1/2 Processing API retrieval | Completed | NRT surveillance pipeline |
| **NB10.1** | Imja Extent Calibration | Dual-sensor calibration, speckle filtering, threshold tuning on Imja Tsho | Completed | Extent calibration protocol |
| **NB10.2** | Five-Lake Scaling | Controlled evaluation across Imja, Thulagi, Tsho Rolpa, Lower Barun, Sabai | Completed | `10_2_five_lake_monitoring_qa.csv` |

---

## 5. Dataset Description

The project integrates multiple Earth Observation and geospatial datasets:

1. **GLO v1.02 (Glacial Lake Observatory):**  
   - Source: Rawlins et al. (2025), Zenodo (`10.5281/zenodo.17802333`).
   - Annual lake polygons across Nepal and transboundary catchments from 2017 to 2024.
   - Provides separate Sentinel-1 SAR and Sentinel-2 Optical inventories.
   - Equal-area projection: `ESRI:102025` (Asia North Albers Equal Area Conic).
2. **ERA5-Land Monthly Reanalysis (ECMWF / Copernicus CDS):**  
   - 0.1° spatial resolution (~9 km), monthly aggregates (2016–2023).
   - Variables: 2m temperature (`t2m`), total precipitation (`tp`), runoff (`ro`), snowmelt (`smlt`).
3. **CHIRPS v2.0 (Climate Hazards Center):**  
   - 0.05° high-resolution quasi-global monthly precipitation grids.
4. **Randolph Glacier Inventory (RGI 6.0 / RGI 7.0):**  
   - Glacier outlines and morphological attributes (Region 15 — South Asia East).
   - Used to compute lake-to-glacier terminus distances and connectivity indices.
5. **Operational Sentinel-1 SAR (CDSE / Copernicus):**  
   - Level-1 Ground Range Detected (GRD), C-band synthetic aperture radar (VV/VH polarizations).
   - All-weather day/night capability; insensitive to cloud cover.
6. **Operational Sentinel-2 Optical (CDSE / Copernicus):**  
   - Level-2A Bottom-Of-Atmosphere (BOA) reflectance.
   - 10m/20m spatial resolution with Scene Classification Layer (SCL) cloud/shadow masks.

> [!NOTE]
> **Literature Reference vs. GLO Vector Reference:**  
> Historical papers frequently cite lake surface areas from older single-date sensors (e.g., historical literature often lists Imja Tsho as ~1.32 km²). In this repository, all reference polygons are derived from the official GLO v1.02 vectorized annual survey (where Imja Tsho in 2023 is measured at **1.7646 km²**). Do not confuse older literature references with the standardized 2023 GLO ground truth.

---

## 6. Target Definition

The machine learning prediction target is defined as the **annual surface area change**:

$$\text{NEXT\_AREA\_CHANGE} = \text{AREA}(t+1) - \text{AREA}(t)$$

### Temporal Consistency Rules
- Observations are paired strictly across consecutive calendar years:
  $$\text{NEXT\_YEAR} - \text{AREA\_YEAR} == 1$$
- Samples with missing intermediate years (e.g., $t+2$ without an observation at $t+1$) are excluded to prevent multi-year rate contamination.
- The target is measured in square kilometers ($\text{km}^2$).

---

## 7. Temporal Evaluation

To eliminate data leakage, models are evaluated exclusively using a **strict temporal forward-split**:

| Split | Time Window | Role | Constraints |
|---|---|---|---|
| **TRAIN** | $\text{AREA\_YEAR} \le 2021$ | Feature scaling & Model fitting | 8,527 lake-year observations |
| **VALIDATION** | $\text{AREA\_YEAR} = 2022$ | Hyperparameter selection & Early stopping | 1,580 lake-year observations |
| **TEST** | $\text{AREA\_YEAR} = 2023$ | Final confirmatory benchmark | **1,742 lake-year observations (FROZEN)** |

The 2023 test set was strictly frozen throughout feature development and exploratory modeling. It was evaluated only during confirmatory validation in Notebook 09.

---

## 8. Current Best ML Model

The primary validated predictive model is the **Cross-Sensor Fused XGBoost Regressor** (`reports/models/09_final_fused_xgboost.pkl`), which combines SAR backscatter morphology from Sentinel-1 with optical spectral features from Sentinel-2.

### Validated Matched-Test Benchmark (2023 Frozen Set)

| Metric | Validated Score | Scientific Interpretation |
|---|---|---|
| **MAE** | **0.004792 km²** | Typical error is under $4,800\text{ m}^2$ (~less than half a Sentinel-2 pixel on lake diameter) |
| **RMSE** | **0.015079 km²** | Robust penalization against large outlier expansions |
| **$R^2$** | **0.3366** | Explains 33.7% of annual expansion variance across diverse Himalayan lake types |

> [!IMPORTANT]
> **Statistical Clarification:**  
> In accordance with rigorous remote sensing standards, **$R^2$ must NOT be referred to as "accuracy"**. $R^2$ represents the coefficient of determination (explained variance). Because natural glacial lake change includes stochastic calving and moraine dam slumping, an $R^2$ of 0.337 with an MAE of 0.0048 km² represents a robust predictive baseline. Notebook 09 confirmed statistical significance via 1,000-iteration bootstrap confidence intervals: MAE 95% CI $[0.0042, 0.0054]\text{ km}^2$.

---

## 9. Near-Real-Time Monitoring

The operational surveillance layer (Notebooks 10, 10.1, and 10.2) interfaces with the Copernicus Data Space Ecosystem to track priority glacial lakes:

### Operational Processing Protocol
1. **OAuth 2.0 Client Credentials Authentication:**  
   Authenticates against the Copernicus Keycloak endpoint (`identity.dataspace.copernicus.eu`) with automatic fallback to Sentinel Hub OAuth.
2. **Spatiotemporal STAC Query:**  
   Searches the CDSE STAC Catalog for `sentinel-1-grd` and `sentinel-2-l2a` collections over targeted AOIs.
3. **Optical Extraction (Sentinel-2):**  
   - Cloud, cloud-shadow, and cirrus screening using the Scene Classification Layer (SCL).
   - Modified Normalized Difference Water Index:
     $$\text{MNDWI} = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}} = \frac{B03 - B11}{B03 + B11}$$
   - Water threshold calibrated to $\text{MNDWI} > 0.0$.
4. **Radar Extraction (Sentinel-1):**  
   - C-band SAR Level-1 GRD with thermal and border noise removal.
   - 5x5 Lee Speckle Filtering to smooth wind roughening and speckle noise.
   - Calibrated backscatter threshold: $\sigma^0_{\text{VV}} < -12\text{ dB}$.
5. **Spatial Geometry Prior:**  
   Extracts water pixels within a calibrated buffer around the known GLO lake boundary to prevent false positives from surrounding shadowed valleys or proglacial streams.

---

## 10. Local Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- Python `3.12.*`
- [`uv`](https://docs.astral.sh/uv/) (recommended high-performance package manager) or standard `pip`
- Git

### Installation Steps (Windows PowerShell)

```powershell
# 1. Clone the existing repository
git clone https://github.com/surya1290000-tech/GlacierGuard-AI.git
cd GlacierGuard-AI

# 2. Synchronize virtual environment with uv
uv sync

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### Verification
Run the verification suite to ensure dependencies and scripts compile cleanly:

```powershell
python -c "import geopandas, shapely, sklearn, xgboost; print('Environment OK!')"
python scratch/audit_secrets.py
```

---

## 11. Databricks Setup

GlacierGuard-AI supports distributed data processing via **Databricks Connect** and Databricks Asset Bundles:

1. **Bundle Definition:** Defined in `databricks.yml` targeting the workspace host `https://dbc-0be231d6-04fb.cloud.databricks.com`.
2. **Local Profile:** Databricks CLI authentication should be configured locally using:
   ```powershell
   databricks auth login --host https://<your-databricks-instance>.cloud.databricks.com
   ```
3. **Execution Separation:** Heavy exploratory spatial joins and catalog profiling can execute on Databricks clusters via PySpark (`notebooks/02_glo_databricks_eda.ipynb`), while ML modeling and NRT satellite processing execute locally.
4. **Security Notice:** Databricks authentication tokens are stored in the local profile (`~/.databrickscfg`) and must **NEVER** be committed to Git.

---

## 12. API Credential Setup

GlacierGuard-AI requires credentials for Copernicus CDS (ERA5) and Copernicus Data Space (NRT Satellite Imagery). All credentials reside outside the Git repository.

### 1. CDSE / Sentinel Hub OAuth (for NRT Imagery)
Register an account at [Copernicus Data Space](https://dataspace.copernicus.eu/). Create an OAuth client under your dashboard.

Save credentials in your user home directory at `~/.glacierguard/cdse_credentials.json`:
```json
{
  "client_id": "your-cdse-client-id",
  "client_secret": "your-cdse-client-secret"
}
```
Alternatively, set them in your terminal session:
```powershell
$env:CDSE_CLIENT_ID="your-client-id"
$env:CDSE_CLIENT_SECRET="your-client-secret"
```
Verify your configuration using the helper tool:
```powershell
python scratch/setup_cdse_credentials.py
```

### 2. Climate Data Store (CDS API for ERA5-Land)
Create an account at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). Save your key in `~/.cdsapirc`:
```ini
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key
```

> [!WARNING]
> **Never commit real credentials.** The repository `.gitignore` strictly blocks all `.env` files, `.cdsapirc`, and `.glacierguard/` directories.

---

## 13. How to Run

### Execution Sequence

To replicate or build upon the pipeline:

```
RESEARCH PIPELINE:
  NB01 (Inspection) ➔ NB02 (Databricks EDA) ➔ NB03 (Features) ➔ NB04 (Baselines)
         ➔ NB05 (Uncertainty) ➔ NB06 (Model Dev) ➔ NB07 (Environmental)
         ➔ NB08 (Sensor Fusion) ➔ NB09 (Validation Baseline)

OPERATIONAL SURVEILLANCE PIPELINE:
  NB10 (NRT Architecture) ➔ NB10.1 (Calibration) ➔ NB10.2 (Multi-Lake QA)
```

> [!TIP]
> **Collaborator Note:**  
> All notebooks, reports, figures, and model weights are already committed and validated in this repository. You do **NOT** need to rerun expensive historical stages (NB01–NB09) unless you are explicitly modifying feature engineering or training architectures.

---

## 14. Outputs

Key artifacts produced by the platform:

- **Reports (`reports/*.csv`):** 38+ quantitative assessment tables detailing model ablations, size-stratified error metrics, paired sensor comparisons, and bootstrap confidence intervals.
- **Figures (`reports/figures/*.png`):** 66+ analytical plots including residual distributions, feature importance rankings, and threshold sensitivity curves.
- **Model Checkpoints (`reports/models/*.pkl`):** Serialized scikit-learn and XGBoost pipelines for immediate inference.
- **Operational Dashboards (`reports/figures/10_*.png`):** Multi-sensor time-series plots and visual water masks for monitored glacial lakes.

---

## 15. Limitations

Collaborating developers and researchers should remain aware of known physical and methodological limitations:

1. **Annual Prediction vs. Instantaneous Events:** The machine learning model forecasts annual net surface area change. It does not predict the exact day or hour of moraine dam failure.
2. **Satellite Pass Cadence:** Near-real-time monitoring is governed by orbital pass schedules (Sentinel-1: 6–12 day repeat; Sentinel-2: 5 day repeat). Cloud cover during monsoon months significantly limits optical acquisitions.
3. **SAR Mountain Geometry Distortions:** Radar backscatter can experience foreshortening, layover, and shadowing in steep Himalayan valleys (observed at Tsho Rolpa).
4. **Wind-Roughened Water Surfaces:** High winds induce capillary waves on lake surfaces, increasing SAR backscatter and causing localized under-segmentation if not filtered.
5. **Calibrated Prior Sensitivity:** Water extraction currently relies on spatial buffering around known lake extents. Lakes undergoing unprecedented massive expansion require dynamic adaptive buffering.

---

## 16. Two-Developer Collaboration

To ensure clean, non-conflicting collaboration between two developers on this existing repository, follow this standardized branching and integration workflow.

### Git Branching Model

```
main (protected baseline)
  │
  ├── feature/developer-a-monitoring
  └── feature/developer-b-climate-features
```

- **`main`**: Represents the stable, validated project baseline. Direct commits to `main` should be minimized.
- **`feature/<task-name>`**: Dedicated branch for new experiments, notebook extensions, or script additions.
- **`fix/<issue-name>`**: Dedicated branch for bug fixes.

### Step-by-Step Workflow

#### 1. Before Starting Work (Sync with Remote)
Always pull latest changes into your local branch before editing:
```powershell
git checkout main
git fetch origin
git pull --ff-only origin main
```

#### 2. Create a Feature Branch
```powershell
# Developer A:
git checkout -b feature/nrt-scaling

# Developer B:
git checkout -b feature/era5-runoff-tuning
```

#### 3. Commit Work Incrementally
```powershell
git status
git add notebooks/ reports/ scratch/
git commit -m "feat: implement adaptive thresholding for high-altitude SAR"
```

#### 4. Push Feature Branch
```powershell
git push -u origin feature/<task-name>
```

#### 5. Merge into `main` (Preserving History)
When a feature is verified and ready for integration:
```powershell
# Switch to main and update
git checkout main
git pull --ff-only origin main

# Merge feature branch with a merge commit (no fast-forward)
git merge --no-ff feature/<task-name>

# Push updated main to remote
git push origin main
```

#### 6. Resolving Merge Conflicts
If both developers touched the same file:
1. Inspect conflicting files: `git status`
2. Open the file, locate the `<<<<<<<`, `=======`, `>>>>>>>` markers, and reconcile.
3. Mark resolved: `git add <file>`
4. Finalize merge: `git commit`
5. **NEVER force-push (`git push --force`) to shared branches.**

---

*GlacierGuard-AI is developed for remote sensing research and environmental risk assessment in High Mountain Asia.*
