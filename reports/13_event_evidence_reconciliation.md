# GlacierGuard-AI — Event Evidence Reconciliation & Spatial Audit Hardening
**Document ID:** GG-RPT-13-RECON
**Version:** 1.2.0 (Phase 9.1.3.2 Security Scanner Coverage & Scientific Provenance Closure)
**Date:** 2026-11-01
**Lead Geospatial ML Engineer:** Lead Geospatial ML Engineer
**Branch:** `feature/phase9-dual-ensemble`

---

## 1. Executive Summary & Purpose

Phase 9.1.3.2 establishes rigorous security scanning coverage, completes scientific provenance closure, resolves elevation ambiguities, and separates distinct research preprints across the registered 2026 Nepal cryospheric events:
1. **26 August 2026 Langtang Lirung / Lhende Khola Rock–Ice Avalanche**
2. **11 October 2026 Budhi Gandaki River Surge / Cascading Flood (near Chumchet, Gorkha)**

### Core Scientific Commitments Preserved
- **Strictly Audit & Reconciliation:** Zero ML model training performed. No labels generated.
- **Model Isolation:** Frozen XGBoost artifact (`reports/models/09_final_fused_xgboost.pkl`, SHA-256: `5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f`) preserved byte-for-byte.
- **GLO Dataset Integrity:** Neither event is integrated into the GLO 2017–2024 annual lake area training dataset.
- **PR #2 Preserved:** PR #2 remains open, untouched, and unmerged.
- **Security Remediation:** Current working tree verified with zero detected secret matches across all tracked text files (including notebooks, configuration files, and code). No Git history rewrite was executed.

---

## 2. Reconciling the 11 October 2026 Budhi Gandaki Event

### 2.1 Evidence Evolution Timeline

```text
2026-10-11: Initial Flood Wave Observed near Chumchet
    │
    ▼ (ICIMOD Initial Press Release, 11 Oct 2026)
Provisional Interpretation: Temporary Valley Landslide Dam Breach (Superseded)
    │
    ▼ (ICIMOD Formal Briefing, 13 Oct 2026)
Updated Finding: High-Mountain Rock Avalanche & Glacier Detachment Confirmed
    │
    ▼ (Nepal Red Cross Society Sitrep 3 & 4, 17 & 29 Oct 2026)
Field Geomorphic Mapping: Ice-and-debris avalanche in upper catchment;
debris accumulation at Ungdilung / Syar confluence causing
temporary river damming and multi-pulse outburst releases
    │
    ▼ (EarthArXiv Preprint 15540, 25 Oct 2026)
Preprint Remote Sensing: No detectable pre-collapse SAR backscatter precursor
(Study-specific negative finding; unreviewed preprint)
    │
    ▼ (HGR Secondary Satellite Analysis, 20 Oct 2026)
Candidate Source Region: Southern slopes of Shringri Himal upstream of Udilung Khola
```

### 2.2 Reconciled Event Classification & Cascade Sequence
- **Primary Hazard:** Reconciled from `LANDSLIDE_DAM_BREACH` to **`ROCK_ICE_AVALANCHE`** based on multi-agency agreement (ICIMOD 13 Oct, Red Cross Sitrep 4).
- **Cascading Stages:**
  1. `ROCK_ICE_AVALANCHE` (Primary detachment in high-relief headwalls)
  2. `DEBRIS_FLOW` (Channelized sediment and boulder transport down Udilung/Ungdilung gorge; not classified as hyperconcentrated absent rheological measurement)
  3. `TEMPORARY_RIVER_BLOCKAGE` (Debris damming at Syar/Shiar confluence)
  4. `DOWNSTREAM_FLOOD` (Surge pulses down Budhi Gandaki mainstem)
- **Temporary River Blockage vs. Landslide Dam:** The river obstruction was caused by tributary debris-flow accumulation at a stream junction, not a classic valley wall rotational landslide dam. It is classified under the distinct process type `TEMPORARY_RIVER_BLOCKAGE`.
- **Preservation of History:** The initial 11 October ICIMOD interpretation is preserved in the event catalogue revision history with confidence tag `PRELIMINARY_SUPERSEDED`.

### 2.3 Strict Separation of Source vs. Reach Locations
- **Source Region:** High-relief slopes near Shringri Himal upstream of Udilung Khola. **Coordinates set to null/unresolved.** No exact point coordinate is fabricated from secondary text.
- **Affected Reach:** Chumchet village / confluence reach (`28.457° N, 84.975° E`, ±2,500 m precision). Retained strictly as an impact-reach reference.
- **Blockage Location:** Ungdilung / Syar stream confluence (approximate confluence reach).

---

## 3. Langtang Lirung Spatial Uncertainty & Elevation Provenance

### 3.1 Point-in-Polygon vs. Coordinate Precision Reality
- **Reported Source Point:** 28.288708° N, 85.528159° E
- **Nearest RGI 6.0 Glacier (RGI60-15.04173):** Separated by **42.4 m**
- **Reported Coordinate Precision:** **150.0 m**
- **Uncertainty Semantics:** **UNKNOWN** (the source publication does not specify whether 150 m represents circular error probable, standard deviation, bounding-box precision, or grid quantization).

$$\text{Distance to Polygon } (42.4\,\text{m}) < \text{Coordinate Precision Metric } (150.0\,\text{m})$$

> [!IMPORTANT]
> **Scientific Uncertainty Stance:**
> 1. The raw `point_in_polygon == False` test confirms the nominal point falls 42.4 m outside RGI 6.0 polygon `RGI60-15.04173`.
> 2. **Hypothetical Scenario:** If the 150 m precision is hypothetically modeled as an isotropic circular error radius, the error circle intersects glacierized terrain.
> 3. However, because coordinate uncertainty semantics are **UNKNOWN**, treating 150 m as a confirmed isotropic buffer is scientifically unverified.
> 4. Therefore, available evidence **neither proves nor disproves** that the detachment initiated inside glacierized terrain. The spatial status is classified as **`SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION`** with glacier containment status **`UNRESOLVED`**.

### 3.2 Glacier Inventory Metadata Provenance
- **RGI 6.0:** Randolph Glacier Inventory 6.0 (Region 15 South Asia East) based on Landsat composites from **2000–2010**. These historical outlines do not reflect instantaneous 2026 ice boundaries or high-altitude hanging glacier retreat.
- **RGI 7.0 Comparison:** Tabular attributes in `raw/RGI/rgi7_region15_glaciers.csv` list nearby glacier `RGI2000-v7.0-G-15-05732` (centroid 1,461 m south at 28.2780° N, 85.5368° E, area 9.842 km²). Because official RGI 7.0 vector polygon shapefiles for Region 15 are pending in the local workspace, spatial testing remains reproducibly anchored on RGI 6.0 vector geometry.

### 3.3 Elevation Provenance & Unresolved Statuses
A citation is not considered traceable merely because a report URL exists; specific tabular, cartographic, or altimetric references are required. Conflicting elevation claims are deconstructed below:

| Elevation Segment | Value / Range | Source & Evidence Context | Scientific Provenance Status |
| :--- | :--- | :--- | :--- |
| **Source Coordinate (Point)** | ~6,200 m | Historical catalogue nominal approximation on NW headwall face; unverified by in-situ survey or DEM differencing. | `NOMINAL_CATALOGUE_UNVERIFIED` |
| **Detachment Scar (Working Note)** | 5,500–6,800 m | Preliminary working notes; lacks specific page, figure, or table attribution in published ICIMOD report. | `UNVERIFIED_PENDING_PAGE_CITATION` |
| **Detachment Slope (ICIMOD Briefing)** | 5,000–5,500 m | [ICIMOD (16 Sept 2026)](https://www.icimod.org/news/learning-from-disaster-icimod-convenes-international-experts-following-rasuwa-gyirong-flood/) public event briefing describing north-face detachment slope. | `PUBLIC_EVENT_SUMMARY_ESTIMATE` |
| **Lower Runout / Breakline** | 5,000–5,200 m | Early situational media reports describing the lower plunge throat into the gorge; exact primary source unidentified. | `PROVISIONAL_UNVERIFIED` |
| **Valley Floor Deposition** | ~3,800 m | Topographic map estimation of Lhende Khola upper gorge floor; unverified by DEM difference mapping. | `APPROXIMATE_UNVERIFIED` |

- Conflicting figures (5,000–5,500 m vs 5,500–6,800 m vs 6,200 m) are explicitly preserved rather than silently harmonized.
- Precise 3D detachment geometry and volume profiling remain **`UNRESOLVED`** pending post-event stereoscopic DEM differencing.

---

## 4. Distinct Sentinel-1 Research Preprints (Sah 2026)

Two distinct preprints authored by Ajay Kumar Sah address separate aspects of Sentinel-1 radar capabilities for the 26 August 2026 Langtang Lirung event. These studies are separate records with different DOIs and publication dates:

### 4.1 Amplitude Precursor Analysis (DOI `10.31223/X59N5N`)
- **Title:** *No detectable precursor in Sentinel-1 backscatter before the 26 August 2026 Langtang Lirung rock–ice avalanche, Nepal*
- **DOI / URL:** [https://doi.org/10.31223/X59N5N](https://doi.org/10.31223/X59N5N)
- **Publication Date:** **6 September 2026** (updated **8 September 2026**)
- **Category:** `PREPRINT` (Status: `PREPRINT_UNREVIEWED`)
- **Core Claim:** Systematic analysis of Sentinel-1 Level-1 GRD radar backscatter amplitude ($\sigma^0$) at 12-day intervals showed no anomalous precursory deviation prior to collapse. Coherent sub-centimeter creep does not alter macroscopic surface roughness.
- **Guardrail:** Phase-unwrapping findings are **not** attributed to this DOI.

### 4.2 Interferometric Unwrapping Limits (DOI `10.31223/X5GZ3J`)
- **Indexed Title:** *The single-pair unwrapping ceiling is a property of the track, not of the radar: Sentinel-1 evidence from rock–ice avalanches at Langtang Lirung and Ronti Peak*
- **Author:** Ajay Kumar Sah
- **Indexed Study Findings:** Demonstrates across ascending and descending tracks over Langtang Lirung (2026) and Ronti Peak (Chamoli) that single-pair C-band InSAR phase unwrapping ceilings and geometric decorrelation are governed by track acquisition geometry (look angle, heading, local incidence, and slope aspect) rather than radar frequency in isolation. Phase aliasing occurs when displacement exceeds $\sim \lambda/4$ per pixel, preventing 2D phase unwrapping in steep relief (>45°).
- **Operational Model Interpretation:** Single-pair C-band InSAR cannot operate as an autonomous standalone early-warning trigger for steep hanging glacier detachments without multi-track ascending/descending combinations, multi-temporal InSAR (MTI) persistence scatterers, or optical offset tracking.
- **DOI / URL:** [https://doi.org/10.31223/X5GZ3J](https://doi.org/10.31223/X5GZ3J)
- **Publication Date:** **18 September 2026**
- **Category:** `PREPRINT` (Status: `PREPRINT_UNREVIEWED`)
- **Core Claim:** High-relief steep mountain slopes impose severe phase unwrapping ceilings on 12-day C-band interferograms when displacement gradients exceed $\lambda / (2 \cdot \Delta x)$. Phase aliasing prevents standard 2D unwrappers from resolving rapid pre-collapse acceleration.

---

## 5. Comprehensive Evidence Provenance Table

| Category | Organization / Author | Publication Date | URL / DOI | Key Claim Supported | Scientific Status & Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Official Report** | **ICIMOD** | 2026-10-13 | [lib.icimod.org/records/qdvc1-tc123](https://lib.icimod.org/records/qdvc1-tc123) | Pre-collapse InSAR LOS displacement up to 40 cm between 7 July and 24 Aug 2026. | `HIGH (Confirmed Official Report)` |
| **Official Report** | **ICIMOD** | 2026-10-11 | [icimod.org/.../landslide-dam-breach...](https://www.icimod.org/press-releases/landslide-dam-breach-may-have-triggered-budhi-gandaki-flash-flood/) | Initial flood surge observation; hypothesized landslide dam. | `PRELIMINARY_SUPERSEDED` |
| **Official Report** | **ICIMOD** | 2026-10-13 | [icimod.org/.../second-major-himalayan-flood...](https://www.icimod.org/press-releases/second-major-himalayan-flood-in-seven-weeks-raises-new-questions-about-mountain-risk/) | Confirmed high-mountain rock avalanche and glacier collapse trigger for Budhi Gandaki. | `HIGH (Confirmed Official Report)` |
| **Field Report** | **Nepal Red Cross** | 2026-10-17 | [nrcs.org/.../gorkha-sitrep3/](https://nrcs.org/resources/news-and-events/gorkha-sitrep3/) | Downstream flood corridor impact, infrastructure damage in Gorkha. | `HIGH (Field Humanitarian)` |
| **Field Report** | **Nepal Red Cross** | 2026-10-29 | [nrcs.org/.../gorkha-floodresponse-sitrep4/](https://nrcs.org/resources/news-and-events/gorkha-floodresponse-sitrep4/) | Upper catchment ice-debris avalanche; debris damming at Ungdilung/Syar confluence; multiple surge pulses. | `HIGH (Field Geomorphic)` |
| **Preprint** | **A. K. Sah (EarthArXiv)** | 2026-09-06 *(upd 2026-09-08)* | [doi.org/10.31223/X59N5N](https://doi.org/10.31223/X59N5N) | GRD amplitude backscatter shows no detectable amplitude precursor prior to Langtang collapse. | `PREPRINT_UNREVIEWED (Study-Specific)` |
| **Preprint** | **A. K. Sah (EarthArXiv)** | 2026-09-18 | [doi.org/10.31223/X5GZ3J](https://doi.org/10.31223/X5GZ3J) | Track-dependent Sentinel-1 interferometric unwrapping ceiling and detection limits in steep terrain. | `PREPRINT_UNREVIEWED (Study-Specific)` |
| **Preprint** | **EarthArXiv 15540** | 2026-10-25 | [eartharxiv.org/repository/view/15540/](https://eartharxiv.org/repository/view/15540/) | Study-specific SAR amplitude backscatter analysis reporting no detectable pre-collapse precursor. | `PREPRINT_UNREVIEWED (Study-Specific)` |
| **Secondary Research** | **HGR (N. Parajuli)** | 2026-10-20 | [parajulinawaraj.com.np/.../budi-gandaki-2026/](https://www.parajulinawaraj.com.np/HGR/events/budi-gandaki-2026/index.html) | Candidate detachment zone on Shringri Himal flank via 6 Oct vs 13 Oct optical comparisons. | `CANDIDATE_SECONDARY (Unresolved Geometry)` |

---

## 6. Current-Tree vs. Git-History Secret Audit Findings

### 6.1 Current Working Tree Scan
- **Methodology:** Content-based binary detection (`b'\x00'` null-byte evaluation) on all tracked files from `git ls-files`.
- **Scan Coverage:**
  - Total Tracked Files: **287**
  - Text Files Scanned: **205** (including 14 Jupyter notebooks, all Python code, Markdown reports, JSON catalogues, CSVs, YAML, TOML, and shell scripts)
  - Binary Files Safely Skipped: **80** (images, model pickle files, shapefiles, archives)
  - Allowlisted Files: **2** (`.env.example` and `README.md`, containing explicit documentation placeholder templates)
- **Current Tree Result:** **Zero detected matches.**

### 6.2 Git History Audit
- **Findings:** Historical commit `1187161` added literal secret fixtures to `tests/test_phase9_1_3_event_reconciliation.py`. These were eliminated in commit `cddef45`.
- **Policy Enforcement:** Git history was not rewritten in this task.
- **Mandatory Credential Rotation Advisory:** Any credentials committed to Git history in commit `1187161` must be independently rotated and revoked in the respective service provider accounts (Copernicus CDSE and CDS portals).
