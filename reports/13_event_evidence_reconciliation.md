# GlacierGuard-AI — Event Evidence Reconciliation & Spatial Audit Hardening
**Document ID:** GG-RPT-13-RECON
**Version:** 1.1.0 (Phase 9.1.3.1 Security & Scientific Corrections)
**Date:** 2026-11-01
**Lead Geospatial ML Engineer:** Lead Geospatial ML Engineer
**Branch:** `feature/phase9-dual-ensemble`

---

## 1. Executive Summary & Purpose

Phase 9.1.3.1 establishes scientific defensibility, strict spatial uncertainty semantics, elevation provenance, and security remediation across the registered 2026 Nepal cryospheric events:
1. **26 August 2026 Langtang Lirung / Lhende Khola Rock–Ice Avalanche**
2. **11 October 2026 Budhi Gandaki River Surge / Cascading Flood (near Chumchet, Gorkha)**

### Core Scientific Commitments Preserved
- **Strictly Audit & Reconciliation:** Zero ML model training performed. No labels generated.
- **Model Isolation:** Frozen XGBoost artifact (`reports/models/09_final_fused_xgboost.pkl`, SHA-256: `5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f`) preserved byte-for-byte.
- **GLO Dataset Integrity:** Neither event is integrated into the GLO 2017–2024 annual lake area training dataset.
- **PR #2 Preserved:** PR #2 remains open, untouched, and unmerged.
- **Security Remediation:** Zero literal credentials stored in test code or fixtures. Generic repository-wide secret scanning enforced.

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
- **RGI 7.0 Comparison:** Tabular attributes in `raw/RGI/rgi7_region15_glaciers.csv` list nearby glacier `RGI2000-v7.0-G-15-05732` (centroid at 28.2780° N, 85.5368° E, area 9.842 km², 1,461 m south). Official RGI 7.0 vector polygon shapefiles for Region 15 are pending in the local workspace; reproducible spatial testing remains anchored on RGI 6.0 vector geometry.

### 3.3 Elevation Provenance & Reconciliation
Every elevation value is explicitly partitioned and cross-referenced against traceable sources:

| Elevation Segment | Value / Range | Traceable Source & Citation | Status |
| :--- | :--- | :--- | :--- |
| **Source Coordinate (Point)** | ~6,200 m | Nominal catalogue point estimate for NW face detachment scarp | `NOMINAL_ESTIMATE` |
| **Source Detachment Scar** | 5,500–6,800 m | ICIMOD (13 Oct 2026), *Understanding the Gyirong–Rasuwa disaster*, [lib.icimod.org/records/qdvc1-tc123](https://lib.icimod.org/records/qdvc1-tc123) | `TRACEABLE_OFFICIAL_REPORT` |
| **Lower Runout / Breakline** | 5,000–5,200 m | Preliminary news bulletins and situational reports describing plunge throat into gorge | `HISTORICAL_CLAIM_PROVISIONAL` |
| **Valley Floor Deposition** | ~3,800 m | Topographic map estimation of Lhende Khola upper gorge floor | `APPROXIMATE_RUNOUT_FLOOR` |

---

## 4. Satellite Catalogue Product vs. Acquisition Normalization

### 4.1 Langtang Lirung Satellite Normalization
- **Total Product Records in Catalogue:** **92 products**
  - Sentinel-1 SLC: 23 records
  - Sentinel-1 GRD: 23 records
  - Sentinel-2 L2A: 46 records
- **Total Independent Physical Overpass Acquisitions:** **68 acquisitions**
  - **Sentinel-1:** Exactly **23 physical acquisitions** across Orbits 19, 85, and 121. Every S1 SLC has a concurrent S1 GRD representing the identical physical overpass at different processing levels.
  - **Sentinel-2:** Exactly **45 physical acquisitions** (44 single-granule overpasses + 1 dual-granule overpass on 2026-08-14, both tiles preserved under `ACQ_S2B_20260814_0446_ORB076`).
- **Deliverables:**
  - `reports/13_satellite_scene_availability.csv`: Product-level records with stable `acquisition_group_id`.
  - `reports/13_acquisition_group_summary.csv`: Acquisition-level summary table.

### 4.2 Budhi Gandaki Satellite Scene Inventory
- **Separate AOI Bounding Box:** `[84.78°E, 28.02°N, 85.08°E, 28.58°N]` (`BUDHI_GANDAKI_BASIN`, completely separate from `LANGTANG_LIRUNG_AOI`).
- **Total Product Records Discovered:** **50 products**
  - Pre-Event (1 Sep – 10 Oct 2026): 6 S1 SLC, 6 S1 GRD, 20 S2 L2A (32 products)
  - Event Day (11 Oct 2026): 1 S1 SLC, 1 S1 GRD, 2 S2 L2A (4 products; S2 cloud cover 15.5% and 20.4%)
  - Post-Event (12 Oct – 31 Oct 2026): 2 S1 SLC, 2 S1 GRD, 10 S2 L2A (14 products)
- **Deliverable:** `reports/13_budhi_gandaki_scene_availability.csv`.
- **Processing Status:** All records catalogued as `DISCOVERED_IN_CATALOGUE`.

---

## 5. Comprehensive Evidence Provenance Table

| Category | Organization / Author | Publication Date | URL | Key Claim Supported | Scientific Status & Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Official Report** | **ICIMOD** | 2026-10-13 | [lib.icimod.org/records/qdvc1-tc123](https://lib.icimod.org/records/qdvc1-tc123) | Pre-collapse InSAR LOS displacement up to 40 cm between 7 July and 24 Aug 2026; detachment scar on NW face spanning ~5,500–6,800 m. | `HIGH (Confirmed Official Report)` |
| **Official Report** | **ICIMOD** | 2026-10-11 | [icimod.org/.../landslide-dam-breach...](https://www.icimod.org/press-releases/landslide-dam-breach-may-have-triggered-budhi-gandaki-flash-flood/) | Initial flood surge observation; hypothesized landslide dam. | `PRELIMINARY_SUPERSEDED` |
| **Official Report** | **ICIMOD** | 2026-10-13 | [icimod.org/.../second-major-himalayan-flood...](https://www.icimod.org/press-releases/second-major-himalayan-flood-in-seven-weeks-raises-new-questions-about-mountain-risk/) | Confirmed high-mountain rock avalanche and glacier collapse trigger for Budhi Gandaki. | `HIGH (Confirmed Official Report)` |
| **Field Report** | **Nepal Red Cross** | 2026-10-17 | [nrcs.org/.../gorkha-sitrep3/](https://nrcs.org/resources/news-and-events/gorkha-sitrep3/) | Downstream flood corridor impact, infrastructure damage in Gorkha. | `HIGH (Field Humanitarian)` |
| **Field Report** | **Nepal Red Cross** | 2026-10-29 | [nrcs.org/.../gorkha-floodresponse-sitrep4/](https://nrcs.org/resources/news-and-events/gorkha-floodresponse-sitrep4/) | Upper catchment ice-debris avalanche; debris damming at Ungdilung/Syar confluence; multiple surge pulses. | `HIGH (Field Geomorphic)` |
| **Preprint** | **A. K. Sah (EarthArXiv)** | 2026-09-06 *(upd 2026-09-08)* | [doi.org/10.31223/x59n5n](https://doi.org/10.31223/x59n5n) | Reconciles GRD amplitude vs InSAR phase; documents C-band phase unwrapping limits in steep relief. | `PREPRINT_UNREVIEWED (Study-Specific)` |
| **Preprint** | **EarthArXiv 15540** | 2026-10-25 | [eartharxiv.org/repository/view/15540/](https://eartharxiv.org/repository/view/15540/) | Study-specific SAR amplitude backscatter analysis reporting no detectable pre-collapse precursor. | `PREPRINT_UNREVIEWED (Study-Specific)` |
| **Secondary Research** | **HGR (N. Parajuli)** | 2026-10-20 | [parajulinawaraj.com.np/.../budi-gandaki-2026/](https://www.parajulinawaraj.com.np/HGR/events/budi-gandaki-2026/index.html) | Candidate detachment zone on Shringri Himal flank via 6 Oct vs 13 Oct optical comparisons. | `CANDIDATE_SECONDARY (Unresolved Geometry)` |
