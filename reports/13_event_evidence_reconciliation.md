# GlacierGuard-AI — Event Evidence Reconciliation & Spatial Audit Hardening
**Document ID:** GG-RPT-13-RECON  
**Date:** 2026-11-01  
**Lead Geospatial ML Engineer:** Lead Geospatial ML Engineer  
**Branch:** `feature/phase9-dual-ensemble`

---

## 1. Executive Summary & Purpose

Phase 9.1.3 resolves scientific discrepancies and hardens spatial uncertainty for the two registered 2026 cryospheric events:
1. **26 August 2026 Langtang Lirung / Lhende Khola Rock–Ice Avalanche**
2. **11 October 2026 Budhi Gandaki River Surge / Cascading Flood (near Chumchet, Gorkha)**

### Core Commitments Preserved
- **Strictly Audit Only:** Zero ML model training performed. No labels generated.
- **Model Isolation:** Frozen XGBoost artifact (`reports/models/09_final_fused_xgboost.pkl`) preserved byte-for-byte.
- **GLO Dataset Integrity:** Neither event is integrated into the GLO 2017–2024 annual lake area training dataset.
- **PR #2 Preserved:** PR #2 remains open, untouched, and unmerged.

---

## 2. Reconciling the 11 October 2026 Budhi Gandaki Event

### 2.1 Evidence Evolution Timeline

```text
2026-10-11: Initial Flood Wave Observed near Chumchet
    │
    ▼ (ICIMOD Initial Press Release, 11 Oct 2026)
Provisional Interpretation: Temporary Valley Landslide Dam Breach
    │
    ▼ (ICIMOD Formal Briefing, 13 Oct 2026)
Updated Finding: High-Mountain Rock Avalanche & Glacier Detachment Confirmed
    │
    ▼ (Nepal Red Cross Society Sitrep 3 & 4, 17 & 29 Oct 2026)
Field Geomorphic Mapping: Ice-and-debris avalanche in upper catchment;
massive debris accumulation at Ungdilung / Syar confluence causing
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
  2. `DEBRIS_FLOW` (Hyperconcentrated transport down Udilung/Ungdilung gorge)
  3. `TEMPORARY_RIVER_BLOCKAGE` (Debris damming at Syar/Shiar confluence)
  4. `DOWNSTREAM_FLOOD` (Surge pulses down Budhi Gandaki mainstem)
- **Temporary River Blockage vs. Landslide Dam:** The river obstruction was caused by tributary debris-flow accumulation at a stream junction, not a classic valley wall rotational landslide dam. It is classified under the distinct process type `TEMPORARY_RIVER_BLOCKAGE`.
- **Preservation of History:** The initial 11 October ICIMOD interpretation is preserved in the event catalogue revision history.

### 2.3 Strict Separation of Source vs. Reach Locations
- **Source Region:** High-relief slopes near Shringri Himal upstream of Udilung Khola. **Coordinates set to null/unresolved.** No exact point coordinate is fabricated from secondary text.
- **Affected Reach:** Chumchet village / confluence reach (`28.457° N, 84.975° E`, ±2,500 m precision). Retained strictly as an impact-reach reference.
- **Blockage Location:** Ungdilung / Syar stream confluence (approximate confluence reach).

---

## 3. Langtang Lirung Spatial Uncertainty & Elevation Reconciliation

### 3.1 Point-in-Polygon vs. Coordinate Precision Reality
- **Reported Source Point:** 28.288708° N, 85.528159° E
- **Nearest RGI Glacier (RGI60-15.04173):** Separated by **42.4 m**
- **Reported Coordinate Precision:** **150.0 m**

$$\text{Distance to Polygon } (42.4\,\text{m}) < \text{Coordinate Uncertainty } (150.0\,\text{m})$$

> [!IMPORTANT]
> Because the separation distance (42.4 m) is substantially smaller than the spatial precision uncertainty (150 m), the raw `point_in_polygon == False` result **does not prove** that detachment occurred outside glaciated terrain.  
> Under a conservative 150 m buffer, the source coordinate overlaps RGI glacier ice.  
> The spatial status is formally updated to **`SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION`**.

### 3.2 RGI Inventory Metadata Provenance
- The glacier outline comes from **Randolph Glacier Inventory (RGI) 6.0** (Region 15 South Asia East).
- RGI 6.0 outlines are based primarily on Landsat imagery from the **2000–2010** era.
- **RGI outlines do NOT represent instantaneous 2026 ice boundaries.**

### 3.3 Elevation Discrepancy Reconciliation
- **Catalogue Elevation:** ~6,200 m
- **Alternative News/Reports:** 5,000–5,200 m
- **Physical Reconciliation:** The collapse headwall crown initiated high on the northwest face of Langtang Lirung between 5,800 m and 6,800 m (median scarp ~6,200 m). The lower detachment breakline and plunge throat into the Lhende Khola gorge occurred at 5,000–5,200 m, with deposition extending down to 3,800 m. Both values describe valid elevations along the continuous vertical mass-wasting profile.

---

## 4. Satellite Catalogue Product vs. Acquisition Normalization

### 4.1 Langtang Lirung Satellite Normalization
- **Total Product Records in Catalogue:** **92 products**
  - Sentinel-1 SLC: 23 records
  - Sentinel-1 GRD: 23 records
  - Sentinel-2 L2A: 46 records
- **Total Independent Physical Overpass Acquisitions:** **68 acquisitions**
  - **Sentinel-1:** Exactly **23 physical acquisitions** across Orbit 19, 85, and 121. Every S1 SLC has a concurrent S1 GRD representing the identical physical overpass at different processing levels.
  - **Sentinel-2:** Exactly **45 physical acquisitions** (44 single-granule overpasses + 1 dual-granule overpass on 2026-08-14).
- **Deliverables:**
  - `reports/13_satellite_scene_availability.csv`: Product-level records with stable `acquisition_group_id`.
  - `reports/13_acquisition_group_summary.csv`: Acquisition-level summary table.

### 4.2 Budhi Gandaki Satellite Scene Inventory
- **Separate AOI Bounding Box:** `[84.78°E, 28.02°N, 85.08°E, 28.58°N]` (Upper Catchment + Downstream Gorge Corridor)
- **Total Product Records Discovered:** **50 products**
  - Pre-Event (1 Sep – 10 Oct 2026): 6 S1 SLC, 6 S1 GRD, 20 S2 L2A (32 products)
  - Event Day (11 Oct 2026): 1 S1 SLC, 1 S1 GRD, 2 S2 L2A (4 products; S2 cloud cover 15.5% and 20.4%)
  - Post-Event (12 Oct – 31 Oct 2026): 2 S1 SLC, 2 S1 GRD, 10 S2 L2A (14 products)
- **Deliverable:** `reports/13_budhi_gandaki_scene_availability.csv`.

---

## 5. Comprehensive Evidence Provenance Table

| Organization | Publication Date | Evidence Type | URL | Key Claim Supported | Confidence & Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ICIMOD** | 2026-10-11 | OFFICIAL_REPORT | [icimod.org/press-releases/landslide-dam-breach...](https://www.icimod.org/press-releases/landslide-dam-breach-may-have-triggered-budhi-gandaki-flash-flood/) | Initial flood surge observation; hypothesized landslide dam. | PRELIMINARY (Superseded by 13 Oct finding) |
| **ICIMOD** | 2026-10-13 | OFFICIAL_REPORT | [icimod.org/press-releases/second-major-himalayan-flood...](https://www.icimod.org/press-releases/second-major-himalayan-flood-in-seven-weeks-raises-new-questions-about-mountain-risk/) | Confirmed high-mountain rock avalanche and glacier collapse trigger. | HIGH (Confirmed) |
| **Nepal Red Cross** | 2026-10-17 | FIELD_REPORT | [nrcs.org/.../gorkha-sitrep3/](https://nrcs.org/resources/news-and-events/gorkha-sitrep3/) | Downstream flood corridor impact, infrastructure damage in Gorkha. | HIGH (Confirmed) |
| **Nepal Red Cross** | 2026-10-29 | FIELD_REPORT | [nrcs.org/.../gorkha-floodresponse-sitrep4/](https://nrcs.org/resources/news-and-events/gorkha-floodresponse-sitrep4/) | Upper catchment ice-debris avalanche; debris damming at Ungdilung/Syar confluence; multiple burst pulses. | HIGH (Confirmed field geomorphic evidence) |
| **EarthArXiv** | 2026-10-25 | PREPRINT | [eartharxiv.org/repository/view/15540/](https://eartharxiv.org/repository/view/15540/) | Study-specific SAR amplitude backscatter analysis reporting no detectable pre-collapse precursor. | PREPRINT_UNREVIEWED (Study-specific result) |
| **HGR (N. Parajuli)**| 2026-10-20 | SECONDARY_RESEARCH| [parajulinawaraj.com.np/.../budi-gandaki-2026/](https://www.parajulinawaraj.com.np/HGR/events/budi-gandaki-2026/index.html) | Candidate detachment zone on Shringri Himal flank via 6 Oct vs 13 Oct optical comparisons. | CANDIDATE_SECONDARY (Unresolved geometry) |
| **ICIMOD (Langtang)**| 2026-10-26 | OFFICIAL_REPORT | [icimod.org/reports/langtang-lirung-2026](https://www.icimod.org) | Pre-collapse InSAR LOS displacement up to 40 cm between 7 July and 24 August 2026. | HIGH (Confirmed InSAR sequence) |
| **Sah (EarthArXiv)** | 2026-10-20 | PREPRINT | [doi.org/10.31223/x59n5n](https://doi.org/10.31223/x59n5n) | Reconciles GRD amplitude vs InSAR phase; documents C-band unwrapping limits in steep relief. | PREPRINT_RECONCILED |
