# GlacierGuard-AI — Multi-Hazard Cryospheric Monitoring Pipeline Design
**Document ID:** GG-DES-13-MULTIHZ
**Date:** 2026-11-01
**Lead ML Engineer & Geospatial Architect:** Lead ML Engineer
**Status:** PROPOSED RESEARCH SPECIFICATION (Audit Only — Training Prohibited)

---

## 1. Architectural Independence & Separation of Concerns

GlacierGuard-AI's existing machine learning stack is strictly partitioned into three independent tiers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GlacierGuard-AI Machine Learning Systems                              │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Annual Lake Area Regression (System A)                             │
│    Target: NEXT_AREA_CHANGE in km²                                     │
│    Dataset: 36-feature cross-sensor fused GLO lake polygons (N=11,849) │
│    Model: Fused XGBoost / Random Forest Regressor                      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Annual Lake Condition Classification (System B)                    │
│    Target: STABLE / EXPANDING / UNUSUAL_CHANGE                         │
│    Dataset: Uncertainty-aware hybrid classes on GLO lake polygons      │
│    Model: Cost-sensitive ensemble classifier                           │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Multi-Hazard Cryospheric Surveillance (System C - NEW WORKSTREAM)  │
│    Target: Mass movement susceptibility & valley impoundment risk      │
│    Unit of Analysis: Mountain slope patches & river corridor reaches   │
│    Data Sources: S1 InSAR/GRD, S2 MSI, Copernicus 30m DEM, Seismicity  │
└────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Zero Architectural Contamination:**
> Systems A and B predict annual lake area evolution on closed water polygons.
> System C operates on steep bedrock/ice flanks, hanging glaciers, and river valleys.
> **No training of System C may occur until event and control sample feasibility is established.**

---

## 2. Standardized Multi-Hazard Process Taxonomy

The cryospheric hazard catalogue establishes six mutually distinct process types:

1. **`ROCK_ICE_AVALANCHE`**: Sudden high-velocity detachment and free-fall/slide of rock and glacial ice masses from steep mountain headwalls and cirques.
2. **`GLACIAL_LAKE_OUTBURST` (GLOF)**: Rapid, uncontrolled drainage of a proglacial, supraglacial, or moraine-dammed lake due to dam collapse, piping, or displacement wave.
3. **`LANDSLIDE_DAM_BREACH`**: Catastrophic incision and failure of an impounded natural dam formed by valley wall slope collapse or rotational slide.
4. **`DEBRIS_FLOW`**: High-velocity saturated slurry of water, mud, gravel, boulders, and pulverized ice traversing tributary channels or gorges (not classified as hyperconcentrated without rheological proof).
5. **`DOWNSTREAM_FLOOD`**: High-discharge fluvial surge traveling along mainstream river valleys, producing bank overtopping and infrastructure destruction.
6. **`TEMPORARY_RIVER_BLOCKAGE`**: Short-duration obstruction of a river channel caused by tributary debris deposition, avalanche debris, or transient ponding that creates multi-pulse releases, distinct from consolidated rotational landslide dams.

---

## 3. Units of Analysis: Event Units vs. Lake Monitoring Units

To eliminate ambiguity between lake-level models and multi-hazard surveillance, units of analysis are strictly differentiated:

| Attribute | Lake Monitoring Unit (Systems A & B) | Multi-Hazard Cryospheric Unit (System C) |
| :--- | :--- | :--- |
| **Primary Geographic Entity** | Closed lake surface water polygon | Steep mountain face patch ($500\,\text{m} \times 500\,\text{m}$) or river corridor reach ($1\,\text{km}$) |
| **Identifier Scheme** | Lake ID (e.g., `GLO_ID_10425` or `GL085448E28236N`) | Event ID (e.g., `NEPAL-2026-08-26-LANGTANG`) |
| **Domain Scope** | Lacustrine water boundaries ($\le 15^\circ$ slope) | High-relief headwalls ($> 3,500\,\text{m}$, slope $> 30^\circ$) & valley channels |
| **Temporal Frequency** | Annual composites & multi-week cadence | Sub-daily to weekly satellite overpasses & seismic feeds |
| **Prediction Target** | Lake area change ($\text{km}^2$) / lake condition class | Slope detachment susceptibility / valley impoundment index |
| **Training Pipeline** | GLO 2017–2024 annual lake database | Independent multi-hazard inventory (never mixed with GLO) |

---

## 4. Metadata Discovery vs. Raster Processing Semantics

All satellite inventories distinguish catalogued metadata from downloaded/processed raster data:

- **`DISCOVERED_IN_CATALOGUE`**: The satellite scene has been queried, resolved, and verified via the Copernicus Data Space Ecosystem (CDSE) STAC catalogue. Metadata, asset keys, sensing geometries, cloud cover, and access feasibility have been audited and catalogued. No local raster extraction has been executed.
- **`DOWNLOADED_PENDING_PREPROCESSING`**: Full product assets (e.g., SAFE or GeoTIFF) are staged on local disk or cloud cache.
- **`PREPROCESSED_ANALYSIS_READY`**: Calibrated, coregistered, terrain-corrected, and reprojected raster grids ready for feature extraction.

---

## 5. Event-Level Dataset Schema & Sampling Strategy

### 5.1 Positive Event Cohort (Case Studies)
Positive training samples require independently documented, expert-verified historical cryospheric mass wasting events in the Hindu Kush Himalaya:
1. **2026 Langtang Lirung** (Rock–ice avalanche & debris surge, Nepal)
2. **2026 Budhi Gandaki** (Rock–ice avalanche, tributary blockage & cascading surge, Nepal)
3. **2021 Chamoli Disaster** (Ronti peak hanging glacier / rock detachment, India)
4. **2021 Melamchi Disaster** (Bremdang cirque sediment surge & river damming, Nepal)
5. **2018 Sedongpu Landslide/Avalanche** (Repeated Yarlung Tsangpo blockage, Tibet)
6. **2017 Bhotekoshi Outburst** (Tibetan transboundary breach)

### 5.2 Non-Event Control Sampling (Negative Cohort)
To prevent extreme selection bias and false positives:
- **Terrain-Matched Controls:** Sample high-altitude hanging glaciers and steep mountain flanks with similar elevation ($> 5,000\,\text{m}$), slope ($35^\circ\text{--}60^\circ$), aspect, and glacier thermal regime that did **not** experience catastrophic failure.
- **Ratio:** Minimum of **$10 : 1$** control patches to event patches.
- **Stratification:** Matched across Western, Central, and Eastern Himalayan climatic zones.

---

## 6. Preventing Spatial and Temporal Leakage

1. **Strict Temporal Cutoff:**
   - All pre-collapse predictors for an event must be extracted strictly from $t \le t_{\text{event}} - 48\,\text{hours}$.
   - Post-event imagery (scars, deposition) must never enter precursor prediction features.
2. **Spatial Basin Isolation:**
   - Cross-validation must strictly use **Leave-One-Mountain-Massif-Out (LOMO)**:
     - Fold 1: Train on Karakoram + Western Himalaya; Validate on Central Himalaya (Langtang / Ganesh).
     - Fold 2: Train on Central Himalaya; Validate on Eastern Himalaya / Tibet.
   - Spatial buffers of at least **$25\,\text{km}$** around any event source must be quarantined to prevent spatial autocorrelation leakage into control patches.

---

## 7. Feature Space for Multi-Hazard Monitoring & Research Literature Provenance

### 7.1 Kinematic / InSAR Features & Scientific Limits
- 12-day and 24-day line-of-sight displacement velocity ($v_{\text{LOS}}$ in $\text{mm/day}$).
- Temporal coherence loss gradient ($\Delta \gamma / \Delta t$).
- Spatial strain rate / displacement gradient ($\partial d / \partial x$).

> [!NOTE]
> **Interferometric Detection Limits (DOI `10.31223/X5GZ3J`):**
> *Indexed Title:* Ajay Kumar Sah (EarthArXiv, 18 Sept 2026, `PREPRINT_UNREVIEWED`), *"The single-pair unwrapping ceiling is a property of the track, not of the radar: Sentinel-1 evidence from rock–ice avalanches at Langtang Lirung and Ronti Peak"*.
> - **Indexed Study Findings:** Demonstrates across ascending and descending tracks over Langtang Lirung (2026) and Ronti Peak (2021) that single-pair C-band InSAR phase unwrapping ceilings and geometric decorrelation are governed by track acquisition geometry (look angle, heading, local incidence, and slope aspect) rather than radar frequency in isolation. Phase aliasing occurs when displacement exceeds $\sim \lambda/4$ per pixel, preventing 2D phase unwrapping in steep relief (>45°).
> - **Operational Model Interpretation:** Single-pair C-band InSAR cannot operate as an autonomous standalone early-warning trigger for steep hanging glacier detachments without multi-track ascending/descending combinations, multi-temporal InSAR (MTI) persistence scatterers, or optical offset tracking.

### 7.2 SAR Amplitude Sensitivity & Negative Precursor Provenance
- SAR backscatter anomalous drop ($\Delta \sigma^0 < -3\,\text{dB}$) indicating river ponding/impoundment.

> [!NOTE]
> **Backscatter Amplitude Precursor Baseline (DOI `10.31223/X59N5N`):**
> *Indexed Title:* Ajay Kumar Sah (EarthArXiv, 6 Sept 2026, updated 8 Sept 2026, `PREPRINT_UNREVIEWED`), *"No detectable precursor in Sentinel-1 backscatter before the 26 August 2026 Langtang Lirung rock–ice avalanche, Nepal"*.
> - **Indexed Study Findings:** Sentinel-1 Level-1 GRD backscatter amplitude ($\sigma^0$) time series analysis in VV and VH polarizations showed **no anomalous precursory deviation** over the Langtang detachment face prior to collapse.
> - **Stated Temporal & Detection Limits:** Routine 12-day orbital repeat sampling and C-band surface penetration depth constrain sensitivity to sub-surface mechanical degradation or rapid pre-failure deformation occurring between satellite passes. Amplitude features cannot serve as early kinematic creep predictors.

### 7.3 Topographic / Geomorphic Features
- Slope angle, profile curvature, planform curvature.
- Upstream contributing catchment area and relief energy ($H_{\text{peak}} - H_{\text{toe}}$).
- Distance to major structural faults / shear zones.

### 7.4 Thermal & Cryospheric Features
- Permafrost thaw index / active layer thickness anomalies (ERA5-Land reanalysis).
- Hanging glacier thickness and thermal state (cold-based vs. polythermal).
- Cumulative positive degree days (PDD) prior to event.

### 7.5 Hydrological Corridor Features
- Valley confinement ratio (channel width vs. valley floor width).
- Tributary junction blockage susceptibility index.

---

## 8. Gate Criteria Before Phase 9.4 Implementation
- [ ] At least 15 verified high-mountain mass-wasting event polygons catalogued with field/satellite ground truth.
- [ ] 150 terrain-matched negative control patches established.
- [ ] Containerized InSAR coregistration and coherence pipeline deployed.
- [ ] Separate attribution of amplitude baseline (`10.31223/X59N5N`) vs interferometric limits (`10.31223/X5GZ3J`).
- [ ] Zero overlap with GLO annual lake area models.
