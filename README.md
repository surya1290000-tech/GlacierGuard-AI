# GlacierGuard AI — GLO Dataset Discovery Report

> **Status:** Dataset Inspection Only — No data modified, no models built.  
> **Generated:** 2026-08-27  
> **Notebook:** `notebooks/01_glo_dataset_inspection.ipynb`

---

## What We Have

Two GeoPackage files from the **Glacial Lake Observatory (GLO) v1.02** dataset, located in `raw/GLO/`:

| File | Sensor | Size |
|---|---|---|
| `S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg` | Sentinel-1 (SAR) | ~7.0 MB |
| `S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg` | Sentinel-2 (Optical) | ~7.8 MB |
| `README (1).md` | Documentation | ~11 KB |

**Source:** Rawlins et al. (2025), Zenodo. https://doi.org/10.5281/zenodo.17802333  
**Licence:** CC BY 4.0

---

## Dataset Overview

| Property | Value |
|---|---|
| **Dataset name** | Glacial Lake Observatory (GLO) Annual Dataset v1.02 |
| **Coverage** | Nepal + transboundary catchments (China, India borders) |
| **Time span** | 2017–2024 (8 years of observations, encoded as expansion stats) |
| **Project** | GLO-FHICC — Glacial Lake Observatory for Flood Hazards Impacted by Changing Climate |

---

## Files: Sentinel-1 Unique Lakes

| Property | Value |
|---|---|
| **File** | `S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg` |
| **Features (lakes)** | **2,967** |
| **Geometry type** | `MultiPolygon` (dissolved maximum lake extents) |
| **CRS** | `ESRI:102025` — Asia North Albers Equal Area Conic |
| **Longitude range** | 80.03° → 88.88° E |
| **Latitude range** | 27.47° → 30.59° N |
| **Unique GLO_IDs** | 2,967 (one per lake — perfectly unique) |
| **Columns** | 18 (17 attribute + geometry) |

### Country Breakdown (S1)
| Country | Lakes |
|---|---|
| Nepal | 1,578 |
| China | 1,353 |
| India | 36 |

### Basin Breakdown (S1, top 3)
| Basin | Lakes |
|---|---|
| Koshi | 1,722 |
| Karnali | 878 |
| Gandaki | 367 |

### Connectivity (S1)
| Type | Count |
|---|---|
| Glacier-fed | 1,701 |
| Non Glacier-fed | 1,266 |

### Expansion Stats (S1)
| Category | Count |
|---|---|
| Expanding (rate > 0) | 981 |
| Shrinking (rate < 0) | 965 |
| Stable (rate = 0) | 1,021 |
| Statistically significant expansion | **349** |

---

## Files: Sentinel-2 Unique Lakes

| Property | Value |
|---|---|
| **File** | `S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg` |
| **Features (lakes)** | **4,150** |
| **Geometry type** | `MultiPolygon` (dissolved maximum lake extents) |
| **CRS** | `ESRI:102025` — Asia North Albers Equal Area Conic |
| **Longitude range** | 80.04° → 88.79° E |
| **Latitude range** | 27.47° → 30.58° N |
| **Unique GLO_IDs** | 4,150 (one per lake — perfectly unique) |
| **Columns** | 18 (17 attribute + geometry) |

### Country Breakdown (S2)
| Country | Lakes |
|---|---|
| Nepal | 2,347 |
| China | 1,745 |
| India | 58 |

### Basin Breakdown (S2, top 3)
| Basin | Lakes |
|---|---|
| Koshi | 2,306 |
| Karnali | 1,231 |
| Gandaki | 613 |

### Connectivity (S2)
| Type | Count |
|---|---|
| Glacier-fed | 2,535 |
| Non Glacier-fed | 1,615 |

### Expansion Stats (S2)
| Category | Count |
|---|---|
| Expanding (rate > 0) | 1,242 |
| Shrinking (rate < 0) | 803 |
| Stable (rate = 0) | 2,099 |
| Statistically significant expansion | **367** |

---

## Column Schema (Both Files)

Both files share the same 18 columns (with minor differences noted):

| Column | Type | Description |
|---|---|---|
| `GLO_ID` | Text | **Primary unique identifier** — format `GLO_<lon>_<lat>` |
| `COUNTRY` | Text | Country of lake location (Nepal / China / India) |
| `BASIN` | Text | Hydrological basin name |
| `CONNECTIVITY` | Text | Glacier-fed or Non Glacier-fed |
| `DATA_SOURCE` | Text | Sentinel-1 or Sentinel-2 |
| `ELEVATION_MEAN` | Float | Mean lake-surface elevation (m) from ALOS AW3D30 DEM |
| `ELEVATION_MIN` | Float | Minimum lake-surface elevation (m) |
| `ELEVATION_MEDIAN` | Float | Median lake-surface elevation (m) |
| `LONGITUDE` | Float | Representative lake centre longitude (decimal degrees) |
| `LATITUDE` | Float | Representative lake centre latitude (decimal degrees) |
| `GTNG_REGION_O2` | Text | GTN-G region code (02 = High Mountain Asia) |
| `AREA_DISSOLVED` | Float | Dissolved lake area in km² (max extent across years) |
| `PERIMETER_DISSOLVED` | Float | Perimeter of the dissolved extent (km) |
| `EXPANSION_RATE` | Float | Lake area change rate 2017–2024 (km² yr⁻¹) |
| `EXPANSION_UNCERTAINTY` | Float | Uncertainty in expansion rate (km² yr⁻¹) |
| `EXPANSION_RATE_SIG` | Boolean | TRUE if expansion is statistically significant |
| `S2_MSTAT` | Text | **(S1 only)** Flag if S1 lake also detected in S2 |
| `REF_MSTAT` | Text | **(S2 only)** Flag comparing to reference validation datasets |
| `geometry` | MultiPolygon | Dissolved maximum lake polygon (in ESRI:102025) |

---

## Key Technical Findings

### Geometry
- Both files contain **MultiPolygon** geometries (not simple Polygons or Points)
- Each feature = one **unique lake** represented by its dissolved maximum spatial extent
- The README also describes annual polygon files and a centroid file — **these are NOT in this folder**

### CRS
- Both polygon files use **`ESRI:102025` — Asia North Albers Equal Area Conic**
- This equal-area projection ensures accurate `AREA_DISSOLVED` and `PERIMETER_DISSOLVED` calculations in metres
- For web mapping, reproject to `EPSG:4326` (WGS84)

### Unique Lake Identification
- **`GLO_ID`** is the canonical unique lake identifier
- Format: `GLO_<longitude>_<latitude>` (e.g., `GLO_80.03311_30.32452`)
- Verified unique: every row in both files has a distinct `GLO_ID`
- Can be used to cross-reference S1 and S2 datasets

### Missing Data (Nulls)
| File | Column | Missing | Reason |
|---|---|---|---|
| S1 | `EXPANSION_RATE_SIG` | 800 | Lakes with <5 valid years excluded from time-series |
| S2 | `EXPANSION_RATE_SIG` | 1,872 | Same reason |
| S2 | `EXPANSION_RATE` | 6 | Insufficient observations |
| S2 | `EXPANSION_UNCERTAINTY` | 6 | Insufficient observations |

### Area Range
- **S1:** 0.00103 – 5.487 km² (median: 0.019 km²) — mostly small proglacial lakes
- **S2:** 0.00108 – 5.532 km² (median: 0.014 km²) — similar but slightly smaller median (higher detection sensitivity)

---

## What Is NOT in This Folder (Per README)

The GLO v1.02 dataset includes 5 GeoPackages total. We have 2 of them:

| File | Status |
|---|---|
| `S1_20172024_NTB_GLOID_v1.02.gpkg` (annual S1 polygons) | ❌ Not present |
| `S1_20172024_NTB_GLOID_UniqueLakesv1.02.gpkg` | ✅ Present |
| `S2_20172024_NTB_GLOID_v1.02.gpkg` (annual S2 polygons) | ❌ Not present |
| `S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg` | ✅ Present |
| `S2_20172024_NTB_GLOID_UniqueLakes_centroids_v1.02.gpkg` (EPSG:4326 points) | ❌ Not present |
| Deep learning model ZIPs & classification script | ❌ Not present |

---

## Recommended Next Steps for GlacierGuard AI

1. **Cross-reference S1 and S2 lakes** using `GLO_ID` to identify lakes detected by both sensors
2. **Hazard ranking**: use `EXPANSION_RATE_SIG=TRUE` + highest `EXPANSION_RATE` to flag priority lakes
3. **Reproject** to `EPSG:4326` for all web-mapping, API, and dashboard use cases
4. **Request remaining files** (annual time-series GeoPackages) to enable year-by-year change analysis
5. **Elevation filtering**: use `ELEVATION_MEAN` to stratify lakes by altitude for climate modelling

---

*Original dataset files are read-only and were not modified.*
