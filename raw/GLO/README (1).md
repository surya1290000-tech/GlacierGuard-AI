# README - Glacial Lake Observatory (GLO): Annual dataset of glacial lakes in Nepal and transboundary catchments (2017–2024) - v1.02

*Correspondence:* C. Scott Watson (C.S.Watson@leeds.ac.uk)

|-----------------------------------------------------------------------------------------------|

## OVERVIEW
This dataset provides a spatially consistent inventory of glacial lakes across Nepal-transboundary catchments derived from Sentinel-1 (S1) and Sentinel-2 (S2) satellite imagery for the period 2017–2024.
The work is part of the *Glacial Lake Observatory for Flood Hazards Impacted by Changing Climate (GLO-FHICC) project* (https://glacial-lake-observatory.org/). 

|-----------------------------------------------------------------------------------------------|
## DATASET

The dataset contains **five GeoPackage (.gpkg)** files that represent both annual lake outlines and unique lakes (merged across years) derived from the S1 and S2 datasets. Two ZIP files contain the deep learning models used to classify lakes with an associated Python classification script.

| Data | Description |
|-------------|-------------|

| **S1_20172024_NTB_GLOID_v1.02.gpkg** | All Sentinel-1 detected glacial lake polygons (2017–2024). Each feature corresponds to an annual delineation. |

| **S1_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg** | Spatially dissolved unique lake *Polygons* from Sentinel-1. |

| **S2_20172024_NTB_GLOID_UniqueLakes_centroids_v1.02.gpkg** |  Spatially dissolved unique lake *Centroids* from Sentinel-2 – representative centroid points for each unique lake.|

| **S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg** | Spatially dissolved unique lake *Polygons* from Sentinel-2. Polygons represent the merged maximum lake extents per unique lake ID (`GLO_ID`). |

| **S2_20172024_NTB_GLOID_v1.02.gpkg** | All Sentinel-2 detected glacial lake polygons (2017–2024). Each feature corresponds to an annual delineation. |

| **GLO_DLv3_HMA_Sentinel_1.zip** | Sentinel-1 DeepLabV3 model used to classify lakes |

| **GLO_DLv3_HMA_Sentinel_2.zip** | Sentinel-2 DeepLabV3 model used to classify lakes |

| **GLO_lakes_classification_v1.0.py** | Script calling the models used to classify lakes |

|-----------------------------------------------------------------------------------------------|
## ATTRIBUTE INFORMATION

Each GeoPackage includes attribute fields describing the spatial, temporal, and physical characteristics of the glacial lakes. While many attributes are shared across datasets, certain fields are specific to the **All Lakes**, **Unique Lakes**, or **Centroid** layers.

---

### Common Attributes (present in all GeoPackage datasets)

| Field name | Description | Units / Type |
|-------------|-------------|---------------|
| `GLO_ID` | Unique Glacial Lake Observatory identifier for each lake (GLO_lon_lat) | Text |
| `COUNTRY` | Country in which the lake is located | Text |
| `BASIN` | Basin name | Text |
| `CONNECTIVITY` | Lake connectivity type (e.g., glacier-fed or non-glacier-fed) | Text |
| `ELEVATION_MEAN`, `ELEVATION_MIN`, `ELEVATION_MEDIAN` | Lake-surface elevation statistics derived from ALOS DSM (AW3D30 v4.1; 30 m) | metres (m) |
| `LONGITUDE`, `LATITUDE` | Representative lake centre coordinates | Decimal degrees |
| `DATA_SOURCE` | Data origin or sensor (Sentinel-1 / Sentinel-2) | Text |
| `START_DATE`, `END_DATE` | Start and end dates of the observation period | YYYY-MM-DD |
| `GTNG_REGION` | GTN-G (Global Terrestrial Network for Glaciers) region code (02 region) | Text |
| `TS_OUTLIER` | Boolean flag indicating a temporal outlier in the time series | TRUE / FALSE |

---

### Attributes in **All Lakes** datasets  
*(S1_20172024_NTB_GLOID_v1.02.gpkg and S2_20172024_NTB_GLOID_v1.02.gpkg)*

| Field name | Description | Units / Type |
|-------------|-------------|---------------|
| `AREA_YEAR` | Year of lake delineation (2017 – 2024) | Integer |
| `AREA` | Lake surface area | km² |
| `AREA_UNCERTAINTY` | Estimated uncertainty in lake area | km² |
| `PERIMETER` | Lake perimeter length | km |
| `REF_MSTAT` | Flag indicating if a lake is 'new' (dl_new) in the GLO dataset or 'NA' when compared to validation (reference) datasets of **TZhang et al. (2024a)** and **Kumar et al. (2025)** | Text |
| `S2_MSTAT` | Flag indicating if lake is in the S2 data (S1-only); NA if no match | Text |
| *(plus all Common Attributes)* |  |  |

---

### Attributes in **Unique Lakes (polygon)** datasets  
*(S1_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg and S2_20172024_NTB_GLOID_UniqueLakes_v1.02.gpkg)*

| Field name | Description | Units / Type |
|-------------|-------------|---------------|
| `AREA_DISSOLVED` | Total lake area after dissolving annual outlines by `GLO_ID` | km² |
| `PERIMETER_DISSOLVED` | Perimeter of the dissolved (maximum) lake extent | km |
| `EXPANSION_RATE` | Rate of lake area change between 2017 – 2024 | km² yr⁻¹ |
| `EXPANSION_UNCERTAINTY` | Uncertainty in expansion rate estimate | km² yr⁻¹ |
| `EXPANSION_RATE_SIG` | Significance flag for detected expansion | TRUE / FALSE |
| `S2_MSTAT` | Flag indicating if lake is in the S2 data (S1-only) | Text |
| *(plus all Common Attributes)* |  |  |

---

### Attributes in **Centroid** dataset
*(S2_20172024_TB_GLOID_UniqueLakes_centroids_v1.02.gpkg)*

| Field name | Description | Units / Type |
|-------------|-------------|---------------|
| `GLO_ID` | Identifier corresponding to each unique lake polygon | Text |
| `CENTROID_LON`, `CENTROID_LAT` | Harmonised centroid coordinates (from S2 where overlapping) | Decimal degrees |
| *(plus all attributes from the matching Unique Lakes polygon layer, excluding area and perimeter calculations)* |  |  |

---

### Attribute Notes

Output coordinate systems
| Layer type | CRS | Purpose |
|-------------|-----|----------|
| Polygon layers | **ESRI:102025 – Asia North Albers Equal Area Conic** | Accurate area and perimeter calculation |
| Centroid layers | **EPSG:4326 – WGS 84 (geographic)** | Global compatibility and integration with external datasets |


- Centroid layers are stored in **EPSG:4326 (WGS 84)** for compatibility with global datasets.  
- Elevation metrics are derived from the **ALOS DSM (AW3D30 v4.1; 30 m)**.  
- The `GLO_ID` value encodes spatial position and unique lake identity (`GLO_lon_lat`).  
- Boolean fields (`TRUE / FALSE`) indicate classification or significance status.

|-----------------------------------------------------------------------------------------------|

## DATA QUALITY AND LIMITATIONS

### Data quality assurance
- All classifications were produced using a fully automated deep learning workflow (DeepLabV3 architecture with ResNet backbone) applied to annual Sentinel-1 (SAR) and Sentinel-2 (optical) mosaics for 2017–2024.  
- Quality control included:
  - Removal of erroneous classifications at image edges and steep terrain (identified by anomalously high elevation standard deviation values or DEM artefacts).  
  - Exclusion of lakes with fewer than five valid years of observations from time-series analyses.  
  - Outlier detection in lake-area change rates using standardised residuals (>2σ threshold) and bootstrap resampling (1,000 iterations) to derive 95% confidence intervals.   
  - Verification of all coordinate reference systems (ESRI:102025 for polygons; EPSG:4326 for centroids) before data export.  

### Accuracy assessment
- A **10% stratified random sample** of classified lakes (n = 895 outlines, from 239 lakes) was manually digitised at 1:3,000–1:5,000 scale by two independent analysts using Sentinel-2 imagery (2020).  
- Comparison of independent manual digitisation produced an **internal F1 score = 0.95**, confirming consistency of the validation dataset.  
- Validation of deep learning classifications against manual reference outlines yielded **F1 = 0.82 (Sentinel-1)** and **F1 = 0.92 (Sentinel-2)** for 2020; Sentinel-2 performance remained stable (F1 ≈ 0.91) for 2017 and 2024.  
- Direct comparison of Sentinel-1 and Sentinel-2 inventories produced **F1 = 0.85** and **R² = 0.95** for shared lakes, indicating strong cross-sensor spatial consistency.  
- Comparison with the regional inventories of **Zhang et al. (2024a)** and **Kumar et al. (2025)** yielded **F1 = 0.79–0.85**, demonstrating good agreement across datasets and sensors.  

### Positional accuracy
- Polygon area and perimeter were derived in **ESRI:102025 (WGS 1984 Albers for Northern Asia)** to maintain equal-area properties.  
- Centroid coordinates were derived in **EPSG:4326 (WGS 84)** with precision to **five decimal places (~1 m precision)**, consistent with the 10 m native pixel resolution of Sentinel imagery.
- Where both Sentinel-1 and Sentinel-2 lakes shared the same `GLO_ID`, centroid positions were harmonised using the Sentinel-2 coordinates to ensure sub-pixel alignment (<10 m offset).  
- Positional uncertainty therefore reflects both the native satellite resolution and DEM-based elevation accuracy used for ancillary attributes (ALOS DSM (AW3D30 v4.1; 30 m), ±5 m RMSE).  

### Limitations
- Snow, icebergs, and seasonal freezing can reduce classification precision, particularly in SAR imagery or for supraglacial lakes.  
- Some temporal variability in mapped area arises from differences in observation date, water level fluctuations, and image mosaicking artefacts.  
- Users comparing with other inventories should note possible discrepancies in mapping resolution, date, and hydrological connectivity definitions.  

Overall, accuracy metrics (F1 = 0.79–0.92) and positional consistency (<10 m).

|-----------------------------------------------------------------------------------------------|
## CITATION, ACKNOWLEDGEMENTS, AND LICENCE

### Citation
If using this dataset, please cite as:

Rawlins, L., Watson, C. S., Bhambri, R., Khadka, N., & Chand, M. B. (2025). Datasets supporting - Glacial Lake Observatory (GLO): A dataset of glacial lakes in Nepal and transboundary catchments (2017–2024) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17802333

### Acknowledgements
This dataset will form part of the **Glacial Lake Observatory (GLO)** open-access database.  
This work was supported by a UK Research and Innovation Future Leaders Fellowship [grant number MR/Y016564/1].
We thank contributors to the **Sentinel-1** and **Sentinel-2** missions, and the **ALOS AW3D30 DEM** project for open-access data provision.
The authors declare that they have no conflict of interest.

### Licence
This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.  

|-----------------------------------------------------------------------------------------------|

