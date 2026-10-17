# GlacierGuard-AI — GLOF Classification Research Plan & Data Feasibility Audit

> **Status:** EXPLORATORY RESEARCH TRACK — MODEL TRAINING GATE: **NO-GO**
> **Date:** October 2026
> **Author:** GlacierGuard-AI Research & Engineering Team
> **Audited Dataset:** High Mountain Asia Glacial Lake Outburst Flood Database (HMAGLOFDB)
> **Source Repository:** [https://github.com/fidelsteiner/HMAGLOFDB](https://github.com/fidelsteiner/HMAGLOFDB)
> **File SHA-256:** `b4c84bb933f63a398001ac3838d9eddbfa53015c1ff173867d5f8f5dfe8b290c`

---

## Executive Summary & Architectural Separation

GlacierGuard-AI maintains a strict architectural boundary across three distinct scientific tasks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GlacierGuard-AI Architecture                          │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Tier 1: Annual Area-     │ Tier 2: Operational      │ Tier 3: GLOF Hazard & │
│ Change Regression        │ Satellite Surveillance   │ Risk Research         │
│ (NB01–NB09, NB11)        │ (NB10.1, NB10.2)         │ (HMAGLOFDB Audit)     │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ Task:                    │ Task:                    │ Task:                 │
│ NEXT_AREA_CHANGE =       │ NRT dual-sensor (S1/S2)  │ Event susceptibility  │
│ AREA(t+1) - AREA(t)      │ lake water extraction    │ & impact severity     │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ Status:                  │ Status:                  │ Status:               │
│ PRODUCTION-READY         │ OPERATIONAL CANDIDATE    │ RESEARCH GATE: NO-GO  │
│ Validated XGBoost        │ 5-lake calibrated QA     │ Data audit proves ML  │
│ (MAE 0.00479 km²)        │ surveillance pipeline    │ classifier unfeasible │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

This research plan documents the data-driven audit of **HMAGLOFDB**, evaluates the feasibility of machine learning classifiers for GLOF occurrence and impact severity, establishes strict feature-leakage boundaries, and delivers an evidence-based **NO-GO** gate decision for model training.

---

## 1. Dataset Provenance & Profile

| Property | Value |
|---|---|
| **Dataset Name** | High Mountain Asia Glacial Lake Outburst Flood Database (HMAGLOFDB) |
| **Citation** | Steiner, F. et al. (2024). *A comprehensive database of glacial lake outburst floods in High Mountain Asia*. |
| **License** | Open Access (Creative Commons / Academic Research) |
| **Source File** | `raw/GLOF/HMAGLOFDB.csv` (excluded from Git tracking via `.gitignore`) |
| **File SHA-256** | `b4c84bb933f63a398001ac3838d9eddbfa53015c1ff173867d5f8f5dfe8b290c` |
| **Total Records** | **774 rows × 59 columns** |
| **Unique Identifiers** | 774 unique `GF_ID`s (0 duplicates, 0 nulls) |
| **Temporal Span** | 1533 to 2026 (172 recorded events from year 2000 onwards) |
| **Encoding Note** | Encoded in Latin-1; contains non-breaking spaces (`\xa0`) and author diacritics |

### '+' Count Convention
In HMAGLOFDB, the character `+` is an explicit metadata convention denoting a **positive-but-unknown count** (e.g., casualties or destroyed buildings confirmed to have occurred, but where exact numbers are unknown).
- `Lives_total`: 4 entries recorded as `+`
- `Livestock`: 5 entries recorded as `+`
- `Residential_destroyed`: 26 entries recorded as `+`
- `Agricultural`: 17 entries recorded as `+`

**Non-negotiable Rule:** Under no circumstances may `+` be silently coerced to zero or treated as an exact count of 1.

---

## 2. Evaluation of Proposed Machine Learning Tasks

### Task 1 — GLOF Occurrence / Susceptibility Classification
*Goal:* Predict whether a glacial lake will experience a GLOF event within a specified future temporal window.

**Scientific Assessment:** **BLOCKED (NO-GO)**
1. **Event-Only Inventory (Absence of Negatives):** HMAGLOFDB is an inventory of recorded positive events (774 GLOFs). It contains zero verified non-event observations.
2. **Severe Historical Reporting Bias:** A lake without a recorded GLOF in historical literature cannot be assumed to be a true negative. Small, remote, or ancient outburst events frequently went unrecorded.
3. **Severe Missingness in Pre-Event Geometry:**
   - Pre-event lake area (`Area`) is available for only **38 out of 774 events (4.9%)**.
   - Pre-event lake volume (`Volume`) is available for only **108 out of 774 events (13.9%)**.
4. **Temporal Cutoff Violation:** Merging static HMAGLOFDB attributes with modern GLO (2017–2024) features would cause temporal misalignment across centuries of geomorphic change.

### Task 2 — GLOF Impact Severity Classification
*Goal:* Predict the downstream impact severity (e.g., Low / Moderate / High) of a GLOF event.

**Scientific Assessment:** **BLOCKED (NO-GO)**
1. **Extreme Missingness in Consequence Metrics:**
   - `Injured_total`: 0.3% reporting (2 / 774)
   - `Displaced_total`: 1.2% reporting (9 / 774)
   - `Econ_damage`: 2.1% reporting (16 / 774)
   - `Hydropower`: 2.8% reporting (22 / 774)
   - `Impact` (narrative text): 3.6% reporting (28 / 774)
   - `Lives_total`: 4.1% reporting (32 numeric + 4 `+` counts)
   - `Discharge_water`: 10.1% reporting (78 / 774)
   - `Infra`: 11.4% reporting (88 / 774)
2. **Prohibition of Subjective Label Bins:** Partitioning narrative descriptions into arbitrary "LOW / MEDIUM / HIGH" classes creates an invented, unverified ground truth. Missing records cannot be assumed to represent zero damage.
3. **Post-Event Leakage:** Most reported fields describe downstream aftermath, not pre-breach characteristics.

---

## 3. Comprehensive Feature & Leakage Registry

All 59 columns in HMAGLOFDB have been formally audited and classified:

### 1. Identifiers (`IDENTIFIER`)
- `GF_ID`, `Lake_name`, `Glacier_name`, `GL_ID`, `LakeDB_ID`, `G_ID`

### 2. Event Metadata (`EVENT_METADATA`)
- `Year_approx`, `Year_exact`, `Month`, `Day`, `Transboundary`, `Repeat`, `Region_RGI`, `Region_HiMAP`, `Country`, `Province`, `River_Basin`, `Ref_scientific`, `Ref_scientific_full`, `Ref_other`, `Remarks`

### 3. Eligible Pre-Event Candidates (`PRE_EVENT_CANDIDATE`)
*Characteristics existing prior to an event, supportable if observation cutoff is respected:*
- `Lat_lake`, `Lon_lake` (100.0% coverage)
- `Elev_lake` (99.1% coverage)
- `Lake_type` (96.1% coverage)
- `Driver_lake` (7.6% coverage)

### 4. Sparse & Temporally Unverified (`SPARSE_UNVERIFIED_TIMING`)
- `Area` (4.9% coverage): Unverified whether measured prior to drainage or post-event.
- `Volume` (13.9% coverage): Unverified pre- vs post-drainage volume.

### 5. Post-Event Leakage (`POST_EVENT_LEAKAGE`)
*Strictly prohibited as predictive inputs; describe event physics or consequences:*
- **Trigger & Physics:** `Driver_GLOF` (12.7%), `Mechanism` (25.1%), `Discharge_water` (10.1%), `Discharge_solid` (4.0%)
- **Downstream Consequences:** `Impact` (3.6%), `Lives_total` (4.1%), `Lives_male`, `Lives_female`, `Lives_disabilities`, `Injured_total`, `Displaced_total`, `Livestock`, `Residential_destroyed`, `Commerical_destroyed`, `Residential_damaged`, `Commerical_damaged`, `Infra` (11.4%), `Agricultural`, `Hydropower`, `Econ_damage`
- **Impact Coordinates & Evidence:** `Lat_impact`, `Lon_impact`, `Elev_impact`, `Impact_type`, `Sat_evidence`

---

## 4. Temporal Quality Control Audit & Anomalous Year Inspection

### 4.1 Database Maintainer Documentation Notice
As documented in the official repository README ([https://github.com/fidelsteiner/HMAGLOFDB](https://github.com/fidelsteiner/HMAGLOFDB)):
> *"The database has been quality controlled up to 2025/12/31. Events with a later date in the database need to be evaluated and cross checked."*

### 4.2 Inspection of Anomalous & Extreme Dates (1533–2026)
An exhaustive inspection of all 774 records in `raw/GLOF/HMAGLOFDB.csv` identified three distinct temporal cohorts requiring explicit quality control categorization:

#### 1. Unverified Post-2025 Events ($N = 2$)
Two entries have recorded dates in calendar year 2026:
- **`GF_ID = 773`:** Date `2026-07-26`, Lake: *Cogalongco* (China). Source: Steiner et al. `(this study)`.
- **`GF_ID = 775`:** Date `2026-08-09`, Lake: *Unnamed* (Bhutan). Source: Steiner et al. `(this study)`, secondary news citation `https://www.bbs.bt/246122/` (Bhutan Broadcasting Service report).
- **QC Classification:** `UNVERIFIED_POST_2025_REQUIRES_CROSS_CHECK`
- **Review Status:** `PENDING_MAINTAINER_EVALUATION`
- **Scientific Treatment:** In accordance with repository constraints, **original raw records are preserved untouched**. They are neither deleted nor silently edited, but are explicitly quarantined from any automated operational assumptions and must not be represented as definitively confirmed historical events until formal maintainer cross-checking is complete.

#### 2. Pre-Instrumental Historical Documentary Accounts ($N = 11$, 1533–1846)
- **`GF_ID = 1` (1533):** Upper Shyok valley, India. Source: Hedin (1917), Hewitt (2014), Bazai et al. (2021). Earliest recorded outburst event in the inventory.
- **`GF_ID = 2` (1560):** Seti River valley, Nepal. Source: Ives, Shrestha and Mool (2010), ICIMOD (2011).
- **`GF_ID = 3` (1661):** Braldu River valley, Pakistan. Source: Godwin-Austen (1864).
- **`GF_ID = 4` (1780):** Upper Shyok valley, India. Source: Strachey (1853), Bazai et al. (2021). Documentary interview with an 80-year-old resident regarding the bursting of Kumdan glacier lake.
- **`GF_ID = 5–11` (1833–1846):** Early 19th-century accounts of Kumdan and Karakoram ice-dam outbursts.
- **QC Classification:** `HISTORICAL_PRE_INSTRUMENTAL_DOCUMENTARY`
- **Scientific Treatment:** These entries represent valuable historical documentary evidence, but lack modern geometric boundary measurements or verified pre-event lake areas.

#### 3. Undated Events ($N = 148$, 19.1%)
- 148 entries lack both an exact year (`Year_exact`) and an approximate year (`Year_approx`).
- **QC Classification:** `UNDATED_HISTORICAL_RECORD`

All 774 records have been profiled and exported with explicit quality-control flags, source citations, review statuses, and notes in:
`reports/glof_research/hmaglofdb_temporal_qc_audit.csv`.

---

## 5. Model-Training Gate Decision: STRICT NO-GO

Based on the empirical evidence from `scratch/audit_glof_event_database.py`, the field completeness report `reports/glof_research/hmaglofdb_field_completeness.csv`, and the temporal QC audit `reports/glof_research/hmaglofdb_temporal_qc_audit.csv`:

1. **Gate Decision:** **STRICT NO-GO**
2. **Action Taken:** Do **NOT** train an XGBoost or Random Forest GLOF classifier on HMAGLOFDB.
3. **Rationale:** Training a model on this dataset would require either:
   - Inventing synthetic negative samples, creating false ground truth; or
   - Converting subjective text into uncalibrated target classes; or
   - Utilizing post-event leakage features to achieve artificially high validation metrics; or
   - Depending on unverified post-2025 provisional entries or pre-instrumental documentary dates.
4. **Operational Directive:** GlacierGuard-AI will continue to provide GLOF hazard surveillance through **physical change detection** (Tier 2: lake expansion, calving terminus dynamics, freeboard reduction) and **annual regression forecasting** (Tier 1: frozen NB09 XGBoost and NB11 benchmarks), which rest on sound, validated satellite observations.
