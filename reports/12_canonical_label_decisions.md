# GlacierGuard-AI — Canonical Lake Classification Label Decisions & Scientific Reconciliation
**Phase:** 9.1.4 / 9.1.4.1 Canonical Classification Label & Uncertainty Semantics Closure
**Date:** 2026-11-18
**Author:** Lead Geospatial ML Engineer
**Git Branch:** `feature/phase9-dual-ensemble`

---

## 1. Executive Summary & Purpose

Before commencing Phase 9.2 multi-class classifier training, this audit establishes the **authoritative, observation-level canonical ground truth** for glacial lake condition classification, closes the scientific semantics of GLO `AREA_UNCERTAINTY` (Phase 9.1.4.1), aligns ambiguous-category diagnostic boundaries, and enforces strict vocabulary separation between physical ML classes and operational `RiskEngine` statuses.

Previously generated Phase 12 draft artifacts contained numerical inconsistencies across reports:
- `reports/12_label_sensitivity.csv` (Row 8 Hybrid): `STABLE=10,760`, `EXPANDING=77`, `UNUSUAL_CHANGE=115`, `AMBIGUOUS=897`
- Legacy `reports/12_class_distribution.csv` (superseded): `STABLE=11,157`, `EXPANDING=577`, `UNUSUAL_CHANGE=115`, `AMBIGUOUS=0`
- Legacy draft `reports/12_validation_design.md` & `reports/12_classification_data_audit.json` (superseded): `STABLE=10,764`, `EXPANDING=577`, `UNUSUAL_CHANGE=508`, `AMBIGUOUS=0`

This document provides the definitive mathematical formulation, uncertainty-semantics closure, exact residual ambiguous-category boundaries, reconciled partition counts, and strict Phase 9.2 validation guardrails.

---

## 2. Root-Cause Analysis of Previous Inconsistencies

All earlier count discrepancies stemmed from **how the 897 ambiguous observations ($7.57\%$ of the matched cohort) were handled**:

1. **`reports/12_label_sensitivity.csv` (Row 8 Hybrid):**
   - Evaluated the strict conjunctions of the hybrid definition and explicitly isolated 897 ambiguous observations (`10,760 / 77 / 115 / 897`).
2. **Legacy `reports/12_class_distribution.csv` (superseded):**
   - Initialized labels to `STABLE` and assigned `cond_expanding = (~cond_stable) & (~cond_unusual) & (d > 0)`, forcing 500 positive ambiguous observations into `EXPANDING` ($77 + 500 = 577$) and 397 negative ambiguous observations into `STABLE` ($10,760 + 397 = 11,157$).
3. **Legacy draft `reports/12_validation_design.md` & `reports/12_classification_data_audit.json` (superseded):**
   - Reassigned Category C small-lake negative contractions ($\%\Delta A < -20\%$, $N=393$) into `UNUSUAL_CHANGE` ($115 + 393 = 508$), subtracting them from `STABLE` ($11,157 - 393 = 10,764$) while leaving `EXPANDING` at 577. That draft distribution contradicted the written rule conjunctions.

---

## 3. Uncertainty Semantics Closure & Canonical Mathematical Formulation

### 3.1 GLO `AREA_UNCERTAINTY` Semantics (Summary of `reports/12_uncertainty_semantics_note.md`)
- In Rawlins et al. (2026, *Earth System Science Data*, 18, 5143–5165, `doi:10.5194/essd-18-5143-2026`; Zenodo `10.5281/zenodo.17802333`), `AREA_UNCERTAINTY` is **not** a $1\sigma$ Gaussian standard deviation or confidence interval. It is an **empirical area-dependent power-law delineation-error estimate** ($\text{error}\% = 4.88 A^{-0.463} \implies \text{round}(0.0488 A^{0.537}, 3)\,\text{km}^2$), verified at **100.0000% exact match** across all 40,423 GLO v1.02 records (`18,129` S1 + `22,294` S2).
- Statistical independence across sensors and consecutive years is **not** justified and is empirically falsified in the matched cohort (lag-1 autocorrelation of $A_{S1} - A_{S2}$ is $+0.5489$; lag-1 autocorrelation of $\Delta A$ is $-0.2700$, $N=9,392$).
- Accordingly, the combined scale $U_{\text{RMS}}$ is interpreted strictly as the **descriptive Root-Mean-Square (RMS) empirical delineation-error scale** across the four paired observations (requiring no Gaussian or independence assumption), and thresholds $1.0\times U_{\text{RMS}}$ and $3.0\times U_{\text{RMS}}$ are engineering multipliers — **not** statistical significance tests.

Let:
$$\Delta A = \text{FUSED\_TARGET} = 0.5 \cdot (\Delta A_{S1} + \Delta A_{S2}) \quad [\text{km}^2]$$
$$\bar{A} = \text{MEAN\_AREA} = 0.5 \cdot (A_{S1} + A_{S2}) \quad [\text{km}^2]$$
$$\%\Delta A = \text{RELATIVE\_CHANGE\_PCT\_FUSED} = 100.0 \cdot \frac{\Delta A}{\bar{A}} \quad [\%]$$
$$U_{\text{RMS}} = \sqrt{\frac{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2}{4}} = 0.5 \cdot \sqrt{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2} \quad [\text{km}^2]$$
*(Stored in schema column `FUSED_SIGMA_DELTA` for backward schema compatibility; represents $U_{\text{RMS}}$.)*

### 3.2 STABLE ($N=10,760$, $90.81\%$ of overall matched cohort; $98.25\%$ of unambiguous subcohort)
$$\text{STABLE} \iff (|\Delta A| \le 1.0\,U_{\text{RMS}}) \lor (|\%\Delta A| \le 5.0\%) \lor (\Delta A < 0 \land \Delta A \ge -3.0\,U_{\text{RMS}} \land \%\Delta A \ge -20.0\%)$$
- **Interpretation:** Captures annual changes within the $1.0\times U_{\text{RMS}}$ empirical delineation-error floor, within $\pm 5.0\%$ relative change, or within moderate annual contraction bounds. **Grouping moderate annual contraction into `STABLE` is a project-defined classification convention**; annual area observations alone do not prove physical causes (such as evaporation or border freezing).

### 3.3 EXPANDING ($N=77$, $0.65\%$ of overall matched cohort; $0.70\%$ of unambiguous subcohort)
$$\text{EXPANDING} \iff \neg \text{STABLE} \land \Delta A > 1.0\,U_{\text{RMS}} \land \Delta A \le 3.0\,U_{\text{RMS}} \land \%\Delta A > 5.0\% \land \%\Delta A \le 20.0\%$$
- **Interpretation:** Positive annual expansion exceeding both the $1.0\times U_{\text{RMS}}$ empirical delineation-error floor and $+5.0\%$ relative change, bounded above by $3.0\times U_{\text{RMS}}$ and $+20.0\%$.

### 3.4 UNUSUAL_CHANGE ($N=115$, $0.97\%$ of overall matched cohort; $1.05\%$ of unambiguous subcohort)
$$\text{UNUSUAL\_CHANGE} \iff (\Delta A > 3.0\,U_{\text{RMS}} \land \%\Delta A > 20.0\%) \lor (\Delta A < -3.0\,U_{\text{RMS}} \land \%\Delta A < -20.0\%)$$
- **Extreme Annual Expansion ($N=68$):** Threshold-defined unusual change exceeding both $3.0\times U_{\text{RMS}}$ and $+20.0\%$ relative area growth.
- **Extreme Annual Contraction ($N=47$):** Threshold-defined unusual change exceeding both $-3.0\times U_{\text{RMS}}$ and $-20.0\%$ relative area contraction.
- **Scientific Guardrail:** `UNUSUAL_CHANGE` represents **threshold-defined unusual change** in annual area observations, **not** independently verified hazard events or GLOFs. Cryospheric event catalogues and independent event evidence remain strictly separate from this annual-area training target.

### 3.5 AMBIGUOUS ($N=897$, $7.57\%$ of overall matched cohort — Exclusion State, NOT a 4th Learned Physical Class)
$$\text{AMBIGUOUS} \iff \neg (\text{STABLE} \lor \text{EXPANDING} \lor \text{UNUSUAL\_CHANGE})$$
- **Interpretation:** Observations with conflicting relative-change magnitude and empirical delineation-error scale. Excluded from 3-class supervised training so noisy or conflicting cases are never forced into `STABLE` or `EXPANDING`.

---

## 4. Exact Residual Taxonomy of Ambiguous Observations

Assuming positive `MEAN_AREA` and positive $U_{\text{RMS}}$, the exact mutually exclusive and exhaustive residual regions after applying `STABLE`, `EXPANDING`, and `UNUSUAL_CHANGE` partition all 897 `AMBIGUOUS` observations cleanly into four categories (non-ambiguous observations strictly receive `None`):

| Category | Code | Exact Residual Boundary Formula | Count ($N$) | Pct of Ambig | Interpretation |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Category A** | `CATEGORY_A_SMALL_LAKE_POS_HIGH_REL_SUB_3SIGMA` | $1.0\,U_{\text{RMS}} < \Delta A \le 3.0\,U_{\text{RMS}} \;\land\; \%\Delta A > +20.0\%$ | 497 | 55.41% | Small lakes with high positive relative change amplified by small denominator, below $3.0\times U_{\text{RMS}}$. |
| **Category B** | `CATEGORY_B_LARGE_LAKE_POS_HIGH_SIGMA_SUB_20PCT` | $\Delta A > 3.0\,U_{\text{RMS}} \;\land\; +5.0\% < \%\Delta A \le +20.0\%$ | 3 | 0.33% | Large lakes exceeding $3.0\times U_{\text{RMS}}$ in absolute expansion with moderate relative percentage. |
| **Category C** | `CATEGORY_C_SMALL_LAKE_NEG_HIGH_REL_SUB_3SIGMA` | $-3.0\,U_{\text{RMS}} \le \Delta A < -1.0\,U_{\text{RMS}} \;\land\; \%\Delta A < -20.0\%$ | 393 | 43.81% | Small lakes with large negative relative change amplified by small denominator, within $3.0\times U_{\text{RMS}}$. |
| **Category D** | `CATEGORY_D_LARGE_LAKE_NEG_HIGH_SIGMA_SUB_20PCT` | $\Delta A < -3.0\,U_{\text{RMS}} \;\land\; -20.0\% \le \%\Delta A < -5.0\%$ | 4 | 0.45% | Large lakes exceeding $-3.0\times U_{\text{RMS}}$ in absolute contraction with moderate relative percentage. |
| **Total** | | | **897** | **100.00%** | Every ambiguous observation has exactly one category; 0 non-ambiguous observations receive a reason. |

---

## 5. Reconciled Partition Distribution

### 5.1 Overall Matched Cohort ($N = 11,849$, Including Explicit `AMBIGUOUS` Exclusions)

| Partition | Total $N$ | STABLE ($N$, %) | EXPANDING ($N$, %) | UNUSUAL_CHANGE ($N$, %) | AMBIGUOUS ($N$, %) | Unambiguous $N$ | Imbalance Ratio (Stable : Unusual) |
| :--- | ---: | :--- | :--- | :--- | :--- | ---: | ---: |
| **Train ($\le 2021$)** | 8,527 | 7,744 (90.82%) | 47 (0.55%) | 89 (1.04%) | 647 (7.59%) | 7,880 | 87.01 : 1 |
| **Val (2022)** | 1,580 | 1,440 (91.14%) | 11 (0.70%) | 9 (0.57%) | 120 (7.59%) | 1,460 | 160.00 : 1 |
| **Test (2023 — Exhausted Historical Reference)** | 1,742 | 1,576 (90.47%) | 19 (1.09%) | 17 (0.98%) | 130 (7.46%) | 1,612 | 92.71 : 1 |
| **All Matched (2017–2023)** | **11,849** | **10,760 (90.81%)** | **77 (0.65%)** | **115 (0.97%)** | **897 (7.57%)** | **10,952** | **93.57 : 1** |

### 5.2 Unambiguous Three-Class Subcohort ($N = 10,952$, Excluding `AMBIGUOUS`)

| Partition | Unambiguous $N$ | STABLE ($N$, %) | EXPANDING ($N$, %) | UNUSUAL_CHANGE ($N$, %) |
| :--- | ---: | :--- | :--- | :--- |
| **Train Unambiguous ($\le 2021$)** | 7,880 | 7,744 (98.27%) | 47 (0.60%) | 89 (1.13%) |
| **Val Unambiguous (2022)** | 1,460 | 1,440 (98.63%) | 11 (0.75%) | 9 (0.62%) |
| **Test Unambiguous (2023 — Exhausted Historical Reference)** | 1,612 | 1,576 (97.77%) | 19 (1.18%) | 17 (1.05%) |
| **All Unambiguous (2017–2023)** | **10,952** | **10,760 (98.25%)** | **77 (0.70%)** | **115 (1.05%)** |

---

## 6. Strict Separation from Operational Heuristics (`RiskEngine`)

Physical ML classes (`PHYSICAL_CLASSES = ["STABLE", "EXPANDING", "UNUSUAL_CHANGE"]`) and operational alert tiers (`OPERATIONAL_HEURISTIC_STATES = ["SAFE", "NORMAL", "WATCH", "ELEVATED", "ELEVATED_ALERT", "ELEVATED ALERT", "STALE", "STALE_DATA", "STALE DATA", "INSUFFICIENT_DATA", "INSUFFICIENT DATA", "PROCESSING", "SYSTEM_ERROR"]`) are **strictly disjoint vocabularies** (`set(PHYSICAL_CLASSES).isdisjoint(set(OPERATIONAL_HEURISTIC_STATES)) == True`).

- Operational statuses (`WATCH`, `ELEVATED_ALERT`, `STALE`, `INSUFFICIENT_DATA`, etc.) combine heuristics such as data staleness, sensor discordance, and operational thresholds; they must **NEVER** be used as supervised training targets.
- For backward compatibility with existing operational pipelines and dashboard fixtures, `RiskEngine` retains `"UNUSUAL_CHANGE"` solely as a legacy internal alias mapped to user-facing `"ELEVATED ALERT"` (canonical operational identifier: `ELEVATED_ALERT`); this legacy alert code is semantically distinct from the physical ML class `CLASS_UNUSUAL_CHANGE`.

---

## 7. Phase 9.2 Classifier Training Guidelines & Guardrails

1. **Class Imbalance Mitigation (Training-Only Class Weights):**
   - Balanced inverse-frequency class weights $w_c = \frac{N_{\text{train}}}{3 \cdot N_{c,\text{train}}}$ must be computed strictly from the **unambiguous training partition** (`Train <=2021`, $N = 7,880$):
     - `STABLE`: $7,880 / (3 \times 7,744) = 0.339187$
     - `EXPANDING`: $7,880 / (3 \times 47) = 55.886525$
     - `UNUSUAL_CHANGE`: $7,880 / (3 \times 89) = 29.513109$
   - Primary metrics: **Macro-F1** and **PR-AUC** for `EXPANDING` and `UNUSUAL_CHANGE` (overall accuracy is prohibited as a selection metric because an all-`STABLE` trivial classifier achieves $90.81\%$ on the overall cohort and $98.25\%$ on the unambiguous subcohort).
   - Synthetic oversampling (e.g., SMOTE) on validation or test partitions is strictly prohibited.

2. **Target-Derived Feature Leakage Exclusion:**
   - The following columns are explicitly excluded from all model predictors: `FUSED_TARGET`, `CANONICAL_LABEL`, `AMBIGUOUS_REASON`, `RELATIVE_CHANGE_PCT_FUSED`, `FUSED_SIGMA_DELTA`, `FUSED_U_RMS`, `NEXT_AREA_CHANGE`, `NEXT_UNCERTAINTY`, `S1_NEXT_AREA_CHANGE`, `S2_NEXT_AREA_CHANGE`, `S1_NEXT_UNCERTAINTY`, `S2_NEXT_UNCERTAINTY`.

3. **Validation & Partition Integrity:**

> [!IMPORTANT]
> **2023 Test Partition Exhausted — Strict Prohibition.** The 2023 historical test partition ($N=1,742$ overall; $N=1,612$ unambiguous) was already evaluated in NB09 and NB11. **Any further test-set evaluation, threshold selection, hyperparameter tuning, or model selection using the 2023 partition is strictly prohibited.** Previously reported regression benchmark metrics ($\text{MAE}=0.004792\,\text{km}^2$, $\text{RMSE}=0.015079\,\text{km}^2$, $R^2=0.3366$) are preserved as historical references only.

- **2024 Status:** 2024 area observations were consumed to construct the 2023 target ($\text{AREA}(2024) - \text{AREA}(2023)$) and lack a 2025 observation; **2024 remains unavailable as an untouched future holdout**.
- **Spatial/Lake Leakage Prevention:** Random row-level splitting is strictly prohibited. Cross-validation must use lake-grouped splitting (`GroupKFold(groups=GLO_ID)`).

---

## 8. Decision Gate & Recommendation

- **Phase 9.2 Classification Training:** **GO**
- **Justification:** GLO `AREA_UNCERTAINTY` semantics are verified against Rawlins et al. (2026) and local GeoPackages, $U_{\text{RMS}}$ thresholds replace unsupported Gaussian sigma terminology, residual ambiguous boundaries A–D are mathematically aligned, operational and physical vocabularies are strictly disjoint, and all Phase 12 artifacts are reconciled.
