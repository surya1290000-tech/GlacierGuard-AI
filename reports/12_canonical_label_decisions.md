# GlacierGuard-AI — Canonical Lake Classification Label Decisions & Scientific Reconciliation
**Phase:** 9.1.4 Canonical Classification Label Reconciliation
**Date:** 2026-11-02
**Author:** Lead Geospatial ML Engineer
**Git Branch:** `feature/phase9-dual-ensemble`

---

## 1. Executive Summary & Purpose

Before commencing Phase 9.2 multi-class classifier training, this audit establishes the **authoritative, observation-level canonical ground truth** for glacial lake change classification.

Previously generated Phase 12 artifacts contained numerical inconsistencies across reports:
- `reports/12_label_sensitivity.csv` (Row 8 Hybrid): `STABLE=10,760`, `EXPANDING=77`, `UNUSUAL=115`, `AMBIGUOUS=897`
- `reports/12_class_distribution.csv`: `STABLE=11,157`, `EXPANDING=577`, `UNUSUAL=115`, `AMBIGUOUS=0`
- `reports/12_validation_design.md` & `reports/12_classification_data_audit.json`: `STABLE=10,764`, `EXPANDING=577`, `UNUSUAL=508`, `AMBIGUOUS=0`

This document provides the definitive mathematical derivation, traces the precise root cause of every past inconsistency, defines the canonical four-state formulation (`STABLE`, `EXPANDING`, `UNUSUAL_CHANGE`, `AMBIGUOUS`), and produces reconciled counts across all partitions.

---

## 2. Root-Cause Analysis of Previous Inconsistencies

An audit of the data generation routines revealed that all inconsistencies stemmed from **how ambiguous observations ($N=897$, $7.57\%$ of the matched cohort) were handled**:

1. **`reports/12_label_sensitivity.csv` (Row 8 Hybrid):**
   - Correctly evaluated the strict conjunctions of the hybrid definition.
   - Identified 10,760 STABLE, 77 EXPANDING, 115 UNUSUAL_CHANGE, and **897 AMBIGUOUS** observations.

2. **`reports/12_class_distribution.csv` & `notebooks/12_dual_ensemble_data_audit.ipynb`:**
   - The script initialized labels to `"STABLE"` and assigned `cond_expanding = (~cond_stable) & (~cond_unusual) & (d > 0)`.
   - Consequently, all 500 positive ambiguous observations (mostly small lakes with $>20\%$ relative growth but $\le 3\sigma$ noise clearance) were forced into `EXPANDING` ($77 + 500 = 577$).
   - The remaining 397 negative ambiguous observations (small-lake shrinkage $<-20\%$ within $3\sigma$) defaulted to `"STABLE"` ($10,760 + 397 = 11,157$).
   - Ambiguous observations were thus suppressed to zero.

3. **`reports/12_validation_design.md` & `reports/12_classification_data_audit.json`:**
   - An intermediate script considered Category C small-lake negative shrinkage ($\%\Delta A < -20\%$, $N=393$) as anomalous drainage and grouped them into `UNUSUAL_CHANGE` ($115 + 393 = 508$).
   - These 393 observations were subtracted from STABLE ($11,157 - 393 = 10,764$), while EXPANDING remained at 577.
   - This created the $10,764$ / $577$ / $508$ distribution ($90.84\%$ / $4.87\%$ / $4.29\%$), which directly contradicted the formal mathematical definitions documented in Section 4.1.

---

## 3. Canonical Label Semantics & Mathematical Formulation

The canonical formulation enforces that **the three physical classes are strictly mutually exclusive**, and observations that cannot confidently be assigned to a physical class are explicitly assigned to **`AMBIGUOUS`**.

Let:
$$\Delta A = \text{FUSED\_TARGET} = 0.5 \cdot (\Delta A_{S1} + \Delta A_{S2}) \quad [\text{km}^2]$$
$$\bar{A} = \text{MEAN\_AREA} = 0.5 \cdot (A_{S1} + A_{S2}) \quad [\text{km}^2]$$
$$\%\Delta A = \text{RELATIVE\_CHANGE\_PCT\_FUSED} = 100.0 \cdot \frac{\Delta A}{\bar{A}} \quad [\%]$$
$$\sigma_{\Delta A} = \text{FUSED\_SIGMA\_DELTA} = 0.5 \cdot \sqrt{\sigma_{S1,t}^2 + \sigma_{S1,t+1}^2 + \sigma_{S2,t}^2 + \sigma_{S2,t+1}^2} \quad [\text{km}^2]$$

### 3.1 STABLE ($N=10,760$, $90.81\%$)
$$\text{STABLE} \iff (|\Delta A| \le 1.0\,\sigma_{\Delta A}) \lor (|\%\Delta A| \le 5.0\%) \lor (\Delta A < 0 \land \Delta A \ge -3.0\,\sigma_{\Delta A} \land \%\Delta A \ge -20.0\%)$$
- **Physical Basis:** Delineation uncertainty for Sentinel-1/2 lake boundaries is approximately 1-2 edge pixels. Fluctuations within $1.0\,\sigma_{\Delta A}$ or $\pm 5\%$ represent sensor segmentation noise. Contraction up to $3\sigma$ and down to $-20\%$ represents normal seasonal evaporation or border freezing, not an emergency.

### 3.2 EXPANDING ($N=77$, $0.65\%$)
$$\text{EXPANDING} \iff \neg \text{STABLE} \land \Delta A > 1.0\,\sigma_{\Delta A} \land \Delta A \le 3.0\,\sigma_{\Delta A} \land \%\Delta A > 5.0\% \land \%\Delta A \le 20.0\%$$
- **Physical Basis:** Statistically verified glacial lake expansion clearing measurement uncertainty without exhibiting rapid multi-sigma surge.

### 3.3 UNUSUAL_CHANGE ($N=115$, $0.97\%$)
$$\text{UNUSUAL\_CHANGE} \iff (\Delta A > 3.0\,\sigma_{\Delta A} \land \%\Delta A > 20.0\%) \lor (\Delta A < -3.0\,\sigma_{\Delta A} \land \%\Delta A < -20.0\%)$$
- **Exceptional Expansion ($N=68$):** Verified rapid expansion exceeding both $3\sigma$ noise and $+20\%$ relative area growth (e.g., massive calving or moraine dam destabilization).
- **Catastrophic Drainage ($N=47$):** Sudden drainage exceeding both $3\sigma$ noise and $-20\%$ relative shrinkage (e.g., GLOF breach, piping, or tunnel drainage).

### 3.4 AMBIGUOUS ($N=897$, $7.57\%$)
$$\text{AMBIGUOUS} \iff \neg (\text{STABLE} \lor \text{EXPANDING} \lor \text{UNUSUAL\_CHANGE})$$
- **Physical Basis:** Conflicting physical evidence between relative magnitude and measurement uncertainty. Forcing these observations into STABLE or EXPANDING induces severe label noise. They are excluded from 3-class model training and reserved for operational quality abstention.

---

## 4. Physical Taxonomy of Ambiguous Observations

The 897 ambiguous observations fall deterministically into four distinct physical categories:

| Category | Code | Count ($N$) | Pct of Ambig | Physical Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Category A** | `CAT_A_SMALL_LAKE_POS` | 497 | 55.41% | Small lakes with $\%\Delta A > +20\%$, but $\Delta A \le 3.0\,\sigma_{\Delta A}$. Delineation edge-pixel noise inflated by tiny denominator. |
| **Category B** | `CAT_B_LARGE_LAKE_POS` | 3 | 0.33% | Large lakes with $\Delta A > 3.0\,\sigma_{\Delta A}$, but $\%\Delta A \le +20\%$. Statistically significant absolute growth, but modest percentage. |
| **Category C** | `CAT_C_SMALL_LAKE_NEG` | 393 | 43.81% | Small lakes with $\%\Delta A < -20\%$, but $\Delta A \ge -3.0\,\sigma_{\Delta A}$. Apparent severe shrinkage driven by small lake denominator; within 3-sigma noise floor. |
| **Category D** | `CAT_D_LARGE_LAKE_NEG` | 4 | 0.45% | Large lakes with $\Delta A < -3.0\,\sigma_{\Delta A}$, but $\%\Delta A \ge -20\%$. Statistically significant contraction exceeding 3 sigma, but non-catastrophic percentage. |
| **Total** | | **897** | **100.00%** | |

---

## 5. Reconciled Partition Distribution

The canonical counts across all partitions are verified to reconcile exactly:

| Partition | Total $N$ | STABLE ($N$, %) | EXPANDING ($N$, %) | UNUSUAL_CHANGE ($N$, %) | AMBIGUOUS ($N$, %) | Unambiguous $N$ | Imbalance Ratio (Stable:Unusual) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (<= 2021)** | 8,527 | 7,744 (90.82%) | 47 (0.55%) | 89 (1.04%) | 647 (7.59%) | 7,880 | 87.01 : 1 |
| **Val (2022)** | 1,580 | 1,440 (91.14%) | 11 (0.70%) | 9 (0.57%) | 120 (7.59%) | 1,460 | 160.00 : 1 |
| **Test (2023 Reused)** | 1,742 | 1,576 (90.47%) | 19 (1.09%) | 17 (0.98%) | 130 (7.46%) | 1,612 | 92.71 : 1 |
| **All Matched (2017-2023)** | **11,849** | **10,760 (90.81%)** | **77 (0.65%)** | **115 (0.97%)** | **897 (7.57%)** | **10,952** | **93.57 : 1** |

### Unambiguous 3-Class Subcohort (Excluding Ambiguous)
When training and evaluating the 3-class classifier on unambiguous observations:
- **Total Unambiguous:** $N=10,952$ ($98.25\%$ STABLE, $0.70\%$ EXPANDING, $1.05\%$ UNUSUAL_CHANGE)
- **Train Subcohort:** $N=7,880$ ($98.27\%$ STABLE, $0.60\%$ EXPANDING, $1.13\%$ UNUSUAL_CHANGE)
- **Val Subcohort:** $N=1,460$ ($98.63\%$ STABLE, $0.75\%$ EXPANDING, $0.62\%$ UNUSUAL_CHANGE)
- **Test Subcohort:** $N=1,612$ ($97.77\%$ STABLE, $1.18\%$ EXPANDING, $1.05\%$ UNUSUAL_CHANGE)

---

## 6. Strict Separation from Operational Heuristics (RiskEngine)

Operational alert tiers in `src/glacierguard/risk_engine.py` (`SAFE/NORMAL`, `WATCH`, `ELEVATED/UNUSUAL_CHANGE`, `STALE`, `INSUFFICIENT_DATA`) serve a distinct operational purpose:
- They combine heuristics including sensor discordance, missing dates, and uncalibrated threshold rules.
- **Rule:** Operational statuses must **NEVER** be used as training labels.
- `WATCH`, `INSUFFICIENT_DATA`, and `STALE` are pipeline quality/abstention states, not physical ground-truth classes.

---

## 7. Phase 9.2 Classifier Training Guidelines & Guardrails

1. **Class Imbalance Mitigation:**
   - With an empirical imbalance ratio of $\sim 94:1$ (STABLE to UNUSUAL) and $\sim 140:1$ (STABLE to EXPANDING), accuracy is a prohibited evaluation metric.
   - Optimization must target **Macro-F1** and **PR-AUC (Precision-Recall AUC)** for `EXPANDING` and `UNUSUAL_CHANGE`.
   - Cost-sensitive loss weighting ($w_c = \frac{N}{3 \cdot N_c}$) or Focal Loss must be employed.
   - **Prohibition:** Synthetic oversampling (e.g., SMOTE) is strictly prohibited on validation and test sets.

2. **Ambiguous Handling Strategy:**
   - *Phase 9.2 Primary Approach:* Train 3-class classifier on unambiguous observations ($N=7,880$ train, $N=1,460$ val).
   - *Abstention Rule:* At inference, predict probabilities $P(y=c|X)$. If $\max_c P(y=c|X) < 0.50$ or entropy is high, output `INCONCLUSIVE_PREDICTION` rather than forcing a prediction.

3. **Validation & Partition Integrity:**
   - **2023 Historical Benchmark:** An exhausted benchmark from NB09. Evaluating 2023 for model selection, thresholding, or hyperparameter tuning is strictly prohibited.
   - **2024 Status:** 2024 area values were consumed to form the 2023 target; 2024 has no consecutive next year (2025 unavailable) and cannot be treated as an untouched future holdout.
   - **Spatial Leakage Prevention:** Random row splitting is prohibited. Cross-validation must strictly use lake-level grouping (`GroupKFold(groups=GLO_ID)`).

---

## 8. Decision Gate & Recommendation

- **Phase 9.2 Classification Training:** **GO**
- **Justification:** Canonical ground truth is fully reconciled, reproducible, and mathematically grounded in physical measurement uncertainty. All prior report discrepancies are resolved.
