# GlacierGuard-AI — Phase 9.1.4.1 / 9.1.4.2 Final Validation Report
# Scientific Uncertainty, Observed Serial Dependence, and Runtime Label Taxonomy Closure

**Phases:** 9.1.4.1 & 9.1.4.2 — Scientific Uncertainty, Statistical Interpretation, and Runtime Taxonomy Cleanup
**Date:** 2026-11-18
**Branch:** `feature/phase9-dual-ensemble`
**Base Commits:** `c91626c8fa831fc2c12c6e6af611d45a057c30f8` (9.1.4.1) → `2eb248c55498452a9fcb56a424ee6c4f8e9cb4c8` (9.1.4.2)
**Scope:** Scientific audit, uncertainty semantics verification, dependence-diagnostic wording correction, migration-safe RiskEngine runtime taxonomy cleanup, hardened training-only class-weight validation, Phase 9.2 predictor-leakage validator, documentation reconciliation, and regression testing. **No Phase 9.2 classifier training was started or executed.**

---

## 1. Executive Summary

Phases 9.1.4.1 and 9.1.4.2 closed all scientific, statistical, runtime-taxonomic, and pre-training leakage gaps prior to Phase 9.2:

1. **GLO `AREA_UNCERTAINTY` Semantics Closed:** Verified from Rawlins et al. (2026, *Earth System Science Data*, 18, 5143–5165, `doi:10.5194/essd-18-5143-2026`, Eq. 1) and confirmed at **100.0000% exact match across all 40,423 GLO v1.02 records** (`18,129` S1 + `22,294` S2) that `AREA_UNCERTAINTY` is a deterministic empirical power-law delineation-error estimate (`round(0.0488 * A^0.537, 3)` km²), **not** a $1\sigma$ Gaussian standard deviation. All Gaussian $1\sigma$/$3\sigma$ and variance-propagation claims have been withdrawn and replaced with the descriptive Root-Mean-Square empirical delineation-error scale $U_{\text{RMS}}$ (engineering multipliers $1.0\times U_{\text{RMS}}$ and $3.0\times U_{\text{RMS}}$). See `reports/12_uncertainty_semantics_note.md`.
2. **Observed Serial Dependence Diagnostics Precisely Scoped (Phase 9.1.4.2):** Reproducibly computed via `compute_observed_serial_dependence_diagnostics()` on consecutive matched pairs (`N = 9,392`: `r = +0.5489` for observed S1–S2 discrepancy; `r = -0.2700` for observed fused annual change). Wording corrected across all code and reports: observed serial dependence diagnostics indicate that the derived cross-sensor discrepancies and annual changes are not temporally independent as observed quantities; they do **not** independently identify the covariance structure of the underlying measurement errors, nor do they alone prove worst-case additive bounds are too conservative.
3. **Migration-Safe RiskEngine Runtime Transition Completed (Phase 9.1.4.2):** Audited `risk_engine.py`, `models.py`, `db.py` (no SQL CHECK/enum constraint on `monitoring_alerts.status`), `dashboard_data.py`, and `app/dashboard.py`. Updated `RiskEngine._evaluate_rules()` Rule 2 to emit canonical **`ELEVATED_ALERT`** for new elevated operational alerts while retaining legacy `"UNUSUAL_CHANGE"` read compatibility in `map_status_to_user_facing()`, `dashboard_data.validate_qa_schema()`, `db.py`, and `app/dashboard.py` (preserving `"ELEVATED ALERT"` user-facing output across both canonical and legacy records with zero DB migration required).
4. **Hardened Training-Only Class-Weight Computation (Phase 9.1.4.2):** Updated `compute_training_class_weights()` to explicitly reject empty Series, null/missing labels (`None`/`NaN`/`pd.NA`), `AMBIGUOUS`, unknown/operational labels, and zero-support physical classes before computing `w_c = N_train / (K * N_c)` strictly on validated unambiguous training labels (`STABLE=0.339187`, `EXPANDING=55.886525`, `UNUSUAL_CHANGE=29.513109` on `Train <=2021`, `N = 7,880`).
5. **Runtime Predictor-Leakage Validator Locked for Phase 9.2 (Phase 9.1.4.2):** Added `CANONICAL_CLASSIFICATION_PREDICTOR_COLUMNS` (36 pre-transition predictors) and reusable `validate_classification_training_inputs(X, y, ...)` enforcing rejection of all `EXCLUDED_TARGET_DERIVED_COLUMNS`, duplicate/mismatched columns, missing targets, `AMBIGUOUS` targets, and any non-physical labels before classifier fitting.
6. **Ambiguous Residual Boundaries & Canonical Counts Preserved:** Exact residual Category A–D boundaries verified (all 897 `AMBIGUOUS` rows uniquely categorized; all 10,952 non-ambiguous rows `None`). Overall matched cohort counts remain byte-for-byte preserved: `STABLE=10,760`, `EXPANDING=77`, `UNUSUAL_CHANGE=115`, `AMBIGUOUS=897` (`Total=11,849`; `Train <=2021` unambiguous `7,880`; `Val 2022` unambiguous `1,460`).

---

## 2. Epistemic Classification

### 2.1 Confirmed Scientific Facts
- **Source Formula of `AREA_UNCERTAINTY`:** Rawlins et al. (2026, *ESSD*, 18, 5143–5165, Eq. 1) defines percentage area error from Sentinel-2 DeepLabV3 vs. manual digitizations ($n=895$) as $\text{error}\% = 4.88 \times A^{-0.463}$, yielding $\text{AREA\_UNCERTAINTY}(A) = \text{round}(0.0488 \times A^{0.537}, 3)\,\text{km}^2$.
- **Exact Local GeoPackage Match:** $\text{round}(0.0488 \times A^{0.537}, 3)$ matches **18,129 / 18,129 (100.0000%)** Sentinel-1 records and **22,294 / 22,294 (100.0000%)** Sentinel-2 records (**40,423 / 40,423**, maximum unrounded residual `0.000500 km²`).
- **Non-One-Sigma Semantics:** `AREA_UNCERTAINTY` is an empirical area-dependent delineation-error estimate applied identically to S1 and S2 despite lower S1 validation F1 (`0.82` vs. S2 `0.92`). It is not a $1\sigma$ standard deviation or parametric confidence interval.
- **Canonical Cohort Counts:** Overall matched cohort $N=11,849$ (`STABLE=10,760` [90.81%], `EXPANDING=77` [0.65%], `UNUSUAL_CHANGE=115` [0.97%: 68 extreme annual expansions, 47 extreme annual contractions], `AMBIGUOUS=897` [7.57%: Cat A=497, Cat B=3, Cat C=393, Cat D=4]); unambiguous three-class subcohort $N=10,952$ (`Train <=2021`: `7,880`; `Val 2022`: `1,460`; `Exhausted 2023 Ref`: `1,612`).
- **Frozen Model Immutability:** `reports/models/09_final_fused_xgboost.pkl` SHA-256 remains byte-for-byte identical: `5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f`.

### 2.2 Observed Empirical Diagnostics
- Computed reproducibly via `compute_observed_serial_dependence_diagnostics()` on consecutive matched lake-year pairs (`AREA_YEAR - prev_AREA_YEAR == 1`, `N = 9,392`):
  - Lag-1 autocorrelation of observed S1–S2 area discrepancy $(A_{S1,t} - A_{S2,t})$: **`r = +0.5489`**. Describes serial persistence in the observed cross-sensor area discrepancy; **not** a direct estimate of cross-sensor measurement-error covariance.
  - Lag-1 autocorrelation of observed consecutive fused annual area changes (`FUSED_TARGET`): **`r = -0.2700`**. Describes serial dependence in consecutive observed differences (which share endpoint $A_t$, and to which physical lake dynamics may also contribute).
  - Median observed cross-sensor discrepancy $|A_{S1,t} - A_{S2,t}| = 0.004769\,\text{km}^2$ (`0.6695×` baseline $\bar{u}_t$, `N = 11,849`).
- **Scope Boundary:** Observed serial dependence diagnostics indicate that the derived cross-sensor discrepancies and annual changes are not temporally independent as observed quantities. These diagnostics do **not** independently identify the covariance structure of the underlying measurement errors.

### 2.3 Project-Defined Engineering Conventions
- Retaining $U_{\text{RMS}} = \sqrt{\frac{1}{4}(u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2)}$ strictly as a descriptive RMS empirical delineation-error scale (bounded by $[\min u_i, \max u_i]$, 99.49% label agreement with arithmetic mean $\bar{u}$; avoids the minority-class collapse to 12 `EXPANDING` / 25 `UNUSUAL_CHANGE` under $U_{\text{additive}} = 2\bar{u}$).
- Multipliers $1.0\times U_{\text{RMS}}$ and $3.0\times U_{\text{RMS}}$, relative thresholds $\pm 5.0\%$ and $\pm 20.0\%$, grouping moderate annual contraction into `STABLE`, and isolating conflicting cases into `AMBIGUOUS` (exclusion state, not a fourth learned class).
- Runtime operational taxonomy: new `RiskEngine` elevated alerts emit canonical `ELEVATED_ALERT` while legacy persisted `UNUSUAL_CHANGE` records remain readable as user-facing `"ELEVATED ALERT"`.
- Pre-fitting enforcement: training-only class weights via `compute_training_class_weights()` (rejecting nulls/empty/AMBIGUOUS/unknown/missing classes) and feature/target leakage enforcement via `validate_classification_training_inputs()`.

### 2.4 Unresolved Scientific Questions
- Separate empirical area-error power-law parameters for Sentinel-1 SAR (which has lower validation F1 `0.82` vs. S2 `0.92`) are not published in GLO v1.02.
- The exact original-space coverage quantile of Rawlins et al.'s power-law fit and the true cross-sensor/temporal measurement-error covariance matrix cannot be identified without independent multi-year reference delineations.
- Genuinely unconsumed post-2024 holdout evaluation requires a future GLO release containing 2025+ observations (2024 areas were consumed to form the 2023 target; 2023 is exhausted and strictly prohibited from any further experimental use).

---

## 3. Summary of Phase 9.1.4.2 Changes

| File | Key Changes |
| :--- | :--- |
| `src/glacierguard/risk_engine.py` | Rule 2 now emits canonical `"ELEVATED_ALERT"` (previously `"UNUSUAL_CHANGE"`); `map_status_to_user_facing()` preserves `"ELEVATED ALERT"` for both `"ELEVATED_ALERT"` and legacy persisted `"UNUSUAL_CHANGE"`. |
| `src/glacierguard/dashboard_data.py` | Updated `validate_qa_schema()` `valid_statuses` to accept canonical `"ELEVATED_ALERT"` and `"STALE_DATA"` alongside legacy `"UNUSUAL_CHANGE"`. |
| `app/dashboard.py` | Added `"ELEVATED_ALERT"` alongside `"ELEVATED ALERT"` and legacy `"UNUSUAL_CHANGE"` in CSS badge, KPI summary counter, map color map, and alert banner. |
| `src/glacierguard/models.py` | Updated `MonitoringAlert.status` comment to mark `"UNUSUAL_CHANGE"` strictly as legacy persisted alias. |
| `src/glacierguard/classification_labels.py` | Hardened `compute_training_class_weights()` (explicitly rejects empty, null/missing, AMBIGUOUS, unknown, and zero-support inputs); added `CANONICAL_CLASSIFICATION_PREDICTOR_COLUMNS` (36 features), expanded `EXCLUDED_TARGET_DERIVED_COLUMNS`, added reusable `validate_classification_training_inputs()` and `compute_observed_serial_dependence_diagnostics()`; corrected dependence-diagnostic interpretation in docstrings and audit JSON generator. |
| `src/glacierguard/__init__.py` | Exported `CANONICAL_CLASSIFICATION_PREDICTOR_COLUMNS`, `compute_observed_serial_dependence_diagnostics`, and `validate_classification_training_inputs`. |
| `reports/12_uncertainty_semantics_note.md`, `reports/12_canonical_label_audit.json`, `reports/12_canonical_label_decisions.md`, `reports/12_classification_data_audit.json`, `reports/12_validation_design.md`, `notebooks/12_dual_ensemble_data_audit.ipynb` | Corrected dependence-diagnostic wording, documented completed RiskEngine migration, null-label weight validation, and Phase 9.2 runtime leakage validator. |
| `tests/test_risk_engine.py`, `tests/test_phase9_canonical_labels.py` | Added regression tests for canonical `ELEVATED_ALERT`, legacy `UNUSUAL_CHANGE` compatibility (including DB round-trip & dashboard validation), null/empty/missing-class/AMBIGUOUS/unknown class-weight rejection, every excluded predictor column rejection, feature/target validator checks, and reproducible serial dependence diagnostics. |

---

## 4. Verification & Test Results

- **Local Full Test Suite:**
  - Exact Command: `.venv\Scripts\python.exe -m pytest tests/ -q`
  - **Local Result:** **`287 passed, 2 warnings in 56.88s`** (0 failures, 0 errors; the 2 warnings are the pre-existing `sklearn.impute.SimpleImputer` median-skip warnings on early-year lag columns in `tests/test_ensemble_benchmark.py::TestTemporalOOF::test_oof_uses_only_past_years`).
- **GitHub Actions CI Status:** Remote workflows execute asynchronously on GitHub after pushing to `origin/feature/phase9-dual-ensemble`; no remote CI pass is claimed prior to push.
- **`git diff --check`:** Clean (exit code `0`).
- **Repository Secret Scan (`scan_repository`):** `0` violations.
- **Frozen Baseline XGBoost SHA-256 (`reports/models/09_final_fused_xgboost.pkl`):** `5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f` (byte-for-byte identical).

---

## 5. Phase 9.2 Readiness Recommendation

- **Phase 9.2 Classification Training Gate:** **GO** *(not started in this phase)*.
- **Mandatory Phase 9.2 Entry Controls Verified:**
  1. Fit strictly on unambiguous `Train <=2021` (`N = 7,880`) validated via `validate_classification_training_inputs(X_train, y_train, expected_features=CANONICAL_CLASSIFICATION_PREDICTOR_COLUMNS)`.
  2. Compute balanced weights strictly via `compute_training_class_weights(y_train)` (`0.339187` / `55.886525` / `29.513109`).
  3. Validate strictly via lake-grouped CV (`GroupKFold(groups=GLO_ID)`) and `Val 2022` (`N = 1,460` unambiguous) using Macro-F1 and PR-AUC.
  4. Zero evaluation, threshold selection, hyperparameter tuning, or model selection on the exhausted 2023 partition.
