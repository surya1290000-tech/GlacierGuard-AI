# GlacierGuard-AI — Phase 9 Validation & Experimental Design
**Author:** Lead ML Engineer
**Date:** 2026-11-18 *(reconciled in Phase 9.1.4.1)*
**Target Systems:**
- System A: Regression Ensemble (Predicting `NEXT_AREA_CHANGE` in $\text{km}^2$)
- System B: Classification Ensemble (Predicting `STABLE`, `EXPANDING`, `UNUSUAL_CHANGE`)

---

## 1. Executive Summary & Audit Baseline

GlacierGuard-AI has established a frozen **36-feature cross-sensor fused XGBoost regression model** (`reports/models/09_final_fused_xgboost.pkl`), which achieved a previously reported historical test benchmark of:
$$\text{MAE} = 0.004792\,\text{km}^2 \quad (4,792\,\text{m}^2), \quad \text{RMSE} = 0.015079\,\text{km}^2, \quad R^2 = 0.3366$$
on the $N=1,742$ matched 2023 cohort (preserved strictly as a historical reference).

Previous experiment NB11 explored Random Forest vs. XGBoost on **20-feature single-sensor datasets**. It demonstrated that:
1. Equal-weight voting was suboptimal.
2. S1 Random Forest showed minor empirical improvement over XGBoost ($\Delta \text{MAE} = -529\,\text{m}^2$), but its 95% bootstrap confidence interval $[-1183.5, +20.7]\,\text{m}^2$ crossed zero (statistically inconclusive).
3. On optical S2 features, Persistence (zero change) beat all single-sensor ML models.
4. The 20-feature single-sensor models are **not** directly comparable to the 36-feature fused production model.

This document establishes the rigorous validation framework for **Phase 9.2 (Classification)** and **Phase 9.3 (Regression Ensemble)**.

---

## 2. Partitioning Strategy & Leakage Prevention

### 2.1 The Cardinal Rule: Strict Prohibition on Any Further Use of the 2023 Partition

> [!IMPORTANT]
> **2023 Test Partition Exhausted — Strictly Prohibited.** The 2023 historical test partition ($N=1,742$ overall matched pairs; $N=1,612$ unambiguous pairs) was already used in NB09 and NB11. **No further test-set evaluation, threshold selection, hyperparameter tuning, or model selection using the 2023 partition is permitted.** Previously reported benchmark metrics ($\text{MAE}=0.004792\,\text{km}^2$, $\text{RMSE}=0.015079\,\text{km}^2$, $R^2=0.3366$) are preserved strictly as historical references only.

Under scientific integrity rules:
- All feature selection, hyperparameter tuning, class-weight estimation (strictly from training observations), loss calibration, probability thresholding, ensemble blending weights $\alpha$, and model evaluation in Phase 9 must rely strictly on **lake-grouped cross-validation within Train ($\le 2021$)**, **temporal expanding-window validation**, **basin spatial holdout**, and the **2022 validation partition**.
- No Phase 9.2 or Phase 9.3 experiment may evaluate, tune, threshold, or select models against the exhausted 2023 partition.

### 2.2 Status of 2024 Data
The GLO v1.02 catalogue contains observations through year 2024. However:
1. For year 2024 observations, `NEXT_YEAR` (2025) is absent from the dataset; hence `NEXT_AREA_CHANGE` is undefined (`NaN`).
2. The 2024 area values were already consumed to construct the target for the 2023 partition:
   $$\text{NEXT\_AREA\_CHANGE}(2023) = \text{AREA}(2024) - \text{AREA}(2023)$$
3. **Conclusion:** **2024 remains unavailable as an untouched future holdout.** There is no unconsumed future holdout in GLO v1.02.

### 2.3 Chronological Split Design (Overall Cohort vs. Unambiguous Subcohort)

| Partition | Observation Year | Overall Matched $N$ | Unambiguous 3-Class $N$ | Ambiguous Excluded $N$ | Role in Phase 9 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Historical Train** | $2017 \le t \le 2021$ | 8,527 | 7,880 | 647 | Base model training, class-weight estimation, lake-grouped CV, expanding-window OOF generation |
| **Validation Split** | $t = 2022$ | 1,580 | 1,460 | 120 | Hyperparameter tuning, blend weight optimization ($\alpha$), probability calibration, decision thresholds |
| **Exhausted Historical Reference** | $t = 2023$ | 1,742 | 1,612 | 130 | **PROHIBITED for any further evaluation, tuning, thresholding, or model selection** (NB09/NB11 historical reference only) |
| **Total Matched Cohort** | $2017 \le t \le 2023$ | **11,849** | **10,952** | **897** | Full historical matched dataset |

### 2.4 Spatial Holdout Protocol (Basin Generalization) & Lake Grouping
To test geographic transferability and prevent lake-level leakage:
- **Primary Basin Partition (overall matched cohort $N=11,849$):**
  - Training / Validation basins: **Koshi** ($N=7,622$, 64.3%) and **Karnali** ($N=2,845$, 24.0%)
  - Spatial Holdout basin: **Gandaki** ($N=1,382$, 11.7%)
- **Lake-Grouped Cross-Validation:**
  - Random row-level splitting is strictly prohibited.
  - Multi-fold cross-validation must strictly use `GroupKFold(n_splits=5, groups=GLO_ID)`. All multi-year observations for any single lake remain strictly within either train or validation folds.

### 2.5 Explicitly Excluded Target-Derived Predictors
All target-derived and future-observation columns are strictly prohibited from model feature matrices:
`FUSED_TARGET`, `CANONICAL_LABEL`, `AMBIGUOUS_REASON`, `RELATIVE_CHANGE_PCT_FUSED`, `FUSED_SIGMA_DELTA`, `FUSED_U_RMS`, `NEXT_AREA_CHANGE`, `NEXT_UNCERTAINTY`, `S1_NEXT_AREA_CHANGE`, `S2_NEXT_AREA_CHANGE`, `S1_NEXT_UNCERTAINTY`, `S2_NEXT_UNCERTAINTY`.

---

## 3. Regression Ensemble Design (System A)

### 3.1 Base Model Architecture
Both base models will train on the **exact same 36-feature matrix** and preprocessing pipeline (fitted strictly on training folds):
- **Model 1: Random Forest Regressor** (`n_estimators=400`, `max_features='sqrt'`, `min_samples_leaf=5`)
- **Model 2: XGBoost Regressor** (`n_estimators=100` frozen baseline / `n_estimators=600` tuned, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.7`)

### 3.2 Combination Candidates
1. **Equal-Weight Voting:** $\hat{y}_{\text{vote}} = 0.5\,\hat{y}_{\text{RF}} + 0.5\,\hat{y}_{\text{XGB}}$
2. **Validation-Optimal Weighted Blend:**
   $$\hat{y}_{\text{blend}}(\alpha) = \alpha\,\hat{y}_{\text{RF}} + (1-\alpha)\,\hat{y}_{\text{XGB}}$$
   where $\alpha \in [0.0, 1.0]$ is selected strictly on the 2022 validation split (and lake-grouped CV) to minimize validation MAE.
3. **Temporal Out-of-Fold Stacking:**
   - Expanding-window meta-features generated strictly using past years ($t_{\text{train}} < t$).
   - Meta-learner: Ridge regression with non-negative coefficients.

### 3.3 Statistical Significance Gate
A new ensemble will be accepted as superior to the frozen baseline **only if** (evaluated strictly on the 2022 validation split and lake-grouped CV — never on 2023):
1. $\text{Val\_MAE}_{\text{ensemble}} < \text{Val\_MAE}_{\text{frozen}} - 10\,\text{m}^2$.
2. The 95% paired bootstrap confidence interval for $\Delta \text{MAE} = \text{MAE}_{\text{ensemble}} - \text{MAE}_{\text{frozen}}$ is strictly negative (upper bound $< 0$).

---

## 4. Classification Ensemble Design (System B)

### 4.1 Ground-Truth Formulation (Canonical Empirical-Uncertainty Schema)
Supervised lake condition classes are defined strictly from consecutive annual area changes and the GLO v1.02 empirical delineation-error scale $U_{\text{RMS}}$ (Rawlins et al., 2026 Eq. 1: $\text{AREA\_UNCERTAINTY}(A) = \text{round}(0.0488 A^{0.537}, 3)\,\text{km}^2$; see `reports/12_uncertainty_semantics_note.md`):
$$U_{\text{RMS}} = \sqrt{\frac{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2}{4}} = 0.5\,\sqrt{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2}$$
*(Descriptive RMS empirical delineation-error scale; multipliers $1.0\times U_{\text{RMS}}$ and $3.0\times U_{\text{RMS}}$ are engineering thresholds, not Gaussian $1\sigma$/$3\sigma$ significance levels.)*

1. **STABLE** ($N=10,760$, $90.81\%$ overall cohort / $98.25\%$ unambiguous subcohort; Train $7,744$, Val $1,440$, Exhausted 2023 Ref $1,576$):
   $$|\Delta A| \le 1.0\,U_{\text{RMS}} \quad \text{OR} \quad |\%\Delta A| \le 5.0\% \quad \text{OR} \quad (\Delta A < 0 \land \Delta A \ge -3.0\,U_{\text{RMS}} \land \%\Delta A \ge -20.0\%)$$
   *(Grouping moderate contraction into `STABLE` is a project-defined convention; annual area observations alone do not prove physical causes.)*
2. **EXPANDING** ($N=77$, $0.65\%$ overall cohort / $0.70\%$ unambiguous subcohort; Train $47$, Val $11$, Exhausted 2023 Ref $19$):
   $$\Delta A > 1.0\,U_{\text{RMS}} \quad \text{AND} \quad \%\Delta A > 5.0\% \quad \text{AND} \quad \Delta A \le 3.0\,U_{\text{RMS}} \quad \text{AND} \quad \%\Delta A \le 20.0\%$$
3. **UNUSUAL_CHANGE** ($N=115$, $0.97\%$ overall cohort / $1.05\%$ unambiguous subcohort; Train $89$, Val $9$, Exhausted 2023 Ref $17$):
   $$(\Delta A > 3.0\,U_{\text{RMS}} \land \%\Delta A > 20.0\%) \quad \text{OR} \quad (\Delta A < -3.0\,U_{\text{RMS}} \land \%\Delta A < -20.0\%)$$
   *(Threshold-defined unusual change: 68 extreme annual expansions and 47 extreme annual contractions; NOT independently verified GLOF or hazard events.)*
4. **AMBIGUOUS** ($N=897$, $7.57\%$ overall cohort; Train $647$, Val $120$, Exhausted 2023 Ref $130$):
   $$\neg (\text{STABLE} \lor \text{EXPANDING} \lor \text{UNUSUAL\_CHANGE})$$
   Label-exclusion state (not a fourth learned physical class); excluded from 3-class supervised training.

Operational alert tiers in `RiskEngine` (`SAFE`, `NORMAL`, `WATCH`, `ELEVATED_ALERT`, `STALE`, `INSUFFICIENT_DATA`) are strictly separated from physical ML labels and must never be used as training targets.

### 4.2 Handling Class Imbalance
With an empirical overall-cohort imbalance ratio of **93.57 : 1** (`STABLE` to `UNUSUAL_CHANGE`; **87.01 : 1** in Train $\le 2021$):
- **Training-Only Loss Weighting:** Class weights are calculated strictly from unambiguous training observations (`Train <=2021`, $N_{\text{train}} = 7,880$):
  $$w_c = \frac{N_{\text{train}}}{3 \cdot N_{c,\text{train}}}$$
  yielding $w_{\text{STABLE}} = 0.339187$, $w_{\text{EXPANDING}} = 55.886525$, $w_{\text{UNUSUAL\_CHANGE}} = 29.513109$.
- **Evaluation Metrics:**
  - Primary Metrics: **Macro-Averaged F1-Score** and **PR-AUC (Precision-Recall AUC)** for `EXPANDING` and `UNUSUAL_CHANGE`.
  - Secondary: Balanced Accuracy.
  - Overall accuracy is **prohibited** as a selection metric (a trivial all-`STABLE` classifier scores **$90.81\%$** on the overall matched cohort and **$98.25\%$** on the unambiguous 3-class subcohort).
- **Prohibition:** Synthetic oversampling (e.g., SMOTE) on validation or test partitions is strictly prohibited.

### 4.3 Probability Calibration & Abstention
- Multi-class predicted probabilities $P(y=c|X)$ will be calibrated using Platt Scaling or `CalibratedClassifierCV` strictly on training OOF folds / 2022 validation split.
- **Abstention Zone:** If $\max_c P(y=c|X) < 0.50$, the system flags `INCONCLUSIVE_PREDICTION` rather than forcing a low-confidence call.

---

## 5. Decision Gates for Phase 9

| Workstream | Recommendation | Justification |
| :--- | :--- | :--- |
| **Phase 9.2: Classification Training** | **GO** | Canonical 3-class + explicit `AMBIGUOUS` exclusion taxonomy closed on $N=11,849$ matched cohort ($N=10,952$ unambiguous; $N=7,880$ train unambiguous, $N=1,460$ val unambiguous) with verified GLO empirical delineation-error scale $U_{\text{RMS}}$, training-only class weights, strict macro-F1 / PR-AUC optimization, and abstention handling for 93.57:1 overall (87.01:1 train) imbalance. |
| **Phase 9.3: Regression Ensemble** | **GO** | 100% data parity established on 36-feature representation. Random Forest and XGBoost ready to train on identical inputs with lake-grouped CV and 2022 validation selection (zero reuse of exhausted 2023 partition). |
