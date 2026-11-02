# GlacierGuard-AI — Phase 9 Validation & Experimental Design
**Author:** Lead ML Engineer  
**Date:** 2026-11-01  
**Target Systems:**
- System A: Regression Ensemble (Predicting `NEXT_AREA_CHANGE` in $\text{km}^2$)
- System B: Classification Ensemble (Predicting `STABLE`, `EXPANDING`, `UNUSUAL_CHANGE`)

---

## 1. Executive Summary & Audit Baseline

GlacierGuard-AI has successfully established a validated **36-feature cross-sensor fused XGBoost regression model** (`reports/models/09_final_fused_xgboost.pkl`), achieving an official test benchmark of:
$$\text{MAE} = 0.004792\,\text{km}^2 \quad (4,792\,\text{m}^2), \quad \text{RMSE} = 0.015079\,\text{km}^2, \quad R^2 = 0.3366$$
on the audited $N=1,742$ matched historical test cohort (2023).

Previous experiment NB11 explored Random Forest vs. XGBoost on **20-feature single-sensor datasets**. It demonstrated that:
1. Equal-weight voting was suboptimal.
2. S1 Random Forest showed minor empirical improvement over XGBoost ($\Delta \text{MAE} = -529\,\text{m}^2$), but its 95% bootstrap confidence interval $[-1183.5, +20.7]\,\text{m}^2$ crossed zero (statistically inconclusive).
3. On optical S2 features, Persistence (zero change) beat all single-sensor ML models.
4. The 20-feature single-sensor models are **not** directly comparable to the 36-feature fused production model.

This document establishes the rigorous validation framework for **Phase 9.2 (Classification)** and **Phase 9.3 (Regression Ensemble)**.

---

## 2. Partitioning Strategy & Leakage Prevention

### 2.1 The Cardinal Rule: No Re-use of the 2023 Historical Benchmark
The 2023 historical test partition ($N=1,742$ matched pairs) was evaluated in NB09 and NB11. **It is an exhausted benchmark partition.** Under scientific integrity rules:
- All feature selection, hyperparameter tuning, loss function calibration, probability thresholding, and ensemble blending weights $\alpha$ must be determined **strictly** on the 2022 validation split or internal training cross-validation folds.
- The 2023 test set may only be evaluated **once** at the conclusion of model development as a confirmatory historical comparison against the frozen NB09 artifact.
- Any model tuning on the 2023 partition constitutes test leakage and is strictly prohibited.

### 2.2 Status of 2024 Data
The GLO v1.02 catalogue contains observations through year 2024. However:
1. For year 2024 observations, `NEXT_YEAR` (2025) is absent from the dataset; hence `NEXT_AREA_CHANGE` is undefined (NaN).
2. The 2024 area values were already consumed to construct the target for the 2023 test partition:
   $$\text{NEXT\_AREA\_CHANGE}(2023) = \text{AREA}(2024) - \text{AREA}(2023)$$
3. **Conclusion:** 2024 is **not** an untouched future holdout. There is no unconsumed future holdout in the current dataset.

### 2.3 Chronological & Spatial Split Design

| Partition | Observation Year | Sample Count (Matched) | Role in Phase 9 |
| :--- | :--- | :--- | :--- |
| **Historical Train** | $2017 \le t \le 2021$ | 8,527 | Base model training; expanding-window OOF generation |
| **Validation Split** | $t = 2022$ | 1,580 | Hyperparameter tuning, blend weight optimization ($\alpha$), probability calibration, decision thresholds |
| **Historical Holdout** | $t = 2023$ | 1,742 | Single-pass confirmation against frozen NB09 baseline |

### 2.4 Spatial Holdout Protocol (Basin Generalization)
To test geographic transferability and ensure models do not overfit to specific regional topography:
- **Primary Basin Partition:**
  - Training / Validation: **Koshi** ($N=7,622$, 64.3%) and **Karnali** ($N=2,845$, 24.0%)
  - Spatial Holdout: **Gandaki** ($N=1,382$, 11.7%)
- **Lake-Grouped Cross-Validation:**
  - Random row-level splitting is strictly prohibited.
  - Multi-fold cross-validation must use `GroupKFold(n_splits=5, groups=GLO_ID)`. All multi-year observations for any single lake remain strictly within either train or validation.

---

## 3. Regression Ensemble Design (System A)

### 3.1 Base Model Architecture
Both base models will train on the **exact same 36-feature matrix** and preprocessing pipeline:
- **Model 1: Random Forest Regressor**
  - Estimators: $n=400$
  - Max features: `'sqrt'`
  - Min samples leaf: 5
- **Model 2: XGBoost Regressor**
  - Estimators: $n=100$ (frozen baseline) / $n=600$ (tuned)
  - Max depth: 4, Learning rate: 0.05
  - Subsample: 0.8, Colsample by tree: 0.7

### 3.2 Combination Candidates
1. **Equal-Weight Voting:** $\hat{y}_{\text{vote}} = 0.5\,\hat{y}_{\text{RF}} + 0.5\,\hat{y}_{\text{XGB}}$
2. **Validation-Optimal Weighted Blend:**
   $$\hat{y}_{\text{blend}}(\alpha) = \alpha\,\hat{y}_{\text{RF}} + (1-\alpha)\,\hat{y}_{\text{XGB}}$$
   where $\alpha \in [0.0, 1.0]$ is selected to minimize $\text{MAE}_{\text{val}}$ on the 2022 validation split.
3. **Temporal Out-of-Fold Stacking:**
   - Expanding-window meta-features generated strictly using past years ($t_{\text{train}} < t$).
   - Meta-learner: Ridge regression with $\alpha=1.0$ constrained to non-negative coefficients.

### 3.3 Statistical Significance Gate
A new ensemble will be accepted as superior to the frozen baseline **only if**:
1. $\text{Val\_MAE}_{\text{ensemble}} < \text{Val\_MAE}_{\text{frozen}} - 10\,\text{m}^2$.
2. The 95% paired bootstrap confidence interval for $\Delta \text{MAE} = \text{MAE}_{\text{ensemble}} - \text{MAE}_{\text{frozen}}$ is strictly negative (upper bound $< 0$).

---

## 4. Classification Ensemble Design (System B)

### 4.1 Ground-Truth Formulation (Canonical Hybrid Schema)
Supervised lake condition classes are defined strictly from physical consecutive changes and GLO measurement uncertainty:
$$\sigma_{\Delta A} = 0.5\,\sqrt{\sigma_{S1,t}^2 + \sigma_{S1,t+1}^2 + \sigma_{S2,t}^2 + \sigma_{S2,t+1}^2}$$

1. **STABLE ($N=10,760$, $90.81\%$ overall cohort; $N=7,744$ train, $N=1,440$ val, $N=1,576$ test):**
   $$|\Delta A| \le 1.0\,\sigma_{\Delta A} \quad \text{OR} \quad |\%\Delta A| \le 5.0\% \quad \text{OR} \quad \text{moderate shrinkage } (-3\sigma \le \Delta A < 0, \%\Delta A \ge -20\%)$$
2. **EXPANDING ($N=77$, $0.65\%$ overall cohort; $N=47$ train, $N=11$ val, $N=19$ test):**
   $$\Delta A > 1.0\,\sigma_{\Delta A} \quad \text{AND} \quad \%\Delta A > 5.0\% \quad \text{AND} \quad \Delta A \le 3.0\,\sigma_{\Delta A} \quad \text{AND} \quad \%\Delta A \le 20.0\%$$
3. **UNUSUAL_CHANGE ($N=115$, $0.97\%$ overall cohort; $N=89$ train, $N=9$ val, $N=17$ test):**
   $$(\Delta A > 3.0\,\sigma_{\Delta A} \text{ and } \%\Delta A > 20.0\%) \quad \text{OR} \quad (\Delta A < -3.0\,\sigma_{\Delta A} \text{ and } \%\Delta A < -20.0\%)$$
4. **AMBIGUOUS ($N=897$, $7.57\%$ overall cohort; $N=647$ train, $N=120$ val, $N=130$ test):**
   $$\neg (\text{STABLE} \lor \text{EXPANDING} \lor \text{UNUSUAL\_CHANGE})$$
   Excluded from 3-class model supervision; reserved for operational quality abstention.

### 4.2 Handling Class Imbalance
With an empirical imbalance ratio of 93.6:1 (Stable to Unusual):
- **Loss Weighting:** Class weights set inversely proportional to frequencies:
  $$w_c = \frac{N}{3 \cdot N_c}$$
- **Evaluation Metrics:**
  - Primary Metric: **Macro-Averaged F1-Score** and **PR-AUC (Precision-Recall AUC)** for `EXPANDING` and `UNUSUAL_CHANGE`.
  - Secondary: Balanced Accuracy.
  - Overall accuracy is **prohibited** as a selection metric (a trivial all-stable classifier scores $90.84\%$).
- **Prohibition:** Synthetic oversampling (e.g. SMOTE) on validation or test sets is strictly prohibited.

### 4.3 Probability Calibration & Abstention
- Multi-class predicted probabilities $P(y=c|X)$ will be calibrated using Platt Scaling (Logistic Regression on validation logits) or CalibratedClassifierCV on temporal OOF folds.
- **Abstention Zone:** If $\max_c P(y=c|X) < 0.50$, the system flags the observation as `INCONCLUSIVE_PREDICTION` rather than forcing a low-confidence call.

---

## 5. Decision Gates for Phase 9

| Workstream | Recommendation | Justification |
| :--- | :--- | :--- |
| **Phase 9.2: Classification Training** | **CONDITIONAL GO** | High feasibility on $N=11,849$ matched cohort with Hybrid uncertainty labels. Requires strict macro-F1 / PR-AUC optimization, class weighting, and abstention handling to handle 21:1 imbalance. |
| **Phase 9.3: Regression Ensemble** | **GO** | 100% data parity established on 36-feature representation. Random Forest and XGBoost ready to train on identical inputs. Blend weights must be chosen strictly on 2022 validation split. |
