# GLO `AREA_UNCERTAINTY` Semantics — Scientific Evidence Note

**Phase:** 9.1.4.1 / 9.1.4.2 — Scientific Uncertainty and Label Taxonomy Closure
**Date:** 2026-11-18
**Author:** Lead Geospatial ML Engineer
**Branch:** `feature/phase9-dual-ensemble`
**Primary Dataset:** Glacial Lake Observatory (GLO) v1.02 (2017–2024), Zenodo DOI `10.5281/zenodo.17802333`
**Companion Publication:** Rawlins, L. D., Watson, C. S., Bhambri, R., Khadka, N., & Chand, M. B. (2026). *Glacial Lake Observatory (GLO): annual dataset of glacial lakes in Nepal and transboundary catchments (2017–2024)*. Earth System Science Data, 18, 5143–5165. `https://doi.org/10.5194/essd-18-5143-2026`

---

## 1. Executive Finding

`AREA_UNCERTAINTY` in GLO v1.02 is **NOT a one-sigma ($1\sigma$) Gaussian standard deviation** and **NOT a parametric confidence interval**.

By inspecting the official GLO v1.02 README, the companion *Earth System Science Data* publication (Rawlins et al., 2026), and all 40,423 polygon records in the local GeoPackage files, we establish with 100.0000% numerical verification that `AREA_UNCERTAINTY` is a **deterministic, area-dependent empirical delineation-error estimate** derived from a power-law regression fitted to Sentinel-2 deep-learning vs. manual digitization discrepancies ($n = 895$ outlines) and applied identically to both Sentinel-2 and Sentinel-1 annual polygons.

Consequently:
1. The previous statistical interpretation of `0.5 * sqrt(u_S1_t^2 + u_S1_t1^2 + u_S2_t^2 + u_S2_t1^2)` as Gaussian variance propagation of independent $1\sigma$ standard deviations is **unsupported by the data source and withdrawn**.
2. Classification thresholds **must not** be described as "$1\sigma$" or "$3\sigma$" statistical significance levels.
3. The combined scale $U_{\text{RMS}} = \sqrt{\frac{1}{4}(u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2)}$ is retained solely as a **descriptive Root-Mean-Square (RMS) empirical delineation-error scale** requiring no distributional or independence assumption, with thresholds expressed as engineering multiples **$1.0 \times U_{\text{RMS}}$** and **$3.0 \times U_{\text{RMS}}$**.

---

## 2. Source Definition and Empirical Verification

### 2.1 Dataset Metadata (`raw/GLO/README (1).md`)
In the GLO v1.02 README (Section *Attribute Information → Attributes in All Lakes datasets*, line 66), `AREA_UNCERTAINTY` is defined verbatim as:

> `AREA_UNCERTAINTY` — *"Estimated uncertainty in lake area"* — Units: `km²`

The README provides no statistical distribution, confidence level, or formula for `AREA_UNCERTAINTY`. The GeoPackage internal metadata tables (`gpkg_metadata`, `gpkg_contents`) contain only QGIS CRS definitions (`ESRI:102025`) and no field-level statistical descriptions.

*(Note: Line 123 of the README mentions a `>2σ` standardized-residual outlier threshold and 1,000-iteration bootstrap 95% confidence intervals. Those apply to long-term `EXPANSION_RATE` time-series regression in the Unique Lakes layers, NOT to single-observation `AREA_UNCERTAINTY`.)*

### 2.2 Companion Publication Definition (Rawlins et al., 2026, ESSD)
In Rawlins et al. (2026, *Earth System Science Data*, 18, 5143–5165, Equation 1), the authors quantified delineation error by comparing Sentinel-2 DeepLabV3 lake polygons against manually digitized outlines ($n = 895$ outlines from 239 lakes) and fitted a log-log linear power-law model of percentage area error as a function of lake area $A$ (in $\text{km}^2$):

$$\text{error}\% = 4.88 \times A^{-0.463}$$

Converting percentage error to absolute area uncertainty in $\text{km}^2$ gives:

$$\text{AREA\_UNCERTAINTY}(A) = \text{round}\!\left(\frac{4.88}{100} \times A^{1 - 0.463},\; 3\right) = \text{round}\!\left(0.0488 \times A^{0.537},\; 3\right) \quad [\text{km}^2]$$

### 2.3 Exact Verification Against Local GLO v1.02 GeoPackages
Testing $\text{round}(0.0488 \times A^{0.537}, 3)$ directly against every observation in the local GLO v1.02 files yields:

| Source GeoPackage | Total Records | Exact Match to $\text{round}(0.0488 A^{0.537}, 3)$ | Max $|U - 0.0488 A^{0.537}|$ (unrounded) |
| :--- | ---: | ---: | ---: |
| `S1_20172024_NTB_GLOID_v1.02.gpkg` | 18,129 | **18,129 / 18,129 (100.0000%)** | `0.000500 km²` |
| `S2_20172024_NTB_GLOID_v1.02.gpkg` | 22,294 | **22,294 / 22,294 (100.0000%)** | `0.000500 km²` |
| **Combined GLO v1.02** | **40,423** | **40,423 / 40,423 (100.0000%)** | **`0.000500 km²`** |

---

## 3. Independence Justification and Observed Serial Dependence Diagnostics

Standard Gaussian variance propagation for $\Delta A = \frac{1}{2}(A_{S1,t+1} - A_{S1,t}) + \frac{1}{2}(A_{S2,t+1} - A_{S2,t})$ requires all four observation errors $(\epsilon_{S1,t}, \epsilon_{S1,t+1}, \epsilon_{S2,t}, \epsilon_{S2,t+1})$ to be independent, zero-mean random variables with known standard deviations.

### 3.1 Why Independence Cannot Be Assumed From Source Documentation
1. **Sentinel-1 vs. Sentinel-2 area errors:** Both sensors observe the same lake in the same year under shared site-specific confounders (partial snow/ice cover, debris, turbid shorelines, terrain shadow), and GLO v1.02 assigns `AREA_UNCERTAINTY` using the identical Sentinel-2-fitted power law rather than independent sensor-specific error variance estimates.
2. **Consecutive annual observations ($t$ vs. $t+1$):** Source documentation does not establish temporal error independence; repeated observations of the same lake share persistent topography, viewing geometry, and model characteristics.
3. **Current vs. next-year uncertainty values:** Because $u_t = \text{round}(0.0488 A_t^{0.537}, 3)$ and $u_{t+1} = \text{round}(0.0488 A_{t+1}^{0.537}, 3)$ are deterministic functions of nearly equal consecutive areas ($A_{t+1} \approx A_t$ for most lakes), the four inputs $(u_{S1,t}, u_{S1,t+1}, u_{S2,t}, u_{S2,t+1})$ are nearly collinear.

### 3.2 Reproducible Observed Serial Dependence Diagnostics
Using `glacierguard.classification_labels.compute_observed_serial_dependence_diagnostics()` on the canonical matched cohort (`N = 11,849`, sorted by `GLO_ID`, `AREA_YEAR`), restricted to consecutive matched lake-year pairs (`AREA_YEAR - prev_AREA_YEAR == 1`, `N = 9,392` pairs):

- **Lag-1 autocorrelation of observed S1–S2 area discrepancy** $(A_{S1,t} - A_{S2,t})$: **`r = +0.5489`** (`N = 9,392`).
- **Lag-1 autocorrelation of observed fused annual area change** (`FUSED_TARGET`): **`r = -0.2700`** (`N = 9,392`).
- **Median observed cross-sensor absolute discrepancy** $|A_{S1,t} - A_{S2,t}|$: `0.004769 km²` (`0.6695×` baseline mean $\bar{u}_t$, `N = 11,849`).

### 3.3 Precise Interpretation and Scope Limits of These Diagnostics
Observed serial dependence diagnostics indicate that the derived cross-sensor discrepancies and annual changes are not temporally independent as observed quantities. These diagnostics do not independently identify the covariance of the underlying measurement errors.

Specifically:
1. The `+0.5489` statistic describes serial persistence in the **observed S1–S2 area discrepancy**. It is **not** a direct estimate of cross-sensor measurement-error covariance.
2. The `-0.2700` statistic describes serial dependence in **consecutive observed fused annual area changes**. Consecutive differences share an endpoint ($A_t$), and physical lake dynamics may also contribute.
3. These statistics alone do **not** establish the complete covariance structure of the four delineation-error inputs $(\epsilon_{S1,t}, \epsilon_{S1,t+1}, \epsilon_{S2,t}, \epsilon_{S2,t+1})$.
4. Because no independent ground-truth reference areas are available across the full 2017–2024 cohort, these diagnostics do **not** directly test underlying measurement-error independence against reference observations.

---

## 4. Defensible Alternative Without Invented Distributional or Covariance Assumptions

Because we do not invent a Gaussian distribution or an unverified cross-sensor/temporal covariance matrix, any combination of the four documented empirical error estimates $(u_{S1,t}, u_{S1,t+1}, u_{S2,t}, u_{S2,t+1})$ must be framed strictly as a **descriptive empirical error scale**.

We evaluated four candidate non-parametric scales on the full matched cohort ($N = 11,849$):

| Candidate Scale Formula | Mathematical Interpretation | Median ($\text{km}^2$) | Ratio to $U_{\text{RMS}}$ (Med [P05, P95]) | STABLE / EXPANDING / UNUSUAL / AMBIGUOUS | Agreement with $U_{\text{RMS}}$ |
| :--- | :--- | ---: | :--- | :--- | ---: |
| **$U_{\text{RMS}} = \sqrt{\frac{1}{4}\sum_{i=1}^4 u_i^2} = 0.5\sqrt{u_{S1,t}^2+u_{S1,t+1}^2+u_{S2,t}^2+u_{S2,t+1}^2}$** | **Descriptive RMS ($L_2$ mean) of the 4 empirical delineation-error estimates** | `0.007000` | `1.000 [1.000, 1.000]` | `10,760 / 77 / 115 / 897` | **100.00%** |
| $\bar{u} = \frac{1}{4}(u_{S1,t}+u_{S1,t+1}+u_{S2,t}+u_{S2,t+1})$ | Arithmetic mean ($L_1$ mean) of the 4 empirical delineation-error estimates | `0.007000` | `0.997 [0.948, 1.000]` | `10,716 / 78 / 132 / 923` | 99.49% |
| $\bar{u}_t = \frac{1}{2}(u_{S1,t}+u_{S2,t})$ | Baseline year-$t$ mean empirical delineation error | `0.007000` | `0.999 [0.870, 1.077]` | `10,693 / 87 / 151 / 918` | 98.05% |
| $U_{\text{additive}} = \frac{1}{2}\sum_{i=1}^4 u_i = 2\bar{u}$ | Linear additive sum of the four halved error estimates | `0.014000` | `1.994 [1.896, 2.000]` | `11,519 / 12 / 25 / 293` | 92.81% |

### Engineering Decision
We retain **$U_{\text{RMS}}$** as the canonical descriptive empirical delineation-error scale:

$$U_{\text{RMS}} = \sqrt{\frac{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2}{4}} = 0.5\,\sqrt{u_{S1,t}^2 + u_{S1,t+1}^2 + u_{S2,t}^2 + u_{S2,t+1}^2} \quad [\text{km}^2]$$

**Justification:**
1. **No distributional or independence assumption required:** Algebraically, $U_{\text{RMS}}$ is simply the quadratic mean (RMS) of the four documented empirical delineation-error estimates, bounded strictly by $\min_i u_i \le \bar{u} \le U_{\text{RMS}} \le \max_i u_i$.
2. **Close agreement with the arithmetic mean ($\bar{u}$):** Median ratio is `0.9972` with **99.49%** label agreement across $N = 11,849$ observations, while slightly weighting larger-error endpoints when S1/S2 areas diverge or when large transitions occur.
3. **Practical supervised-learning feasibility vs. $U_{\text{additive}} = 2\bar{u}$:** Under $U_{\text{additive}} = 2\bar{u}$, `EXPANDING` collapses to 12 observations and `UNUSUAL_CHANGE` to 25 across the entire 7-year matched cohort, leaving insufficient minority-class support for multi-year supervised training and validation. *(Note: This engineering choice is based on scale interpretability and minority-class support, NOT on inferring error-cancellation from observed autocorrelation statistics.)*
4. **Terminology closure:** All "$1\sigma$" and "$3\sigma$" statistical significance labels are replaced with **$1.0 \times U_{\text{RMS}}$** and **$3.0 \times U_{\text{RMS}}$**. Legacy code/column names (`FUSED_SIGMA_DELTA`, `compute_combined_sigma_delta`) are retained strictly as backward-compatible aliases documented as representing $U_{\text{RMS}}$.

---

## 5. Epistemic Classification Summary

| Category | Statements |
| :--- | :--- |
| **Confirmed Scientific Facts** | (1) GLO v1.02 `AREA_UNCERTAINTY` equals $\text{round}(0.0488 A^{0.537}, 3)\,\text{km}^2$ (Rawlins et al., 2026 Eq. 1) for 40,423/40,423 (100.0000%) S1 and S2 records. (2) It is an empirical Sentinel-2 DL-vs-manual power-law delineation-error estimate, NOT a $1\sigma$ Gaussian SD or parametric confidence interval. (3) The identical S2-fitted formula is applied to S1 despite lower S1 validation F1 (0.82 vs. S2 0.92). (4) 90.07% of matched annual changes satisfy $|\Delta A| \le 1.0 \times U_{\text{RMS}}$. |
| **Observed Empirical Diagnostics** | (1) On consecutive matched lake-year pairs (`N = 9,392`), lag-1 autocorrelation of observed S1–S2 area discrepancy is `r = +0.5489`. (2) Lag-1 autocorrelation of observed consecutive fused annual changes (`FUSED_TARGET`) is `r = -0.2700`. (3) Median observed $|A_{S1,t} - A_{S2,t}| = 0.004769\,\text{km}^2$ (`0.6695×` baseline $\bar{u}_t$). These describe observed derived quantities and do not independently identify the covariance structure of underlying measurement errors. |
| **Project-Defined Engineering Conventions** | (1) Retaining $U_{\text{RMS}}$ strictly as a descriptive RMS empirical delineation-error scale without Gaussian or independence claims. (2) Threshold multipliers $1.0\times U_{\text{RMS}}$ and $3.0\times U_{\text{RMS}}$, relative boundaries $\pm 5.0\%$ and $\pm 20.0\%$, and grouping moderate annual contraction into `STABLE`. (3) Keeping canonical reconciled counts (`10,760 / 77 / 115 / 897`) and `AMBIGUOUS` exclusion state unchanged. |
| **Unresolved Scientific Questions** | (1) Separate empirical area-error power-law parameters for Sentinel-1 SAR are not published in GLO v1.02. (2) The exact original-space coverage probability of Rawlins et al.'s power-law curve and the true cross-sensor/temporal measurement-error covariance matrix cannot be identified without independent multi-year reference delineations. (3) Genuinely unconsumed post-2024 holdout evaluation requires a future GLO release containing 2025+ observations. |
