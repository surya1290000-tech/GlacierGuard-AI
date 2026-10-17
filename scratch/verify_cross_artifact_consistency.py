import json
import pandas as pd
from pathlib import Path

reconciled_csv = pd.read_csv("reports/11_reconciled_matched_test_metrics.csv")
comparison_csv = pd.read_csv("reports/11_ensemble_regression_comparison.csv")
paired_csv     = pd.read_csv("reports/11_ensemble_paired_comparison.csv")

with open("reports/11_ensemble_experiment_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open("notebooks/11_rf_xgboost_ensemble_benchmark.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

roadmap_text = Path("docs/IMPLEMENTATION_ROADMAP.md").read_text(encoding="utf-8")
readme_text  = Path("README.md").read_text(encoding="utf-8")

print("Checking Cross-Artifact Value Consistency...")

# 1. Fused XGBoost on Matched Cohort (NB09 baseline)
fused_row = reconciled_csv[reconciled_csv["Family"].str.startswith("Fused_Baseline")].iloc[0]
assert round(fused_row["MAE"], 6) == 0.004792, f"Fused MAE mismatch: {fused_row['MAE']}"
assert round(fused_row["RMSE"], 6) == 0.015079, f"Fused RMSE mismatch: {fused_row['RMSE']}"
assert round(fused_row["R2"], 4) == 0.3366, f"Fused R2 mismatch: {fused_row['R2']}"
assert fused_row["N_Test"] == 1742, f"Fused N_Test mismatch: {fused_row['N_Test']}"

# Check in manifest
fused_manifest = manifest["sample_reconciliation"]["reconciled_matched_test_results_n1742"]["fused_xgboost_nb09"]
assert fused_manifest["test_mae"] == 0.004792
assert fused_manifest["test_rmse"] == 0.015079
assert fused_manifest["test_r2"] == 0.3366

# Check in README and roadmap
assert "0.004792" in readme_text
assert "0.004792" in roadmap_text
assert "1742" in readme_text
assert "1742" in roadmap_text

# 2. S1 RandomForest on Matched Cohort
s1_rf_matched = reconciled_csv[(reconciled_csv["Family"] == "S1_Candidate") & (reconciled_csv["Model"] == "RandomForest")].iloc[0]
assert round(s1_rf_matched["MAE"], 6) == 0.006756
assert round(s1_rf_matched["Consensus_MAE"], 6) == 0.005412

# Check in manifest
s1_rf_manifest = manifest["sample_reconciliation"]["reconciled_matched_test_results_n1742"]["s1_random_forest"]
assert s1_rf_manifest["test_mae"] == 0.006756
assert s1_rf_manifest["consensus_mae"] == 0.005412

# 3. S2 Persistence on Matched Cohort
s2_pers_matched = reconciled_csv[(reconciled_csv["Family"] == "S2_Candidate") & (reconciled_csv["Model"] == "Persistence")].iloc[0]
assert round(s2_pers_matched["MAE"], 6) == 0.005082
assert round(s2_pers_matched["Consensus_MAE"], 6) == 0.004982

# Check in manifest
s2_pers_manifest = manifest["sample_reconciliation"]["reconciled_matched_test_results_n1742"]["s2_persistence"]
assert s2_pers_manifest["test_mae"] == 0.005082
assert s2_pers_manifest["consensus_mae"] == 0.004982

# 4. Paired Bootstrap S1 RF vs XGBoost
s1_paired = paired_csv[(paired_csv["Sensor"] == "S1") & (paired_csv["Candidate"] == "RandomForest")].iloc[0]
assert round(s1_paired["Delta_TestMAE"], 6) == -0.000529
assert round(s1_paired["CI_95_Lower"], 6) == -0.001184
assert round(s1_paired["CI_95_Upper"], 6) == 0.000021

# Unit conversion check: 0.000529 km² = 529 m², not 0.529 m²
assert abs(round(s1_paired["Delta_TestMAE"] * 1e6, 1)) == 529.1

# Check unit conversion in manifest
rf_status = manifest["candidate_model_status"]["11_best_s1_pkl"]["paired_bootstrap_test_vs_xgboost"]
assert rf_status["delta_test_mae_km2"] == -0.000529
assert rf_status["delta_test_mae_m2"] == -529.1
assert rf_status["ci_95_lower_km2"] == -0.001184
assert rf_status["ci_95_upper_km2"] == 0.000021
assert "529 m²" in rf_status["unit_conversion"]
assert "0.529 m²" in rf_status["unit_conversion"]

# Check in NB11 cell 6
cell_6_text = "".join(nb["cells"][6]["source"])
assert "529" in cell_6_text
assert "0.529" in cell_6_text
assert "-0.001184" in cell_6_text
assert "+0.000021" in cell_6_text
assert "matched lake-year observations" in cell_6_text
assert "reused historical benchmark" in cell_6_text
assert "best observed candidate" in cell_6_text

# Check in roadmap and README
assert "529" in roadmap_text
assert "0.529" in roadmap_text
assert "-0.001184" in roadmap_text
assert "+0.000021" in roadmap_text
assert "matched lake-year observations" in roadmap_text
assert "reused historical benchmark" in roadmap_text
assert "best observed candidate" in roadmap_text

assert "529" in readme_text
assert "0.529" in readme_text
assert "-0.001184" in readme_text
assert "+0.000021" in readme_text
assert "matched lake-year observations" in readme_text
assert "reused historical benchmark" in readme_text
assert "best observed candidate" in readme_text

print("ALL VALUE CONSISTENCY CHECKS PASSED PERFECTLY!")
