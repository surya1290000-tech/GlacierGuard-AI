import pandas as pd
import pathlib

reports = pathlib.Path("reports")

print("=== 08_ablation_results.csv ===")
abl_df = pd.read_csv(reports / "08_ablation_results.csv")
print(abl_df.to_string())

print("\n=== 08_sensor_fusion_results.csv ===")
sf_df = pd.read_csv(reports / "08_sensor_fusion_results.csv")
print(sf_df.to_string())

print("\n=== 08_error_analysis.csv ===")
ea_df = pd.read_csv(reports / "08_error_analysis.csv")
print(ea_df.to_string())
