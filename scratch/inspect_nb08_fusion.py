import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if any(k in src.lower() for k in ["s1+s2_fused", "sensor_fusion", "§18"]):
        print(f"=== Cell {i} ({cell['cell_type']}) ===")
        print(src)
        print("\n" + "="*50 + "\n")
