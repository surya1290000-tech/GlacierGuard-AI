import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [15, 19, 25]:
    if idx < len(nb["cells"]):
        print(f"=== CELL {idx} ===")
        print("".join(nb["cells"][idx]["source"]))
        print("\n" + "="*50 + "\n")
