import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(25, 30):
    if idx < len(nb["cells"]):
        print(f"=== CELL {idx} ===")
        src = "".join(nb["cells"][idx]["source"])
        print(src.encode("ascii", errors="replace").decode("ascii"))
        print("\n" + "="*50 + "\n")
