import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 34 ===")
print("".join(nb["cells"][34]["source"]).encode("ascii", errors="replace").decode("ascii"))
