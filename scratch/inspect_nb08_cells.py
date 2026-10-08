import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(38, 48):
    if i < len(nb["cells"]):
        cell = nb["cells"][i]
        print(f"=== Cell {i} ({cell['cell_type']}) ===")
        src = "".join(cell["source"])
        print(src[:1000])
        print("\n" + "="*50 + "\n")
