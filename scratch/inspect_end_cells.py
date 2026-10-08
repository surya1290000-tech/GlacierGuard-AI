import json
import ast
import pathlib

nb_path = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for i in range(58, len(nb["cells"])):
    c = nb["cells"][i]
    src = "".join(c["source"])
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    lines = src.strip().split("\n")
    print("\n".join(lines[:6]))
    print("...")
