import json
import pathlib

nb_path = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for idx in [61, 62, 64]:
    c = nb["cells"][idx]
    src = "".join(c["source"])
    print(f"==================== CELL {idx} ====================")
    print(src)
