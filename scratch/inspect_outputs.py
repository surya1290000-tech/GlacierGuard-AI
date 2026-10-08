import json
import pathlib

nb_path = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for idx in [59, 64]:
    cell = nb["cells"][idx]
    print(f"==================== OUTPUT OF CELL {idx} ====================")
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            print("".join(out.get("text", [])))
        elif "data" in out and "text/plain" in out["data"]:
            print("".join(out["data"]["text/plain"]))
