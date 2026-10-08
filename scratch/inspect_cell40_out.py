import json

with open("notebooks/08_multisource_advanced_modeling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][40]
print("=== CELL 40 OUTPUT ===")
for out in cell.get("outputs", []):
    if out.get("output_type") == "stream":
        print("".join(out.get("text", [])))
