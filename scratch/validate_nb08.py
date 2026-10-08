import json
import ast
import pathlib

nb_path = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

errors = []
code_cells = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_cells += 1
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append((i, str(e)))

print(f"Total cells: {len(nb['cells'])}, Code cells: {code_cells}")
if errors:
    print(f"Found {len(errors)} syntax errors:")
    for idx, err in errors:
        print(f"  Cell {idx}: {err}")
else:
    print("ALL CODE CELLS HAVE VALID PYTHON SYNTAX!")
