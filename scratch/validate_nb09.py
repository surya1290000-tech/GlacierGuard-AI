import json
import ast

with open('notebooks/09_final_model_validation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
code_count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code_count += 1
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append((i, str(e)))

print(f'Total cells: {len(nb["cells"])} | Code cells: {code_count}')
if errors:
    print(f'Syntax errors found in {len(errors)} cells:')
    for idx, err in errors:
        print(f'  Cell {idx}: {err}')
else:
    print('ALL CODE CELLS HAVE VALID PYTHON SYNTAX!')
