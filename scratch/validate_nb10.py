import json, ast

with open('notebooks/10_nrt_lake_monitoring.ipynb', encoding='utf-8') as f:
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
            errors.append((i, str(e), src[:120]))

print(f'Total cells: {len(nb["cells"])} | Code cells: {code_count}')
if errors:
    print(f'Syntax errors in {len(errors)} cells:')
    for idx, err, snip in errors:
        print(f'  Cell {idx}: {err}')
        print(f'    Source: {snip[:80]}')
else:
    print('ALL CODE CELLS HAVE VALID PYTHON SYNTAX!')
    # Also check credential layer is present
    cred_found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if '_CRED_FILE' in src and '_resolve_credentials' in src:
                print('Credential layer: PRESENT')
                cred_found = True
                break
    if not cred_found:
        print('WARNING: Credential layer not found in any cell!')
