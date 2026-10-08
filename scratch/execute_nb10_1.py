"""
Executes all code cells of notebooks/10_1_imja_extent_calibration.ipynb in order.
Captures stdout/stderr, records execution_count, updates notebook JSON.
"""
import json
import io
import sys
import traceback
import pathlib
import time

nb_path = pathlib.Path('notebooks/10_1_imja_extent_calibration.ipynb')
nb = json.load(open(nb_path, encoding='utf-8'))

glob_env = {}
exec_count = 0
errors = []

t0 = time.time()
print(f"Starting execution of {nb_path}...")

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    exec_count += 1
    code_text = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture = io.StringIO()
    sys.stdout = capture
    sys.stderr = capture

    cell_err = None
    try:
        exec(code_text, glob_env)
    except Exception as e:
        cell_err = (idx, exec_count, traceback.format_exc())
        errors.append(cell_err)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    captured_text = capture.getvalue()
    cell['execution_count'] = exec_count

    outputs = []
    if captured_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": captured_text.splitlines(keepends=True)
        })
    if cell_err:
        outputs.append({
            "ename": "Exception",
            "evalue": str(cell_err[2].splitlines()[-1]),
            "output_type": "error",
            "traceback": cell_err[2].splitlines(keepends=True)
        })
        print(f"  [!] ERROR in cell {idx} (code #{exec_count}):")
        print(cell_err[2])
    else:
        print(f"  [OK] Cell {idx:2d} (code #{exec_count:2d}) finished in {time.time()-t0:.1f}s")

    cell['outputs'] = outputs

json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1)
print(f"\nExecution finished in {time.time()-t0:.1f}s. Total errors: {len(errors)}")
if errors:
    print("Execution FAILED with errors.")
    sys.exit(1)
else:
    print("Notebook executed and updated successfully with all cell outputs!")
