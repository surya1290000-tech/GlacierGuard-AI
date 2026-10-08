"""
Executes all code cells of notebooks/07_environmental_forcing.ipynb in order.
Captures stdout/stderr, records execution_count, updates notebook JSON.
"""
import json
import io
import sys
import traceback
import pathlib
import time

nb_path = pathlib.Path('notebooks/07_environmental_forcing.ipynb')
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
        print(f"ERROR in cell {idx} (exec {exec_count}):")
        print(cell_err[2])
        print("Continuing execution of remaining cells...")
        # Don't break — continue to see all errors

    cell['outputs'] = outputs
    n_lines = len(captured_text.splitlines()) if captured_text else 0
    print(f"Cell {idx} (exec {exec_count}): {n_lines} output lines" +
          (" [ERROR]" if cell_err else ""), flush=True)

t1 = time.time()
print(f"\nExecution finished in {t1 - t0:.2f}s. Total errors: {len(errors)}", flush=True)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Saved notebook: {nb_path} ({nb_path.stat().st_size / 1024:.1f} KB)")

if errors:
    print(f"\nSummary of {len(errors)} errors:")
    for idx, ec, tb in errors:
        print(f"  Cell {idx} (exec {ec}): {tb.splitlines()[-1]}")
