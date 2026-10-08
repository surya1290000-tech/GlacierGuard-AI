"""
Patch NB08 §27.2 figures cell: fix f-string newline syntax errors.
Two literal newlines were embedded inside f-strings when the builder
wrote the cell source as a Python string — they become invalid syntax
at notebook execution time. Replace with string concatenation.
"""
import json, pathlib

NB_PATH = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

fixed_count = 0

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    # Fix 1: f-string with embedded newline in xticklabels
    # Bad:  f"{r['Config']}\n{r['Model']}"
    # (where \n is a real newline character, not escaped)
    BAD1  = "ax.set_xticklabels([f\"{r['Config']}\n{r['Model']}\" for _, r in fusion_df.iterrows()], fontsize=8)"
    GOOD1 = "ax.set_xticklabels([r['Config'] + '\\n' + r['Model'] for _, r in fusion_df.iterrows()], fontsize=8)"

    # Fix 2: f-string with embedded newline in ax.text
    # Bad:  f'Mean={np.mean(resid):.4f}\nStd={np.std(resid):.4f}'
    BAD2  = "ax.text(0.97, 0.95, f'Mean={np.mean(resid):.4f}\nStd={np.std(resid):.4f}',"
    GOOD2 = "ax.text(0.97, 0.95, 'Mean=%.4f\\nStd=%.4f' % (np.mean(resid), np.std(resid)),"

    changed = False
    if BAD1 in src:
        src = src.replace(BAD1, GOOD1)
        changed = True
        print(f"  Fixed xticklabels f-string in cell {i}")
    if BAD2 in src:
        src = src.replace(BAD2, GOOD2)
        changed = True
        print(f"  Fixed ax.text f-string in cell {i}")

    if changed:
        nb["cells"][i]["source"] = src
        fixed_count += 1

print(f"Total cells patched: {fixed_count}")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved: {NB_PATH}")
