"""
Patch NB08: fix ROOT path detection and add cell IDs.
"""
import json, uuid, pathlib

NB_PATH = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Replacement source for the paths cell
new_paths_src = (
    "# == §3.2  Paths (works from notebooks/ CWD or project root CWD) ==\n"
    "def _find_root():\n"
    "    \"\"\"Probe common execution CWD patterns for the project root.\"\"\"\n"
    "    import pathlib as _pl\n"
    "    marker = _pl.Path('raw') / 'GLO' / 'S1_20172024_NTB_GLOID_v1.02.gpkg'\n"
    "    for candidate in [_pl.Path('..'), _pl.Path('.'), _pl.Path('/Workspace')]:\n"
    "        if (candidate / marker).exists():\n"
    "            return candidate.resolve()\n"
    "    raise FileNotFoundError(\n"
    "        'Cannot locate project root. '\n"
    "        'Searched: .., ., /Workspace — none contain raw/GLO GeoPackage.'\n"
    "    )\n"
    "\n"
    "ROOT        = _find_root()\n"
    "GLO_DIR     = ROOT / 'raw' / 'GLO'\n"
    "ERA5_DIR    = ROOT / 'raw' / 'ERA5'\n"
    "CHIRPS_DIR  = ROOT / 'raw' / 'CHIRPS'\n"
    "RGI_DIR     = ROOT / 'raw' / 'RGI'\n"
    "REPORTS_DIR = ROOT / 'reports'\n"
    "FIGURES_DIR = ROOT / 'reports' / 'figures'\n"
    "MODELS_DIR  = ROOT / 'reports' / 'models'\n"
    "\n"
    "for d in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "assert GLO_DIR.exists(),    f'GLO data missing: {GLO_DIR}'\n"
    "assert ERA5_DIR.exists(),   f'ERA5 data missing: {ERA5_DIR}'\n"
    "assert CHIRPS_DIR.exists(), f'CHIRPS data missing: {CHIRPS_DIR}'\n"
    "assert RGI_DIR.exists(),    f'RGI data missing: {RGI_DIR}'\n"
    "\n"
    "print(f'Project root : {ROOT}')\n"
    "print('All required directories verified.')\n"
    "\n"
    "DARK_BG    = '#0d1117'\n"
    "ACCENT     = '#58a6ff'\n"
    "TEXT_COLOR = '#e6edf3'\n"
    "GRID_COLOR = '#21262d'\n"
)

replaced = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "GLO_DIR" in src and "ROOT" in src and "pathlib.Path('.').resolve()" in src:
        nb["cells"][i]["source"] = new_paths_src
        replaced += 1
        print(f"Patched paths cell at index {i}")
        break

assert replaced == 1, f"Expected 1 replacement, got {replaced}"

# Add cell IDs (required by nbformat >= 4.5)
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = str(uuid.uuid4())[:8]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved: {NB_PATH}")
print("All cell IDs added.")
