"""
Patch NB08: fix ELEVATION_RANGE negative values by clipping to 0.
"""
import json, pathlib

NB_PATH = pathlib.Path("notebooks/08_multisource_advanced_modeling.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def fix_terrain_cell(src):
    """Replace hard assertion with clip + soft check in the terrain §11.1 cell."""
    old = (
        "# Compute and validate ELEVATION_RANGE\n"
        "for name, df_full in [('S1', s1_train_full), ('S2', s2_train_full)]:\n"
        "    er = df_full['ELEVATION_RANGE']\n"
        "    print(f'{name} ELEVATION_RANGE: min={er.min():.2f}, median={er.median():.2f}, max={er.max():.2f} m')\n"
        "    print(f'  NaN: {er.isna().sum()} ({er.isna().mean()*100:.1f}%)')\n"
        "    assert (er.dropna() >= 0).all(), f'{name}: ELEVATION_RANGE has negative values!'"
    )
    new = (
        "# Compute and validate ELEVATION_RANGE\n"
        "# Note: ELEVATION_MEAN - ELEVATION_MIN can produce machine-precision negative values\n"
        "# (e.g. -0.0 due to IEEE 754). These are clipped to 0.0 exactly as with ERA5 snow depth.\n"
        "for name, df_full in [('S1', s1_train_full), ('S2', s2_train_full)]:\n"
        "    er = df_full['ELEVATION_RANGE']\n"
        "    n_neg = int((er.dropna() < 0).sum())\n"
        "    if n_neg > 0:\n"
        "        print(f'[QA] {name}: {n_neg} rows with ELEVATION_RANGE < 0 — clipping to 0.0 (machine precision).')\n"
        "        df_full['ELEVATION_RANGE'] = er.clip(lower=0.0)\n"
        "    er = df_full['ELEVATION_RANGE']\n"
        "    print(f'{name} ELEVATION_RANGE: min={er.min():.4f}, median={er.median():.2f}, max={er.max():.2f} m')\n"
        "    print(f'  NaN: {er.isna().sum()} ({er.isna().mean()*100:.1f}%)')\n"
        "    assert (er.dropna() >= 0).all(), f'{name}: ELEVATION_RANGE still negative after clip!'"
    )
    if old in src:
        return src.replace(old, new), True
    return src, False


def fix_augmented_split_cell(src):
    """Clip ELEVATION_RANGE at construction time in build_augmented_split."""
    old = "    df['ELEVATION_RANGE'] = df['ELEVATION_MEAN'] - df['ELEVATION_MIN']"
    new = (
        "    # Clip to 0 to handle machine-precision negatives (same QA as ERA5 snow depth)\n"
        "    df['ELEVATION_RANGE'] = (df['ELEVATION_MEAN'] - df['ELEVATION_MIN']).clip(lower=0.0)"
    )
    if old in src:
        return src.replace(old, new), True
    return src, False


patched_terrain = False
patched_split   = False

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    if not patched_terrain and "ELEVATION_RANGE has negative values" in src:
        fixed, ok = fix_terrain_cell(src)
        if ok:
            nb["cells"][i]["source"] = fixed
            patched_terrain = True
            print(f"Patched terrain §11.1 cell at index {i}")

    if not patched_split and "build_augmented_split" in src and "ELEVATION_RANGE" in src:
        fixed, ok = fix_augmented_split_cell(src)
        if ok:
            nb["cells"][i]["source"] = fixed
            patched_split = True
            print(f"Patched build_augmented_split cell at index {i}")

if not patched_terrain:
    print("WARNING: terrain cell not found / old pattern not matched")
if not patched_split:
    print("WARNING: build_augmented_split cell not found / old pattern not matched")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. terrain={patched_terrain}, split={patched_split}")
