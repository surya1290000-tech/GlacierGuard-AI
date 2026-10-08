"""
Extract RGI 7.0 Region 15 (South Asia East) glacier data from OGGM global attributes CSV.
The CSV contains centroid coordinates and area for all glaciers globally.
We filter to Region 15 and compute distances to lakes.
"""
import pandas as pd
import pathlib
import requests
import time
import io

RGI_DIR = pathlib.Path("raw/RGI")
RGI_DIR.mkdir(parents=True, exist_ok=True)
RGI7_CSV = RGI_DIR / "RGI2000-v7.0-G-global-attributes.csv"
RGI7_R15 = RGI_DIR / "rgi7_region15_glaciers.csv"

# Download the global CSV if not present
if not RGI7_CSV.exists():
    url = "https://cluster.klima.uni-bremen.de/~oggm/rgi/RGI2000-v7.0-G-global-attributes.csv"
    print(f"Downloading RGI 7.0 global attributes ({url})...")
    t0 = time.time()
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(RGI7_CSV, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print(f"  Downloaded in {time.time()-t0:.1f}s ({RGI7_CSV.stat().st_size/1e6:.1f} MB)")
else:
    print(f"RGI 7.0 global CSV already exists: {RGI7_CSV.stat().st_size/1e6:.1f} MB")

# Load and filter to Region 15
print("\nLoading RGI 7.0 global attributes...")
t0 = time.time()
df = pd.read_csv(RGI7_CSV, low_memory=False)
print(f"  Loaded {len(df)} glaciers in {time.time()-t0:.1f}s")
print(f"  Columns: {list(df.columns[:20])}")

# Find the region column
region_col = None
for c in df.columns:
    if "region" in c.lower() or "o1" in c.lower() or "o1region" in c.lower():
        unique_vals = df[c].unique()
        if len(unique_vals) < 30:
            print(f"  Candidate region column '{c}': {sorted(unique_vals)[:20]}")
            if 15 in unique_vals or "15" in [str(x) for x in unique_vals]:
                region_col = c
                break

# Also check rgi_id for region prefix
if region_col is None:
    id_col = None
    for c in ["rgi_id", "RGI_Id", "rgi7_id", "RGIId"]:
        if c in df.columns:
            id_col = c
            break
    if id_col:
        sample = df[id_col].head()
        print(f"  RGI ID column '{id_col}' samples: {list(sample)}")
        # RGI IDs typically have format "RGI2000-v7.0-G-15-xxxxx"
        r15_mask = df[id_col].str.contains("-15-", na=False)
        r15 = df[r15_mask]
        print(f"  Region 15 glaciers (by ID pattern): {len(r15)}")
    else:
        print("  No RGI ID column found. Columns:", list(df.columns))

# If we found region col, filter
if region_col:
    r15 = df[df[region_col].astype(str) == "15"]
    print(f"\n  Region 15 glaciers: {len(r15)}")
elif "r15_mask" in dir():
    pass  # already filtered above
else:
    print("  Cannot identify Region 15. Dumping first 5 rows:")
    print(df.head().to_string())
    import sys
    sys.exit(1)

# Find coordinate and area columns
coord_cols = {}
for c in r15.columns:
    cl = c.lower()
    if "cenlon" in cl or "cenlng" in cl or "center_lon" in cl:
        coord_cols["lon"] = c
    elif "cenlat" in cl or "center_lat" in cl:
        coord_cols["lat"] = c
    elif cl == "area_km2" or cl == "area":
        coord_cols["area"] = c

print(f"  Coordinate columns: {coord_cols}")

if "lat" in coord_cols and "lon" in coord_cols:
    lat_col, lon_col = coord_cols["lat"], coord_cols["lon"]
    print(f"  Lat range: {r15[lat_col].min():.3f} to {r15[lat_col].max():.3f}")
    print(f"  Lon range: {r15[lon_col].min():.3f} to {r15[lon_col].max():.3f}")
    area_col = coord_cols.get("area")
    if area_col:
        print(f"  Total area: {r15[area_col].sum():.1f} km²")
        print(f"  Mean area: {r15[area_col].mean():.3f} km²")

    # Save Region 15 subset
    r15.to_csv(RGI7_R15, index=False)
    print(f"\n  Saved Region 15: {RGI7_R15} ({r15.shape})")
    print(f"  RGI 7.0 Region 15 extraction complete.")
else:
    print("  Could not find coordinate columns. Available:", list(r15.columns))
