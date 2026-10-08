import pathlib, os, sys

# 1. Check installed packages
print("=== KEY PACKAGES ===")
for pkg in ["sklearn", "xgboost", "lightgbm", "catboost", "xarray", "netCDF4", "rasterio", "cdsapi", "shap"]:
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "installed")
        print(f"  {pkg}: {v}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")

# 2. Check CDS API config
print("\n=== CDS API ===")
cds_file = pathlib.Path.home() / ".cdsapirc"
print(f"  ~/.cdsapirc exists: {cds_file.exists()}")
print(f"  CDSAPI_URL env: {os.environ.get('CDSAPI_URL', 'not set')}")
cds_key = os.environ.get("CDSAPI_KEY")
print(f"  CDSAPI_KEY env: {'set' if cds_key else 'not set'}")

# 3. Check raw data
print("\n=== RAW DATA ===")
for d in ["raw/GLO", "raw/ERA5", "raw/CHIRPS", "raw/RGI"]:
    p = pathlib.Path(d)
    if p.exists():
        files = [f for f in p.iterdir() if f.is_file()]
        print(f"  {d}/: {len(files)} files")
        for f in files:
            sz = f.stat().st_size / 1e6
            print(f"    {f.name} ({sz:.2f} MB)")
    else:
        print(f"  {d}/: MISSING")

# 4. Check reports
print("\n=== REPORTS ===")
for f in sorted(pathlib.Path("reports").glob("*.csv")):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
figs = list(pathlib.Path("reports/figures").glob("*.png"))
print(f"  Figures: {len(figs)}")

# 5. Check notebooks
print("\n=== NOTEBOOKS ===")
for f in sorted(pathlib.Path("notebooks").glob("*.ipynb")):
    sz = f.stat().st_size / 1024
    print(f"  {f.name} ({sz:.1f} KB)")
