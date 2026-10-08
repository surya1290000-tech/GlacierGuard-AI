import numpy as np
from scipy.ndimage import uniform_filter
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
sys.path.insert(0, str(pathlib.Path('scratch').resolve()))
import test_glo_extraction as tge

def lee_filter(img_linear, size=5, num_looks=1):
    """
    Lee filter for SAR speckle reduction in linear intensity domain.
    img_linear: 2D array of linear power / intensity
    size: window size (odd integer, e.g. 3, 5, 7)
    num_looks: equivalent number of looks (default 1 for 1-look or ~4.4 for GRD)
    """
    img = np.maximum(img_linear, 1e-10)
    # Local mean
    mean = uniform_filter(img, size=size)
    # Local mean square
    sqr_mean = uniform_filter(img**2, size=size)
    # Local variance
    variance = np.maximum(sqr_mean - mean**2, 0)
    
    # Noise variance for multiplicative speckle model: sigma_v^2 = 1 / num_looks
    noise_var = (mean**2) / num_looks
    
    # Weight
    weight = variance / (variance + noise_var + 1e-10)
    weight = np.clip(weight, 0.0, 1.0)
    
    # Filtered output
    filtered = mean + weight * (img - mean)
    return np.maximum(filtered, 1e-10)

# Compare raw vs Lee filter (3x3, 5x5, 7x7) on S1 2026-09-28
vv_lin = tge.vv_linear
vv_raw_db = tge.vv_db
b0 = tge.buffer_masks[0]
b50 = tge.buffer_masks[50]
b100 = tge.buffer_masks[100]

print("=== S1 SPECKLE FILTER COMPARISON (2026-09-28) ===")
filters = {
    'RAW': vv_raw_db,
    'Lee 3x3': 10.0 * np.log10(lee_filter(vv_lin, size=3, num_looks=4.4)),
    'Lee 5x5': 10.0 * np.log10(lee_filter(vv_lin, size=5, num_looks=4.4)),
    'Lee 7x7': 10.0 * np.log10(lee_filter(vv_lin, size=7, num_looks=4.4)),
}

for fname, f_db in filters.items():
    print(f"\n--- Filter: {fname} ---")
    val_in_lake = f_db[b0 & tge.valid_s1]
    print(f"In lake (0m): Mean={np.mean(val_in_lake):.2f} dB, Std={np.std(val_in_lake):.2f} dB, Median={np.median(val_in_lake):.2f} dB")
    for thresh in [-15.0, -14.0, -13.0, -12.0, -11.0]:
        water_px = (val_in_lake < thresh).sum()
        area = water_px * tge.pixel_area_km2
        frac = water_px / len(val_in_lake)
        print(f"  VV < {thresh:.1f} dB: {water_px:5d} px ({frac*100:5.1f}%) -> {area:.4f} km2")
