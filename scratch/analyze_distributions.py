import numpy as np
import io, json, pathlib, rasterio
from scipy import ndimage

# Load the saved or fetched arrays from the previous step
# Let's inspect vv_db and mndwi distributions inside bmask_0m, bmask_50m, bmask_100m
# We can re-run with statistics and quantiles
import sys
sys.path.insert(0, str(pathlib.Path('.').resolve()))
sys.path.insert(0, str(pathlib.Path('scratch').resolve()))
import test_glo_extraction as tge

b0 = tge.buffer_masks[0]
b50 = tge.buffer_masks[50]
b100 = tge.buffer_masks[100]

print("=== S1 VV dB inside GLO 0m polygon ===")
vv_in_lake = tge.vv_db[b0 & tge.valid_s1]
print(f"Count: {len(vv_in_lake)}")
print(f"Mean: {np.mean(vv_in_lake):.2f} dB, Std: {np.std(vv_in_lake):.2f} dB")
print(f"Min: {np.min(vv_in_lake):.2f}, 10%: {np.percentile(vv_in_lake, 10):.2f}, 25%: {np.percentile(vv_in_lake, 25):.2f}")
print(f"Median: {np.median(vv_in_lake):.2f}, 75%: {np.percentile(vv_in_lake, 75):.2f}, 90%: {np.percentile(vv_in_lake, 90):.2f}, Max: {np.max(vv_in_lake):.2f}")

for thresh in [-18, -17, -16, -15, -14, -13, -12, -11, -10]:
    cnt = (vv_in_lake < thresh).sum()
    area = cnt * tge.pixel_area_km2
    print(f"VV < {thresh:3d} dB: {cnt:5d} px ({cnt/len(vv_in_lake)*100:5.1f}%) -> {area:.4f} km2")

print("\n=== S2 MNDWI & NDWI inside GLO 0m polygon ===")
mndwi_in_lake = tge.mndwi[b0 & tge.valid_s2]
ndwi_in_lake = tge.ndwi[b0 & tge.valid_s2]
print(f"MNDWI Mean: {np.mean(mndwi_in_lake):.3f}, Median: {np.median(mndwi_in_lake):.3f}, 10%: {np.percentile(mndwi_in_lake, 10):.3f}, 90%: {np.percentile(mndwi_in_lake, 90):.3f}")
print(f"NDWI  Mean: {np.mean(ndwi_in_lake):.3f}, Median: {np.median(ndwi_in_lake):.3f}, 10%: {np.percentile(ndwi_in_lake, 10):.3f}, 90%: {np.percentile(ndwi_in_lake, 90):.3f}")

for thresh in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    cnt = (mndwi_in_lake > thresh).sum()
    area = cnt * tge.pixel_area_km2
    print(f"MNDWI > {thresh:4.1f}: {cnt:5d} px ({cnt/len(mndwi_in_lake)*100:5.1f}%) -> {area:.4f} km2")

print("\n=== S2 MNDWI in Buffer 50m - Buffer 0m (the 50m annulus around lake) ===")
annulus_50 = (b50 & ~b0) & tge.valid_s2
mndwi_annulus = tge.mndwi[annulus_50]
print(f"Annulus pixels: {len(mndwi_annulus)}")
print(f"MNDWI Mean: {np.mean(mndwi_annulus):.3f}, Median: {np.median(mndwi_annulus):.3f}")
for thresh in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    cnt = (mndwi_annulus > thresh).sum()
    print(f"Annulus MNDWI > {thresh:4.1f}: {cnt:5d} / {len(mndwi_annulus)} px ({cnt/len(mndwi_annulus)*100:5.1f}%)")
