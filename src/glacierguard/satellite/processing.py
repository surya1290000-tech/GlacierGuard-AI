"""Raster-backed Sentinel observation extraction.

The functions intentionally accept already-accessed local files.  That makes
the science path testable without CDSE while keeping network access separate.
"""
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np

from glacierguard.satellite.optical import assess_optical_usability, classify_optical_water, compute_mndwi, screen_scl_clouds
from glacierguard.satellite.sar import assess_sar_validity, classify_sar_water, lee_filter, linear_to_db


class RasterProcessingError(RuntimeError):
    pass


def _open_band(path: Path):
    try:
        import rasterio
    except ImportError as exc:
        raise RasterProcessingError("rasterio is required for GeoTIFF processing") from exc
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.is_geographic:
            raise RasterProcessingError("Raster must have a projected CRS for defensible pixel-area calculation")
        data = src.read(1, masked=True)
        if data.ndim != 2:
            raise RasterProcessingError("Expected a single-band raster")
        return np.asarray(data.filled(np.nan), dtype=float), src.crs.to_string(), abs(src.transform.a * src.transform.e), abs(src.transform.a)


def process_sentinel2(paths: Dict[str, Path], min_valid: float = .70) -> Tuple[np.ndarray, Dict[str, float | str]]:
    missing = {"B03", "B11", "SCL"} - set(paths)
    if missing:
        raise RasterProcessingError(f"Missing Sentinel-2 required bands: {sorted(missing)}")
    green, crs, pixel_m2, resolution = _open_band(paths["B03"])
    swir, crs2, pixel_m2b, _ = _open_band(paths["B11"])
    scl, crs3, _, _ = _open_band(paths["SCL"])
    if green.shape != swir.shape or green.shape != scl.shape or crs != crs2 or crs != crs3:
        raise RasterProcessingError("Sentinel-2 bands are not co-registered")
    cloud, valid = screen_scl_clouds(scl)
    valid &= np.isfinite(green) & np.isfinite(swir)
    valid_frac = float(valid.mean())
    cloud_frac = float(cloud.mean())
    if not assess_optical_usability(cloud_frac, valid_frac, min_valid=min_valid):
        raise RasterProcessingError("Optical QA rejected insufficient clear valid pixels")
    mask = classify_optical_water(compute_mndwi(green, swir), valid)
    if not mask.any():
        raise RasterProcessingError("No valid water pixels extracted")
    return mask, {"area_km2": float(mask.sum() * pixel_m2 / 1e6), "valid_frac": valid_frac, "cloud_frac": cloud_frac, "crs": crs, "resolution_m": resolution}


def process_sentinel1(paths: Dict[str, Path], sar_reliable: bool = True) -> Tuple[np.ndarray, Dict[str, float | str]]:
    if "VV" not in paths:
        raise RasterProcessingError("Missing Sentinel-1 VV polarization")
    if not sar_reliable:
        raise RasterProcessingError("SAR extraction requires review for this lake geometry")
    vv, crs, pixel_m2, resolution = _open_band(paths["VV"])
    # GRD backscatter fixtures/products must already be calibrated linear sigma0.
    vv_db = linear_to_db(lee_filter(vv))
    valid, median, _ = assess_sar_validity(vv_db)
    if not valid:
        raise RasterProcessingError("SAR geometric or wind-quality screening failed")
    mask = classify_sar_water(vv_db)
    if not mask.any():
        raise RasterProcessingError("No SAR water pixels extracted")
    return mask, {"area_km2": float(mask.sum() * pixel_m2 / 1e6), "valid_frac": float(np.isfinite(vv).mean()), "vv_median_db": median, "crs": crs, "resolution_m": resolution}
