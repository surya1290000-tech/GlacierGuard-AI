"""
Raster-Backed Sentinel Observation Extraction and Radiometric Processing
========================================================================
Implements geographically correct, calibrated lake water extraction for
Sentinel-1 GRD and Sentinel-2 L2A observations:

- Resolves actual CDSE STAC asset keys (e.g. B03_20m, B03_10m, B11_20m, SCL_20m, VV).
- Co-registers and reprojects multispectral bands onto a consistent 20m grid.
- Masks and crops strictly to verified lake study area boundaries (0m core prior vs 50m envelope).
- Refined Sentinel-2 SCL cloud, shadow, snow/ice, and no-data screening.
- Validated Sentinel-1 radiometric conversion and localized lake layover assessment.
- Area calculation performed in projected coordinates within the lake ROI.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from glacierguard.satellite.lake_geometry import rasterize_lake_roi
from glacierguard.satellite.morphology import clean_mask
from glacierguard.satellite.optical import (
    assess_optical_usability,
    classify_optical_water,
    compute_mndwi,
)
from glacierguard.satellite.sar import (
    assess_sar_validity,
    classify_sar_water,
    lee_filter,
    linear_to_db,
)


class RasterProcessingError(RuntimeError):
    """Raised when raster processing fails QA, validity, or alignment checks."""


def _open_band(path: Path) -> Tuple[np.ndarray, str, Any, float, float]:
    """Read a GeoTIFF band and return (data, crs, transform, pixel_m2, resolution)."""
    try:
        import rasterio
    except ImportError as exc:
        raise RasterProcessingError("rasterio is required for GeoTIFF processing") from exc

    with rasterio.open(path) as src:
        if src.crs is None or src.crs.is_geographic:
            raise RasterProcessingError(
                "Raster must have a projected CRS for defensible pixel-area calculation"
            )
        data = src.read(1, masked=True)
        if data.ndim != 2:
            raise RasterProcessingError("Expected a single-band 2D raster")

        arr = np.asarray(data.filled(np.nan), dtype=np.float32)

        # Apply raster scale and offset if present in metadata tags
        if src.scales and len(src.scales) > 0 and src.scales[0] not in (None, 1.0):
            scale = float(src.scales[0])
            offset = float(src.offsets[0]) if src.offsets and src.offsets[0] is not None else 0.0
            valid_vals = np.isfinite(arr)
            arr[valid_vals] = arr[valid_vals] * scale + offset

        crs_str = src.crs.to_string()
        transform = src.transform
        pixel_m2 = abs(transform.a * transform.e)
        resolution = abs(transform.a)

        return arr, crs_str, transform, pixel_m2, resolution


def _reproject_to_reference(
    src_data: np.ndarray,
    src_crs: str,
    src_transform: Any,
    dst_crs: str,
    dst_transform: Any,
    dst_shape: Tuple[int, int],
    is_discrete: bool = False,
) -> np.ndarray:
    """Reproject a raster array onto a reference grid using rasterio.warp."""
    from rasterio.warp import reproject, Resampling

    dst_arr = np.full(
        dst_shape,
        fill_value=0.0 if is_discrete else np.nan,
        dtype=np.float32,
    )

    reproject(
        source=src_data,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest if is_discrete else Resampling.bilinear,
    )

    return dst_arr


def _resolve_s2_paths(paths: Dict[str, Path]) -> Tuple[Path, Path, Path]:
    """Resolve actual Sentinel-2 STAC asset keys for Green, SWIR, and SCL."""
    # Priority order for green band (prefer 20m if available to match B11/SCL)
    green_candidates = [
        "B03_20m", "b03_20m", "B03_10m", "b03_10m", "B03", "b03", "green", "B03_60m"
    ]
    swir_candidates = [
        "B11_20m", "b11_20m", "B11", "b11", "swir16", "swir", "B11_60m"
    ]
    scl_candidates = [
        "SCL_20m", "scl_20m", "SCL", "scl", "SCL_60m"
    ]

    green_path: Optional[Path] = None
    for cand in green_candidates:
        if cand in paths:
            green_path = paths[cand]
            break

    swir_path: Optional[Path] = None
    for cand in swir_candidates:
        if cand in paths:
            swir_path = paths[cand]
            break

    scl_path: Optional[Path] = None
    for cand in scl_candidates:
        if cand in paths:
            scl_path = paths[cand]
            break

    missing = []
    if green_path is None:
        missing.append("B03/Green")
    if swir_path is None:
        missing.append("B11/SWIR")
    if scl_path is None:
        missing.append("SCL")

    if missing:
        raise RasterProcessingError(f"Missing Sentinel-2 required bands: {missing}")

    return green_path, swir_path, scl_path


def process_sentinel2(
    paths: Dict[str, Path],
    lake_id: Optional[str] = None,
    buffer_m: float = 50.0,
    min_valid: float = 0.50,
    min_valid_fraction: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process Sentinel-2 multispectral imagery into a verified lake water mask.

    Parameters
    ----------
    paths : Dict[str, Path]
        Dictionary mapping asset keys (e.g. B03_20m, B11_20m, SCL_20m) to local files.
    lake_id : Optional[str]
        Monitored lake identifier for spatial ROI clipping (e.g. 'imja_tsho').
    buffer_m : float, default=50.0
        Expansion search envelope buffer distance in meters.
    min_valid : float, default=0.50
        Minimum fraction of clear valid surface pixels inside lake ROI.
    min_valid_fraction : Optional[float]
        Alias for min_valid.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (water_mask, metrics_dict)
    """
    if min_valid_fraction is not None:
        min_valid = min_valid_fraction
    green_p, swir_p, scl_p = _resolve_s2_paths(paths)

    green, crs_g, trans_g, p_m2_g, res_g = _open_band(green_p)
    swir, crs_s, trans_s, p_m2_s, res_s = _open_band(swir_p)
    scl, crs_c, trans_c, p_m2_c, res_c = _open_band(scl_p)

    # Reference grid: prefer SWIR 20m grid
    ref_shape = swir.shape
    ref_crs = crs_s
    ref_trans = trans_s
    pixel_m2 = p_m2_s
    resolution = res_s

    # Align Green band if transform or dimensions differ
    if green.shape != ref_shape or trans_g != ref_trans or crs_g != ref_crs:
        green = _reproject_to_reference(
            green, crs_g, trans_g, ref_crs, ref_trans, ref_shape, is_discrete=False
        )

    # Align SCL band if transform or dimensions differ (nearest neighbour for discrete codes)
    if scl.shape != ref_shape or trans_c != ref_trans or crs_c != ref_crs:
        scl = _reproject_to_reference(
            scl, crs_c, trans_c, ref_crs, ref_trans, ref_shape, is_discrete=True
        )

    # Scene Classification Layer (SCL) quality screening
    scl_int = np.round(scl).astype(int)
    nodata = np.isin(scl_int, [0, 1])  # 0: NO_DATA, 1: SATURATED_DEFECTIVE
    clouds = np.isin(scl_int, [3, 7, 8, 9, 10])  # shadows, low/med/high clouds, cirrus
    snow = np.isin(scl_int, [11])  # 11: SNOW_OR_ICE

    valid = ~nodata & ~clouds
    valid &= np.isfinite(green) & np.isfinite(swir) & (green > 0) & (swir > 0)

    # Modified Normalized Difference Water Index (MNDWI)
    mndwi = compute_mndwi(green, swir)

    # Geographically correct lake ROI clipping
    if lake_id is not None:
        m0, m_buf, _ = rasterize_lake_roi(
            lake_id=lake_id,
            out_shape=ref_shape,
            transform=ref_trans,
            crs=ref_crs,
            buffer_m=buffer_m,
        )

        if not m_buf.any():
            raise RasterProcessingError(
                f"Lake '{lake_id}' ROI does not intersect the provided raster footprint"
            )

        m0_count = max(int(m0.sum()), 1)
        m_buf_count = max(int(m_buf.sum()), 1)

        lake_cloud_frac = float((clouds & m0).sum() / m0_count)
        lake_snow_frac = float((snow & m0).sum() / m0_count)
        lake_invalid_frac = float((nodata & m0).sum() / m0_count)
        lake_valid_frac = float((valid & m0).sum() / m0_count)

        if not assess_optical_usability(lake_cloud_frac, lake_valid_frac, min_valid=min_valid):
            raise RasterProcessingError(
                f"Optical QA rejected: clear valid fraction ({lake_valid_frac:.1%}) below "
                f"{min_valid:.1%} or cloud cover ({lake_cloud_frac:.1%}) exceeds threshold"
            )

        # Classify water strictly inside the lake ROI; exclude snow and outer areas
        raw_water = (mndwi > 0.0) & valid & (~snow)
        w0 = clean_mask(raw_water & m0)
        w_buf = clean_mask(raw_water & m_buf)

        if not w_buf.any():
            raise RasterProcessingError("No valid water pixels extracted inside lake ROI")

        area_0m_km2 = float(w0.sum() * pixel_m2 / 1e6)
        area_50m_km2 = float(w_buf.sum() * pixel_m2 / 1e6)
        mask = w_buf

        metrics = {
            "area_km2": area_50m_km2,
            "area_0m_km2": area_0m_km2,
            "area_50m_km2": area_50m_km2,
            "valid_frac": lake_valid_frac,
            "cloud_frac": lake_cloud_frac,
            "snow_frac": lake_snow_frac,
            "invalid_frac": lake_invalid_frac,
            "crs": ref_crs,
            "resolution_m": resolution,
            "geometry_source": "GLO_v1.02_2023",
            "roi_buffer_m": buffer_m,
        }
    else:
        # Fallback to whole-raster candidate extraction
        valid_frac = float(valid.mean())
        cloud_frac = float(clouds.mean())
        snow_frac = float(snow.mean())

        if not assess_optical_usability(cloud_frac, valid_frac, min_valid=min_valid):
            raise RasterProcessingError("Optical QA rejected insufficient clear valid pixels")

        water = clean_mask((mndwi > 0.0) & valid & (~snow))
        if not water.any():
            raise RasterProcessingError("No valid water pixels extracted")

        mask = water
        area_km2 = float(mask.sum() * pixel_m2 / 1e6)
        metrics = {
            "area_km2": area_km2,
            "area_0m_km2": area_km2,
            "area_50m_km2": area_km2,
            "valid_frac": valid_frac,
            "cloud_frac": cloud_frac,
            "snow_frac": snow_frac,
            "crs": ref_crs,
            "resolution_m": resolution,
            "roi_type": "whole_raster_candidate_region",
        }

    return mask, metrics


def _resolve_s1_paths(paths: Dict[str, Path]) -> Path:
    """Resolve actual Sentinel-1 VV STAC asset key."""
    vv_candidates = ["VV", "vv", "s1-grd-vv", "vv-grd", "measurement", "amplitude-vv"]
    for cand in vv_candidates:
        if cand in paths:
            return paths[cand]
    raise RasterProcessingError("Missing Sentinel-1 VV polarization")


def process_sentinel1(
    paths: Dict[str, Path],
    lake_id: Optional[str] = None,
    sar_reliable: bool = True,
    buffer_m: float = 50.0,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process Sentinel-1 GRD SAR imagery into a calibrated lake water mask.

    Parameters
    ----------
    paths : Dict[str, Path]
        Dictionary mapping asset keys (e.g. 'VV') to local file paths.
    lake_id : Optional[str]
        Monitored lake identifier.
    sar_reliable : bool, default=True
        Physical reliability flag (False for steep gorge layover like Tsho Rolpa).
    buffer_m : float, default=50.0
        Expansion search envelope buffer distance in meters.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (water_mask, metrics_dict)
    """
    vv_path = _resolve_s1_paths(paths)

    if lake_id == "tsho_rolpa" or not sar_reliable:
        raise RasterProcessingError(
            "SAR extraction requires review for this lake geometry "
            "(mountain layover/shadow in steep Rolwaling gorge)"
        )

    vv, crs, transform, pixel_m2, resolution = _open_band(vv_path)

    # Check radiometric representation
    valid_finite = vv[np.isfinite(vv)]
    if len(valid_finite) == 0:
        raise RasterProcessingError("SAR VV raster contains no finite data")

    med_val = float(np.median(valid_finite))
    min_val = float(np.min(valid_finite))

    if min_val < 0.0 and med_val < 5.0 and med_val > -50.0:
        # Already calibrated in decibels (dB); physical linear intensity cannot be negative
        vv_db = vv
    elif med_val > 100.0:
        # Raw uncalibrated integer digital numbers (DN) amplitude
        raise RasterProcessingError(
            "Calibration metadata or physical validity cannot be established for raw SAR DN amplitude"
        )
    else:
        # Calibrated linear backscatter intensity -> apply Lee speckle filter and dB conversion
        vv_db = linear_to_db(lee_filter(vv))

    valid_s1 = (vv_db > -35.0) & (vv_db < 5.0) & np.isfinite(vv_db)

    if lake_id is not None:
        m0, m_buf, _ = rasterize_lake_roi(
            lake_id=lake_id,
            out_shape=vv.shape,
            transform=transform,
            crs=crs,
            buffer_m=buffer_m,
        )

        if not m_buf.any():
            raise RasterProcessingError(
                f"Lake '{lake_id}' ROI does not intersect SAR raster footprint"
            )

        # Screen for SAR geometric layover strictly inside the lake core polygon
        vv_lake = vv_db[m0]
        is_valid, median_vv, p10_vv = assess_sar_validity(vv_lake)
        if not is_valid:
            raise RasterProcessingError(
                f"SAR geometric or wind-quality screening failed: lake median backscatter "
                f"({median_vv:.1f} dB) exceeds threshold (-8.0 dB)"
            )

        w0 = clean_mask((vv_db < -12.0) & valid_s1 & m0)
        w_buf = clean_mask((vv_db < -12.0) & valid_s1 & m_buf)

        if not w_buf.any():
            raise RasterProcessingError("No SAR water pixels extracted inside lake ROI")

        area_0m_km2 = float(w0.sum() * pixel_m2 / 1e6)
        area_50m_km2 = float(w_buf.sum() * pixel_m2 / 1e6)
        mask = w_buf

        metrics = {
            "area_km2": area_50m_km2,
            "area_0m_km2": area_0m_km2,
            "area_50m_km2": area_50m_km2,
            "valid_frac": float(np.isfinite(vv_lake).mean()),
            "vv_median_db": median_vv,
            "crs": crs,
            "resolution_m": resolution,
            "geometry_source": "GLO_v1.02_2023",
            "roi_buffer_m": buffer_m,
        }
    else:
        valid, median_vv, _ = assess_sar_validity(vv_db)
        if not valid:
            raise RasterProcessingError("SAR geometric or wind-quality screening failed")

        water = clean_mask((vv_db < -12.0) & valid_s1)
        if not water.any():
            raise RasterProcessingError("No SAR water pixels extracted")

        mask = water
        area_km2 = float(mask.sum() * pixel_m2 / 1e6)
        metrics = {
            "area_km2": area_km2,
            "area_0m_km2": area_km2,
            "area_50m_km2": area_km2,
            "valid_frac": float(np.isfinite(vv).mean()),
            "vv_median_db": median_vv,
            "crs": crs,
            "resolution_m": resolution,
            "roi_type": "whole_raster_candidate_region",
        }

    return mask, metrics
