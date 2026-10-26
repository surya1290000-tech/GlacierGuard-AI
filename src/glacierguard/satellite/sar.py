"""
SAR Radar Processing and Physical Feasibility
==============================================
Reproduces Sentinel-1 C-band processing from NB10.1 and NB10.2:
- Lee 5x5 speckle filtering on linear amplitude
- Backscatter calibration to decibels (dB)
- Valid water classification thresholding
- Physical validity screening (topographic layover and wind chop detection)
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter


def lee_filter(
    img_linear: np.ndarray,
    size: int = 5,
    num_looks: float = 4.4,
) -> np.ndarray:
    """Apply standard Lee speckle filter on linear SAR backscatter intensity.

    Parameters
    ----------
    img_linear : np.ndarray
        2D array of linear radar backscatter.
    size : int, default=5
        Filter window size (size x size).
    num_looks : float, default=4.4
        Equivalent number of looks for Sentinel-1 GRD.

    Returns
    -------
    np.ndarray
        Speckle-reduced linear intensity array.
    """
    img = np.maximum(img_linear, 1e-10)
    mean = uniform_filter(img, size=size)
    sqr_mean = uniform_filter(img ** 2, size=size)
    variance = np.maximum(sqr_mean - mean ** 2, 0.0)
    noise_var = (mean ** 2) / num_looks
    weight = np.clip(variance / (variance + noise_var + 1e-10), 0.0, 1.0)
    filtered = mean + weight * (img - mean)
    return np.maximum(filtered, 1e-10)


def linear_to_db(img_filtered: np.ndarray) -> np.ndarray:
    """Convert linear SAR intensity to decibels (dB)."""
    return 10.0 * np.log10(np.maximum(img_filtered, 1e-10))


def classify_sar_water(
    vv_db: np.ndarray,
    threshold_db: float = -12.0,
    min_valid_db: float = -35.0,
    max_valid_db: float = 5.0,
) -> np.ndarray:
    """Classify water pixels from calibrated VV decibel backscatter.

    Parameters
    ----------
    vv_db : np.ndarray
        2D array of filtered backscatter in dB.
    threshold_db : float, default=-12.0
        Calibrated calm open water threshold from NB10.1.
    min_valid_db : float, default=-35.0
        Sensor noise floor threshold.
    max_valid_db : float, default=5.0
        Sensor saturation threshold.

    Returns
    -------
    np.ndarray
        Boolean mask where True denotes classified water.
    """
    valid = (vv_db > min_valid_db) & (vv_db < max_valid_db) & np.isfinite(vv_db)
    return (vv_db < threshold_db) & valid


def assess_sar_validity(
    vv_lake_pixels: np.ndarray,
    layover_threshold_db: float = -8.0,
) -> Tuple[bool, float, float]:
    """Screen for SAR geometric failure (radar layover / severe katabatic wind chop).

    When a lake is situated in a steep narrow east-west gorge (e.g., Tsho Rolpa),
    topographic radar layover elevates the median VV backscatter above calm water
    thresholds (often to -5.2 dB). This function flags such scenes to prevent
    spurious reports of lake disappearance.

    Parameters
    ----------
    vv_lake_pixels : np.ndarray
        1D or 2D array of backscatter values strictly inside the lake prior polygon.
    layover_threshold_db : float, default=-8.0
        Threshold above which water is obscured by radar geometry or wind chop.

    Returns
    -------
    Tuple[bool, float, float]
        (is_physically_valid, median_vv_db, p10_vv_db)
    """
    valid_vals = vv_lake_pixels[np.isfinite(vv_lake_pixels)]
    if len(valid_vals) == 0:
        return False, np.nan, np.nan

    median_vv = float(np.median(valid_vals))
    p10_vv = float(np.percentile(valid_vals, 10))
    is_valid = median_vv <= layover_threshold_db
    return is_valid, median_vv, p10_vv
