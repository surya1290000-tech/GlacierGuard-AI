"""
Optical Multispectral Processing and Cloud Screening
=====================================================
Reproduces Sentinel-2 L2A optical processing from NB10.1 and NB10.2:
- Modified Normalized Difference Water Index (MNDWI = (Green - SWIR) / (Green + SWIR))
- Scene Classification Layer (SCL) cloud/shadow screening
- Usable scene quality gating
"""

from typing import Sequence, Tuple
import numpy as np

# SCL cloud classes according to ESA Sentinel-2 L2A specification:
# 0: NO_DATA, 1: SATURATED_OR_DEFECTIVE, 3: CLOUD_SHADOWS,
# 8: CLOUD_MEDIUM_PROBABILITY, 9: CLOUD_HIGH_PROBABILITY, 10: THIN_CIRRUS
DEFAULT_CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)


def compute_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute Modified Normalized Difference Water Index (MNDWI).

    Parameters
    ----------
    green : np.ndarray
        Sentinel-2 Band 3 (Green) reflectance array.
    swir : np.ndarray
        Sentinel-2 Band 11 (SWIR-1) reflectance array.
    eps : float, default=1e-10
        Small numerical stability epsilon.

    Returns
    -------
    np.ndarray
        MNDWI index array with values bounded in [-1, +1].
    """
    g = np.asarray(green, dtype=float)
    s = np.asarray(swir, dtype=float)
    denominator = g + s + eps
    return (g - s) / denominator


def screen_scl_clouds(
    scl: np.ndarray,
    cloud_classes: Sequence[int] = DEFAULT_CLOUD_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify cloud/shadow pixels and generate a valid observation mask.

    Parameters
    ----------
    scl : np.ndarray
        2D array of Sentinel-2 L2A Scene Classification Layer codes.
    cloud_classes : Sequence[int]
        Class codes that represent non-usable cloud or shadow.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cloud_mask, valid_mask) as boolean numpy arrays.
    """
    scl_rounded = np.round(scl).astype(int)
    cloud_mask = np.isin(scl_rounded, cloud_classes)
    valid_mask = ~cloud_mask
    return cloud_mask, valid_mask


def classify_optical_water(
    mndwi: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Extract water pixels using calibrated MNDWI threshold and cloud screening.

    Parameters
    ----------
    mndwi : np.ndarray
        MNDWI array.
    valid_mask : np.ndarray
        Boolean array where True indicates clear, cloud-free surface.
    threshold : float, default=0.0
        Calibrated water threshold from NB10.1.

    Returns
    -------
    np.ndarray
        Boolean array where True indicates classified water.
    """
    return (mndwi > threshold) & valid_mask


def assess_optical_usability(
    cloud_frac: float,
    valid_frac: float,
    min_valid: float = 0.70,
    max_cloud: float = 0.30,
) -> bool:
    """Check if an optical acquisition meets minimum QA criteria for lake measurement.

    Parameters
    ----------
    cloud_frac : float
        Fraction of lake area covered by cloud or cloud shadow [0.0 to 1.0].
    valid_frac : float
        Fraction of clear valid observations [0.0 to 1.0].
    min_valid : float, default=0.70
        Minimum clear valid fraction required (70%).
    max_cloud : float, default=0.30
        Maximum tolerable cloud fraction (30%).

    Returns
    -------
    bool
        True if the scene is usable for defensible area measurement.
    """
    return (valid_frac >= min_valid) and (cloud_frac <= max_cloud)
