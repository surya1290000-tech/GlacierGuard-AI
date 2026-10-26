"""
Morphological Post-Processing
=============================
Reproduces mask cleaning routines from NB10.1 and NB10.2:
- Small island noise suppression
- Enclosed interior hole filling
"""

import numpy as np
from scipy import ndimage


def clean_mask(
    mask: np.ndarray,
    min_size: int = 10,
    max_hole: int = 5,
) -> np.ndarray:
    """Filter small isolated artifacts and fill small enclosed holes in water masks.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array of extracted water pixels.
    min_size : int, default=10
        Minimum connected component size (pixels) to retain.
    max_hole : int, default=5
        Maximum enclosed background hole size (pixels) to fill.

    Returns
    -------
    np.ndarray
        Cleaned 2D boolean mask.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    labeled, n_feat = ndimage.label(mask_bool)
    if n_feat == 0:
        return mask_bool.copy()

    sizes = ndimage.sum(mask_bool, labeled, range(1, n_feat + 1))
    clean = mask_bool.copy()
    for i, s in enumerate(sizes, start=1):
        if s < min_size:
            clean[labeled == i] = False

    inv = ~clean
    labeled_inv, n_inv = ndimage.label(inv)
    if n_inv == 0:
        return clean

    # Determine border labels so we never fill outer background
    border = set(
        np.unique(
            np.concatenate([
                labeled_inv[0, :],
                labeled_inv[-1, :],
                labeled_inv[:, 0],
                labeled_inv[:, -1],
            ])
        )
    )
    sizes_inv = ndimage.sum(inv, labeled_inv, range(1, n_inv + 1))
    for i, s in enumerate(sizes_inv, start=1):
        if i not in border and s <= max_hole:
            clean[labeled_inv == i] = True

    return clean
