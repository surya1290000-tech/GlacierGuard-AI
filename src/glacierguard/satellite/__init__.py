"""
GlacierGuard-AI Satellite and QA Subpackage
"""

from glacierguard.satellite.morphology import clean_mask
from glacierguard.satellite.optical import (
    DEFAULT_CLOUD_CLASSES,
    assess_optical_usability,
    classify_optical_water,
    compute_mndwi,
    screen_scl_clouds,
)
from glacierguard.satellite.qa import (
    assign_lake_status,
    evaluate_cross_sensor_agreement,
    match_temporal_pair,
)
from glacierguard.satellite.sar import (
    assess_sar_validity,
    classify_sar_water,
    lee_filter,
    linear_to_db,
)

__all__ = [
    "lee_filter",
    "linear_to_db",
    "classify_sar_water",
    "assess_sar_validity",
    "compute_mndwi",
    "screen_scl_clouds",
    "classify_optical_water",
    "assess_optical_usability",
    "DEFAULT_CLOUD_CLASSES",
    "clean_mask",
    "match_temporal_pair",
    "evaluate_cross_sensor_agreement",
    "assign_lake_status",
]
