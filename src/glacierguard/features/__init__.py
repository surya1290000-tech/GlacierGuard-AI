"""
GlacierGuard-AI Feature Engineering Subpackage
"""

from glacierguard.features.constants import (
    CATEGORICAL_FEATURES,
    FUSED_FEATURES,
    FUSED_NUMERICAL_FEATURES,
    GLO_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    S2_PREFIX,
    TARGET_COL,
    TEST_YEAR,
    TRAIN_MAX_YEAR,
    VAL_YEAR,
)
from glacierguard.features.pipeline import (
    build_gold_dataset,
    build_preprocessor,
    create_temporal_splits,
)

__all__ = [
    "TARGET_COL",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "GLO_FEATURES",
    "FUSED_NUMERICAL_FEATURES",
    "FUSED_FEATURES",
    "S2_PREFIX",
    "TRAIN_MAX_YEAR",
    "VAL_YEAR",
    "TEST_YEAR",
    "RANDOM_STATE",
    "build_gold_dataset",
    "create_temporal_splits",
    "build_preprocessor",
]
