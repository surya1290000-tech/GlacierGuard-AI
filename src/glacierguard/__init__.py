"""
GlacierGuard-AI — Core Machine Learning and Remote Sensing Package
===================================================================
A defensible framework for proglacial lake monitoring, multi-source
tabular area-change prediction, and multi-sensor satellite surveillance.
"""

__version__ = "0.1.0"
__author__ = "GlacierGuard-AI Team"

from glacierguard.features.constants import (
    TARGET_COL,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    GLO_FEATURES,
)
from glacierguard.features.pipeline import (
    build_gold_dataset,
    create_temporal_splits,
    build_preprocessor,
)
from glacierguard.inference.fused_model import (
    FusedXGBoostPredictor,
    load_fused_xgboost_model,
)

__all__ = [
    "__version__",
    "TARGET_COL",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "GLO_FEATURES",
    "build_gold_dataset",
    "create_temporal_splits",
    "build_preprocessor",
    "FusedXGBoostPredictor",
    "load_fused_xgboost_model",
]
