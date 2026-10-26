"""
GlacierGuard-AI Inference Subpackage
"""

from glacierguard.inference.fused_model import (
    DEFAULT_MODEL_PATH,
    FusedXGBoostPredictor,
    load_fused_xgboost_model,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "FusedXGBoostPredictor",
    "load_fused_xgboost_model",
]
