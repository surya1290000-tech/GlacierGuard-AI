"""
GlacierGuard-AI — Core Machine Learning and Remote Sensing Package
===================================================================
A defensible framework for proglacial lake monitoring, multi-source
tabular area-change prediction, and multi-sensor satellite surveillance.
"""

__version__ = "0.2.0"
__author__ = "GlacierGuard-AI Team"

from glacierguard.config import Settings, default_settings
from glacierguard.features.constants import (
    CATEGORICAL_FEATURES,
    GLO_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
)
from glacierguard.features.pipeline import (
    build_gold_dataset,
    build_preprocessor,
    create_temporal_splits,
)
from glacierguard.inference.fused_model import (
    FusedXGBoostPredictor,
    load_fused_xgboost_model,
)
from glacierguard.models import (
    MonitoringAlert,
    ObservationRecord,
    PipelineRun,
    SatelliteProduct,
)
from glacierguard.db import (
    get_latest_monitoring_alerts,
    get_observations,
    get_pipeline_runs,
    get_satellite_products,
    init_db,
    seed_from_qa_csv,
)
from glacierguard.cdse_client import CDSEClient
from glacierguard.pipeline import ProcessingPipeline
from glacierguard.risk_engine import RiskEngine, evaluate_lake_status, map_status_to_user_facing
from glacierguard.worker import MonitoringWorker

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
    "Settings",
    "default_settings",
    "SatelliteProduct",
    "ObservationRecord",
    "MonitoringAlert",
    "PipelineRun",
    "init_db",
    "seed_from_qa_csv",
    "get_latest_monitoring_alerts",
    "get_observations",
    "get_pipeline_runs",
    "get_satellite_products",
    "CDSEClient",
    "ProcessingPipeline",
    "RiskEngine",
    "evaluate_lake_status",
    "map_status_to_user_facing",
    "MonitoringWorker",
]
