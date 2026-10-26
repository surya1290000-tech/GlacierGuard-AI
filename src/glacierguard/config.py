"""
GlacierGuard-AI Configuration and Operational Settings
======================================================
Centralized configuration management for CDSE STAC access,
SQLite persistence, operational risk evaluation rules,
and lake spatial boundaries.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "glacierguard.db"
DEFAULT_FROZEN_MODEL_PATH = REPO_ROOT / "reports" / "models" / "09_final_fused_xgboost.pkl"
DEFAULT_QA_CSV_PATH = REPO_ROOT / "reports" / "10_2_five_lake_monitoring_qa.csv"

# ── Lake Bounding Boxes [min_lon, min_lat, max_lon, max_lat] ─────────────────
# Bounding boxes sized ~5-8 km around known lake perimeters
LAKE_BOUNDING_BOXES: Dict[str, List[float]] = {
    "imja_tsho": [86.90, 27.88, 86.96, 27.92],
    "thulagi": [84.45, 28.46, 84.52, 28.52],
    "tsho_rolpa": [86.44, 27.83, 86.52, 27.89],
    "lower_barun": [87.05, 27.81, 87.12, 27.87],
    "sabai_tsho": [86.58, 27.76, 86.66, 27.82],
}

# ── GLO 2023 Vector Baseline Areas [km²] (NB10.2 verified) ───────────────────
LAKE_BASELINE_AREAS: Dict[str, float] = {
    "imja_tsho": 1.764624,
    "thulagi": 1.015503,
    "tsho_rolpa": 1.663245,
    "lower_barun": 0.419424,
    "sabai_tsho": 0.455666,
}


@dataclass
class LakeConfig:
    """Configuration for an individual monitored glacier lake."""

    lake_id: str
    display_name: str
    glo_id: str
    lat: float
    lon: float
    bbox: List[float]
    baseline_area_km2: float
    utm_epsg: int = 32645
    dam_type: str = ""
    region: str = ""
    elevation_m: int = 0
    sar_physically_reliable: bool = True


@dataclass
class OperationalThresholds:
    """Thresholds governing data freshness and automated risk tiering."""

    freshness_current_days: int = 30
    freshness_aging_days: int = 90
    max_temporal_pair_days: int = 10
    sensor_disagreement_threshold_pct: float = 20.0
    area_change_elevated_pct: float = 20.0
    area_change_watch_pct: float = 10.0


@dataclass
class Settings:
    """GlacierGuard-AI platform settings loaded from environment or defaults."""

    # Persistence
    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("GLACIERGUARD_DB_PATH", str(DEFAULT_DB_PATH))
        )
    )

    # CDSE STAC API endpoints
    stac_api_url: str = field(
        default_factory=lambda: os.getenv(
            "CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1"
        )
    )
    fallback_stac_api_url: str = field(
        default_factory=lambda: os.getenv(
            "CDSE_FALLBACK_STAC_URL",
            "https://catalogue.dataspace.copernicus.eu/stac",
        )
    )
    token_url: str = field(
        default_factory=lambda: os.getenv(
            "CDSE_TOKEN_URL",
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        )
    )

    # CDSE Credentials (optional for catalogue search, required for raw data download)
    client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("CDSE_CLIENT_ID")
    )
    client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("CDSE_CLIENT_SECRET")
    )

    # Collections to poll
    collections: List[str] = field(
        default_factory=lambda: ["sentinel-1-grd", "sentinel-2-l2a"]
    )

    # Polling & Worker
    poll_interval_hours: float = field(
        default_factory=lambda: float(os.getenv("POLL_INTERVAL_HOURS", "6.0"))
    )
    stac_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("STAC_TIMEOUT_SECONDS", "20"))
    )
    stac_max_retries: int = field(
        default_factory=lambda: int(os.getenv("STAC_MAX_RETRIES", "3"))
    )

    # Frozen ML Model Artifact
    model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("GLACIERGUARD_MODEL_PATH", str(DEFAULT_FROZEN_MODEL_PATH))
        )
    )

    # Legacy QA CSV fallback
    qa_csv_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("GLACIERGUARD_QA_CSV", str(DEFAULT_QA_CSV_PATH))
        )
    )

    # Operational thresholds
    thresholds: OperationalThresholds = field(default_factory=OperationalThresholds)

    def get_lake_config(self, lake_id: str) -> LakeConfig:
        """Retrieve complete configuration for a monitored lake."""
        from glacierguard.dashboard_data import LAKE_METADATA

        if lake_id not in LAKE_METADATA:
            raise KeyError(f"Lake '{lake_id}' is not in configured lake metadata.")

        meta = LAKE_METADATA[lake_id]
        bbox = LAKE_BOUNDING_BOXES.get(
            lake_id,
            [meta["lon"] - 0.04, meta["lat"] - 0.03, meta["lon"] + 0.04, meta["lat"] + 0.03],
        )
        baseline = LAKE_BASELINE_AREAS.get(lake_id, 1.0)
        sar_reliable = lake_id != "tsho_rolpa"  # Rolwaling gorge layover causes C-band failure

        return LakeConfig(
            lake_id=lake_id,
            display_name=meta.get("display_name", lake_id),
            glo_id=meta.get("glo_id", ""),
            lat=meta.get("lat", 0.0),
            lon=meta.get("lon", 0.0),
            bbox=bbox,
            baseline_area_km2=baseline,
            utm_epsg=meta.get("utm_epsg", 32645),
            dam_type=meta.get("dam_type", ""),
            region=meta.get("region", ""),
            elevation_m=meta.get("elevation_m", 0),
            sar_physically_reliable=sar_reliable,
        )

    def list_lake_ids(self) -> List[str]:
        """Return canonical list of monitored lake IDs."""
        from glacierguard.dashboard_data import CANONICAL_LAKE_ORDER
        return list(CANONICAL_LAKE_ORDER)


# Global default settings instance
default_settings = Settings()
