"""
GlacierGuard-AI Domain Models
=============================
Typed dataclasses representing satellite products, lake observations,
pipeline executions, and operational risk alerts.
"""

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SatelliteProduct:
    """Satellite product discovered from CDSE STAC catalogue."""

    product_id: str
    collection: str  # e.g. "sentinel-1-grd", "sentinel-2-l2a"
    acquisition_date: str  # ISO 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ'
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    lake_id: Optional[str] = None
    cloud_cover_pct: Optional[float] = None
    platform: Optional[str] = None  # e.g. "SENTINEL-1A", "SENTINEL-2B"
    orbit_direction: Optional[str] = None  # e.g. "ASCENDING", "DESCENDING"
    discovered_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert product to serializable dictionary."""
        return {
            "product_id": self.product_id,
            "collection": self.collection,
            "acquisition_date": self.acquisition_date,
            "bbox": self.bbox,
            "lake_id": self.lake_id,
            "cloud_cover_pct": self.cloud_cover_pct,
            "platform": self.platform,
            "orbit_direction": self.orbit_direction,
            "discovered_at": self.discovered_at,
            "metadata": self.metadata,
        }


@dataclass
class ObservationRecord:
    """Surface area extraction observation derived from a satellite product."""

    observation_id: str
    lake_id: str
    sensor: str  # "SENTINEL-1" or "SENTINEL-2"
    acquisition_date: str  # 'YYYY-MM-DD'
    product_id: Optional[str] = None
    area_0m_km2: Optional[float] = None
    area_50m_km2: Optional[float] = None
    valid_frac: Optional[float] = None
    cloud_frac: Optional[float] = None
    vv_median_db: Optional[float] = None
    created_at: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "observation_id": self.observation_id,
            "lake_id": self.lake_id,
            "sensor": self.sensor,
            "acquisition_date": self.acquisition_date,
            "product_id": self.product_id,
            "area_0m_km2": self.area_0m_km2,
            "area_50m_km2": self.area_50m_km2,
            "valid_frac": self.valid_frac,
            "cloud_frac": self.cloud_frac,
            "vv_median_db": self.vv_median_db,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass
class MonitoringAlert:
    """Operational monitoring status evaluation and risk assessment."""

    alert_id: str
    lake_id: str
    timestamp: str
    status: str  # Internal status: NORMAL, WATCH, UNUSUAL_CHANGE, STALE_DATA, INSUFFICIENT_DATA
    user_facing_status: str  # SAFE, WATCH, ELEVATED ALERT, STALE DATA, INSUFFICIENT DATA, PROCESSING, SYSTEM ERROR
    primary_area_km2: Optional[float] = None
    baseline_area_km2: Optional[float] = None
    area_change_km2: Optional[float] = None
    area_change_pct: Optional[float] = None
    sensor_agreement_status: Optional[str] = None
    cross_sensor_disagreement_pct: Optional[float] = None
    temporal_pair_days: Optional[int] = None
    temporal_matching_flag: Optional[str] = None
    data_quality: str = "LOW"
    staleness_days: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return {
            "alert_id": self.alert_id,
            "lake_id": self.lake_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "user_facing_status": self.user_facing_status,
            "primary_area_km2": self.primary_area_km2,
            "baseline_area_km2": self.baseline_area_km2,
            "area_change_km2": self.area_change_km2,
            "area_change_pct": self.area_change_pct,
            "sensor_agreement_status": self.sensor_agreement_status,
            "cross_sensor_disagreement_pct": self.cross_sensor_disagreement_pct,
            "temporal_pair_days": self.temporal_pair_days,
            "temporal_matching_flag": self.temporal_matching_flag,
            "data_quality": self.data_quality,
            "staleness_days": self.staleness_days,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "limitations": self.limitations,
        }


@dataclass
class PipelineRun:
    """Traceable execution record of the GlacierGuard processing pipeline."""

    run_id: str
    lake_id: str
    start_time: str
    stage: str  # Stage name
    end_time: Optional[str] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED
    input_product_ids: List[str] = field(default_factory=list)
    model_features: Optional[Dict[str, Any]] = None
    model_prediction: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline run to dictionary."""
        return {
            "run_id": self.run_id,
            "lake_id": self.lake_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stage": self.stage,
            "status": self.status,
            "input_product_ids": self.input_product_ids,
            "model_features": self.model_features,
            "model_prediction": self.model_prediction,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }
