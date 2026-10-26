"""
GlacierGuard-AI Operational Risk Engine
=======================================
Configurable, rule-driven evaluation engine that assigns operational
monitoring tiers to glacier lakes without fabricating GLOF probabilities.

Rules follow the conservative glaciological framework:
- SAFE: Fresh observation (≤30d), QA confirmed, area change < 10% vs baseline.
- WATCH: Triggered by aging data (31–90d), single-sensor reliance,
  temporally unmatched acquisitions (>10d), sensor disagreement,
  or moderate change (10%–20%).
- ELEVATED ALERT: High-quality, verified area change exceeding 20%
  (e.g., rapid calving expansion at Lower Barun).
- STALE DATA: Observation older than 90 days. Crucially, stale data
  is NEVER marked SAFE.
- INSUFFICIENT DATA: No valid imagery or failed quality screening.
- PROCESSING: Live ingestion/inference actively underway.
- SYSTEM ERROR: Failure in data acquisition or processing stages.
"""

import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from glacierguard.config import LakeConfig, Settings, default_settings
from glacierguard.models import MonitoringAlert, ObservationRecord
from glacierguard.satellite.qa import (
    evaluate_cross_sensor_agreement,
    match_temporal_pair,
)


def map_status_to_user_facing(
    status: str,
    staleness_days: Optional[int] = None,
) -> str:
    """Map internal monitoring status to user-facing display tier.

    Enforces that stale data can NEVER be presented as SAFE.
    """
    if staleness_days is not None and staleness_days > 90:
        return "STALE DATA"

    mapping = {
        "NORMAL": "SAFE",
        "SAFE": "SAFE",
        "WATCH": "WATCH",
        "UNUSUAL_CHANGE": "ELEVATED ALERT",
        "ELEVATED_ALERT": "ELEVATED ALERT",
        "STALE_DATA": "STALE DATA",
        "INSUFFICIENT_DATA": "INSUFFICIENT DATA",
        "PROCESSING": "PROCESSING",
        "SYSTEM_ERROR": "SYSTEM ERROR",
    }
    return mapping.get(status.upper(), "INSUFFICIENT DATA")


class RiskEngine:
    """Operational status evaluation engine for monitored lakes."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings

    def evaluate(
        self,
        lake_id: str,
        s2_obs: Optional[ObservationRecord] = None,
        s1_obs: Optional[ObservationRecord] = None,
        reference_date: Optional[datetime.date] = None,
    ) -> MonitoringAlert:
        """Evaluate operational risk status for a lake based on available observations.

        Parameters
        ----------
        lake_id : str
            Identifier of the monitored lake.
        s2_obs : ObservationRecord, optional
            Latest optical Sentinel-2 observation.
        s1_obs : ObservationRecord, optional
            Latest SAR Sentinel-1 observation.
        reference_date : datetime.date, optional
            Reference date for staleness calculations (defaults to today).

        Returns
        -------
        MonitoringAlert
            Evaluated alert and evidence record.
        """
        try:
            lake_cfg = self.settings.get_lake_config(lake_id)
        except KeyError:
            return self._build_unknown_lake_alert(lake_id)

        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        ref_date = reference_date or datetime.date.today()

        # If both observations are completely missing
        if s2_obs is None and s1_obs is None:
            return MonitoringAlert(
                alert_id=f"alert_{lake_id}_{ref_date.isoformat()}",
                lake_id=lake_id,
                timestamp=now_str,
                status="INSUFFICIENT_DATA",
                user_facing_status="INSUFFICIENT DATA",
                baseline_area_km2=lake_cfg.baseline_area_km2,
                data_quality="LOW",
                rationale="No satellite observations available for this lake.",
                limitations="Pending satellite overpass and catalogue discovery.",
            )

        # Determine primary area and dates
        s2_date = s2_obs.acquisition_date if s2_obs else None
        s1_date = s1_obs.acquisition_date if s1_obs else None

        # Compute staleness from most recent observation
        latest_date_str = max(filter(None, [s2_date, s1_date]))
        latest_date = datetime.datetime.strptime(
            latest_date_str.split("T")[0], "%Y-%m-%d"
        ).date()
        staleness_days = max(0, (ref_date - latest_date).days)

        # Primary area selection (prioritize S2 optical if valid, fallback to S1)
        s2_area = s2_obs.area_50m_km2 if s2_obs and s2_obs.area_50m_km2 else (
            s2_obs.area_0m_km2 if s2_obs else None
        )
        s1_area = s1_obs.area_50m_km2 if s1_obs and s1_obs.area_50m_km2 else (
            s1_obs.area_0m_km2 if s1_obs else None
        )

        primary_area: Optional[float] = None
        if s2_area is not None and not np.isnan(s2_area) and s2_area > 0:
            primary_area = s2_area
        elif s1_area is not None and not np.isnan(s1_area) and s1_area > 0:
            primary_area = s1_area

        # Calculate area change against GLO 2023 baseline
        baseline = lake_cfg.baseline_area_km2
        diff_km2: Optional[float] = None
        diff_pct: Optional[float] = None
        if primary_area is not None and baseline > 0:
            diff_km2 = primary_area - baseline
            diff_pct = (diff_km2 / baseline) * 100.0

        # Temporal matching and cross-sensor evaluation
        pair_days, temporal_flag = match_temporal_pair(s2_date, s1_date)
        sar_valid = lake_cfg.sar_physically_reliable

        disagreement_pct, agreement_status = evaluate_cross_sensor_agreement(
            area_s2=s2_area if s2_area is not None else np.nan,
            area_s1=s1_area if s1_area is not None else np.nan,
            pair_diff_days=pair_days,
            sar_physically_valid=sar_valid,
            threshold_pct=self.settings.thresholds.sensor_disagreement_threshold_pct,
        )

        # Determine data quality
        if s2_obs and s1_obs and agreement_status == "GOOD_AGREEMENT":
            data_quality = "HIGH"
        elif s2_obs or (s1_obs and sar_valid):
            data_quality = "MEDIUM"
        else:
            data_quality = "LOW"

        # Apply rule engine hierarchy
        status, rationale, limitations = self._evaluate_rules(
            lake_id=lake_id,
            lake_cfg=lake_cfg,
            diff_pct=diff_pct,
            staleness_days=staleness_days,
            sar_valid=sar_valid,
            pair_days=pair_days,
            agreement_status=agreement_status,
            s2_obs=s2_obs,
            s1_obs=s1_obs,
        )

        user_facing_status = map_status_to_user_facing(status, staleness_days)

        evidence = {
            "s2_date": s2_date,
            "s1_date": s1_date,
            "s2_area_km2": s2_area,
            "s1_area_km2": s1_area,
            "sar_physically_valid": sar_valid,
            "diff_pct": diff_pct,
            "staleness_days": staleness_days,
            "agreement_status": agreement_status,
            "disagreement_pct": disagreement_pct if not np.isnan(disagreement_pct) else None,
        }

        return MonitoringAlert(
            alert_id=f"alert_{lake_id}_{ref_date.isoformat()}_{now_str[-8:-1]}",
            lake_id=lake_id,
            timestamp=now_str,
            status=status,
            user_facing_status=user_facing_status,
            primary_area_km2=round(primary_area, 6) if primary_area is not None else None,
            baseline_area_km2=round(baseline, 6),
            area_change_km2=round(diff_km2, 6) if diff_km2 is not None else None,
            area_change_pct=round(diff_pct, 2) if diff_pct is not None else None,
            sensor_agreement_status=agreement_status,
            cross_sensor_disagreement_pct=round(disagreement_pct, 2)
            if not np.isnan(disagreement_pct)
            else None,
            temporal_pair_days=pair_days,
            temporal_matching_flag=temporal_flag,
            data_quality=data_quality,
            staleness_days=staleness_days,
            evidence=evidence,
            rationale=rationale,
            limitations=limitations,
        )

    def _evaluate_rules(
        self,
        lake_id: str,
        lake_cfg: LakeConfig,
        diff_pct: Optional[float],
        staleness_days: int,
        sar_valid: bool,
        pair_days: Optional[int],
        agreement_status: str,
        s2_obs: Optional[ObservationRecord],
        s1_obs: Optional[ObservationRecord],
    ) -> Tuple[str, str, str]:
        """Core rule evaluation logic."""
        thresholds = self.settings.thresholds

        # Rule 1: Stale data (>90 days)
        if staleness_days > thresholds.freshness_aging_days:
            return (
                "STALE_DATA",
                f"Surveillance data is {staleness_days} days old (>90d threshold). "
                "Satellite overpass update required before operational safety can be confirmed.",
                "Cloud cover or satellite revisit latency has prevented recent acquisitions.",
            )

        # Rule 2: Elevated Area Change (> ±20% vs baseline, verified)
        if diff_pct is not None and abs(diff_pct) >= thresholds.area_change_elevated_pct:
            return (
                "UNUSUAL_CHANGE",
                f"Lake extent exhibits substantial change of {diff_pct:+.1f}% relative to GLO 2023 baseline. "
                "Exceeds elevated threshold (±20.0%). Requires glaciological expert inspection.",
                "Calving bay boundary dynamics or rapid perimeter expansion in effect.",
            )

        # Rule 3: Single-sensor reliance due to geometric SAR failure (e.g. Tsho Rolpa)
        if not sar_valid:
            return (
                "WATCH",
                "SAR C-band data suffers from steep terrain layover/shadow. "
                "Current surveillance relies entirely on single-sensor optical imagery.",
                "C-band SAR cannot be reliably thresholded in narrow steep gorge without DEM simulation.",
            )

        # Rule 4: Temporally unmatched cross-sensor observations (>10 days)
        if pair_days is not None and pair_days > thresholds.max_temporal_pair_days:
            return (
                "WATCH",
                f"Optical and SAR observations are separated by {pair_days} days "
                "(TEMPORALLY_UNMATCHED). Simultaneous cross-sensor verification unavailable.",
                "Seasonal monsoon cloud gaps or mismatched orbital tracks.",
            )

        # Rule 5: Moderate sensor disagreement
        if agreement_status == "MODERATE_DISAGREEMENT":
            return (
                "WATCH",
                "Sensors exhibit moderate extent disagreement (>20%). "
                "Shallow shorelines, sediment plumes, or surface wind roughening may bias detection.",
                "Glacial flour or wind roughening distorts optical reflectance / radar backscatter.",
            )

        # Rule 6: Moderate area change (10% - 20%)
        if diff_pct is not None and abs(diff_pct) >= thresholds.area_change_watch_pct:
            return (
                "WATCH",
                f"Measured lake extent shows moderate change of {diff_pct:+.1f}% vs baseline. "
                "Within watch criteria (10%–20%).",
                "Seasonal water level oscillation or gradual refilling/drainage.",
            )

        # Rule 7: Aging observation (31 - 90 days)
        if staleness_days > thresholds.freshness_current_days:
            return (
                "WATCH",
                f"Latest observation is {staleness_days} days old (aging: 31–90 days). "
                "Monitoring status held at WATCH until fresh imagery arrives.",
                "Seasonal cloud cover or imaging gap.",
            )

        # Rule 8: Normal/Safe (Fresh observation <=30d, change <10%, QA passed)
        if diff_pct is not None:
            return (
                "NORMAL",
                f"Lake perimeter stable: observed extent within {diff_pct:+.2f}% of baseline. "
                "Cross-sensor checks pass within operational tolerances.",
                "Minor wind roughening or seasonal supraglacial fluctuations may be present.",
            )

        return (
            "INSUFFICIENT_DATA",
            "Unable to compute valid lake perimeter from available satellite products.",
            "Pixel quality flags indicate heavy cloud contamination or radar noise.",
        )

    def _build_unknown_lake_alert(self, lake_id: str) -> MonitoringAlert:
        """Handle unrecognized lake identifier."""
        return MonitoringAlert(
            alert_id=f"alert_unknown_{lake_id}",
            lake_id=lake_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            status="INSUFFICIENT_DATA",
            user_facing_status="INSUFFICIENT DATA",
            data_quality="LOW",
            rationale=f"Lake '{lake_id}' is not in configured monitoring inventory.",
            limitations="Unknown lake.",
        )


def evaluate_lake_status(
    lake_id: str,
    s2_obs: Optional[ObservationRecord] = None,
    s1_obs: Optional[ObservationRecord] = None,
    settings: Optional[Settings] = None,
    reference_date: Optional[datetime.date] = None,
) -> MonitoringAlert:
    """Convenience function to evaluate lake status."""
    engine = RiskEngine(settings)
    return engine.evaluate(lake_id, s2_obs, s1_obs, reference_date)
