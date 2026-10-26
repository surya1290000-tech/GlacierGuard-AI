"""
Tests for GlacierGuard Operational Risk Engine
==============================================
Validates conservative status assignments, configurable rule thresholds,
and false-SAFE prevention for stale or missing satellite observations.
"""

import datetime
from pathlib import Path
import pytest

from glacierguard.config import Settings
from glacierguard.models import ObservationRecord
from glacierguard.risk_engine import RiskEngine, evaluate_lake_status, map_status_to_user_facing


@pytest.fixture
def risk_settings(tmp_path: Path) -> Settings:
    """Fixture with standard operational settings."""
    settings = Settings()
    settings.db_path = tmp_path / "test_risk.db"
    return settings


class TestRiskEngineFalseSafePrevention:
    """Critical tests verifying that stale, missing, or uncertain data is NEVER marked SAFE."""

    def test_stale_data_is_never_marked_safe(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 10, 26)

        # Observation is 120 days old (older than 90d threshold)
        old_obs = ObservationRecord(
            observation_id="OBS_OLD",
            lake_id="imja_tsho",
            sensor="SENTINEL-2",
            acquisition_date="2026-06-25",
            area_50m_km2=1.765,  # Matches baseline perfectly
        )

        alert = engine.evaluate("imja_tsho", s2_obs=old_obs, reference_date=ref_date)
        assert alert.staleness_days > 90
        assert alert.user_facing_status == "STALE DATA"
        assert alert.user_facing_status != "SAFE"
        assert "90d" in alert.rationale

    def test_map_status_blocks_safe_for_stale_data(self):
        # Even if internal status is NORMAL, staleness > 90 must force STALE DATA
        user_status = map_status_to_user_facing("NORMAL", staleness_days=100)
        assert user_status == "STALE DATA"
        assert user_status != "SAFE"

        # Under 30 days, NORMAL maps to SAFE
        assert map_status_to_user_facing("NORMAL", staleness_days=10) == "SAFE"

    def test_missing_observations_produce_insufficient_data(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        alert = engine.evaluate("imja_tsho", s2_obs=None, s1_obs=None)

        assert alert.status == "INSUFFICIENT_DATA"
        assert alert.user_facing_status == "INSUFFICIENT DATA"
        assert alert.user_facing_status != "SAFE"


class TestOperationalRiskRules:
    """Test standard operational rule hierarchy."""

    def test_elevated_alert_on_large_area_change(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 9, 30)

        # Lower Barun with +45% expansion
        obs = ObservationRecord(
            observation_id="OBS_LB",
            lake_id="lower_barun",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-28",
            area_50m_km2=0.609,  # baseline is 0.419 -> +45.3%
        )

        alert = engine.evaluate("lower_barun", s2_obs=obs, reference_date=ref_date)
        assert alert.status == "UNUSUAL_CHANGE"
        assert alert.user_facing_status == "ELEVATED ALERT"
        assert alert.area_change_pct is not None
        assert alert.area_change_pct > 20.0
        assert "calving" in alert.rationale.lower() or "substantial change" in alert.rationale.lower()

    def test_watch_on_sar_gorge_layover_failure(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 9, 30)

        # Tsho Rolpa has known SAR failure in steep gorge
        obs_s2 = ObservationRecord(
            observation_id="OBS_TR_S2",
            lake_id="tsho_rolpa",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-28",
            area_50m_km2=1.735,
        )
        obs_s1 = ObservationRecord(
            observation_id="OBS_TR_S1",
            lake_id="tsho_rolpa",
            sensor="SENTINEL-1",
            acquisition_date="2026-09-28",
            area_50m_km2=0.083,  # severe geometric layover
        )

        alert = engine.evaluate(
            "tsho_rolpa",
            s2_obs=obs_s2,
            s1_obs=obs_s1,
            reference_date=ref_date,
        )
        assert alert.status == "WATCH"
        assert alert.user_facing_status == "WATCH"
        assert "layover" in alert.rationale.lower() or "single-sensor" in alert.rationale.lower()

    def test_watch_on_temporally_unmatched_pair(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 9, 30)

        # Thulagi: optical is from May 2026, SAR is from Sep 2026 (>100 days gap)
        obs_s2 = ObservationRecord(
            observation_id="OBS_TH_S2",
            lake_id="thulagi",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-20",  # fresh enough (10d)
            area_50m_km2=0.993,
        )
        obs_s1 = ObservationRecord(
            observation_id="OBS_TH_S1",
            lake_id="thulagi",
            sensor="SENTINEL-1",
            acquisition_date="2026-09-02",  # 18 days difference
            area_50m_km2=0.638,
        )

        alert = engine.evaluate("thulagi", s2_obs=obs_s2, s1_obs=obs_s1, reference_date=ref_date)
        assert alert.status == "WATCH"
        assert alert.user_facing_status == "WATCH"
        assert alert.temporal_pair_days == 18
        assert "TEMPORALLY_UNMATCHED" in alert.temporal_matching_flag

    def test_watch_on_aging_observation(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 9, 30)

        # 45 days old (aging: 31-90 days)
        obs = ObservationRecord(
            observation_id="OBS_AGING",
            lake_id="imja_tsho",
            sensor="SENTINEL-2",
            acquisition_date="2026-08-16",
            area_50m_km2=1.765,
        )

        alert = engine.evaluate("imja_tsho", s2_obs=obs, reference_date=ref_date)
        assert alert.status == "WATCH"
        assert alert.user_facing_status == "WATCH"
        assert "aging" in alert.rationale.lower()

    def test_safe_on_fresh_stable_lake(self, risk_settings: Settings):
        engine = RiskEngine(risk_settings)
        ref_date = datetime.date(2026, 9, 30)

        # Fresh (2 days old), area within +0.03% of baseline
        obs_s2 = ObservationRecord(
            observation_id="OBS_IMJA_S2",
            lake_id="imja_tsho",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-28",
            area_50m_km2=1.765,
        )
        obs_s1 = ObservationRecord(
            observation_id="OBS_IMJA_S1",
            lake_id="imja_tsho",
            sensor="SENTINEL-1",
            acquisition_date="2026-09-28",
            area_50m_km2=1.633,
        )

        alert = engine.evaluate("imja_tsho", s2_obs=obs_s2, s1_obs=obs_s1, reference_date=ref_date)
        assert alert.status == "NORMAL"
        assert alert.user_facing_status == "SAFE"
        assert alert.area_change_pct is not None
        assert abs(alert.area_change_pct) < 1.0
        assert "stable" in alert.rationale.lower()
