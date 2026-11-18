"""
Tests for GlacierGuard SQLite Persistence Layer
==============================================
Validates schema creation, CRUD operations, indexing, deduplication,
and seeding from baseline QA CSV.
"""

from pathlib import Path
from typing import Optional
import pytest

from glacierguard.db import (
    get_latest_monitoring_alerts,
    get_monitoring_history,
    get_observations,
    get_pipeline_runs,
    get_satellite_products,
    init_db,
    product_exists,
    save_monitoring_alert,
    save_observation,
    save_pipeline_run,
    save_satellite_products,
    seed_from_qa_csv,
)
from glacierguard.models import (
    MonitoringAlert,
    ObservationRecord,
    PipelineRun,
    SatelliteProduct,
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Fixture providing a fresh SQLite database path."""
    db_file = tmp_path / "test_persistence.db"
    init_db(db_file)
    return db_file


class TestDatabaseSchemaAndInit:
    """Test schema creation and table initialization."""

    def test_init_db_creates_tables(self, test_db_path: Path):
        assert test_db_path.is_file()
        import sqlite3
        conn = sqlite3.connect(str(test_db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        conn.close()

        expected = {
            "satellite_products",
            "observations",
            "pipeline_runs",
            "monitoring_alerts",
        }
        assert expected.issubset(table_names)


class TestSatelliteProductsCRUD:
    """Test satellite products persistence and deduplication."""

    def test_save_and_retrieve_products(self, test_db_path: Path):
        prod = SatelliteProduct(
            product_id="TEST_PROD_001",
            collection="sentinel-2-l2a",
            acquisition_date="2026-09-28T05:07:01Z",
            bbox=[86.9, 27.8, 86.95, 27.9],
            lake_id="imja_tsho",
            cloud_cover_pct=1.5,
            platform="SENTINEL-2A",
            orbit_direction="DESCENDING",
        )

        inserted = save_satellite_products([prod], test_db_path)
        assert inserted == 1
        assert product_exists("TEST_PROD_001", test_db_path) is True
        assert product_exists("NON_EXISTENT", test_db_path) is False

        # Retrieve
        retrieved = get_satellite_products("imja_tsho", limit=10, db_path=test_db_path)
        assert len(retrieved) == 1
        assert retrieved[0].product_id == "TEST_PROD_001"
        assert retrieved[0].cloud_cover_pct == pytest.approx(1.5)

    def test_duplicate_products_ignored(self, test_db_path: Path):
        prod = SatelliteProduct(
            product_id="TEST_PROD_DUP",
            collection="sentinel-1-grd",
            acquisition_date="2026-09-28T12:00:00Z",
            bbox=[86.9, 27.8, 86.95, 27.9],
        )

        first_count = save_satellite_products([prod], test_db_path)
        second_count = save_satellite_products([prod], test_db_path)
        assert first_count == 1
        assert second_count == 0


class TestObservationsAndAlertsCRUD:
    """Test observation and alert records."""

    def test_save_and_get_observation(self, test_db_path: Path):
        obs = ObservationRecord(
            observation_id="OBS_001",
            lake_id="imja_tsho",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-28",
            area_0m_km2=1.765,
            area_50m_km2=2.150,
            valid_frac=1.0,
            cloud_frac=0.0,
        )
        save_observation(obs, test_db_path)

        all_obs = get_observations("imja_tsho", limit=5, db_path=test_db_path)
        assert len(all_obs) == 1
        assert all_obs[0].observation_id == "OBS_001"
        assert all_obs[0].area_50m_km2 == pytest.approx(2.150)

    def test_save_and_get_alerts(self, test_db_path: Path):
        alert = MonitoringAlert(
            alert_id="ALERT_001",
            lake_id="lower_barun",
            timestamp="2026-09-28T12:00:00Z",
            status="UNUSUAL_CHANGE",
            user_facing_status="ELEVATED ALERT",
            primary_area_km2=0.609,
            baseline_area_km2=0.419,
            area_change_pct=45.2,
            rationale="Substantial calving expansion",
        )
        save_monitoring_alert(alert, test_db_path)

        alerts_map = get_latest_monitoring_alerts(test_db_path)
        assert "lower_barun" in alerts_map
        assert alerts_map["lower_barun"].user_facing_status == "ELEVATED ALERT"
        assert alerts_map["lower_barun"].area_change_pct == pytest.approx(45.2)

        history = get_monitoring_history("lower_barun", limit=10, db_path=test_db_path)
        assert len(history) == 1
        assert history[0].alert_id == "ALERT_001"

    def test_save_and_get_pipeline_runs(self, test_db_path: Path):
        run = PipelineRun(
            run_id="RUN_001",
            lake_id="imja_tsho",
            start_time="2026-09-28T12:00:00Z",
            stage="COMPLETED",
            status="SUCCESS",
            input_product_ids=["PROD_A", "PROD_B"],
            model_prediction=0.0012,
        )
        save_pipeline_run(run, test_db_path)

        runs = get_pipeline_runs("imja_tsho", limit=5, db_path=test_db_path)
        assert len(runs) == 1
        assert runs[0].run_id == "RUN_001"
        assert runs[0].status == "SUCCESS"
        assert runs[0].input_product_ids == ["PROD_A", "PROD_B"]


class TestSeedFromQACSV:
    """Test seeding SQLite database from research baseline QA CSV."""

    def test_seed_from_qa_csv(self, test_db_path: Path):
        seeded = seed_from_qa_csv(db_path=test_db_path)
        assert seeded == 5  # 5 canonical lakes

        alerts = get_latest_monitoring_alerts(test_db_path)
        assert len(alerts) == 5
        assert "imja_tsho" in alerts
        assert "lower_barun" in alerts
        # Because the baseline date (2026-09-28) is >30 days old relative to current date,
        # it correctly maps to WATCH rather than presenting stale data as current SAFE.
        assert alerts["imja_tsho"].user_facing_status == "WATCH"
        assert alerts["lower_barun"].user_facing_status == "ELEVATED ALERT"

        # Verify historical research provenance
        obs = get_observations("imja_tsho", db_path=test_db_path)
        assert len(obs) > 0
        assert all(o.quality_status == "HISTORICAL_RESEARCH_BASELINE" for o in obs)
        assert all(o.provenance.get("source") == "historical_baseline_research_seed" for o in obs)

        # Re-seeding when data already exists should return 0 (idempotent)
        reseed = seed_from_qa_csv(db_path=test_db_path)
        assert reseed == 0

    @pytest.mark.parametrize(
        "days_ago,expected_user_status",
        [
            (10, "SAFE"),
            (45, "WATCH"),
            (100, "STALE DATA"),
            (None, "INSUFFICIENT DATA"),
        ],
    )
    def test_historical_baseline_staleness_status_mapping(
        self, test_db_path: Path, tmp_path: Path, days_ago: Optional[int], expected_user_status: str
    ):
        """Verify historical NORMAL baseline maps truthfully across age boundaries."""
        import datetime
        import pandas as pd

        if days_ago is not None:
            acq_date = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
        else:
            acq_date = None

        csv_content = pd.DataFrame([{
            "lake_id": "test_lake",
            "lake_name": "Test Lake",
            "glo_id": "GLO_TEST",
            "utm_epsg": 32645,
            "glo_2023_vector_ref_km2": 1.0,
            "old_hist_ref_km2": 0.8,
            "latest_s2_date": acq_date,
            "s2_valid_frac": 1.0,
            "s2_cloud_frac": 0.0,
            "s2_area_0m_km2": 1.0,
            "s2_area_50m_km2": 1.0,
            "s2_median_area_km2": 1.0,
            "s2_cv_pct": 1.0,
            "latest_s1_date": None,
            "s1_area_0m_km2": None,
            "s1_area_50m_km2": None,
            "s1_median_area_km2": None,
            "s1_cv_pct": None,
            "s1_vv_median_db": None,
            "temporal_pair_days": None,
            "temporal_matching_flag": "SINGLE_SENSOR",
            "cross_sensor_disagreement_pct": None,
            "sensor_agreement_status": "OPTICAL_ONLY",
            "primary_monitoring_area_km2": 1.0,
            "diff_from_glo_2023_km2": 0.0,
            "diff_from_glo_2023_pct": 0.0,
            "data_quality": "HIGH",
            "monitoring_status": "NORMAL",
            "status_rationale": "Stable",
            "extraction_limitations": "None",
        }])
        custom_csv = tmp_path / "test_qa.csv"
        csv_content.to_csv(custom_csv, index=False)

        custom_db = tmp_path / f"test_stale_{days_ago}.db"
        seeded = seed_from_qa_csv(csv_path=custom_csv, db_path=custom_db)
        assert seeded == 1

        alerts = get_latest_monitoring_alerts(custom_db)
        assert "test_lake" in alerts
        assert alerts["test_lake"].user_facing_status == expected_user_status
