"""
Tests for GlacierGuard SQLite Persistence Layer
==============================================
Validates schema creation, CRUD operations, indexing, deduplication,
and seeding from baseline QA CSV.
"""

from pathlib import Path
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
        assert alerts["imja_tsho"].user_facing_status == "SAFE"
        assert alerts["lower_barun"].user_facing_status == "ELEVATED ALERT"

        # Re-seeding when data already exists should return 0 (idempotent)
        reseed = seed_from_qa_csv(db_path=test_db_path)
        assert reseed == 0
