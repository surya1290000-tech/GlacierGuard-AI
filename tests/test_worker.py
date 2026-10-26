"""
Tests for GlacierGuard Background Worker
========================================
Validates worker initialization, single surveillance cycle execution,
idempotency, and command-line handling.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from glacierguard.config import Settings
from glacierguard.db import get_pipeline_runs
from glacierguard.worker import MonitoringWorker


@pytest.fixture
def worker_settings(tmp_path: Path) -> Settings:
    """Fixture with isolated worker settings."""
    settings = Settings()
    settings.db_path = tmp_path / "test_worker.db"
    return settings


class TestMonitoringWorker:
    """Test worker execution and scheduling."""

    def test_worker_setup_seeds_database(self, worker_settings: Settings):
        worker = MonitoringWorker(worker_settings)
        worker.setup()

        assert worker_settings.db_path.is_file()
        from glacierguard.db import get_latest_monitoring_alerts
        alerts = get_latest_monitoring_alerts(worker_settings.db_path)
        assert len(alerts) == 5

    def test_worker_run_once_executes_cycle(self, worker_settings: Settings):
        worker = MonitoringWorker(worker_settings)

        with patch.object(worker.pipeline.cdse_client, "discover_and_persist", return_value=(0, [])):
            runs = worker.run_once()

            assert len(runs) == 5
            assert all(r.status == "SUCCESS" for r in runs)

            # Verify runs persisted to DB
            stored_runs = get_pipeline_runs(db_path=worker_settings.db_path)
            assert len(stored_runs) == 5
