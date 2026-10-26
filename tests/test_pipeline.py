"""
Tests for GlacierGuard Processing Pipeline
==========================================
Validates end-to-end pipeline stage progression, provenance recording,
frozen model boundary handling, and failure safety.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from glacierguard.config import Settings
from glacierguard.models import ObservationRecord, PipelineRun, SatelliteProduct
from glacierguard.pipeline import ProcessingPipeline


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    """Fixture with isolated test database."""
    settings = Settings()
    settings.db_path = tmp_path / "test_pipeline.db"
    return settings


class TestProcessingPipelineLifecycle:
    """Test full pipeline lifecycle and stage transitions."""

    def test_run_for_lake_success_path(self, pipeline_settings: Settings):
        pipeline = ProcessingPipeline(pipeline_settings)

        mock_prods = [
            SatelliteProduct(
                product_id="MOCK_S2_001",
                collection="sentinel-2-l2a",
                acquisition_date="2026-09-28T05:07:01Z",
                bbox=[86.9, 27.8, 86.95, 27.9],
                lake_id="imja_tsho",
            )
        ]
        mock_obs = [
            ObservationRecord(
                observation_id="MOCK_OBS_001",
                lake_id="imja_tsho",
                sensor="SENTINEL-2",
                acquisition_date="2026-09-28",
                area_50m_km2=1.765,
                valid_frac=1.0,
            )
        ]

        run = pipeline.run_for_lake(
            lake_id="imja_tsho",
            mock_products=mock_prods,
            mock_observations=mock_obs,
        )

        assert isinstance(run, PipelineRun)
        assert run.status == "SUCCESS"
        assert run.stage == "COMPLETED"
        assert run.end_time is not None
        assert "MOCK_S2_001" in run.input_product_ids
        assert run.model_features["lake_id"] == "imja_tsho"
        assert "alert_id" in run.metadata

    def test_pipeline_failure_preserves_error_trace(self, pipeline_settings: Settings):
        pipeline = ProcessingPipeline(pipeline_settings)

        # Force a failure during DISCOVER
        with patch.object(pipeline.cdse_client, "discover_and_persist", side_effect=ValueError("Test network fault")):
            run = pipeline.run_for_lake("imja_tsho")

            assert run.status == "FAILED"
            assert run.error_message is not None
            assert "ValueError" in run.error_message
            assert run.end_time is not None
            assert "traceback" in run.metadata

    def test_run_all_lakes_processes_canonical_roster(self, pipeline_settings: Settings):
        pipeline = ProcessingPipeline(pipeline_settings)

        # Mock discovery so network calls are avoided
        with patch.object(pipeline.cdse_client, "discover_and_persist", return_value=(0, [])):
            runs = pipeline.run_all_lakes()

            assert len(runs) == 5
            processed_lakes = [r.lake_id for r in runs]
            assert processed_lakes == pipeline_settings.list_lake_ids()
            assert all(r.status == "SUCCESS" for r in runs)
