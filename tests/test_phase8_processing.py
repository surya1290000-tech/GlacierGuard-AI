"""Deterministic fixture tests for the product-to-observation boundary."""
from pathlib import Path
import numpy as np
import pytest

from glacierguard.config import Settings
from glacierguard.db import get_observations, get_product_processing
from glacierguard.models import SatelliteProduct
from glacierguard.pipeline import ProcessingPipeline
from glacierguard.satellite.access import AssetAccessError, CDSEAssetClient


def _raster(path: Path, values: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype="float32", crs="EPSG:32645", transform=from_origin(500000, 3000000, 10, 10)) as dst:
        dst.write(values.astype("float32"), 1)


def test_local_s2_fixture_creates_a_traceable_observation(tmp_path: Path):
    green, swir, scl = (tmp_path / "B03.tif", tmp_path / "B11.tif", tmp_path / "SCL.tif")
    _raster(green, np.full((4, 4), 8000)); _raster(swir, np.full((4, 4), 1000)); _raster(scl, np.full((4, 4), 4))
    settings = Settings(); settings.db_path = tmp_path / "phase8.db"; settings.processing_cache_dir = tmp_path / "cache"
    product = SatelliteProduct("FIXTURE_S2", "sentinel-2-l2a", "2026-09-28T05:00:00Z", [0, 0, 1, 1], "imja_tsho", metadata={"assets": {
        "B03": {"href": green.as_uri()}, "B11": {"href": swir.as_uri()}, "SCL": {"href": scl.as_uri()},
    }})
    pipeline = ProcessingPipeline(settings)
    run = pipeline.run_for_lake("imja_tsho", mock_products=[product])
    observations = get_observations("imja_tsho", db_path=settings.db_path)
    assert run.status == "SUCCESS"
    assert len(observations) == 1
    assert observations[0].quality_status == "QA_VALID"
    assert observations[0].area_0m_km2 == pytest.approx(0.0016)
    assert Path(observations[0].artifact_path).is_file()
    assert get_product_processing("FIXTURE_S2", "s2-mndwi-scl-v1", settings.db_path)["status"] == "COMPLETED"


def test_http_asset_without_credentials_is_an_explicit_access_failure(tmp_path: Path):
    settings = Settings(); settings.processing_cache_dir = tmp_path
    settings.client_id = None; settings.client_secret = None
    with pytest.raises(AssetAccessError, match="credentials"):
        CDSEAssetClient(settings).fetch("P", "B03", {"href": "https://example.invalid/B03.tif"})
