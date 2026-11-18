"""
Phase 8.1: Production-Correct Satellite Asset Processing Integration Tests
==========================================================================
Deterministic tests verifying:
1. CDSE STAC asset resolution (B03_20m, B03_10m, B11_20m, SCL_20m) and 20m grid reprojection.
2. Protocol-aware asset access (file://, https://, s3://, unsupported protocols, content validation).
3. Geographically correct lake extraction (0m core prior vs 50m envelope, exclusion of outside water).
4. Sentinel-1 radiometric validation (dB, linear intensity, raw DN rejection, lake ROI layover, Tsho Rolpa restriction).
5. Sentinel-2 SCL QA (cloud shadows, clouds, snow/ice exclusion from water, clear fraction threshold).
6. Pipeline run outcome semantics (COMPLETED_WITH_NEW_OBSERVATION, COMPLETED_NO_NEW_ELIGIBLE_DATA, etc.).
7. Observation retention across failed runs and dashboard data loader integration.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pytest

from glacierguard.config import Settings
from glacierguard.dashboard_data import load_dashboard_data_from_db
from glacierguard.db import get_observations, get_product_processing, save_observation
from glacierguard.models import ObservationRecord, SatelliteProduct
from glacierguard.pipeline import ProcessingPipeline
from glacierguard.satellite.access import AssetAccessError, CDSEAssetClient
from glacierguard.satellite.lake_geometry import rasterize_lake_roi
from glacierguard.satellite.processing import (
    RasterProcessingError,
    process_sentinel1,
    process_sentinel2,
)


def _create_raster(
    path: Path,
    values: np.ndarray,
    origin: Tuple[float, float] = (492000.0, 3086500.0),
    res: float = 20.0,
    crs: str = "EPSG:32645",
) -> None:
    """Create a deterministic test GeoTIFF."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(values.astype("float32"), 1)


class TestSentinel2AssetResolutionAndReprojection:
    """Test STAC asset key resolution and 20m grid alignment."""

    def test_sentinel2_stac_resolution_and_grid_reprojection(self, tmp_path: Path):
        """Test B03_10m reprojection onto B11_20m reference grid."""
        # B03 at 10m resolution (80 rows x 100 cols)
        b03_10m_path = tmp_path / "B03_10m.tif"
        _create_raster(b03_10m_path, np.full((80, 100), 8000), res=10.0)

        # B11 at 20m resolution (40 rows x 50 cols)
        b11_20m_path = tmp_path / "B11_20m.tif"
        _create_raster(b11_20m_path, np.full((40, 50), 1000), res=20.0)

        # SCL at 20m resolution (40 rows x 50 cols) - class 6 is water
        scl_20m_path = tmp_path / "SCL_20m.tif"
        _create_raster(scl_20m_path, np.full((40, 50), 6), res=20.0)

        paths = {
            "B03_10m": b03_10m_path,
            "B11_20m": b11_20m_path,
            "SCL_20m": scl_20m_path,
        }

        mask, metrics = process_sentinel2(paths, lake_id="imja_tsho")

        assert mask.shape == (40, 50)
        assert metrics["crs"] == "EPSG:32645"
        assert metrics["resolution_m"] == 20.0
        assert metrics["area_0m_km2"] > 0.4
        assert metrics["area_50m_km2"] >= metrics["area_0m_km2"]
        assert metrics["geometry_source"] == "GLO_v1.02_2023"

    def test_missing_required_s2_asset_raises_error(self, tmp_path: Path):
        """Verify missing bands are detected without guessing."""
        b03_path = tmp_path / "B03.tif"
        _create_raster(b03_path, np.full((20, 20), 5000))

        with pytest.raises(RasterProcessingError, match="Missing Sentinel-2 required bands"):
            process_sentinel2({"B03_20m": b03_path}, lake_id="imja_tsho")


class TestLakeAreaExtractionGeographicCorrectness:
    """Critical scientific requirement: only pixels inside lake ROI count toward lake area."""

    def test_outside_water_pixels_do_not_contribute_to_lake_area(self, tmp_path: Path):
        """Input raster contains:
        - Water pixels inside monitored lake ROI.
        - Land pixels inside lake ROI.
        - Unrelated water pixels OUTSIDE the lake ROI.

        Assert: Only water pixels within the lake ROI contribute to the extracted lake area.
        """
        shape = (40, 50)
        res = 20.0
        origin = (492000.0, 3086500.0)
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        trans = from_origin(origin[0], origin[1], res, res)
        m0, m_buf, _ = rasterize_lake_roi("imja_tsho", shape, trans, "EPSG:32645")

        outside_lake = ~m_buf
        assert outside_lake.sum() > 0, "Fixture must have pixels outside lake ROI"
        assert m0.sum() > 0, "Fixture must have pixels inside core lake ROI"

        # Construct MNDWI and SCL arrays:
        # 1. Inside m0: 60% water, 40% land
        # 2. Outside m_buf: 100% water (simulating a separate lake or river)
        # 3. Buffer margin: water
        green = np.full(shape, 2000.0, dtype="float32")
        swir = np.full(shape, 5000.0, dtype="float32")  # MNDWI < 0 (land) by default
        scl = np.full(shape, 4.0, dtype="float32")      # Vegetation / land by default

        # Put open water in core lake (first 600 pixels of m0)
        m0_indices = np.where(m0)
        water_in_m0_count = min(600, len(m0_indices[0]))
        water_m0_idx = (m0_indices[0][:water_in_m0_count], m0_indices[1][:water_in_m0_count])
        green[water_m0_idx] = 8000.0
        swir[water_m0_idx] = 1000.0   # MNDWI = (8000-1000)/(8000+1000) = 0.77 > 0
        scl[water_m0_idx] = 6.0       # SCL = 6 (water)

        # Put unrelated water OUTSIDE the lake ROI
        green[outside_lake] = 9000.0
        swir[outside_lake] = 800.0    # Strong water signal
        scl[outside_lake] = 6.0       # SCL = 6 (water)

        b03_p = tmp_path / "B03_20m.tif"
        b11_p = tmp_path / "B11_20m.tif"
        scl_p = tmp_path / "SCL_20m.tif"

        _create_raster(b03_p, green, origin=origin, res=res)
        _create_raster(b11_p, swir, origin=origin, res=res)
        _create_raster(scl_p, scl, origin=origin, res=res)

        paths = {"B03_20m": b03_p, "B11_20m": b11_p, "SCL_20m": scl_p}
        mask, metrics = process_sentinel2(paths, lake_id="imja_tsho")

        # Pixel area in km² for 20m x 20m
        pixel_km2 = (res * res) / 1e6

        # Expected core lake area
        expected_core_area_km2 = water_in_m0_count * pixel_km2

        # Assert extracted core area matches only lake water pixels
        assert metrics["area_0m_km2"] == pytest.approx(expected_core_area_km2, abs=1e-5)

        # Assert that outside water was NOT added to the lake area
        total_water_in_raster_km2 = ((green > swir) & (scl == 6)).sum() * pixel_km2
        assert metrics["area_50m_km2"] < total_water_in_raster_km2
        assert (mask & outside_lake).sum() == 0, "No outside water pixels may appear in lake mask"

    def test_non_intersecting_lake_raster_is_rejected(self, tmp_path: Path):
        """A raster that does not overlap the lake ROI is rejected."""
        # Coordinates far from Imja Tsho
        b03 = tmp_path / "B03.tif"
        b11 = tmp_path / "B11.tif"
        scl = tmp_path / "SCL.tif"
        _create_raster(b03, np.full((10, 10), 5000), origin=(100000.0, 2000000.0))
        _create_raster(b11, np.full((10, 10), 1000), origin=(100000.0, 2000000.0))
        _create_raster(scl, np.full((10, 10), 6), origin=(100000.0, 2000000.0))

        paths = {"B03_20m": b03, "B11_20m": b11, "SCL_20m": scl}
        with pytest.raises(RasterProcessingError, match="does not intersect"):
            process_sentinel2(paths, lake_id="imja_tsho")


class TestProtocolAwareAssetAccess:
    """Test protocol handling for file://, https://, s3://, and unsupported schemes."""

    def test_file_uri_access_succeeds(self, tmp_path: Path):
        raster_p = tmp_path / "valid.tif"
        _create_raster(raster_p, np.full((5, 5), 10.0))

        client = CDSEAssetClient(Settings())
        accessed = client.fetch("PROD_1", "B03", {"href": raster_p.as_uri()})
        assert accessed.path.is_file()
        assert accessed.byte_count > 0

    def test_http_without_credentials_raises_asset_access_error(self, tmp_path: Path):
        settings = Settings()
        settings.client_id = None
        settings.client_secret = None
        client = CDSEAssetClient(settings)

        with pytest.raises(AssetAccessError, match="credentials"):
            client.fetch("PROD_1", "B03", {"href": "https://catalogue.dataspace.copernicus.eu/band.tif"})

    def test_s3_without_credentials_raises_asset_access_error(self, tmp_path: Path):
        settings = Settings()
        client = CDSEAssetClient(settings)

        # Clear any environment S3 keys
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AssetAccessError, match="CDSE S3 credentials"):
                client.fetch("PROD_1", "B03", {"href": "s3://eodata/Sentinel-2/band.tif"})

    def test_unsupported_protocol_fails_explicitly(self, tmp_path: Path):
        client = CDSEAssetClient(Settings())
        with pytest.raises(AssetAccessError, match="Unsupported asset.*scheme"):
            client.fetch("PROD_1", "B03", {"href": "ftp://example.invalid/band.tif"})

    def test_corrupted_raster_fails_readability_validation(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.tif"
        corrupt_file.write_text("NOT_A_VALID_GEOTIFF_CONTENT")

        client = CDSEAssetClient(Settings())
        with pytest.raises(AssetAccessError, match="not a valid readable raster"):
            client.fetch("PROD_1", "B03", {"href": corrupt_file.as_uri()})


class TestSentinel1RadiometricValidation:
    """Test S1 dB vs linear vs uncalibrated DN, lake layover, and Tsho Rolpa restriction."""

    def test_s1_calibrated_db_processing(self, tmp_path: Path):
        """Input already in decibels (median between -35 and 5 dB)."""
        vv_p = tmp_path / "VV.tif"
        # Imja Tsho water typically has backscatter around -18 dB
        vv_data = np.full((40, 50), -18.0, dtype="float32")
        _create_raster(vv_p, vv_data)

        mask, metrics = process_sentinel1({"VV": vv_p}, lake_id="imja_tsho")
        assert metrics["area_0m_km2"] > 0.4
        assert metrics["vv_median_db"] == pytest.approx(-18.0)

    def test_s1_linear_intensity_conversion(self, tmp_path: Path):
        """Input in linear intensity values (0.01 = -20 dB)."""
        vv_p = tmp_path / "VV.tif"
        vv_data = np.full((40, 50), 0.01, dtype="float32")
        _create_raster(vv_p, vv_data)

        mask, metrics = process_sentinel1({"VV": vv_p}, lake_id="imja_tsho")
        assert metrics["area_0m_km2"] > 0.4
        assert metrics["vv_median_db"] < -12.0

    def test_s1_raw_uncalibrated_dn_rejected(self, tmp_path: Path):
        """Input as raw uncalibrated DN amplitude (values > 100) must be rejected."""
        vv_p = tmp_path / "VV.tif"
        vv_data = np.full((40, 50), 500.0, dtype="float32")
        _create_raster(vv_p, vv_data)

        with pytest.raises(RasterProcessingError, match="Calibration metadata or physical validity"):
            process_sentinel1({"VV": vv_p}, lake_id="imja_tsho")

    def test_s1_tsho_rolpa_restriction_requires_review(self, tmp_path: Path):
        """SAR C-band is known unreliable in steep Rolwaling gorge; must be flagged."""
        vv_p = tmp_path / "VV.tif"
        vv_data = np.full((40, 50), -18.0, dtype="float32")
        _create_raster(vv_p, vv_data)

        with pytest.raises(RasterProcessingError, match="requires review for this lake geometry"):
            process_sentinel1({"VV": vv_p}, lake_id="tsho_rolpa")

    def test_s1_layover_inside_lake_roi_rejected(self, tmp_path: Path):
        """Elevated backscatter (median > -8 dB) inside lake core ROI fails screening."""
        vv_p = tmp_path / "VV.tif"
        vv_data = np.full((40, 50), -5.0, dtype="float32")
        _create_raster(vv_p, vv_data)

        with pytest.raises(RasterProcessingError, match="geometric or wind-quality screening failed"):
            process_sentinel1({"VV": vv_p}, lake_id="imja_tsho")


class TestSentinel2QAAndSnowExclusion:
    """Test cloud, shadow, snow/ice, and valid fraction QA behavior."""

    def test_snow_ice_excluded_from_open_water(self, tmp_path: Path):
        """SCL class 11 (snow/ice) must NOT be counted as open water."""
        green = np.full((40, 50), 8000.0, dtype="float32")
        swir = np.full((40, 50), 1000.0, dtype="float32")
        scl = np.full((40, 50), 11.0, dtype="float32")  # Snow/Ice

        b03_p = tmp_path / "B03.tif"
        b11_p = tmp_path / "B11.tif"
        scl_p = tmp_path / "SCL.tif"
        _create_raster(b03_p, green)
        _create_raster(b11_p, swir)
        _create_raster(scl_p, scl)

        paths = {"B03_20m": b03_p, "B11_20m": b11_p, "SCL_20m": scl_p}
        with pytest.raises(RasterProcessingError, match="No valid water pixels extracted"):
            process_sentinel2(paths, lake_id="imja_tsho")

    def test_low_valid_fraction_rejected_by_optical_qa(self, tmp_path: Path):
        """Clouds (SCL 8) covering > 50% of lake ROI triggers QA rejection."""
        green = np.full((40, 50), 8000.0, dtype="float32")
        swir = np.full((40, 50), 1000.0, dtype="float32")
        scl = np.full((40, 50), 8.0, dtype="float32")  # Medium cloud

        b03_p = tmp_path / "B03.tif"
        b11_p = tmp_path / "B11.tif"
        scl_p = tmp_path / "SCL.tif"
        _create_raster(b03_p, green)
        _create_raster(b11_p, swir)
        _create_raster(scl_p, scl)

        paths = {"B03_20m": b03_p, "B11_20m": b11_p, "SCL_20m": scl_p}
        with pytest.raises(RasterProcessingError, match="Optical QA rejected"):
            process_sentinel2(paths, lake_id="imja_tsho")


class TestEndToEndPipelineSemanticsAndDataRetention:
    """Test full pipeline run outcomes and data loader integration."""

    def test_pipeline_outcome_with_new_observation(self, tmp_path: Path):
        """Successful product processing results in COMPLETED_WITH_NEW_OBSERVATION."""
        settings = Settings()
        settings.db_path = tmp_path / "test.db"
        settings.processing_cache_dir = tmp_path / "cache"

        b03_p = tmp_path / "B03_20m.tif"
        b11_p = tmp_path / "B11_20m.tif"
        scl_p = tmp_path / "SCL_20m.tif"
        _create_raster(b03_p, np.full((40, 50), 8000.0))
        _create_raster(b11_p, np.full((40, 50), 1000.0))
        _create_raster(scl_p, np.full((40, 50), 6.0))

        prod = SatelliteProduct(
            product_id="S2A_TEST_SUCCESS",
            collection="sentinel-2-l2a",
            acquisition_date="2026-10-15T05:00:00Z",
            bbox=[86.9, 27.88, 86.96, 27.92],
            lake_id="imja_tsho",
            metadata={"assets": {
                "B03_20m": {"href": b03_p.as_uri()},
                "B11_20m": {"href": b11_p.as_uri()},
                "SCL_20m": {"href": scl_p.as_uri()},
            }},
        )

        pipeline = ProcessingPipeline(settings)
        run = pipeline.run_for_lake("imja_tsho", mock_products=[prod])

        assert run.status == "COMPLETED_WITH_NEW_OBSERVATION"
        assert run.metadata["num_observations_created"] == 1
        assert run.metadata["num_products_discovered"] == 1

        # Verify observation persisted with provenance
        obs = get_observations("imja_tsho", db_path=settings.db_path)
        assert len(obs) == 1
        assert obs[0].quality_status == "QA_VALID"
        assert obs[0].provenance.get("geometry_source") == "GLO_v1.02_2023"
        assert obs[0].area_0m_km2 > 0.4

        # Verify dashboard data loader reads the resulting observation
        df = load_dashboard_data_from_db(db_path=settings.db_path)
        assert "imja_tsho" in df.index
        assert df.loc["imja_tsho", "latest_s2_date"] == "2026-10-15"

    def test_pipeline_outcome_no_new_eligible_data(self, tmp_path: Path):
        """Pipeline with 0 discovered products results in COMPLETED_NO_NEW_ELIGIBLE_DATA."""
        settings = Settings()
        settings.db_path = tmp_path / "test_no_data.db"
        pipeline = ProcessingPipeline(settings)

        run = pipeline.run_for_lake("imja_tsho", mock_products=[])
        assert run.status == "COMPLETED_NO_NEW_ELIGIBLE_DATA"
        assert run.metadata["num_observations_created"] == 0

    def test_pipeline_retains_last_valid_observation_on_processing_failure(self, tmp_path: Path):
        """If a newer scene fails processing, the previous valid observation is retained."""
        settings = Settings()
        settings.db_path = tmp_path / "test_retention.db"
        settings.processing_cache_dir = tmp_path / "cache"

        # Pre-seed a valid observation
        prior_obs = ObservationRecord(
            observation_id="OBS_PRIOR_VALID",
            lake_id="imja_tsho",
            sensor="SENTINEL-2",
            acquisition_date="2026-09-28",
            area_0m_km2=1.765,
            area_50m_km2=2.150,
            valid_frac=1.0,
            quality_status="QA_VALID",
        )
        save_observation(prior_obs, db_path=settings.db_path)

        # Introduce a newer product with corrupted asset
        corrupt_asset = tmp_path / "corrupt.tif"
        corrupt_asset.write_text("CORRUPTED")
        bad_prod = SatelliteProduct(
            product_id="S2A_NEW_FAILED",
            collection="sentinel-2-l2a",
            acquisition_date="2026-10-25T05:00:00Z",
            bbox=[86.9, 27.88, 86.96, 27.92],
            lake_id="imja_tsho",
            metadata={"assets": {
                "B03": {"href": corrupt_asset.as_uri()},
                "B11": {"href": corrupt_asset.as_uri()},
                "SCL": {"href": corrupt_asset.as_uri()},
            }},
        )

        pipeline = ProcessingPipeline(settings)
        run = pipeline.run_for_lake("imja_tsho", mock_products=[bad_prod])

        assert run.status == "PROCESSING_FAILED"
        assert run.metadata["operational_evaluation_used_previous_obs"] is True

        # Confirm previous valid observation remains intact
        obs = get_observations("imja_tsho", db_path=settings.db_path)
        assert len(obs) == 1
        assert obs[0].observation_id == "OBS_PRIOR_VALID"
        assert obs[0].area_0m_km2 == pytest.approx(1.765)
